#include "TracccGpuStandalone.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

struct cell_order {
    bool operator()(const traccc::io::csv::cell& lhs,
                    const traccc::io::csv::cell& rhs) const {
        if (lhs.channel1 != rhs.channel1) {
            return (lhs.channel1 < rhs.channel1);
        } else {
            return (lhs.channel0 < rhs.channel0);
        }
    }
};  // struct cell_order

std::vector<traccc::io::csv::cell> read_csv(
    const std::string &filename,
    const std::map<int64_t, uint64_t> athenaToDetrayMap,
    bool athenaIDs = false
) {
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    std::cout << "Reading cells from " << filename << std::endl;

    while (reader.read(iocell))
    {
        if (iocell.geometry_id == 0)
        {   
            std::cout << "Warning: Found cell with geometry_id = 0, "
                      << "this may indicate an issue with the input data." 
                      << std::endl;
            continue;
        }

        if (athenaIDs) {
            iocell.geometry_id = athenaToDetrayMap.at(iocell.geometry_id);
        } 

        cells.push_back(iocell);
    }

    return cells;
}

// Pack CSV cells into the raw byte layout expected by
// TracccGpuStandalone::cells_from_buffer: an 8-byte cell-count header
// followed by 5 SoA column blocks of length N (channel0, channel1,
// activation, time, module_index), matching what a real client would send.
std::vector<uint8_t> build_cell_buffer(
    const std::vector<traccc::io::csv::cell> &cells,
    const std::unordered_map<traccc::geometry_id, unsigned int> &geomIdMap)
{
    // Group by (already detray-mapped) geometry_id and sort each group by
    // channel1/channel0, mirroring the ordering the clusterization algorithm
    // expects. Sorting happens on the client side.
    std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> cellsByModule;
    for (const auto &cell : cells) {
        cellsByModule[cell.geometry_id].push_back(cell);
    }
    for (auto &[geometry_id, moduleCells] : cellsByModule) {
        std::sort(moduleCells.begin(), moduleCells.end(), ::cell_order());
    }

    const std::uint64_t num_cells = cells.size();
    std::vector<uint8_t> buffer(sizeof(std::uint64_t) + num_cells * 20u);

    std::memcpy(buffer.data(), &num_cells, sizeof(std::uint64_t));

    uint8_t *base = buffer.data() + sizeof(std::uint64_t);
    uint32_t *channel0 = reinterpret_cast<uint32_t *>(base + 0u * num_cells * 4u);
    uint32_t *channel1 = reinterpret_cast<uint32_t *>(base + 1u * num_cells * 4u);
    float *activation = reinterpret_cast<float *>(base + 2u * num_cells * 4u);
    float *time = reinterpret_cast<float *>(base + 3u * num_cells * 4u);
    uint32_t *module_index = reinterpret_cast<uint32_t *>(base + 4u * num_cells * 4u);

    std::uint64_t i = 0;
    for (const auto &[geometry_id, moduleCells] : cellsByModule) {
        auto it = geomIdMap.find(geometry_id);
        if (it == geomIdMap.end()) {
            throw std::runtime_error(
                "Could not find geometry ID (" + std::to_string(geometry_id) +
                ") in the detector description");
        }
        const unsigned int ddIndex = it->second;

        for (const auto &cell : moduleCells) {
            channel0[i] = cell.channel0;
            channel1[i] = cell.channel1;
            activation[i] = cell.value;
            time[i] = cell.timestamp;
            module_index[i] = ddIndex;
            ++i;
        }
    }

    return buffer;
}

// One-time dump of the server's geometry_id -> module_index mapping for the client
void dump_geom_id_map(
    const std::unordered_map<traccc::geometry_id, unsigned int> &geomIdMap,
    const std::string &outPath)
{
    std::ofstream out(outPath);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + outPath);
    }

    out << "{\n";
    bool first = true;
    for (const auto &[geometry_id, module_index] : geomIdMap) {
        if (!first) out << ",\n";
        first = false;
        out << "  \"" << geometry_id << "\": " << module_index;
    }
    out << "\n}\n";

    std::cout << "Wrote geometry_id -> module_index map (" << geomIdMap.size()
               << " entries) to " << outPath << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceID);
    
    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceID);

    dump_geom_id_map(
        traccc_gpu.getGeomIdMap(),
        (std::filesystem::path(event_file).parent_path() /
         "geom_id_to_module_index.json").string());

    std::vector<traccc::io::csv::cell> cells = read_csv(
        event_file, traccc_gpu.getAthenaToDetrayMap(), true);

    std::vector<uint8_t> cell_buffer = build_cell_buffer(
        cells, traccc_gpu.getGeomIdMap());

    traccc::edm::silicon_cell_collection::host cell_collection =
        traccc_gpu.cells_from_buffer(cell_buffer.data(), cell_buffer.size());

    auto traccc_result = traccc_gpu.run(std::move(cell_collection), true);

    int total_tracks = traccc_result.tracks_and_states.tracks.size();
    int excluded_non_positive_ndf = 0;
    int excluded_not_all_smoothed = 0;
    int excluded_unknown = 0;
    int excluded_no_state = 0;
    int printed_tracks = 0;

    for (size_t i = 0; i < traccc_result.tracks_and_states.tracks.size() && printed_tracks < 5; ++i)
    {
        const auto& track = traccc_result.tracks_and_states.tracks.at(i);

        auto track_fit_outcome = track.fit_outcome();

        std::cout << "Fit outcome: " << static_cast<std::underlying_type<traccc::track_fit_outcome>::type>(track_fit_outcome) << std::endl;

        if (track.constituent_links().size() < 1) {
            excluded_no_state += 1;
            continue;
        }

        const auto& fitted_params = track.params();
        traccc::scalar l0 = fitted_params.bound_local()[0];
        traccc::scalar l1 = fitted_params.bound_local()[1];
        traccc::scalar phi = fitted_params.phi();
        traccc::scalar theta = fitted_params.theta();
        traccc::scalar qop = fitted_params.qop();
        
        std::cout << "Track " << i << ": chi2 = " << track.chi2()
                  << ", ndf = " << track.ndf()
                  << ", l0 = " << l0
                  << ", l1 = " << l1
                  << ", phi = " << phi
                  << ", theta = " << theta  
                  << ", q/p = " << qop 
                  << ", time = " << fitted_params.time()
                  << std::endl;

        const auto& constituent_links = track.constituent_links();
        // for (size_t j = 0; j < constituent_links.size(); ++j)
        // {
        //     const auto& link = constituent_links[j];
            
        //     if (link.type != traccc::edm::track_constituent_link::track_state) {
        //         continue;
        //     }

        //     const auto& state = traccc_result.tracks_and_states.states.at(link.index);
        //     size_t meas_idx = state.measurement_index();

        //     std::cout << "Track is smoothed: " << state.is_smoothed() << std::endl;
        //     if (state.is_smoothed()) {
        //         std::cout << "  Filtered parameters: " << state.smoothed_params() << std::endl;
        //         std::cout << "  Smoothed covariance: " << state.smoothed_params().covariance()[0][1] << std::endl;
        //         std::cout << "  Time: " << state.smoothed_params().time() << std::endl;
        //     }
        //     // std::cout << "  Smoothed parameters: " << state.smoothed_params() << std::endl;

        //     const auto& measurement = traccc_result.measurements.at(meas_idx);

        //     std::cout << "  Measurement ID: " << measurement.identifier()
        //               << ", Detected at detray ID: " << measurement.surface_link().value()
        //               << ", Local Position: (" << measurement.local_position()[0] << ", " 
        //               << measurement.local_position()[1] << ")"
        //               << ", Local Variance: (" << measurement.local_variance()[0] << ", "
        //               << measurement.local_variance()[1] << ")"
        //               << ", Time: " << measurement.time()
        //               << ", Measurement Dimension: " << measurement.dimensions()
        //               << std::endl;
        // }
        ++printed_tracks;
    }

    // Print final exclusion statistics
    std::cout << "\n=== Track Exclusion Summary ===" << std::endl;
    std::cout << "Total tracks processed: " << total_tracks << std::endl;
    std::cout << "Excluded (non-positive NDF): " << excluded_non_positive_ndf << std::endl;
    std::cout << "Excluded (not all smoothed): " << excluded_not_all_smoothed << std::endl;
    std::cout << "Excluded (unknown outcome): " << excluded_unknown << std::endl;
    std::cout << "Excluded (no state): " << excluded_no_state << std::endl;
    std::cout << "Total excluded: " << (excluded_non_positive_ndf + excluded_not_all_smoothed + 
                                         excluded_unknown + excluded_no_state) << std::endl;
    std::cout << "Tracks printed: " << printed_tracks << std::endl;

    return 0;
}