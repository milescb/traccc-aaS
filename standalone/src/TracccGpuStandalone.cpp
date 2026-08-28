#include "traccc/examples/cuda/full_chain_service.hpp"
#include "traccc_aas/wire_format.hpp"

#include "TracccEdmConversion.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"

#include <vecmem/memory/host_memory_resource.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
// traccc_aas::cells_from_buffer: an 8-byte cell-count header
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

// Print the first few tracks of a result, the way a client would once it
// has decoded the TRACKS buffer back into traccc containers.
void print_tracks(const traccc_aas::results &results, std::size_t max_tracks)
{
    const auto &tracks = results.tracks.tracks;

    std::cout << "\nTracks: " << tracks.size()
              << ", states: " << results.tracks.states.size()
              << ", measurements: " << results.measurements.size() << std::endl;

    std::size_t printed_tracks = 0;
    for (std::size_t i = 0; i < tracks.size() && printed_tracks < max_tracks; ++i)
    {
        const auto &track = tracks.at(i);

        if (track.constituent_links().size() < 1) {
            continue;
        }

        const auto &fitted_params = track.params();

        std::cout << "Track " << i << ": chi2 = " << track.chi2()
                  << ", ndf = " << track.ndf()
                  << ", pval = " << track.pval()
                  << ", nholes = " << track.nholes()
                  << ", l0 = " << fitted_params.bound_local()[0]
                  << ", l1 = " << fitted_params.bound_local()[1]
                  << ", phi = " << fitted_params.phi()
                  << ", theta = " << fitted_params.theta()
                  << ", q/p = " << fitted_params.qop()
                  << ", time = " << fitted_params.time()
                  << ", fit outcome = "
                  << static_cast<std::underlying_type<traccc::track_fit_outcome>::type>(
                         track.fit_outcome())
                  << ", links = " << track.constituent_links().size()
                  << std::endl;

        ++printed_tracks;
    }
}

// Confirm that decoding the buffer reproduces the containers it was built from,
// so the driver actually exercises the wire format the server ships.
void check_round_trip(const traccc_aas::results &original,
                      const traccc_aas::results &decoded)
{
    const auto &a = original.tracks;
    const auto &b = decoded.tracks;

    if (a.tracks.size() != b.tracks.size() ||
        a.states.size() != b.states.size() ||
        original.measurements.size() != decoded.measurements.size()) {
        throw std::runtime_error("TRACKS round trip changed the element counts");
    }

    for (std::size_t i = 0; i < a.tracks.size(); ++i) {
        const auto track_a = a.tracks.at(i);
        const auto track_b = b.tracks.at(i);
        if (track_a.chi2() != track_b.chi2() || track_a.ndf() != track_b.ndf() ||
            track_a.pval() != track_b.pval() ||
            track_a.nholes() != track_b.nholes() ||
            track_a.fit_outcome() != track_b.fit_outcome() ||
            track_a.constituent_links().size() !=
                track_b.constituent_links().size()) {
            throw std::runtime_error("TRACKS round trip altered track " +
                                     std::to_string(i));
        }
    }

    for (std::size_t i = 0; i < original.measurements.size(); ++i) {
        const auto meas_a = original.measurements.at(i);
        const auto meas_b = decoded.measurements.at(i);
        if (meas_a.surface_link() != meas_b.surface_link() ||
            meas_a.local_position() != meas_b.local_position() ||
            meas_a.identifier() != meas_b.identifier()) {
            throw std::runtime_error("TRACKS round trip altered measurement " +
                                     std::to_string(i));
        }
    }

    std::cout << "TRACKS buffer round trip verified" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID> [geometry_dir]" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    // The geometry directory is a deployment detail, so take it from the
    // command line (or the environment) rather than baking a path in.
    const char *geo_dir_env = std::getenv("TRACCC_ITK_GEO_DIR");
    const std::string geoDir =
        (argc > 3)      ? std::string(argv[3])
        : (geo_dir_env) ? std::string(geo_dir_env)
                        : std::string("/traccc/itk-geometry/");

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;
    std::cout << "Using geometry from " << geoDir << std::endl;

    vecmem::host_memory_resource host_mr;

    traccc::cuda::full_chain_service traccc_gpu(
        host_mr, traccc_aas::make_itk_config(geoDir), deviceID,
        traccc::getDefaultLogger("TracccGpuStandalone",
                                 traccc::Logging::Level::INFO));
    traccc_gpu.initialize();

    dump_geom_id_map(
        traccc_gpu.module_index_map(),
        (std::filesystem::path(event_file).parent_path() /
         "geom_id_to_module_index.json").string());

    // The client owns the athena -> detray mapping, so load it here rather than
    // in the wrapper: the server never needs it.
    const std::map<int64_t, uint64_t> athena_to_detray =
        read_athena_to_detray_mapping(geoDir + "/athenaIdentifierToDetrayMap.txt");

    std::vector<traccc::io::csv::cell> cells = read_csv(
        event_file, athena_to_detray, true);

    std::vector<uint8_t> cell_buffer = build_cell_buffer(
        cells, traccc_gpu.module_index_map());

    traccc::edm::silicon_cell_collection::host cell_collection =
        traccc_aas::cells_from_buffer(cell_buffer.data(), cell_buffer.size(),
                                      traccc_gpu.cell_memory_resource());

    auto traccc_result = traccc_gpu.run(cell_collection);

    // Serialize exactly as the server does, then decode the buffer back into
    // traccc containers the way a client would, so this driver exercises the
    // full round trip of the TRACKS wire format.
    std::vector<uint8_t> tracks_buffer =
        traccc_aas::tracks_to_buffer(traccc_result);
    std::cout << "\nSerialized tracks into " << tracks_buffer.size()
              << " bytes" << std::endl;

    traccc_aas::results decoded = traccc_aas::tracks_from_buffer(
        tracks_buffer.data(), tracks_buffer.size(), host_mr);

    check_round_trip(traccc_result, decoded);
    print_tracks(decoded, 5);

    return 0;
}