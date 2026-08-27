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

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
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
        // module_index is a row index into the detector description, not an ID,
        // so the cell's geometry ID has to be looked up here.
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

// Rebuild the traccc containers from the buffer the server sends: size the
// containers from the counts, then read each column back in the same order.
// The exact inverse of TracccGpuStandalone::tracks_to_buffer (see that function
// for the byte layout). This lives on the client side of the wire, so the
// backend never compiles it.
TracccResults tracks_from_buffer(const uint8_t *buffer, std::size_t byte_size,
                                 vecmem::memory_resource &mr)
{
    std::uint64_t counts[4] = {0, 0, 0, 0};
    if (buffer == nullptr || byte_size < sizeof(counts)) {
        throw std::runtime_error(
            "TRACKS buffer is too small to contain the element counts");
    }
    std::memcpy(counts, buffer, sizeof(counts));

    TracccResults results{
        traccc::edm::track_container<traccc::default_algebra>::host{mr},
        traccc::edm::measurement_collection::host{mr}};
    auto &measurements = results.measurements;
    auto &tracks = results.tracks_and_states.tracks;
    auto &states = results.tracks_and_states.states;

    // Sizing the containers also sizes every column, which is what tells the
    // reader below how many bytes each one occupies.
    measurements.resize(static_cast<std::size_t>(counts[0]));
    tracks.resize(static_cast<std::size_t>(counts[1]));
    states.resize(static_cast<std::size_t>(counts[2]));

    std::vector<std::uint32_t> link_counts(static_cast<std::size_t>(counts[1]));
    std::vector<traccc::edm::track_constituent_link> links(
        static_cast<std::size_t>(counts[3]));

    std::size_t offset = sizeof(counts);
    auto read = [&](auto &column) {
        using value_type =
            typename std::remove_cvref_t<decltype(column)>::value_type;
        const std::size_t bytes = column.size() * sizeof(value_type);
        if (offset + bytes > byte_size) {
            throw std::runtime_error(
                "TRACKS buffer is truncated: needed " + std::to_string(bytes) +
                " more bytes at offset " + std::to_string(offset) + " of " +
                std::to_string(byte_size));
        }
        if (bytes > 0u) {
            std::memcpy(column.data(), buffer + offset, bytes);
        }
        offset += bytes;
    };

    visit_track_columns(measurements, tracks, states, read);
    read(link_counts);
    read(links);

    // The counts determine the length exactly, so a leftover tail means the
    // sender's columns are not the size this build expects.
    if (offset != byte_size) {
        throw std::runtime_error(
            "TRACKS buffer size mismatch: consumed " + std::to_string(offset) +
            " of " + std::to_string(byte_size) +
            " bytes; the sender's traccc build does not match this one");
    }

    // Re-inflate the jagged constituent_links column.
    auto &constituent_links = tracks.constituent_links();
    std::size_t link_offset = 0;
    for (std::size_t i = 0; i < link_counts.size(); ++i) {
        const std::size_t count = link_counts[i];
        if (link_offset + count > links.size()) {
            throw std::runtime_error(
                "TRACKS buffer has inconsistent constituent link counts");
        }
        constituent_links[i].assign(links.begin() + link_offset,
                                    links.begin() + link_offset + count);
        link_offset += count;
    }

    return results;
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

// Print the first few tracks of a TracccResults, the way a client would once it
// has decoded the TRACKS buffer back into traccc containers.
void print_tracks(const TracccResults &results, std::size_t max_tracks)
{
    const auto &tracks = results.tracks_and_states.tracks;

    std::cout << "\nTracks: " << tracks.size()
              << ", states: " << results.tracks_and_states.states.size()
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
void check_round_trip(const TracccResults &original,
                      const TracccResults &decoded)
{
    const auto &a = original.tracks_and_states;
    const auto &b = decoded.tracks_and_states;

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
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    const std::string geoDir = "/global/homes/m/milescb/tracking/traccc-aaS-gpu/traccc/data/geometries/odd/";

    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceID);

    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceID, geoDir);

    dump_geom_id_map(
        traccc_gpu.getGeomIdMap(),
        (std::filesystem::path(event_file).parent_path() /
         "geom_id_to_module_index.json").string());

    std::vector<traccc::io::csv::cell> cells = read_csv(event_file);

    std::vector<uint8_t> cell_buffer = build_cell_buffer(
        cells, traccc_gpu.getGeomIdMap());

    traccc::edm::silicon_cell_collection::host cell_collection =
        traccc_gpu.cells_from_buffer(cell_buffer.data(), cell_buffer.size());

    auto traccc_result = traccc_gpu.run(std::move(cell_collection), true);

    // Serialize exactly as the server does, then decode the buffer back into
    // traccc containers the way a client would, so this driver exercises the
    // full round trip of the TRACKS wire format.
    std::vector<uint8_t> tracks_buffer = traccc_gpu.tracks_to_buffer(traccc_result);
    std::cout << "\nSerialized tracks into " << tracks_buffer.size()
              << " bytes" << std::endl;

    TracccResults decoded = tracks_from_buffer(
        tracks_buffer.data(), tracks_buffer.size(), host_mr);

    check_round_trip(traccc_result, decoded);
    print_tracks(decoded, 5);

    return 0;
}