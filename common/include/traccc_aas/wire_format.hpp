#ifndef TRACCC_AAS_WIRE_FORMAT_HPP
#define TRACCC_AAS_WIRE_FORMAT_HPP

// The byte layout exchanged between the python client and the Triton backend.
// Both the standalone driver and the backend include this header, so the
// encoder and the decoder of a given buffer are always built from the same
// definition.
//
// These formats are a private contract between client/TracccTritonClient.py
// and this backend. They deliberately do not live upstream in traccc: the
// column lists below enumerate the traccc EDM by hand, so any upstream EDM
// change silently alters the layout.

#include "traccc/examples/cuda/full_chain_service.hpp"

#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_constituent_link.hpp"
#include "traccc/edm/track_container.hpp"

#include <vecmem/memory/memory_resource.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace traccc_aas {

/// The reconstruction result carried over the wire
using results = traccc::cuda::full_chain_service::result;

/// Build the service configuration for the ITk geometry in @c geo_dir
///
/// The file names are ATLAS/ITk specific, which is why they live here rather
/// than in the traccc example library.
inline traccc::cuda::full_chain_service::config make_itk_config(
    const std::string& geo_dir)
{
    // Tolerate a missing trailing slash on the directory.
    const std::string dir =
        (geo_dir.empty() || geo_dir.back() == '/') ? geo_dir : geo_dir + "/";

    traccc::cuda::full_chain_service::config cfg;
    cfg.detector_file = dir + "detray_detector_geometry.json";
    cfg.material_file = dir + "detray_detector_material_maps.json";
    cfg.grid_file = dir + "detray_detector_surface_grids.json";
    cfg.digitization_file = dir + "ITk_digitization_config.json";
    cfg.conditions_file = dir + "ITk_digitization_config.json";
    cfg.magnetic_field_file = dir + "ITk_bfield.cvf";
    return cfg;
}

// Build a silicon_cell_collection directly from the raw client buffer.
// Byte layout (little-endian / native), 1-D of length 8 + 20*N:
//   offset 0        : uint64_t N              (cell count)
//   then 5 contiguous column blocks, each length N (SoA order):
//     channel0[N]     : uint32_t
//     channel1[N]     : uint32_t
//     activation[N]   : float32
//     time[N]         : float32
//     module_index[N] : uint32_t
inline traccc::edm::silicon_cell_collection::host cells_from_buffer(
    const uint8_t* buffer, size_t byte_size, vecmem::memory_resource& mr)
{
    traccc::edm::silicon_cell_collection::host cells(mr);

    // Need at least the 8-byte header carrying the cell count.
    if (buffer == nullptr || byte_size < sizeof(std::uint64_t)) {
        throw std::runtime_error(
            "CELLS buffer is too small to contain the cell count header");
    }

    // Read N with memcpy to avoid an unaligned 8-byte load.
    std::uint64_t num_cells = 0;
    std::memcpy(&num_cells, buffer, sizeof(std::uint64_t));

    // Validate the declared layout: 8-byte header + 5 column blocks of N.
    //   channel0/channel1/module_index : uint32 (4 bytes)
    //   activation/time                : float32 (4 bytes)
    const size_t expected_size =
        sizeof(std::uint64_t) + static_cast<size_t>(num_cells) * 20u;
    if (byte_size != expected_size) {
        throw std::runtime_error(
            "CELLS buffer size mismatch: expected " +
            std::to_string(expected_size) + " bytes for N=" +
            std::to_string(num_cells) + ", got " + std::to_string(byte_size));
    }

    static_assert(sizeof(traccc::channel_id) == 4u,
                  "CELLS wire format assumes 4-byte channel_id");
    static_assert(sizeof(float) == 4u,
                  "CELLS wire format assumes 4-byte float");
    static_assert(sizeof(unsigned int) == 4u,
                  "CELLS wire format assumes 4-byte module_index");

    const std::size_t n = static_cast<std::size_t>(num_cells);
    if (n == 0u) {
        return cells;
    }

    const std::size_t block = n * 4u;
    const uint8_t* base = buffer + sizeof(std::uint64_t);

    cells.resize(n);
    std::memcpy(cells.channel0().data(), base + 0u * block, block);
    std::memcpy(cells.channel1().data(), base + 1u * block, block);
    std::memcpy(cells.activation().data(), base + 2u * block, block);
    std::memcpy(cells.time().data(), base + 3u * block, block);
    std::memcpy(cells.module_index().data(), base + 4u * block, block);

    return cells;
}

// The traccc EDM containers are SoA: every field is its own contiguous,
// trivially-copyable column.
//
// Byte layout (little-endian / native):
//   uint64  n_measurements, n_tracks, n_states, n_links
//   then    every column in visit_track_columns order
//   then    uint32 link_counts[n_tracks]
//   then    track_constituent_link links[n_links]
//
template <typename MEASUREMENTS, typename TRACKS, typename STATES,
          typename VISITOR>
void visit_track_columns(MEASUREMENTS&& measurements, TRACKS&& tracks,
                         STATES&& states, VISITOR&& visit) {

    // measurement_collection
    visit(measurements.local_position());
    visit(measurements.local_variance());
    visit(measurements.dimensions());
    visit(measurements.time());
    visit(measurements.diameter());
    visit(measurements.identifier());
    visit(measurements.surface_link());
    visit(measurements.subspace());
    visit(measurements.cluster_index());

    // track_collection (constituent_links is jagged, handled by the caller)
    visit(tracks.fit_outcome());
    visit(tracks.params());
    visit(tracks.ndf());
    visit(tracks.chi2());
    visit(tracks.pval());
    visit(tracks.nholes());

    // track_state_collection
    visit(states.state());
    visit(states.filtered_chi2());
    visit(states.smoothed_chi2());
    visit(states.backward_chi2());
    visit(states.filtered_params());
    visit(states.smoothed_params());
    visit(states.measurement_index());
}

/// Serialize the reconstruction result into the TRACKS wire buffer
inline std::vector<uint8_t> tracks_to_buffer(const results& reco)
{
    const auto& measurements = reco.measurements;
    const auto& tracks = reco.tracks.tracks;
    const auto& states = reco.tracks.states;

    // Flatten the one jagged column.
    const auto& constituent_links = tracks.constituent_links();
    std::vector<std::uint32_t> link_counts;
    std::vector<traccc::edm::track_constituent_link> links;
    link_counts.reserve(constituent_links.size());
    for (std::size_t i = 0; i < constituent_links.size(); ++i) {
        const auto& track_links = constituent_links[i];
        link_counts.push_back(static_cast<std::uint32_t>(track_links.size()));
        links.insert(links.end(), track_links.begin(), track_links.end());
    }

    const std::uint64_t counts[4] = {measurements.size(), tracks.size(),
                                     states.size(), links.size()};

    std::vector<uint8_t> buffer;
    auto append = [&buffer](const auto& column) {
        using value_type =
            typename std::remove_cvref_t<decltype(column)>::value_type;
        static_assert(std::is_trivially_copyable_v<value_type>,
                      "SoA columns must be trivially copyable to go on the wire");
        if (column.empty()) {
            return;
        }
        const auto* bytes = reinterpret_cast<const uint8_t*>(column.data());
        buffer.insert(buffer.end(), bytes,
                      bytes + column.size() * sizeof(value_type));
    };

    const auto* counts_bytes = reinterpret_cast<const uint8_t*>(counts);
    buffer.insert(buffer.end(), counts_bytes, counts_bytes + sizeof(counts));
    visit_track_columns(measurements, tracks, states, append);
    append(link_counts);
    append(links);

    return buffer;
}

// Rebuild the traccc containers from the buffer the server sends: size the
// containers from the counts, then read each column back in the same order.
// The exact inverse of tracks_to_buffer above.
inline results tracks_from_buffer(const uint8_t* buffer, std::size_t byte_size,
                                  vecmem::memory_resource& mr)
{
    std::uint64_t counts[4] = {0, 0, 0, 0};
    if (buffer == nullptr || byte_size < sizeof(counts)) {
        throw std::runtime_error(
            "TRACKS buffer is too small to contain the element counts");
    }
    std::memcpy(counts, buffer, sizeof(counts));

    results reco{mr};
    auto& measurements = reco.measurements;
    auto& tracks = reco.tracks.tracks;
    auto& states = reco.tracks.states;

    // Sizing the containers also sizes every column, which is what tells the
    // reader below how many bytes each one occupies.
    measurements.resize(static_cast<std::size_t>(counts[0]));
    tracks.resize(static_cast<std::size_t>(counts[1]));
    states.resize(static_cast<std::size_t>(counts[2]));

    std::vector<std::uint32_t> link_counts(static_cast<std::size_t>(counts[1]));
    std::vector<traccc::edm::track_constituent_link> links(
        static_cast<std::size_t>(counts[3]));

    std::size_t offset = sizeof(counts);
    auto read = [&](auto& column) {
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
    auto& constituent_links = tracks.constituent_links();
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

    return reco;
}

}  // namespace traccc_aas

#endif  // TRACCC_AAS_WIRE_FORMAT_HPP
