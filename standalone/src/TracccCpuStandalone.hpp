#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

// algorithms
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/options/detector.hpp"

// algorithm options
//! NOTE: these may not be necessary
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"

// detray
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
//! perhaps not necessary? 
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Definition of the cell_order struct
struct cell_order
{
    bool operator()(const traccc::cell &lhs, const traccc::cell &rhs) const
    {
        if (lhs.module_link != rhs.module_link)
        {
            return lhs.module_link < rhs.module_link;
        }
        else if (lhs.channel1 != rhs.channel1)
        {
            return lhs.channel1 < rhs.channel1;
        }
        else
        {
            return lhs.channel0 < rhs.channel0;
        }
    }
};

// Definition of the get_module function
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry *geom,
                               const traccc::digitization_config *dconfig,
                               const std::uint64_t original_geometry_id)
{
    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr)
    {
        if (!geom->contains(result.surface_link.value()))
        {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr)
    {
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end())
        {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        const auto &binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1)
        {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
        result.pixel.variance_y = geo_it->variance_y;
    }

    return result;
}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
// std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<double>> &data);
std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells);
std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells);
void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, 
                detray::geometry::barcode> *barcode_map, 
                bool deduplicate); 

// Type definitions
using detector_type = detray::detector<detray::default_metadata,
                                           detray::host_container_types>;
using stepper_type =
    detray::rk_stepper<detray::bfield::const_field_t::view_t,
                        detector_type::algebra_type,
                        detray::constrained_step<>>;
using navigator_type = detray::navigator<const detector_type>;
using finding_algorithm =
    traccc::finding_algorithm<stepper_type, navigator_type>;
using fitting_algorithm = traccc::fitting_algorithm<
    traccc::kalman_fitter<stepper_type, navigator_type>>;     

class TracccClusterStandalone
{
private:
    int m_deviceID;
    vecmem::host_memory_resource m_mem;
    // options
    traccc::opts::detector m_detector_opts;
    detector_type detector;
    traccc::opts::track_seeding m_seeding_opts;
    traccc::opts::track_finding m_finding_opts;
    traccc::opts::track_propagation m_propagation_opts;
    detray::propagation::config m_propagation_config;
    detray::io::detector_reader_config cfg;
    finding_algorithm::config_type m_finding_cfg;
    fitting_algorithm::config_type m_fitting_cfg;
    // algorithms
    traccc::host::clusterization_algorithm m_ca;
    traccc::host::spacepoint_formation_algorithm m_sf;
    traccc::seeding_algorithm m_sa;
    traccc::track_params_estimation m_tp;
    traccc::finding_algorithm<stepper_type, navigator_type> m_finding_alg;
    traccc::fitting_algorithm<traccc::kalman_fitter<stepper_type, navigator_type>> m_fitting_alg;
    traccc::greedy_ambiguity_resolution_algorithm m_resolution_alg;
    // geometry
    traccc::geometry m_surface_transforms;
    std::unique_ptr<traccc::digitization_config> m_digi_cfg;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> m_barcode_map;
    // field
    traccc::vector3 field_vec;
    detray::bfield::const_field_t field;

public:
    TracccClusterStandalone(int deviceID = 0)
        : m_deviceID(deviceID), 
          detector(m_mem),
          m_propagation_config(m_propagation_opts),
          m_finding_cfg(m_finding_opts),
          m_ca(m_mem), 
          m_sf(m_mem), 
          m_sa(m_seeding_opts.seedfinder,
               {m_seeding_opts.seedfinder},
                m_seeding_opts.seedfilter, m_mem),
          m_tp(m_mem),
          m_finding_alg(m_finding_cfg),
          m_fitting_alg(m_fitting_cfg)

    {
        initializePipeline();
    }

    ~TracccClusterStandalone() = default;

    void initializePipeline();
    void runPipeline(std::vector<traccc::io::csv::cell> cells);
    std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::uint64_t> &geometry_ids,
                                                        const std::vector<std::vector<double>> &data);
};

void TracccClusterStandalone::initializePipeline()
{
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_geometry_detray.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-digi-geometric-config.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_surface_grids_detray.json";
    m_detector_opts.use_detray_detector = true;

    auto geom_data = traccc::io::read_geometry(m_detector_opts.detector_file, traccc::data_format::json);

    m_surface_transforms = std::move(geom_data.first);
    m_barcode_map = std::move(geom_data.second);

    m_digi_cfg = std::make_unique<traccc::digitization_config>(traccc::io::read_digitization_config(m_detector_opts.digitization_file));

    // setup the detector
    cfg.add_file(m_detector_opts.detector_file);
    cfg.add_file(m_detector_opts.grid_file);
    auto det = detray::io::read_detector<detector_type>(m_mem, cfg);
    detector = std::move(det.first);

    // initialize the field
    field_vec = {0.f, 0.f, m_seeding_opts.seedfinder.bFieldInZ};
    field = detray::bfield::create_const_field(field_vec);

    // initialize the track finding algorithm
    m_finding_cfg.propagation = m_propagation_config;
    m_fitting_cfg.propagation = m_propagation_config;
}

void TracccClusterStandalone::runPipeline(std::vector<traccc::io::csv::cell> cells)
{
    traccc::io::cell_reader_output readOut(&m_mem);

    read_cells(readOut, cells, &m_surface_transforms, m_digi_cfg.get(), m_barcode_map.get(), true);

    traccc::cell_collection_types::host data = readOut.cells;
    traccc::cell_collection_types::host &cells_per_event = readOut.cells;
    traccc::cell_module_collection_types::host &modules_per_event = readOut.modules;

    //
    // ----------------- Clusterization -----------------
    // 
    traccc::host::clusterization_algorithm::output_type measurements_per_event{&m_mem};
    measurements_per_event = m_ca(vecmem::get_data(cells_per_event), 
                                  vecmem::get_data(modules_per_event));

    //
    // ----------------- Spacepoint Formation -----------------
    //
    traccc::host::spacepoint_formation_algorithm::output_type spacepoints_per_event{&m_mem};
    spacepoints_per_event = m_sf(vecmem::get_data(measurements_per_event), 
                                 vecmem::get_data(modules_per_event));

    //
    // ----------------- Seeding -----------------
    //
    traccc::seeding_algorithm::output_type seeds{&m_mem};
    seeds = m_sa(spacepoints_per_event);

    //
    // ----------------- Finding and Fitting -----------------
    //

    // track paramater estimation
    traccc::track_params_estimation::output_type params{&m_mem};
    params = m_tp(spacepoints_per_event, seeds, field_vec);

    // track finding
    finding_algorithm::output_type track_candidates{&m_mem};
    track_candidates = m_finding_alg(detector, field, measurements_per_event, params);

    // track fitting
    fitting_algorithm::output_type track_states{&m_mem};
    track_states = m_fitting_alg(detector, field, track_candidates);

    // resolved tracks
    traccc::greedy_ambiguity_resolution_algorithm::output_type resolved_track_states{&m_mem};
    resolved_track_states = m_resolution_alg(track_states);

    // ----------------- Print Statistics -----------------
    std::cout << " " << std::endl;
    std::cout << "==> Statistics ... " << std::endl;

    // measurement and spacepoints
    auto measurements_size = measurements_per_event.size();
    std::cout << " - number of measurements created: " << measurements_size << std::endl;
    auto spacepoints_size = spacepoints_per_event.size();
    std::cout << " - number of spacepoints created: " << spacepoints_size << std::endl;

    // for (std::size_t i = 0; i < 10; ++i)
    // {
    //     auto measurement = measurements_per_event.at(i);
    //     auto spacepoint = spacepoints_per_event.at(i);
    //     std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
    //     std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl;
    //     std::cout << "Global coordinates: [" << spacepoint.global[0] << ", " << spacepoint.global[1] << ", " << spacepoint.global[2] << "]" << std::endl;
    // }

    // seeding
    auto seeds_size = seeds.size();
    std::cout << " - number of seeds created: " << seeds_size << std::endl;

    // fitting and finding
    auto track_candidates_size = track_candidates.size();
    std::cout << " - number of track candidates: " << track_candidates_size << std::endl;

    auto track_states_size = track_states.size();
    std::cout << " - number of fitted tracks: " << track_states_size << std::endl;

    auto resolved_track_states_size = resolved_track_states.size();
    std::cout << " - number of resolved tracks: " << resolved_track_states_size << std::endl;

}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    while (reader.read(iocell))
    {
        cells.push_back(iocell);
    }

    return cells;
}

std::vector<traccc::io::csv::cell> TracccClusterStandalone::read_from_array(const std::vector<std::uint64_t> &geometry_ids,
                                                                            const std::vector<std::vector<double>> &data)
{
    std::vector<traccc::io::csv::cell> cells;

    if (geometry_ids.size() != data.size())
    {
        throw std::runtime_error("Number of geometry IDs and data rows do not match.");
    }

    for (size_t i = 0; i < data.size(); ++i) 
    {
        const auto& row = data[i];
        if (row.size() != 5)
            continue; 

        traccc::io::csv::cell iocell;

        if (i < geometry_ids.size()) 
        {
            iocell.geometry_id = geometry_ids[i];
        } 
        else 
        {
            continue;
        }

        iocell.hit_id = static_cast<int>(row[0]);
        iocell.channel0 = static_cast<int>(row[1]);
        iocell.channel1 = static_cast<int>(row[2]);
        iocell.timestamp = static_cast<int>(row[3]);
        iocell.value = row[4];

        cells.push_back(iocell);
    }

    return cells;
}

std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, unsigned int &nduplicates)
{
    std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> cellMap;
    nduplicates = 0;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        auto ret = cellMap[iocell.geometry_id].insert({cell, iocell.value});
        if (!ret.second)
        {
            cellMap[iocell.geometry_id].at(cell) += iocell.value;
            ++nduplicates;
        }
    }

    return cellMap;
}

std::map<std::uint64_t, std::vector<traccc::cell>> create_result_container(const std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> &cellMap)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;
    for (const auto &[geometry_id, cells] : cellMap)
    {
        for (const auto &[cell, value] : cells)
        {
            traccc::cell summed_cell{cell};
            summed_cell.activation = value;
            result[geometry_id].push_back(summed_cell);
        }
    }
    return result;
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    unsigned int nduplicates = 0;
    auto cellMap = fill_cell_map(cells, nduplicates);

    if (nduplicates > 0)
    {
        std::cout << "WARNING: " << nduplicates << " duplicate cells found." << std::endl;
    }

    return create_result_container(cellMap);
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        result[iocell.geometry_id].push_back(cell);
    }

    return result;
}

void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map, 
                bool deduplicate)
{
    auto cellsMap = (deduplicate ? read_deduplicated_cells(cells)
                                 : read_all_cells(cells));

    for (const auto &[original_geometry_id, cells] : cellsMap)
    {
        std::uint64_t geometry_id = original_geometry_id;
        if (barcode_map != nullptr)
        {
            const auto it = barcode_map->find(geometry_id);
            if (it != barcode_map->end())
            {
                geometry_id = it->second.value();
            }
            else
            {
                throw std::runtime_error(
                    "Could not find barcode for geometry ID " +
                    std::to_string(geometry_id));
            }
        }

        out.modules.push_back(
            get_module(geometry_id, geom, dconfig, original_geometry_id));
        for (auto &cell : cells)
        {
            out.cells.push_back(cell);
            out.cells.back().module_link = out.modules.size() - 1;
        }
    }
}
