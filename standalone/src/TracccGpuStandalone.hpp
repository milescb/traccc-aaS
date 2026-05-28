#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>
#include <chrono>
#include <unordered_map>

// local includes
#include "TracccEdmConversion.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/alpaka/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/alpaka/seeding/triplet_seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/seed_parameter_estimation_algorithm.hpp"
#include "traccc/alpaka/utils/get_device_info.hpp"
#include "traccc/alpaka/utils/queue.hpp"
#include "traccc/alpaka/utils/vecmem_objects.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/geometry/detector_design_description.hpp"
#include "traccc/geometry/detector_conditions_description.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

// magnetic field include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/io/read_magnetic_field.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"

// algorithm options
#include "traccc/options/detector.hpp"

std::chrono::high_resolution_clock::time_point start_total, end_total;
std::chrono::high_resolution_clock::time_point start_read, end_read;
std::chrono::high_resolution_clock::time_point start_copy_in, end_copy_in;
std::chrono::high_resolution_clock::time_point start_cluster, end_cluster;
std::chrono::high_resolution_clock::time_point start_spacepoint, end_spacepoint;
std::chrono::high_resolution_clock::time_point start_seeding, end_seeding;
std::chrono::high_resolution_clock::time_point start_params, end_params;
std::chrono::high_resolution_clock::time_point start_finding, end_finding;
std::chrono::high_resolution_clock::time_point start_fitting, end_fitting;
std::chrono::high_resolution_clock::time_point start_copy_out, end_copy_out;

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

struct TracccResults {
    traccc::edm::track_container<traccc::default_algebra>::host tracks_and_states;
    traccc::edm::measurement_collection::host measurements;
};

traccc::magnetic_field make_magnetic_field(std::string filename) {
    traccc::magnetic_field result;
    traccc::io::read_magnetic_field(result, filename, traccc::data_format::binary);
    return result;
}

// Type definitions
using host_detector_type = traccc::host_detector;
using device_detector_type = traccc::detector_buffer;
using scalar_type = traccc::default_detector::host::scalar_type;

using bfield_type =
    covfie::field<traccc::const_bfield_backend_t<scalar_type>>;
using stepper_type =
    detray::rk_stepper<bfield_type::view_t,
                        traccc::default_detector::host::algebra_type,
                        detray::constrained_step<scalar_type>>;

using spacepoint_formation_algorithm =
    traccc::alpaka::silicon_pixel_spacepoint_formation_algorithm;
using clustering_algorithm = traccc::alpaka::clusterization_algorithm;
using finding_algorithm =
    traccc::alpaka::combinatorial_kalman_filter_algorithm;
using fitting_algorithm = traccc::alpaka::kalman_fitting_algorithm;

template <typename scalar_t>
using unit = detray::unit<scalar_t>;

static traccc::seedfinder_config create_and_setup_finder_config() {
    traccc::seedfinder_config cfg;
    cfg.zMin = -3000.f * traccc::unit<float>::mm;
    cfg.zMax = 3000.f * traccc::unit<float>::mm;
    cfg.rMax = 320.f * traccc::unit<float>::mm;
    cfg.rMin = 33.f * traccc::unit<float>::mm;
    // 3 sigmas of max beam spot delta z for Run 3
    cfg.collisionRegionMax = 3 * 38 * traccc::unit<float>::mm;
    cfg.collisionRegionMin = -cfg.collisionRegionMax;
    // Based on Pixel barrel layers layout
    // Minimum R distance between 2 layers: 30
    // Max distance between N, N+2 layer (allowing one "hole"): 160
    // Plus some margin (2 mm)
    cfg.deltaRMin = 20.f * traccc::unit<float>::mm;
    cfg.deltaRMax = 100.f * traccc::unit<float>::mm;
    cfg.deltaZMax = 800.f * traccc::unit<float>::mm;
    cfg.minPt = 900.f * traccc::unit<float>::MeV;
    cfg.cotThetaMax = 27.2899f;
    cfg.impactMax = 2.f * traccc::unit<float>::mm;
    cfg.sigmaScattering = 3.0f;
    cfg.maxPtScattering = 10.f * traccc::unit<float>::GeV;
    cfg.radLengthPerSeed = 0.05f;
    cfg.maxSeedsPerSpM = 2;
    cfg.setup();
    return cfg;
}

static traccc::seedfilter_config create_and_setup_filter_config() {
    traccc::seedfilter_config cfg;
    cfg.good_spB_min_radius = 150.f * unit<float>::mm;
    cfg.good_spB_weight_increase = 400.f;
    cfg.good_spT_max_radius = 150.f * unit<float>::mm;
    cfg.good_spT_weight_increase = 200.f;
    cfg.good_spB_min_weight = 380.f;
    cfg.seed_min_weight = 200.f;
    cfg.spB_min_radius = 43.f * unit<float>::mm;
    cfg.compatSeedLimit = 1;
    return cfg;
}

static fitting_algorithm::config_type create_and_setup_fitting_config() {
    fitting_algorithm::config_type cfg;
    return cfg;
}

static finding_algorithm::config_type create_and_setup_finding_config() {
    finding_algorithm::config_type cfg{
        .max_num_branches_per_seed = 3,
        .max_num_branches_per_surface = 1,
        .min_track_candidates_per_track = 7,
        .max_track_candidates_per_track = 20,
        .max_num_skipping_per_cand = 2,
        .max_num_consecutive_skipped = 1,
        .max_num_tracks_per_measurement = 1,
        .min_step_length_for_next_surface = 0.5f * detray::unit<float>::mm,
        .max_step_counts_for_next_surface = 100,
        .chi2_max = 30.f,
        .propagation = {
            .navigation = {
                .intersection = {
                    .overstep_tolerance = -300.f * unit<float>::um
                }
            },
            .stepping = {
                .min_stepsize = 1e-4f * unit<float>::mm,
                .rk_error_tol = 1e-4f * unit<float>::mm,
                .step_constraint = std::numeric_limits<float>::max(),
                .path_limit = 5.f * unit<float>::m,
                .max_rk_updates = 10000u,
                .use_mean_loss = true,
                .use_eloss_gradient = false,
                .use_field_gradient = false,
                .do_covariance_transport = true
            }
        },
        .initial_links_per_seed = 5
    };
    return cfg;
}

class TracccGpuStandalone
{
private:
    /// Device ID to use
    int m_device_id;
    /// Geometry directory path
    std::string m_geoDir;

    /// Logger
    std::unique_ptr<const traccc::Logger> logger;

    /// Alpaka queue (must be declared before vecmem_objects)
    traccc::alpaka::queue m_queue;
    /// VecMem objects (host/device memory resources and copy helpers)
    traccc::alpaka::vecmem_objects m_vo;
    /// (Asynchronous) memory copy object
    vecmem::copy& m_copy;
    /// Unified memory resource struct
    traccc::memory_resource m_mr;
    /// Non-pinned host memory resource for large host-only data (descriptor, detector)
    vecmem::host_memory_resource m_host_mr;

    /// digitization configuration
    std::unique_ptr<traccc::digitization_config> m_digi_cfg;
    /// Athena to detray map
    std::map<int64_t, uint64_t> m_athena_to_detray_map;
    /// Detray to Athena map (reverse mapping)
    std::unordered_map<uint64_t, int64_t> m_detray_to_athena_map;
    /// detector description to geo id map
    std::unordered_map<traccc::geometry_id, unsigned int> m_geomIdMap;

    // program configuration 
    /// detector options
    traccc::opts::detector m_detector_opts;
    /// Configuration for clustering
    traccc::clustering_config m_clustering_config;
    /// Configuration for the seed finding
    traccc::seedfinder_config m_finder_config;
    /// Configuration for the seed filtering
    traccc::seedfilter_config m_filter_config;
    /// Configuration for the spacepoint grid
    traccc::spacepoint_grid_config m_grid_config;

    /// further configuration
    /// Configuration for the track parameter estimation
    traccc::track_params_estimation_config m_track_params_estimation_config;
    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;

    /// host field object
    traccc::magnetic_field m_host_field;
    /// device field object
    traccc::magnetic_field m_field;

    /// Detector design description (segmentation geometry)
    traccc::detector_design_description::host m_det_descr_design;
    /// Detector conditions description (per-module conditions)
    traccc::detector_conditions_description::host m_det_descr_cond;
    /// Detector design description buffer (device)
    traccc::detector_design_description::buffer m_device_det_descr_design;
    /// Detector conditions description buffer (device)
    traccc::detector_conditions_description::buffer m_device_det_cond;

    /// Host detector
    traccc::host_detector m_detector;
    traccc::detector_buffer m_device_detector;

    /// Sub-algorithms used by this full-chain algorithm
    /// Clusterization algorithm
    clustering_algorithm m_clusterization;
    /// Measurement sorting algorithm
    traccc::alpaka::measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    traccc::alpaka::triplet_seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    traccc::alpaka::seed_parameter_estimation_algorithm m_track_parameter_estimation;

    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;

    // Helper function to read in cells
    std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_all_cells(
        const std::vector<traccc::io::csv::cell> &cells);

    void read_cells(traccc::edm::silicon_cell_collection::host &out,
                const std::vector<traccc::io::csv::cell> &cells,
                bool deduplicate,
                bool use_acts_geometry_id);

public:
    TracccGpuStandalone(
        int deviceID = 0,
        const std::string& geoDir = "/eos/project/a/atlas-eftracking/GPU/ITk_data/FinalReport/") :
            m_device_id(deviceID),
            m_geoDir(geoDir),
            logger(traccc::getDefaultLogger("TracccGpuStandalone", traccc::Logging::Level::INFO)),
            m_queue(static_cast<std::size_t>(deviceID)),
            m_vo(m_queue),
            m_copy(m_vo.async_copy()),
            m_mr{m_vo.device_mr(), &m_vo.host_mr()},
            m_clustering_config{256, 16, 8, 256},
            m_finder_config(create_and_setup_finder_config()),
            m_filter_config(create_and_setup_filter_config()),
            m_grid_config(m_finder_config),
            m_finding_config(create_and_setup_finding_config()),
            m_field(make_magnetic_field(geoDir + "ITk_bfield.cvf")),
            m_det_descr_design(m_host_mr),
            m_det_descr_cond(m_host_mr),
            m_clusterization(m_mr, m_vo.async_copy(), m_queue,
                             m_clustering_config,
                             logger->cloneWithSuffix("ClusteringAlg")),
            m_measurement_sorting(m_mr, m_vo.async_copy(), m_queue,
                                   logger->cloneWithSuffix("MeasSortingAlg")),
            m_spacepoint_formation(m_mr, m_vo.async_copy(), m_queue,
                                    logger->cloneWithSuffix("SpFormationAlg")),
            m_seeding(m_finder_config, m_grid_config, m_filter_config,
                      m_mr, m_vo.async_copy(), m_queue,
                      logger->cloneWithSuffix("SeedingAlg")),
            m_track_parameter_estimation(
                m_track_params_estimation_config,
                m_mr, m_vo.async_copy(), m_queue,
                logger->cloneWithSuffix("TrackParEstAlg")),
            m_finding(m_finding_config, m_mr, m_vo.async_copy(), m_queue,
                      logger->cloneWithSuffix("TrackFindingAlg")),
            m_fitting_config(create_and_setup_fitting_config()),
            m_fitting(m_fitting_config, m_mr, m_vo.async_copy(), m_queue,
                      logger->cloneWithSuffix("TrackFittingAlg"))
    {
        std::cout << "Using device ID: " << m_device_id << std::endl;

        initialize();
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    std::vector<traccc::io::csv::cell> read_from_array(
        const int64_t* cell_positions,
        const float* cell_properties,
        size_t num_cells,
        bool athena_ids);

    // getters
    const std::map<int64_t, uint64_t>& getAthenaToDetrayMap() const {
        return m_athena_to_detray_map;
    }

    const std::unordered_map<uint64_t, int64_t>& getDetrayToAthenaMap() const {
        return m_detray_to_athena_map;
    }

    void initialize();

    TracccResults run(std::vector<traccc::io::csv::cell> cells, bool show_stats = false);

};


void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = m_geoDir + "/detray_detector_geometry.json";
    m_detector_opts.digitization_file = m_geoDir + "/ITk_digitization_config.json";
    m_detector_opts.conditions_file = m_geoDir + "/ITk_digitization_config.json";
    m_detector_opts.grid_file = m_geoDir + "/detray_detector_surface_grids.json";
    m_detector_opts.material_file = m_geoDir + "/detray_detector_material_maps.json";

    // Load Athena-to-Detray mapping
    std::string athenaTransformsPath = m_geoDir + "/athenaIdentifierToDetrayMap.txt";
    m_athena_to_detray_map = read_athena_to_detray_mapping(athenaTransformsPath);

    // Create reverse mapping from Detray to Athena
    m_detray_to_athena_map.reserve(m_athena_to_detray_map.size());
    for (const auto& [athena_id, detray_id] : m_athena_to_detray_map) {
        m_detray_to_athena_map[detray_id] = athena_id;
    }

    traccc::io::read_detector_description(
        m_det_descr_design, m_det_descr_cond,
        m_detector_opts.detector_file,
        m_detector_opts.digitization_file,
        m_detector_opts.conditions_file,
        traccc::data_format::json);

    // Setup detector design description buffer
    m_device_det_descr_design = traccc::detector_design_description::buffer(
        [&]() {
            std::vector<unsigned int> sizes(m_det_descr_design.size());
            for (std::size_t i = 0; i < m_det_descr_design.size(); ++i) {
                auto d = m_det_descr_design.at(i);
                sizes[i] = std::max(
                    static_cast<unsigned int>(d.bin_edges_x().size()),
                    static_cast<unsigned int>(d.bin_edges_y().size()));
            }
            return sizes;
        }(),
        m_vo.device_mr());
    m_vo.copy().setup(m_device_det_descr_design)->wait();
    m_vo.copy()(vecmem::get_data(m_det_descr_design), m_device_det_descr_design)->wait();

    // Setup detector conditions description buffer (fixed-size)
    m_device_det_cond = traccc::detector_conditions_description::buffer(
        static_cast<traccc::detector_conditions_description::buffer::size_type>(
            m_det_descr_cond.size()),
        m_vo.device_mr());
    m_vo.copy().setup(m_device_det_cond)->wait();
    m_vo.copy()(vecmem::get_data(m_det_descr_cond), m_device_det_cond,
                vecmem::copy::type::host_to_device)->wait();

    m_queue.synchronize();

    // fill the det description to geometry id map
    m_geomIdMap.clear();
    m_geomIdMap.reserve(m_det_descr_cond.geometry_id().size());
    for (unsigned int i = 0; i < m_det_descr_cond.geometry_id().size(); ++i) {
        m_geomIdMap[m_det_descr_cond.geometry_id()[i].value()] = i;
    }

    traccc::io::read_detector(
        m_detector, m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    m_device_detector =
        traccc::buffer_from_host_detector(m_detector, m_vo.device_mr(), m_vo.copy());
    m_queue.synchronize();

    return;
}

TracccResults TracccGpuStandalone::run(
    std::vector<traccc::io::csv::cell> cells, bool show_stats
) {
    if (show_stats) start_total = std::chrono::high_resolution_clock::now();

    // Read cells
    if (show_stats) start_read = std::chrono::high_resolution_clock::now();
    traccc::edm::silicon_cell_collection::host read_out(m_vo.host_mr());
    read_cells(read_out, cells, true, false);
    if (show_stats) end_read = std::chrono::high_resolution_clock::now();

    // Copy to device
    if (show_stats) start_copy_in = std::chrono::high_resolution_clock::now();
    traccc::edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(read_out.size()), m_mr.main);
    m_copy.setup(cells_buffer)->wait();
    m_copy(vecmem::get_data(read_out), cells_buffer)->wait();
    if (show_stats) end_copy_in = std::chrono::high_resolution_clock::now();

    // Clusterization
    if (show_stats) start_cluster = std::chrono::high_resolution_clock::now();
    auto unsorted_measurements =
        m_clusterization(cells_buffer, m_device_det_descr_design, m_device_det_cond);
    auto measurements =
        m_measurement_sorting(unsorted_measurements);
    m_queue.synchronize();
    if (show_stats) end_cluster = std::chrono::high_resolution_clock::now();

    // Spacepoint formation
    if (show_stats) start_spacepoint = std::chrono::high_resolution_clock::now();
    auto spacepoints =
        m_spacepoint_formation(m_device_detector, measurements);
    m_queue.synchronize();
    if (show_stats) end_spacepoint = std::chrono::high_resolution_clock::now();

    // Seeding
    if (show_stats) start_seeding = std::chrono::high_resolution_clock::now();
    auto seeds = m_seeding(spacepoints);
    m_queue.synchronize();
    if (show_stats) end_seeding = std::chrono::high_resolution_clock::now();

    // // Track parameter estimation
    // if (show_stats) start_params = std::chrono::high_resolution_clock::now();
    // auto track_params =
    //     m_track_parameter_estimation(m_field, measurements, spacepoints, seeds);
    // m_queue.synchronize();
    // if (show_stats) end_params = std::chrono::high_resolution_clock::now();

    // // Track finding
    // if (show_stats) start_finding = std::chrono::high_resolution_clock::now();
    // auto track_candidates = m_finding(
    //     m_device_detector, m_field, measurements, track_params);
    // if (show_stats) end_finding = std::chrono::high_resolution_clock::now();

    // // Track fitting
    // if (show_stats) start_fitting = std::chrono::high_resolution_clock::now();
    // auto track_states = m_fitting(m_device_detector, m_field, track_candidates);
    // if (show_stats) end_fitting = std::chrono::high_resolution_clock::now();

    // // Copy results back to host
    // if (show_stats) start_copy_out = std::chrono::high_resolution_clock::now();
    traccc::edm::track_container<traccc::default_algebra>::host
        track_states_host{m_vo.host_mr()};

    // m_copy(track_states.tracks, track_states_host.tracks)->wait();
    // m_copy(track_states.states, track_states_host.states)->wait();

    // // copy measurements back to host
    traccc::edm::measurement_collection::host measurements_host(m_vo.host_mr());
    // m_copy(measurements, measurements_host)->wait();
    // if (show_stats) end_copy_out = std::chrono::high_resolution_clock::now();

    traccc::edm::measurement_collection::host
        measurements_per_event_alpaka{m_vo.host_mr()};
    traccc::edm::spacepoint_collection::host spacepoints_per_event_alpaka{
        m_vo.host_mr()};
    traccc::edm::seed_collection::host seeds_alpaka{m_vo.host_mr()};
    // traccc::bound_track_parameters_collection_types::host params_alpaka{
    //     &m_vo.host_mr()};
    // traccc::edm::track_collection<traccc::default_algebra>::host
    //     track_candidates_alpaka{m_vo.host_mr()};
    // traccc::edm::track_container<traccc::default_algebra>::host
    //     track_states_alpaka{m_vo.host_mr()};

    m_copy(measurements, measurements_per_event_alpaka)->wait();
    m_copy(spacepoints, spacepoints_per_event_alpaka)->wait();
    m_copy(seeds, seeds_alpaka)->wait();
    // m_copy(params, params_alpaka)->wait();
    // m_copy(track_candidates.tracks, track_candidates_alpaka)->wait();
    // m_copy(track_states.tracks, track_states_alpaka.tracks)->wait();
    // m_copy(track_states.states, track_states_alpaka.states)->wait();
    m_queue.synchronize();

    std::cout << "Number of spacepoints " << spacepoints_per_event_alpaka.size();

    if (show_stats)
    {
        end_total = std::chrono::high_resolution_clock::now();

        // Print timing information
        std::cout << "\n=== Timing Information ===" << std::endl;
        std::cout << "Read cells:          "
                << std::chrono::duration<double, std::milli>(end_read - start_read).count()
                << " ms" << std::endl;
        std::cout << "Copy to device:      "
                << std::chrono::duration<double, std::milli>(end_copy_in - start_copy_in).count()
                << " ms" << std::endl;
        std::cout << "Clusterization:      "
                << std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count()
                << " ms" << std::endl;
        std::cout << "Spacepoint form.:    "
                << std::chrono::duration<double, std::milli>(end_spacepoint - start_spacepoint).count()
                << " ms" << std::endl;
        std::cout << "Seeding:             "
                << std::chrono::duration<double, std::milli>(end_seeding - start_seeding).count()
                << " ms" << std::endl;
        // std::cout << "Track param. est.:   "
        //         << std::chrono::duration<double, std::milli>(end_params - start_params).count()
        //         << " ms" << std::endl;
        // std::cout << "Track finding:       "
        //         << std::chrono::duration<double, std::milli>(end_finding - start_finding).count()
        //         << " ms" << std::endl;
        // std::cout << "Track fitting:       "
        //         << std::chrono::duration<double, std::milli>(end_fitting - start_fitting).count()
        //         << " ms" << std::endl;
        // std::cout << "Copy to host:        "
        //         << std::chrono::duration<double, std::milli>(end_copy_out - start_copy_out).count()
        //         << " ms" << std::endl;
        // std::cout << "------------------------" << std::endl;
        // std::cout << "Total time:          "
        //         << std::chrono::duration<double, std::milli>(end_total - start_total).count()
        //         << " ms" << std::endl;
        // std::cout << "=========================\n" << std::endl;

        std::cout << "Number of measurements: " << m_copy.get_size(measurements) << std::endl;
        std::cout << "Number of spacepoints: " << m_copy.get_size(spacepoints) << std::endl;
        std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
        // std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
        // std::cout << "Number of found tracks:  " << m_copy.get_size(track_candidates.tracks) << std::endl;
        // std::cout << "Number of fitted tracks: " << track_states_host.tracks.size() << std::endl;

    }

    return {track_states_host, measurements_host};
}

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_from_array(
    const int64_t* cell_positions,
    const float* cell_properties,
    size_t num_cells,
    bool athena_ids = false)
{
    std::vector<traccc::io::csv::cell> cells;
    cells.reserve(num_cells);

    for (size_t i = 0; i < num_cells; ++i)
    {

        traccc::io::csv::cell cell;

        if (athena_ids){
            cell.geometry_id = m_athena_to_detray_map.at(cell_positions[i * 4]);
        } else {
            cell.geometry_id = cell_positions[i * 4];
        }
        cell.measurement_id = cell_positions[i * 4 + 1];
        cell.channel0 = cell_positions[i * 4 + 2];
        cell.channel1 = cell_positions[i * 4 + 3];

        cell.timestamp = cell_properties[i * 2];
        cell.value = cell_properties[i * 2 + 1];

        cells.push_back(cell);
    }

    return cells;
}

std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>>
    TracccGpuStandalone::read_all_cells(
        const std::vector<traccc::io::csv::cell> &cells)
{
    std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>> result;

    // Pre-count cells per geometry_id to avoid reallocations
    std::unordered_map<std::uint64_t, size_t> counts;
    for (const auto &cell : cells) {
        counts[cell.geometry_id]++;
    }

    // Reserve space for each geometry_id
    for (const auto& [geom_id, count] : counts) {
        result[geom_id].reserve(count);
    }

    for (const auto &iocell : cells)
    {
        result[iocell.geometry_id].emplace_back(iocell.geometry_id, iocell.measurement_id,
                            iocell.channel0, iocell.channel1,
                            iocell.timestamp, iocell.value);
    }

    // Sorting happens on the client side!
    // put here again for redundancy
    for (auto& [_, cells] : result)
    {
        std::sort(cells.begin(), cells.end(), ::cell_order());
    }

    return result;
}

void TracccGpuStandalone::read_cells(traccc::edm::silicon_cell_collection::host &out,
                const std::vector<traccc::io::csv::cell> &cells,
                bool deduplicate,
                bool use_acts_geometry_id)
{
    out.resize(0);
    out.reserve(cells.size());

    // get the cells and modules in intermediate format
    auto cellsMap = read_all_cells(cells);

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [geometry_id, cellz] : cellsMap) {

        // Figure out the index of the detector description object for this
        // group of cells.
        unsigned int ddIndex = 0;
        auto it = m_geomIdMap.find(geometry_id);
        if (it == m_geomIdMap.end()) {
            throw std::runtime_error("Could not find geometry ID (" +
                                        std::to_string(geometry_id) +
                                        ") in the detector description");
        }
        ddIndex = it->second;

        // Add the cells to the output.
        for (auto& cell : cellz) {
            out.push_back({cell.channel0, cell.channel1, cell.value,
                             cell.timestamp, ddIndex});
        }
    }
}

#endif
