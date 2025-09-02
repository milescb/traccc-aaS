#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>
#include <chrono>
#include <unordered_map>

// CUDA include(s).
#include <cuda_runtime.h>

// local includes
#include "TracccEdmConversion.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

// magnetic field include(s).
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/io/read_magnetic_field.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/magnetic_field.hpp"

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
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"

// algorithm options
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/threading.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

#include "DataStructures.hpp"

// function to set the CUDA device and get the stream
static traccc::cuda::stream setCudaDeviceAndGetStream(int deviceID)
{
    cudaError_t err = cudaSetDevice(deviceID);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device: \
                " + std::string(cudaGetErrorString(err)));
    }
    return traccc::cuda::stream(deviceID);
}

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

// Definition of the cell_order struct
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

traccc::magnetic_field make_magnetic_field(std::string filename) {
    traccc::magnetic_field result;
    traccc::io::read_magnetic_field(result, filename, traccc::data_format::binary);
    return result;
}

// Type definitions
using host_detector_type = traccc::default_detector::host;
using device_detector_type = traccc::default_detector::device;
using scalar_type = traccc::default_detector::host::scalar_type;

using bfield_type =
    covfie::field<traccc::const_bfield_backend_t<scalar_type>>;
using stepper_type =
    detray::rk_stepper<bfield_type::view_t,
                        traccc::default_detector::host::algebra_type,
                        detray::constrained_step<scalar_type>>;
using navigator_type = detray::navigator<const device_detector_type>;
using device_navigator_type = detray::navigator<const device_detector_type>;

using spacepoint_formation_algorithm =
    traccc::cuda::spacepoint_formation_algorithm<
        traccc::default_detector::device>;
using clustering_algorithm = traccc::cuda::clusterization_algorithm;
using finding_algorithm =
    traccc::cuda::combinatorial_kalman_filter_algorithm;
using fitting_algorithm = traccc::cuda::kalman_fitting_algorithm;

// fitting and finding params
static traccc::seedfinder_config create_and_setup_finder_config() {
    traccc::seedfinder_config cfg;
    // Set desired values to match reference
    cfg.zMin = -3000.f * traccc::unit<float>::mm;
    cfg.zMax = 3000.f * traccc::unit<float>::mm;
    cfg.rMax = 320.f * traccc::unit<float>::mm;
    cfg.rMin = 33.f * traccc::unit<float>::mm;
    cfg.collisionRegionMin = -200.f * traccc::unit<float>::mm;
    cfg.collisionRegionMax = 200.f * traccc::unit<float>::mm;
    cfg.minPt = 900.f * traccc::unit<float>::MeV;  // Changed from 500
    cfg.cotThetaMax = 27.2899f;
    cfg.deltaRMin = 20.f * traccc::unit<float>::mm;
    cfg.deltaRMax = 280.f * traccc::unit<float>::mm;
    cfg.impactMax = 10.f * traccc::unit<float>::mm;  // Changed from 2.f
    cfg.sigmaScattering = 3.0f;  // Changed from 2.0f
    cfg.maxPtScattering = 10.f * traccc::unit<float>::GeV;
    cfg.maxSeedsPerSpM = 2;  // Changed from 3
    cfg.radLengthPerSeed = 0.05f;  // Explicitly set
    // cfg.bFieldInZ uses its default (1.99724f T) unless set here

    cfg.setup(); // Call setup() again with the new values
    return cfg;
}

// Helper function to create and setup seedfilter config
static traccc::seedfilter_config create_and_setup_filter_config() {
    traccc::seedfilter_config cfg;
    cfg.maxSeedsPerSpM = 2;
    cfg.good_spB_min_radius = 150.f * traccc::unit<float>::mm;
    cfg.good_spB_weight_increase = 400.f;
    cfg.good_spT_max_radius = 150.f * traccc::unit<float>::mm;
    cfg.good_spT_weight_increase = 200.f;
    cfg.good_spB_min_weight = 380.f;
    cfg.seed_min_weight = 200.f;
    cfg.spB_min_radius = 43.f * traccc::unit<float>::mm;
    return cfg;
}

// Helper function to create and setup finding_algorithm::config_type
static finding_algorithm::config_type create_and_setup_finding_config() {
    finding_algorithm::config_type cfg{
        .max_num_branches_per_seed = 3,                
        .max_num_branches_per_surface = 5,
        .min_track_candidates_per_track = 7,
        .max_track_candidates_per_track = 20,
        .max_num_skipping_per_cand = 2,
        .min_step_length_for_next_surface = 0.5f * detray::unit<float>::mm,
        .max_step_counts_for_next_surface = 100,
        .chi2_max = 10.f,
        .propagation = { 
            .navigation = { 
                .overstep_tolerance = -300.f * traccc::unit<float>::um
            },
            .stepping = {
                .min_stepsize = 1e-4f * traccc::unit<float>::mm,
                .rk_error_tol = 1e-4f * traccc::unit<float>::mm,
                .step_constraint = std::numeric_limits<float>::max(),
                .path_limit = 5.f * traccc::unit<float>::m,
                .max_rk_updates = 10000u,
                .use_mean_loss = true,
                .use_eloss_gradient = false,
                .use_field_gradient = false,
                .do_covariance_transport = true
            }
        }
    };
    return cfg;
}

// Helper function to create and setup fitting_algorithm::config_type
static fitting_algorithm::config_type create_and_setup_fitting_config() {
    fitting_algorithm::config_type cfg{
        .propagation = { 
            .navigation = {
                .min_mask_tolerance = 1e-5f * traccc::unit<float>::mm,
                .max_mask_tolerance = 3.f * traccc::unit<float>::mm,
                .overstep_tolerance = -300.f * traccc::unit<float>::um,
                .search_window = {0u, 0u}
            }
        }
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
    /// Host memory resource
    vecmem::host_memory_resource *m_host_mr;
    /// CUDA stream to use
    traccc::cuda::stream m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource *m_device_mr;
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// (Asynchronous) memory copy object
    mutable vecmem::cuda::async_copy m_copy;
    /// Memory resource for the host memory
    traccc::memory_resource m_mr;

    /// data configuration
    traccc::geometry m_surface_transforms;
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
    /// propagation options
    traccc::opts::track_propagation m_propagation_opts;
    /// clusterization options
    detray::propagation::config m_propagation_config;
    /// Configuration for clustering
    traccc::clustering_config m_clustering_config;
    /// Configuration for the seed finding
    traccc::seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    traccc::spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    traccc::seedfilter_config m_filter_config;

    /// further configuration
    /// Configuration for ambiguity resolution
    traccc::host::greedy_ambiguity_resolution_algorithm::config_type m_resolution_config;
    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;

    /// Magnetic field options
    traccc::opts::magnetic_field m_bfield_opts;
    /// host field object
    traccc::magnetic_field m_host_field;
    /// device field object
    traccc::magnetic_field m_field;
    /// const. field for the track finding and fitting
    traccc::vector3 m_field_vec;

    /// Detector description
    traccc::silicon_detector_description::host m_det_descr;
    /// Detector description buffer
    traccc::silicon_detector_description::buffer m_device_det_descr;
    /// Host detector
    std::unique_ptr<host_detector_type> m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// Sub-algorithms used by this full-chain algorithm
    /// Clusterization algorithm
    clustering_algorithm m_clusterization;
    /// Measurement sorting algorithm
    traccc::cuda::measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    traccc::cuda::seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    traccc::cuda::track_params_estimation m_track_parameter_estimation;

    /// Resolution algorithm
    traccc::cuda::greedy_ambiguity_resolution_algorithm m_resolution;
    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;
    
    // copy back!
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        m_copy_track_states;

    // Helper function to read in cells
    std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_all_cells(
        const std::vector<traccc::io::csv::cell> &cells);

    void read_cells(traccc::edm::silicon_cell_collection::host &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::silicon_detector_description::host* dd,
                bool deduplicate,
                bool use_acts_geometry_id);

public:
    TracccGpuStandalone( 
        vecmem::host_memory_resource *host_mr,
        vecmem::cuda::device_memory_resource *device_mr,
        int deviceID = 0,
        const std::string& geoDir = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-01/itk-geo/") :
            m_device_id(deviceID), 
            m_geoDir(geoDir),
            logger(traccc::getDefaultLogger("TracccGpuStandalone", traccc::Logging::Level::INFO)),
            m_host_mr(host_mr),
            m_stream(setCudaDeviceAndGetStream(deviceID)),
            m_device_mr(device_mr),
            m_cached_device_mr(
                std::make_unique<vecmem::binary_page_memory_resource>(*m_device_mr)),
            m_copy(m_stream.cudaStream()),
            m_mr{*m_cached_device_mr, m_host_mr},
            m_propagation_config(m_propagation_opts),
            m_clustering_config{256, 16, 8, 256},
            m_finder_config(create_and_setup_finder_config()), // Initialize m_finder_config using the helper
            m_grid_config(m_finder_config), 
            m_filter_config(create_and_setup_filter_config()), 
            m_resolution_config(),
            m_finding_config(create_and_setup_finding_config()), 
            m_fitting_config(create_and_setup_fitting_config()), 
            m_bfield_opts(),
            m_host_field(make_magnetic_field(geoDir + "ITk_bfield.cvf")),
            m_field(traccc::cuda::make_magnetic_field(m_host_field)),
            m_field_vec({0.f, 0.f, m_bfield_opts.value}),
            m_det_descr{*m_host_mr},
            m_clusterization(m_mr, m_copy, m_stream, m_clustering_config),
            m_measurement_sorting(m_mr, m_copy, m_stream, 
                logger->cloneWithSuffix("MeasSortingAlg")),
            m_spacepoint_formation(m_mr, m_copy, m_stream,
                logger->cloneWithSuffix("SpFormationAlg")),
            m_seeding(m_finder_config, m_grid_config, m_filter_config, 
                        m_mr, m_copy, m_stream,
                        logger->cloneWithSuffix("SeedingAlg")),
            m_track_parameter_estimation(m_mr, m_copy, m_stream,
                logger->cloneWithSuffix("TrackParEstAlg")),
            m_resolution(m_resolution_config, m_mr, m_copy, m_stream,
                logger->cloneWithSuffix("ResolutionAlg")),
            m_finding(m_finding_config, m_mr, m_copy, m_stream, 
                logger->cloneWithSuffix("TrackFindingAlg")),
            m_fitting(m_fitting_config, m_mr, m_copy, m_stream, 
                logger->cloneWithSuffix("TrackFittingAlg")),
            m_copy_track_states(m_mr, m_copy, logger->cloneWithSuffix("TrackStateD2HCopyAlg"))
    {
        // Tell the user what device is being used.
        int device = 0;
        CUDA_ERROR_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << "Using CUDA device: " << props.name << " [id: " << device
                << ", bus: " << props.pciBusID
                << ", device: " << props.pciDeviceID << "]" << std::endl;

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

    traccc::track_state_container_types::host run(
        std::vector<traccc::io::csv::cell> cells);

};


void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = m_geoDir + "/ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = m_geoDir + "/ITk_digitization_config_with_strips_with_shift_annulus_flip.json";
    m_detector_opts.grid_file = m_geoDir + "/ITk_DetectorBuilder_surface_grids.json";
    m_detector_opts.material_file = m_geoDir + "/ITk_detector_material.json";
    m_detector_opts.material_file = m_geoDir + "/ITk_detector_material.json";

    // Load Athena-to-Detray mapping
    std::string athenaTransformsPath = m_geoDir + "/athenaIdentifierToDetrayMap.txt";
    m_athena_to_detray_map = read_athena_to_detray_mapping(athenaTransformsPath);

    // Create reverse mapping from Detray to Athena
    m_detray_to_athena_map.reserve(m_athena_to_detray_map.size());
    for (const auto& [athena_id, detray_id] : m_athena_to_detray_map) {
        m_detray_to_athena_map[detray_id] = athena_id;
    }

    // Read the detector description
    traccc::io::read_detector_description(
        m_det_descr, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, traccc::data_format::json);
    traccc::silicon_detector_description::data m_det_descr_data{
        vecmem::get_data(m_det_descr)};
    m_device_det_descr = traccc::silicon_detector_description::buffer(
            static_cast<traccc::silicon_detector_description::buffer::size_type>(
                m_det_descr.size()),
            *m_device_mr);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(m_det_descr_data, m_device_det_descr)->wait();

    // fill the det description to geometry id map
    m_geomIdMap.clear();
    m_geomIdMap.reserve(m_det_descr.geometry_id().size());
    for (unsigned int i = 0; i < m_det_descr.geometry_id().size(); ++i) {
        m_geomIdMap[m_det_descr.geometry_id()[i].value()] = i;
    }

    // Create the detector and read the configuration file
    m_detector = std::make_unique<host_detector_type>(*m_host_mr);
    traccc::io::read_detector(
        *m_detector, *m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    
    // copy it to the device - dereference the unique_ptr to get the actual object
    m_device_detector = detray::get_buffer(*m_detector, *m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    return;
}

traccc::track_state_container_types::host TracccGpuStandalone::run(
    std::vector<traccc::io::csv::cell> cells
) {
    traccc::edm::silicon_cell_collection::host read_out(*m_mr.host);

    // Read the cells from the relevant event into host memory.
    read_cells(read_out, cells, &m_det_descr, true, false);

    traccc::edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(read_out.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(read_out), cells_buffer)->ignore();

    // Clusterization
    traccc::measurement_collection_types::buffer measurements =
        m_clusterization(cells_buffer, m_device_det_descr);
    m_measurement_sorting(measurements);
    
    // Spacepoint formation
    traccc::edm::spacepoint_collection::buffer spacepoints =
        m_spacepoint_formation(m_device_detector_view, measurements);

    // Seeding and track param est.
    traccc::edm::seed_collection::buffer seeds = m_seeding(spacepoints);
    m_stream.synchronize();

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds, m_field_vec);
    m_stream.synchronize();
                     
    // Run the track finding
    const finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);       

    //  // Run the resolution algorithm on the candidates
    // traccc::edm::track_candidate_collection<traccc::default_algebra>::buffer 
    //     res_track_candidates = m_resolution({track_candidates, measurements});

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, 
            {track_candidates, measurements});

    // Print fitting stats
    // // TODO: remove this in production code, add ability to select at initialization
    // std::cout << "Number of measurements: " << m_copy.get_size(measurements) << std::endl;
    // std::cout << "Number of spacepoints: " << m_copy.get_size(spacepoints) << std::endl;
    // std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
    // std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
    // std::cout << "Number of track candidates: " << m_copy.get_size(track_candidates) << std::endl;
    // std::cout << "Number of resolved track candidates: " << m_copy.get_size(res_track_candidates) << std::endl;
    // std::cout << "Number of fitted tracks: " << track_states.headers.size() << std::endl;

    // copy track states to host
    auto track_states_host = m_copy_track_states(track_states);

    // filter out tracks with ndf < 1
    traccc::track_state_container_types::host filtered_track_states;
    size_t initial_count = track_states_host.size();

    for (size_t i = 0; i < initial_count; ++i) {
        const auto& [header, items] = track_states_host.at(i);
        if (header.trk_quality.ndf >= 1) {
            filtered_track_states.push_back(header, items);
        }
    }

    size_t removed_count = initial_count - filtered_track_states.size();
    if (removed_count > 0) {
        std::cout << "Warning: " << removed_count 
                  << " tracks failed to fit (ndf < 1) and were removed." 
                  << std::endl;
    }

    return filtered_track_states;
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
    // for (auto& [_, cells] : result) 
    // {
    //     std::sort(cells.begin(), cells.end(), ::cell_order());
    // }

    return result;
}

void TracccGpuStandalone::read_cells(traccc::edm::silicon_cell_collection::host &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::silicon_detector_description::host* dd,
                bool deduplicate,
                bool use_acts_geometry_id)
{
    out.resize(0);
    out.reserve(cells.size());

    // get the cells and modules in intermediate format
    auto cellsMap = read_all_cells(cells);

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [geometry_id, cellz] : cellsMap) {

        // Figure out the index of the detector description object, for this
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