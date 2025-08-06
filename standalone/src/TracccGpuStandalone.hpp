#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>

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
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/bfield.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

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
    traccc::cuda::finding_algorithm<stepper_type, navigator_type>;
using fitting_algorithm = traccc::cuda::fitting_algorithm<
    traccc::kalman_fitter<stepper_type, navigator_type>>;

// fitting and finding params
static traccc::seedfinder_config create_and_setup_finder_config() {
    traccc::seedfinder_config cfg;
    // Set desired values
    cfg.zMin = -3000.f * traccc::unit<float>::mm;
    cfg.zMax = 3000.f * traccc::unit<float>::mm;
    cfg.rMax = 320.f * traccc::unit<float>::mm;
    cfg.rMin = 33.f * traccc::unit<float>::mm;
    cfg.collisionRegionMin = -200.f * traccc::unit<float>::mm;
    cfg.collisionRegionMax = 200.f * traccc::unit<float>::mm;
    cfg.minPt = 500.f * traccc::unit<float>::MeV;
    cfg.cotThetaMax = 27.2899f;
    cfg.deltaRMin = 20.f * traccc::unit<float>::mm;
    cfg.deltaRMax = 280.f * traccc::unit<float>::mm;
    cfg.impactMax = 2.f * traccc::unit<float>::mm;
    cfg.sigmaScattering = 2.0f;
    cfg.maxPtScattering = 10.f * traccc::unit<float>::GeV;
    cfg.maxSeedsPerSpM = 3;
    // cfg.bFieldInZ uses its default (1.99724f T) unless set here
    // cfg.radLengthPerSeed uses its default (0.05f) unless set here

    cfg.setup(); // Call setup() again with the new values
    return cfg;
}

// Helper function to create and setup finding_algorithm::config_type
static finding_algorithm::config_type create_and_setup_finding_config() {
    finding_algorithm::config_type cfg{
        .max_num_branches_per_seed = 3,                
        .max_num_branches_per_surface = 5,
        .min_track_candidates_per_track = 3,
        .max_track_candidates_per_track = 20,
        .max_num_skipping_per_cand = 3,
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
        // .ptc_hypothesis and .initial_links_per_seed will use their defaults
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

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    covfie::field<traccc::const_bfield_backend_t<traccc::scalar>> m_field;

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

public:
    TracccGpuStandalone( 
        vecmem::host_memory_resource *host_mr,
        vecmem::cuda::device_memory_resource *device_mr,
        int deviceID = 0,
        const std::string& geoDir = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/") :
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
            m_filter_config(), 
            m_resolution_config(),
            m_finding_config(create_and_setup_finding_config()), 
            m_fitting_config(create_and_setup_fitting_config()), 
            m_field_vec{0.f, 0.f, m_finder_config.bFieldInZ},
            m_field(traccc::construct_const_bfield<traccc::scalar>(m_field_vec)),
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

    std::vector<InputData> read_from_array(
        const int64_t* geometry_ids,
        const float* data,
        size_t num_rows,
        size_t num_features);

    // getters
    const std::map<int64_t, uint64_t>& getAthenaToDetrayMap() const {
        return m_athena_to_detray_map;
    }

    const std::unordered_map<uint64_t, int64_t>& getDetrayToAthenaMap() const {
        return m_detray_to_athena_map;
    }

    void initialize();

    traccc::track_state_container_types::host run(
        traccc::edm::spacepoint_collection::host spacepoints_per_event,
        traccc::measurement_collection_types::host measurements_per_event);

};


void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_digitization_config_with_strips.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_surface_grids.json";
    m_detector_opts.material_file = "";

    // Load Athena-to-Detray mapping
    std::string athenaTransformsPath = m_geoDir + "geometry_mappings.csv";
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
    traccc::edm::spacepoint_collection::host spacepoints_per_event,
    traccc::measurement_collection_types::host measurements_per_event)
{   
    // copy spacepoints and measurements to device
    traccc::edm::spacepoint_collection::buffer spacepoints(
        static_cast<unsigned int>(spacepoints_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(spacepoints_per_event), spacepoints)->wait();

    traccc::measurement_collection_types::buffer measurements(
        static_cast<unsigned int>(measurements_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(measurements_per_event), measurements)->wait();

    // Seeding and track param est.
    auto seeds = m_seeding(spacepoints);
    m_stream.synchronize();

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds, m_field_vec);
    m_stream.synchronize();

    // track finding                        
    // Run the track finding
    const finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);

     // Run the resolution algorithm on the candidates
    traccc::edm::track_candidate_collection<traccc::default_algebra>::buffer 
        res_track_candidates = m_resolution({track_candidates, measurements});

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, 
            {res_track_candidates, measurements});

    // Print fitting stats
    // TODO: remove this in production code, add ability to select at initialization
    std::cout << "Number of measurements: " << measurements_per_event.size() << std::endl;
    std::cout << "Number of spacepoints: " << spacepoints_per_event.size() << std::endl;
    std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
    std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
    std::cout << "Number of track candidates: " << m_copy.get_size(track_candidates) << std::endl;
    std::cout << "Number of resolved track candidates: " << m_copy.get_size(res_track_candidates) << std::endl;
    std::cout << "Number of fitted tracks: " << track_states.headers.size() << std::endl;

    // copy track states to host
    auto track_states_host = m_copy_track_states(track_states);

    return track_states_host;
}

std::vector<InputData> TracccGpuStandalone::read_from_array(
    const int64_t* geometry_ids,
    const float* data,
    size_t num_rows,
    size_t num_features)
{
    std::vector<InputData> features;
    features.reserve(num_rows);

    for (size_t i = 0; i < num_rows; ++i) 
    {
        const float* row = &data[i * num_features];

        InputData cluster;

        cluster.athena_id_1 = geometry_ids[i * 2];
        cluster.athena_id_2 = geometry_ids[i * 2 + 1];

        // spacepoint info
        cluster.sp_x = row[0];
        cluster.sp_y = row[1];
        cluster.sp_z = row[2];

        //cluster info
        cluster.loc_eta_1 = row[3];
        cluster.loc_phi_1 = row[4];
        cluster.loc_eta_2 = row[5];
        cluster.loc_phi_2 = row[6];

        features.push_back(cluster);
    }

    return features;
}

#endif 