#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>

// CUDA include(s).
#include <cuda_runtime.h>

// #include "DataHandler.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
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
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
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

struct clusterInfo {
    std::uint64_t detray_id; 
    unsigned int local_key;
    Eigen::Vector3d globalPosition;
    Eigen::Vector2d localPosition;
    bool pixel;
};

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

/// Comparison / ordering operator for measurements
struct measurement_sort_comp{
    bool operator()(clusterInfo& lhs, clusterInfo& rhs){

        if (lhs.detray_id != rhs.detray_id) {
            return lhs.detray_id < rhs.detray_id;
        } else if (lhs.localPosition[0] != rhs.localPosition[0]) {
            return lhs.localPosition[0] < rhs.localPosition[0];
        } else if (lhs.localPosition[1] != rhs.localPosition[1]) {
            return lhs.localPosition[1] < rhs.localPosition[1];
        } 
        return false;
    }
};


// Type definitions
using host_detector_type = traccc::default_detector::host;
using device_detector_type = traccc::default_detector::device;
using scalar_type = device_detector_type::scalar_type;

using stepper_type =
    detray::rk_stepper<detray::bfield::const_field_t<scalar_type>::view_t,
                   device_detector_type::algebra_type,
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
    /// barcode map
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> m_barcode_map;

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
    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;
    /// Ambiguity resolution (on the host)
    traccc::host::greedy_ambiguity_resolution_algorithm::config_type m_resolution_config;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    detray::bfield::const_field_t<traccc::scalar> m_field;

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

    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;
    
    // copy back!
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        m_copy_track_states;

    // ambiguity resolution
    traccc::host::greedy_ambiguity_resolution_algorithm m_resolution_alg;

public:
    TracccGpuStandalone( 
        vecmem::host_memory_resource *host_mr,
        vecmem::cuda::device_memory_resource *device_mr,
        int deviceID = 0) :
            m_device_id(deviceID), 
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
            m_finding_config(create_and_setup_finding_config()), 
            m_fitting_config(create_and_setup_fitting_config()), 
            m_resolution_config(),
            m_field_vec{0.f, 0.f, m_finder_config.bFieldInZ},
            m_field(detray::bfield::create_const_field<host_detector_type::scalar_type>(m_field_vec)),
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
            m_finding(m_finding_config, m_mr, m_copy, m_stream, 
                logger->cloneWithSuffix("TrackFindingAlg")),
            m_fitting(m_fitting_config, m_mr, m_copy, m_stream, 
                logger->cloneWithSuffix("TrackFittingAlg")),
            m_copy_track_states(m_mr, m_copy, logger->cloneWithSuffix("TrackStateD2HCopyAlg")),
            m_resolution_alg(m_resolution_config, m_mr, logger->cloneWithSuffix("AmbiguityResolutionAlg"))
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

    void read_spacepoints(
        traccc::edm::spacepoint_collection::host& spacepoints,
        std::vector<clusterInfo>& detray_clusters, bool do_strip);

    void read_measurements(
        traccc::measurement_collection_types::host& measurements,
        std::vector<clusterInfo>& detray_clusters, bool do_strip);

    std::vector<clusterInfo> read_clusters_from_csv(
        const std::string& filename);

    std::vector<clusterInfo> read_from_array(
        const std::vector<std::uint64_t> &geometry_ids,
        const std::vector<std::vector<double>> &data);

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

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, track_candidates);

    // Print fitting stats
    std::cout << "Number of measurements: " << measurements_per_event.size() << std::endl;
    std::cout << "Number of spacepoints: " << spacepoints_per_event.size() << std::endl;
    std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
    std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
    std::cout << "Number of track candidates: " << track_candidates.items.size() << std::endl;
    std::cout << "Number of fitted tracks: " << track_states.items.size() << std::endl;

    // copy track states to host
    //! Expensive with so many gd track states
    auto track_states_host = m_copy_track_states(track_states);

    // run ambiguity resolution
    // TODO: this should run before fitting, but requires copy back and forth first
    // TODO: ask traccc people about this
    // traccc::track_state_container_types::host resolved_track_states_cuda =
    //     m_resolution_alg(traccc::get_data(track_states_host));

    return track_states_host;
}

void TracccGpuStandalone::read_spacepoints(
    traccc::edm::spacepoint_collection::host& spacepoints,
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    traccc::measurement_collection_types::host measurements;
    read_measurements(measurements, detray_clusters, do_strip);

    std::map<traccc::geometry_id, unsigned int> m;
    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++){
        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel){continue;}

        // Construct the local 3D(2D) position of the measurement.
        // traccc::measurement meas;
        // meas = measurements[i];

        spacepoints.push_back({static_cast<unsigned int>(i), 
            traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX,
            {static_cast<float>(cluster.globalPosition[0]),
            static_cast<float>(cluster.globalPosition[1]),
            static_cast<float>(cluster.globalPosition[2])},
            0.f, 0.f});
    }
}


void TracccGpuStandalone::read_measurements(
    traccc::measurement_collection_types::host& measurements,
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    std::map<traccc::geometry_id, unsigned int> m;
    std::multimap<uint64_t,detray::geometry::barcode> sf_seen;

    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++){

        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel){continue;}

        uint64_t geometry_id = cluster.detray_id;
        const auto& sf = detray::geometry::barcode{geometry_id};
        const detray::tracking_surface surface{*m_detector, sf};
        cluster.localPosition[0] = cluster.localPosition.x();
        cluster.localPosition[1] = cluster.localPosition.y();

        // ATH_MSG_INFO("Traccc measurement at index " << i << ": " << cluster.localPosition[0] << "," << cluster.localPosition[1]);

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<detray::dsize_type<traccc::default_algebra>, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (((cluster.local_key) & (1 << (ipar + 1))) != 0) {

                switch (ipar) {
                    case 0: {
                        meas.local[0] = cluster.localPosition.x();
                        meas.variance[0] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case 1: {
                        meas.local[1] = cluster.localPosition.y();
                        meas.variance[1] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        meas.subs.set_indices(indices);
        meas.surface_link = detray::geometry::barcode{geometry_id};

        // Keeps measurement_id for ambiguity resolution
        meas.measurement_id = i;
        measurements.push_back(meas);
    }
}


std::vector<clusterInfo> TracccGpuStandalone::read_clusters_from_csv(
    const std::string& filename)
{
    std::vector<clusterInfo> clusters;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Read and discard the header line
    if (!std::getline(file, line)) {
         throw std::runtime_error("Could not read header line from file: " + filename);
    }

    // Read data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        clusterInfo cluster;
        int col_index = 0;
        double global_x, global_y, global_z;
        double local_x, local_y;
        int pixel_val;

        try {
            // Read each field separated by a comma
            while (std::getline(ss, field, ',')) {
                // Trim leading/trailing whitespace
                field.erase(0, field.find_first_not_of(" \t\n\r\f\v"));
                field.erase(field.find_last_not_of(" \t\n\r\f\v") + 1);

                switch (col_index) {
                    case 0: // atlas_id (ignored)
                        break;
                    case 1: // detray_id
                        cluster.detray_id = std::stoull(field);
                        break;
                    case 2: // measurement_id (ignored)
                        break;
                    case 3: // local_key
                        cluster.local_key = std::stoul(field);
                        break;
                    case 4: // local_x
                        local_x = std::stod(field);
                        break;
                    case 5: // local_y
                        local_y = std::stod(field);
                        break;
                    case 6: // global_x
                        global_x = std::stod(field);
                        break;
                    case 7: // global_y
                        global_y = std::stod(field);
                        break;
                    case 8: // global_z
                        global_z = std::stod(field);
                        break;
                    case 9: // pixel
                        pixel_val = std::stoi(field);
                        cluster.pixel = (pixel_val != 0);
                        break;
                    default:
                        // Handle unexpected extra columns if necessary
                        break;
                }
                col_index++;
            }
             // Check if we read the expected number of columns
            if (col_index != 10) {
                 std::cerr << "Warning: Row has " << col_index << " columns, expected 10. Line: " << line << std::endl;
                 continue; // Skip this row or handle error as appropriate
            }

            // Assign Eigen vectors after reading all components
            cluster.globalPosition = Eigen::Vector3d(global_x, global_y, global_z);
            cluster.localPosition = Eigen::Vector2d(local_x, local_y);

            clusters.push_back(cluster);

        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid argument during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Out of range during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        }
    }

    file.close();

    // print first 5 clusters to ensure they are read correctly
    std::cout << "Read " << clusters.size() << " clusters from file." << std::endl;
    for (size_t i = 0; i < std::min(clusters.size(), size_t(5)); ++i) {
        std::cout << "Cluster " << i << ": "
                  << "detray_id: " << clusters[i].detray_id
                  << ", local_key: " << clusters[i].local_key
                  << ", globalPosition: (" << clusters[i].globalPosition.transpose() << ")"
                  << ", localPosition: (" << clusters[i].localPosition.transpose() << ")"
                  << ", pixel: " << clusters[i].pixel
                  << std::endl;
    }

    return clusters;
}

std::vector<clusterInfo> TracccGpuStandalone::read_from_array(
    const std::vector<std::uint64_t> &geometry_ids,
    const std::vector<std::vector<double>> &data)
{
    std::vector<clusterInfo> clusters;

    if (geometry_ids.size() != data.size())
    {
        throw std::runtime_error("Number of geometry IDs and data rows do not match.");
    }

    for (size_t i = 0; i < data.size(); ++i) 
    {
        const auto& row = data[i];
        if (row.size() != 7)
            continue; 

        clusterInfo cluster;

        if (i < geometry_ids.size()) 
        {
            cluster.detray_id = geometry_ids[i];
        } 
        else 
        {
            continue;
        }

        cluster.local_key = static_cast<unsigned int>(row[0]);
        cluster.localPosition = Eigen::Vector2d(row[1], row[2]);
        cluster.globalPosition = Eigen::Vector3d(row[3], row[4], row[5]);
        cluster.pixel = (row[6] != 0);
        clusters.push_back(cluster);
    }

    return clusters;
}

#endif 