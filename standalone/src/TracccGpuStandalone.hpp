#include <iostream>
#include <memory>

// CUDA include(s).
#include <cuda_runtime.h>

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

std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells);
std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells);
void read_cells(traccc::edm::silicon_cell_collection::host &out, 
    const std::vector<traccc::io::csv::cell> &cells, 
    const traccc::silicon_detector_description::host* dd,
    bool deduplicate,
    bool use_acts_geometry_id);

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

struct TrackFittingResult
{
    std::vector<float> chi2;
    std::vector<float> ndf;
    std::vector<std::vector<std::array<float, 2>>> local_positions;
    std::vector<std::vector<std::array<float, 2>>> variances;
};

class TracccGpuStandalone
{
private:
    /// Device ID to use
    int m_device_id;

    /// Logger 
    std::unique_ptr<const traccc::Logger> logger;
    /// Host memory resource
    vecmem::host_memory_resource m_host_mr;
    /// CUDA stream to use
    traccc::cuda::stream m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
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

public:
    TracccGpuStandalone(int deviceID = 0) :
        m_device_id(deviceID), 
        logger(traccc::getDefaultLogger("TracccGpuStandalone", traccc::Logging::Level::INFO)),
        m_host_mr(),
        m_stream(setCudaDeviceAndGetStream(deviceID)),
        m_device_mr(deviceID),
        m_cached_device_mr(
            std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
        m_copy(m_stream.cudaStream()),
        m_mr{*m_cached_device_mr, &m_host_mr},
        m_propagation_config(m_propagation_opts),
        m_clustering_config{256, 16, 8, 256},
        m_finder_config(), 
        m_grid_config(m_finder_config), 
        m_filter_config(), 
        m_finding_config(), 
        m_fitting_config(), 
        m_field_vec{0.f, 0.f, m_finder_config.bFieldInZ},
        m_field(detray::bfield::create_const_field<host_detector_type::scalar_type>(m_field_vec)),
        m_det_descr{m_host_mr},
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
            logger->cloneWithSuffix("TrackFittingAlg"))
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

    void initialize();
    TrackFittingResult run(std::vector<traccc::io::csv::cell> cells);
    std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
    std::vector<std::vector<double>> read_from_csv(const std::string &filename);
    std::vector<traccc::io::csv::cell> 
        read_from_array(const std::vector<std::uint64_t> &geometry_ids,
                            const std::vector<std::vector<double>> &data);
};

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data_new/geometries/odd/odd-detray_geometry_detray.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data_new/geometries/odd/odd-digi-geometric-config.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data_new/geometries/odd/odd-detray_surface_grids_detray.json";
    m_detector_opts.material_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data_new/geometries/odd/odd-detray_material_detray.json";
    m_detector_opts.use_detray_detector = true;

    // Read the detector description
    traccc::io::read_detector_description(
        m_det_descr, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, traccc::data_format::json);
    traccc::silicon_detector_description::data m_det_descr_data{
        vecmem::get_data(m_det_descr)};
    m_device_det_descr = traccc::silicon_detector_description::buffer(
            static_cast<traccc::silicon_detector_description::buffer::size_type>(
                m_det_descr.size()),
            m_device_mr);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(m_det_descr_data, m_device_det_descr)->wait();

    // Create the detector and read the configuration file
    m_detector = std::make_unique<host_detector_type>(m_host_mr);
    traccc::io::read_detector(
        *m_detector, m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    
    // copy it to the device - dereference the unique_ptr to get the actual object
    m_device_detector = detray::get_buffer(*m_detector, m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    return;
}

TrackFittingResult TracccGpuStandalone::run(std::vector<traccc::io::csv::cell> cells)
{
    traccc::edm::silicon_cell_collection::host read_out(*m_mr.host);

    // Read the cells from the relevant event file into host memory.
    read_cells(read_out, cells, &m_det_descr, true, true);

    traccc::edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(read_out.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(read_out), cells_buffer)->ignore();

    //
    // ----------------- Clusterization -----------------
    // 
    const traccc::cuda::clusterization_algorithm::output_type measurements =
        m_clusterization(cells_buffer, m_device_det_descr);
    m_measurement_sorting(measurements);
    
    //
    // ----------------- Spacepoint Formation -----------------
    //  
    const spacepoint_formation_algorithm::output_type spacepoints =
        m_spacepoint_formation(m_device_detector_view, measurements);

    //
    // ----------------- Seeding and track param est. -----------
    //
    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
                                    m_seeding(spacepoints), m_field_vec);

    //
    // ----------------- Finding and Fitting -----------------
    //
    // track finding                        
    // Run the track finding (asynchronously).
    const finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);

    // Run the track fitting (asynchronously).
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, track_candidates);

    //
    // ----------------- Return fitted track (headers) -----------------
    // 

    // print number of fitted tracks
    std::cout << "Number of fitted tracks: " << track_states.headers.size() << std::endl;

    // create output type
    TrackFittingResult result;
            
    // // for now, only copy headers back
    // vecmem::vector<traccc::fitting_result<algebra::plugin::array<float>>> headers(*m_mr.host);
    // m_copy(track_states.headers, headers)->wait();

    // std::cout << "Number of headers: " << headers.size() << std::endl;

    // if (!headers.empty()) 
    // {
    //     result.chi2.reserve(headers.size());
    //     result.ndf.reserve(headers.size());
    //     result.local_positions.reserve(headers.size());
    //     result.variances.reserve(headers.size());

    //     for (size_t i = 0; i < headers.size(); ++i) 
    //     {
    //         const auto& header = headers[i];
    //         result.chi2.push_back(header.trk_quality.chi2);
    //         result.ndf.push_back(header.trk_quality.ndf);
    //     }
    // }

    return result;
}

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

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    std::cout << "Reading cells from " << filename << std::endl;

    while (reader.read(iocell))
    {
        cells.push_back(iocell);
    }

    std::cout << "Read " << cells.size() << " cells." << std::endl;

    return cells;
}

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_from_array(
    const std::vector<std::uint64_t> &geometry_ids,
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
        if (row.size() != 6)
            continue; 

        traccc::io::csv::cell iocell;

        iocell.geometry_id = static_cast<int>(row[0]);
        iocell.measurement_id = static_cast<int>(row[1]);
        iocell.channel0 = static_cast<int>(row[2]);
        iocell.channel1 = static_cast<int>(row[3]);
        iocell.timestamp = static_cast<int>(row[4]);
        iocell.value = row[5];

        cells.push_back(iocell);
    }

    return cells;
}

std::map<std::uint64_t, std::map<traccc::io::csv::cell, float, cell_order>> 
    fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, 
        unsigned int &nduplicates)
{
    std::map<std::uint64_t, std::map<traccc::io::csv::cell, float, cell_order>> cellMap;
    nduplicates = 0;

    for (const auto &iocell : cells)
    {
        traccc::io::csv::cell cell{iocell.geometry_id, iocell.measurement_id, 
                                iocell.channel0, iocell.channel1, 
                                iocell.timestamp, iocell.value};
        auto ret = cellMap[iocell.geometry_id].insert({iocell, iocell.value});
        if (ret.second == false) {
            cellMap[iocell.geometry_id].at(iocell) += iocell.value;
            ++nduplicates;
        }
    }

    return cellMap;
}

std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> 
    create_result_container(const std::map<std::uint64_t, 
        std::map<traccc::io::csv::cell, float, cell_order>> &cellMap)
{
    std::map<std::uint64_t, std::vector<traccc::io::csv::cell> > result;
    for (const auto& [geometry_id, cells] : cellMap) 
    {
        for (const auto& [cell, value] : cells) 
        {
            traccc::io::csv::cell summed_cell{cell};
            summed_cell.value = value;
            result[geometry_id].push_back(summed_cell);
        }
    }
    return result;
}

std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_deduplicated_cells(
    const std::vector<traccc::io::csv::cell> &cells)
{
    unsigned int nduplicates = 0;
    auto cellMap = fill_cell_map(cells, nduplicates);

    if (nduplicates > 0)
    {
        std::cout << "WARNING: " << nduplicates << " duplicate cells found." << std::endl;
    }

    return create_result_container(cellMap);
}

std::map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_all_cells(
    const std::vector<traccc::io::csv::cell> &cells)
{
    std::map<std::uint64_t, std::vector<traccc::io::csv::cell> > result;

    for (const auto &iocell : cells)
    {
        traccc::io::csv::cell cell{iocell.geometry_id, iocell.measurement_id, 
                            iocell.channel0, iocell.channel1, 
                            iocell.timestamp, iocell.value};
        result[iocell.geometry_id].push_back(cell);
    }

    // Sort the cells. Deduplication or not, they do need to be sorted.
    for (auto& [_, cells] : result) 
    {
        std::sort(cells.begin(), cells.end(), ::cell_order());
    }

    return result;
}

void read_cells(traccc::edm::silicon_cell_collection::host &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::silicon_detector_description::host* dd,
                bool deduplicate,
                bool use_acts_geometry_id)
{
    // clear output container
    out.resize(0u);

    // get the cells and modules in intermediate format
    auto cellsMap = (deduplicate ? read_deduplicated_cells(cells)
                                 : read_all_cells(cells));

    // If there is a detector description object, build a map of geometry IDs
    // to indices inside the detector description.
    std::map<traccc::geometry_id, unsigned int> geomIdMap;
    if (dd) {
        if (use_acts_geometry_id) {
            for (unsigned int i = 0; i < dd->acts_geometry_id().size(); ++i) {
                geomIdMap[dd->acts_geometry_id()[i]] = i;
            }
        } else {
            for (unsigned int i = 0; i < dd->geometry_id().size(); ++i) {
                geomIdMap[dd->geometry_id()[i].value()] = i;
            }
        }
    }

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [geometry_id, cellz] : cellsMap) {

        // Figure out the index of the detector description object, for this
        // group of cells.
        unsigned int ddIndex = 0;
        if (dd) {
            auto it = geomIdMap.find(geometry_id);
            if (it == geomIdMap.end()) {
                throw std::runtime_error("Could not find geometry ID (" +
                                         std::to_string(geometry_id) +
                                         ") in the detector description");
            }
            ddIndex = it->second;
        }


        // Add the cells to the output.
        for (auto& cell : cellz) {
            out.push_back({cell.channel0, cell.channel1, cell.value,
                             cell.timestamp, ddIndex});
        }
    }
}