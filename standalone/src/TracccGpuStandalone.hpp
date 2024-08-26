#include <iostream>
#include <memory>

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"

// clusterization
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/edm/cell.hpp"

#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
// #include "traccc/performance/collection_comparator.hpp"
// #include "traccc/performance/container_comparator.hpp"
// #include "traccc/performance/timer.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// CUDA include(s).
#include <cuda_runtime.h>

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

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the modules limits with the cell's
/// properties
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry* geom,
                               const traccc::digitization_config* dconfig,
                               const std::uint64_t original_geometry_id) {

    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(result.surface_link.value())) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }

        // Set the value on the module description.
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() >= 2);
        result.pixel = {binning_data[0].min, binning_data[1].min,
                        binning_data[0].step, binning_data[1].step};
    }

    return result;
}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells);
std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells);
void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells,
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map, 
                bool deduplicate);

// Type definitions
using host_detector_type = detray::detector<detray::default_metadata,
                                            detray::host_container_types>;
using device_detector_type =
    detray::detector<detray::default_metadata,
                        detray::device_container_types>;
using stepper_type =
    detray::rk_stepper<detray::bfield::const_field_t::view_t,
                        host_detector_type::algebra_type,
                        detray::constrained_step<>>;
using host_navigator_type = detray::navigator<const host_detector_type>;
using device_navigator_type = detray::navigator<const device_detector_type>;

using host_finding_algorithm =
    traccc::finding_algorithm<stepper_type, host_navigator_type>;
using device_finding_algorithm =
    traccc::cuda::finding_algorithm<stepper_type, device_navigator_type>;

using host_fitting_algorithm = 
    traccc::fitting_algorithm<traccc::kalman_fitter<stepper_type, host_navigator_type>>;
using device_fitting_algorithm =  
    traccc::cuda::fitting_algorithm<traccc::kalman_fitter<stepper_type, device_navigator_type>>;

class TracccGpuStandalone
{
private:
    int m_device_id;
    // memory resources
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr;
    // CUDA types used.
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy;
    // opt inputs
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::detector detector_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    // detray::propagation::config propagation_config;
    // detector options
    traccc::geometry surface_transforms;
    std::unique_ptr<traccc::digitization_config> digi_cfg;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> barcode_map;
    host_detector_type host_detector;
    host_detector_type::buffer_type device_detector;
    host_detector_type::view_type device_detector_view;
    detray::io::detector_reader_config cfg;
    host_finding_algorithm::config_type finding_cfg;
    host_fitting_algorithm::config_type fitting_cfg;
    // algorithms
    traccc::cuda::clusterization_algorithm ca_cuda;
    traccc::cuda::measurement_sorting_algorithm ms_cuda;
    traccc::cuda::spacepoint_formation_algorithm sf_cuda;
    traccc::cuda::seeding_algorithm sa_cuda;
    traccc::cuda::track_params_estimation tp_cuda;
    device_finding_algorithm finding_alg_cuda;
    device_fitting_algorithm fitting_alg_cuda;
    // field
    traccc::vector3 field_vec;
    detray::bfield::const_field_t field;
    // copying to cpu
    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        copy_track_candidates;
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        copy_track_states;

public:
    TracccGpuStandalone(int deviceID = 0) :
        m_device_id(deviceID), 
        host_mr(),
        cuda_host_mr(),
        device_mr(),
        mr{device_mr, &cuda_host_mr},
        stream(),
        copy(stream.cudaStream()),
        host_detector(host_mr),
        ca_cuda(mr, copy, stream, clusterization_opts.target_cells_per_partition),
        ms_cuda(copy, stream),
        sf_cuda(mr, copy, stream),
        sa_cuda(seeding_opts.seedfinder, {seeding_opts.seedfinder}, seeding_opts.seedfilter, mr, copy, stream),
        tp_cuda(mr, copy, stream),
        finding_alg_cuda(finding_cfg, mr, copy, stream),
        fitting_alg_cuda(fitting_cfg, mr, copy, stream),
        copy_track_candidates(mr, copy),
        copy_track_states(mr, copy)
    {
        //! Set the CUDA device totally doesn't work
        // cudaError_t err = cudaSetDevice(m_device_id);
        // if (err != cudaSuccess)
        // {
        //     throw std::runtime_error("Failed to set CUDA device: " \
        //                                 + std::string(cudaGetErrorString(err)));
        // }

        initialize();
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    void initialize();
    void run(std::vector<traccc::io::csv::cell> cells);
    std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
    std::vector<std::vector<double>> read_from_csv(const std::string &filename);
    std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<double>> &data);
};

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_geometry_detray.json";
    detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-digi-geometric-config.json";
    detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_surface_grids_detray.json";
    detector_opts.use_detray_detector = true;

    // read in geometry
    auto geom_data = traccc::io::read_geometry(detector_opts.detector_file, traccc::data_format::json);
    surface_transforms = std::move(geom_data.first);
    barcode_map = std::move(geom_data.second);

    // setup the detector
    cfg.add_file(detector_opts.detector_file);
    cfg.add_file(detector_opts.grid_file);

    // initialize the field
    field_vec = {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ};
    field = detray::bfield::create_const_field(field_vec);

    // Read the detector configuration file
    auto det = detray::io::read_detector<host_detector_type>(host_mr, cfg);
    host_detector = std::move(det.first);

    // Copy it to the device.
    device_detector = detray::get_buffer(detray::get_data(host_detector),
                                            device_mr, copy);
    stream.synchronize();
    device_detector_view = detray::get_data(device_detector);

    // Read the digitization configuration file
    digi_cfg = std::make_unique<traccc::digitization_config>(traccc::io::read_digitization_config(detector_opts.digitization_file));

    // initialize the track finding algorithm
    // finding_cfg.propagation = propagation_config;
    // fitting_cfg.propagation = propagation_config;
    finding_cfg.min_track_candidates_per_track =
        finding_opts.track_candidates_range[0];
    finding_cfg.max_track_candidates_per_track =
        finding_opts.track_candidates_range[1];
    finding_cfg.min_step_length_for_next_surface =
        finding_opts.min_step_length_for_next_surface;
    finding_cfg.max_step_counts_for_next_surface =
        finding_opts.max_step_counts_for_next_surface;
    finding_cfg.chi2_max = finding_opts.chi2_max;
    finding_cfg.max_num_branches_per_seed = finding_opts.nmax_per_seed;
    finding_cfg.max_num_skipping_per_cand =
        finding_opts.max_num_skipping_per_cand;
    propagation_opts.setup(finding_cfg.propagation);

    host_fitting_algorithm::config_type fitting_cfg;
    propagation_opts.setup(fitting_cfg.propagation);

    return;
}

void TracccGpuStandalone::run(std::vector<traccc::io::csv::cell> cells)
{
    traccc::io::cell_reader_output read_out(mr.host);

    // Read the cells from the relevant event file into host memory.
    read_cells(read_out, cells, &surface_transforms, digi_cfg.get(), barcode_map.get(), true);

    const traccc::cell_collection_types::host& cells_per_event =
        read_out.cells;
    const traccc::cell_module_collection_types::host&
        modules_per_event = read_out.modules;

    // create buffers and copy to device
    traccc::cell_collection_types::buffer cells_buffer(
        cells_per_event.size(), mr.main);
    copy(vecmem::get_data(cells_per_event), cells_buffer);
    traccc::cell_module_collection_types::buffer modules_buffer(
        modules_per_event.size(), mr.main);
    copy(vecmem::get_data(modules_per_event), modules_buffer);
    stream.synchronize();

    //
    // ----------------- Clusterization -----------------
    // 
    traccc::measurement_collection_types::buffer measurements_cuda_buffer(0, *mr.host);
    measurements_cuda_buffer = ca_cuda(cells_buffer, modules_buffer);
    ms_cuda(measurements_cuda_buffer);

    stream.synchronize();
    
    //
    // ----------------- Spacepoint Formation -----------------
    //  
    traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(0, *mr.host);
    spacepoints_cuda_buffer = sf_cuda(measurements_cuda_buffer, modules_buffer);

    //
    // ----------------- Seeding Algorithm -----------------
    //
    traccc::seed_collection_types::buffer seeds_cuda_buffer(0, *mr.host);
    seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);

    //
    // ----------------- Finding and Fitting -----------------
    //
    // track params estimation
    traccc::bound_track_parameters_collection_types::buffer params_cuda_buffer(0, *mr.host);
    params_cuda_buffer = tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer, field_vec);

    auto navigation_buffer = detray::create_candidates_buffer(
                                host_detector,
                                finding_cfg.navigation_buffer_size_scaler *
                                    copy.get_size(seeds_cuda_buffer),
                                mr.main, mr.host);

    // track finding                        
    traccc::track_candidate_container_types::buffer track_candidates_buffer;
    track_candidates_buffer = finding_alg_cuda(device_detector_view, field, navigation_buffer,
                                               measurements_cuda_buffer, params_cuda_buffer);

    // track fitting
    traccc::track_state_container_types::buffer track_states_buffer;
    track_states_buffer = fitting_alg_cuda(device_detector_view, field, navigation_buffer,
                     track_candidates_buffer);

    //
    // ----------------- Print Statistics -----------------
    // 
    std::cout << " done! " << std::endl;
    // copy buffer to host
    // traccc::measurement_collection_types::host measurements_per_event_cuda;
    // traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
    // traccc::seed_collection_types::host seeds_cuda;
    // traccc::bound_track_parameters_collection_types::host params_cuda;

    // copy(measurements_cuda_buffer, measurements_per_event_cuda)->wait();
    // copy(spacepoints_cuda_buffer, spacepoints_per_event_cuda)->wait();
    // copy(seeds_cuda_buffer, seeds_cuda)->wait();
    // copy(params_cuda_buffer, params_cuda)->wait();
    // auto track_candidates_cuda =
    //     copy_track_candidates(track_candidates_buffer);
    // auto track_states_cuda = copy_track_states(track_states_buffer);
    // stream.synchronize();

    // // print results
    // std::cout << " " << std::endl;
    // std::cout << "==> Statistics ... " << std::endl;
    // std::cout << " - number of measurements created " << measurements_per_event_cuda.size() << std::endl;
    // std::cout << " - number of spacepoints created " << spacepoints_per_event_cuda.size() << std::endl;
    // std::cout << " - number of seeds created " << seeds_cuda.size() << std::endl;
    // std::cout << " - number of track candidates created " << track_candidates_cuda.size() << std::endl;
    // std::cout << " - number of fitted tracks created " << track_states_cuda.size() << std::endl;

    // for (std::size_t i = 0; i < 10; ++i) {
    //     auto measurement = measurements_per_event_cuda.at(i);
    //     std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
    //     std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl; 
    // }

    return;
}

// deal with input data

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

std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, 
                                                                                    unsigned int &nduplicates)
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
                const std::map<std::uint64_t, 
                detray::geometry::barcode> *barcode_map, 
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

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_from_array(const std::vector<std::vector<double>> &data)
{
    std::vector<traccc::io::csv::cell> cells;

    for (const auto &row : data)
    {
        if (row.size() != 6)
            continue; // ensure each row contains exactly 6 elements
        traccc::io::csv::cell iocell;
        // FIXME needs to decode to the correct type
        iocell.geometry_id = static_cast<std::uint64_t>(row[0]);
        iocell.hit_id = static_cast<int>(row[1]);
        iocell.channel0 = static_cast<int>(row[2]);
        iocell.channel1 = static_cast<int>(row[3]);
        iocell.timestamp = static_cast<int>(row[4]);
        iocell.value = row[5];
        cells.push_back(iocell);
    }

    return cells;
}