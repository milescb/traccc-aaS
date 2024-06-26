// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
// #include "traccc/options/track_finding.hpp"
// #include "traccc/options/track_propagation.hpp"
// #include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"

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

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

class TracccGpuStandalone
{
private:
    int m_device_id;
    // memory resources
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
    // CUDA types used.
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};
    // inputs
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::accelerator accelerator_opts;
    // detector options
    traccc::geometry surface_transforms;
    std::unique_ptr<traccc::digitization_config> digi_cfg;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> barcode_map;

public:
    TracccGpuStandalone(int deviceID = 0) :
        m_device_id(deviceID)
    {
        initialize();
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    void initialize();
    void run();
};

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector file
    detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/trackml-detector.csv";
    detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/default-geometric-config-generic.json";

    // read in geometry
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file, traccc::data_format::csv);

    return;
}

void TracccGpuStandalone::run()
{

    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;

    host_detector_type host_detector{host_mr};
    host_detector_type::buffer_type device_detector;
    host_detector_type::view_type device_detector_view;
    if (detector_opts.use_detray_detector) {
        // Set up the detector reader configuration.
        detray::io::detector_reader_config cfg;
        cfg.add_file(traccc::io::data_directory() +
                     detector_opts.detector_file);
        if (detector_opts.material_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.material_file);
        }
        if (detector_opts.grid_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.grid_file);
        }

        // Read the detector.
        auto det = detray::io::read_detector<host_detector_type>(host_mr, cfg);
        host_detector = std::move(det.first);

        // Copy it to the device.
        device_detector = detray::get_buffer(detray::get_data(host_detector),
                                             device_mr, copy);
        stream.synchronize();
        device_detector_view = detray::get_data(device_detector);
    }

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements_cuda = 0;

    traccc::cuda::clusterization_algorithm ca_cuda(
        mr, copy, stream, clusterization_opts);
    traccc::cuda::measurement_sorting_algorithm ms_cuda(copy, stream);

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::host::clusterization_algorithm::output_type
            measurements_per_event;

        // Instantiate cuda containers/collections
        traccc::measurement_collection_types::buffer measurements_cuda_buffer(
            0, *mr.host);

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                traccc::io::read_cells(read_out_per_event, event,
                                       input_opts.directory, input_opts.format,
                                       &surface_transforms, &digi_cfg,
                                       barcode_map.get());
            }  // stop measuring file reading timer

            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (cuda)
            -----------------------------*/
            // Create device copy of input collections
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization (cuda)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                measurements_cuda_buffer =
                    ca_cuda(cells_buffer, modules_buffer);

                ms_cuda(measurements_cuda_buffer);
                stream.synchronize();

            }  // stop measuring clusterization cuda timer

        }  // Stop measuring wall time


        traccc::measurement_collection_types::host measurements_per_event_cuda;

        copy(measurements_cuda_buffer, measurements_per_event_cuda)->wait();

        // Print out measurements!
        std::cout << measurements_per_event_cuda.size() << std::endl;

        for (std::size_t i = 0; i < 10; ++i) {
            auto measurement = measurements_per_event_cuda.at(i);
            std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
            std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl; 
        }

        /// Statistics
        n_modules += read_out_per_event.modules.size();
        n_cells += read_out_per_event.cells.size();
        n_measurements_cuda += measurements_per_event_cuda.size();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created (cuda)  " << n_measurements_cuda
              << " measurements     " << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return;
}