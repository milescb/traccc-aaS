#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Project include(s).
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/io/read_cells.hpp"

#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << argv[0] << " <event_file>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    vecmem::host_memory_resource mem;
    using detector_type = detray::detector<detray::default_metadata,
                                           detray::host_container_types>;
    detector_type detector{mem};

    traccc::host::sparse_ccl_algorithm cc(mem);
    traccc::host::clusterization_algorithm ca(mem);

    traccc::io::cell_reader_output readOut(&mem);
    traccc::io::read_cells(readOut, event_file);
    traccc::cell_collection_types::host data = readOut.cells;
    // auto data_size = data.size();
    // for (std::size_t i = 0; i < 10; ++i) {
    //     std::cout << data.at(i).channel0 << std::endl;
    // }
    traccc::cell_collection_types::host& cells_per_event =
                readOut.cells;
    traccc::cell_module_collection_types::host& modules_per_event =
        readOut.modules;

    traccc::host::clusterization_algorithm::output_type
            measurements_per_event{&mem};
    measurements_per_event =
                    ca(vecmem::get_data(cells_per_event),
                       vecmem::get_data(modules_per_event));
    auto measurements_size = measurements_per_event.size();
    std::cout << "Number of measurements: " << measurements_size << std::endl;

    // for (const auto& measurement : measurements_per_event) {
    for (std::size_t i = 0; i < 10; ++i) {
        auto measurement = measurements_per_event.at(i);
        std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
        std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl;        // std::cout << "Variance: " << measurement.variance << std::endl;
        // std::cout << "Surface link: " << measurement.surface_link << std::endl;
        // std::cout << "Module link: " << measurement.module_link << std::endl;
        // std::cout << "Cluster link: " << measurement.cluster_link << std::endl;
        // std::cout << "Measurement dimension: " << measurement.meas_dim << std::endl;
        // std::cout << "Subspace: ";
        // for (auto elem : measurement.subs) {
        //     std::cout << elem << " ";
        // }
        // std::cout << std::endl;
    }

    return 0;
}