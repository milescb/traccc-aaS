#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Project include(s).
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/edm/cell.hpp"

#include "traccc/options/detector.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"



#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).


// Function to split a string by a delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// std::vector<std::tuple<long long, int, int, int, int, double>> readCSV(const std::string& filename) {
//     std::ifstream file(filename);
//     std::vector<std::tuple<long long, int, int, int, int, double>> data;
//     std::string line;
    
//     if (file.is_open()) {
//         std::getline(file, line); // Skip the header line
//         while (std::getline(file, line)) {
//             std::vector<std::string> row = split(line, ',');
//             long long geometry_id = std::stoll(row[0]);
//             int hit_id = std::stoi(row[1]);
//             int channel0 = std::stoi(row[2]);
//             int channel1 = std::stoi(row[3]);
//             int timestamp = std::stoi(row[4]);
//             double value = std::stod(row[5]);
//             data.emplace_back(geometry_id, hit_id, channel0, channel1, timestamp, value);
//         }
//         file.close();
//     } else {
//         std::cerr << "Unable to open file";
//     }
    
//     return data;
// }

// FIXME idk how to use the current cell_order struct
// seems annoymouns namespace is not working, so I copy the definition
// this file: /workspace/traccc/io/src/csv/read_cells.cpp
struct cell_order {
    bool operator()(const traccc::cell& lhs, const traccc::cell& rhs) const {
        if (lhs.module_link != rhs.module_link) {
            return lhs.module_link < rhs.module_link;
        } else if (lhs.channel1 != rhs.channel1) {
            return lhs.channel1 < rhs.channel1;
        } else {
            return lhs.channel0 < rhs.channel0;
        }
    }
};

// FIXME idk how to import the get_module function
// this file: /workspace/traccc/io/src/csv/read_cells.cpp
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
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1) {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
        result.pixel.variance_y = geo_it->variance_y;
    }

    return result;
}

// read from cells and make map
std::map<std::uint64_t, std::map<traccc::cell, float, cell_order> >
fill_cell_map(const std::vector<traccc::io::csv::cell>& cells, unsigned int& nduplicates) {
    std::map<std::uint64_t, std::map<traccc::cell, float, cell_order> > cellMap;
    nduplicates = 0;

    for (const auto& iocell : cells) {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        auto ret = cellMap[iocell.geometry_id].insert({cell, iocell.value});
        if (!ret.second) {
            cellMap[iocell.geometry_id].at(cell) += iocell.value;
            ++nduplicates;
        }
    }

    return cellMap;
}

std::map<std::uint64_t, std::vector<traccc::cell> >
create_result_container(const std::map<std::uint64_t, std::map<traccc::cell, float, cell_order> >& cellMap) {
    std::map<std::uint64_t, std::vector<traccc::cell> > result;
    for (const auto& [geometry_id, cells] : cellMap) {
        for (const auto& [cell, value] : cells) {
            traccc::cell summed_cell{cell};
            summed_cell.activation = value;
            result[geometry_id].push_back(summed_cell);
        }
    }
    return result;
}

std::vector<traccc::io::csv::cell> read_csv(const std::string& filename) {
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    while (reader.read(iocell)) {
        cells.push_back(iocell);
    }

    return cells;
}

std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<double>>& data) {
    std::vector<traccc::io::csv::cell> cells;

    for (const auto& row : data) {
        if (row.size() != 6) continue; // ensure each row contains exactly 6 elements
        traccc::io::csv::cell iocell;
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

std::map<std::uint64_t, std::vector<traccc::cell> >
read_deduplicated_cells(const std::vector<traccc::io::csv::cell>& cells) {
    unsigned int nduplicates = 0;
    auto cellMap = fill_cell_map(cells, nduplicates);

    if (nduplicates > 0) {
        std::cout << "WARNING: " << nduplicates << " duplicate cells found." << std::endl;
    }

    return create_result_container(cellMap);
}

std::map<std::uint64_t, std::vector<traccc::cell> >
read_all_cells(const std::vector<traccc::io::csv::cell>& cells) {
    std::map<std::uint64_t, std::vector<traccc::cell> > result;

    for (const auto& iocell : cells) {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        result[iocell.geometry_id].push_back(cell);
    }

    return result;
}

void read_cells(
    traccc::io::cell_reader_output& out, std::vector<traccc::io::csv::cell> cells, const traccc::geometry* geom,
    const traccc::digitization_config* dconfig,
    const std::map<std::uint64_t, detray::geometry::barcode>* barcode_map,
    const bool deduplicate) {

    // auto cells = read_csv(filename);

    // Get the cells and modules into an intermediate format.
    auto cellsMap = (deduplicate ? read_deduplicated_cells(cells)
                                 : read_all_cells(cells));

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [original_geometry_id, cells] : cellsMap) {
        // Modify the geometry ID of the module if a barcode map is
        // provided.
        std::uint64_t geometry_id = original_geometry_id;
        if (barcode_map != nullptr) {
            const auto it = barcode_map->find(geometry_id);
            if (it != barcode_map->end()) {
                geometry_id = it->second.value();
            } else {
                throw std::runtime_error(
                    "Could not find barcode for geometry ID " +
                    std::to_string(geometry_id));
            }
        }

        // Add the module and its cells to the output.
        out.modules.push_back(
            get_module(geometry_id, geom, dconfig, original_geometry_id));
        for (auto& cell : cells) {
            out.cells.push_back(cell);
            // Set the module link.
            out.cells.back().module_link = out.modules.size() - 1;
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << argv[0] << " <event_file>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    vecmem::host_memory_resource mem;

    traccc::opts::detector detector_opts;
    detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/trackml-detector.csv";
    detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/default-geometric-config-generic.json";

    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));
    using detector_type = detray::detector<detray::default_metadata,
                                           detray::host_container_types>;
    detector_type detector{mem};
    // Set up the detector reader configuration.
    detray::io::detector_reader_config cfg;
    cfg.add_file(traccc::io::data_directory() +
                    detector_opts.detector_file);

    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);


    traccc::host::sparse_ccl_algorithm cc(mem);
    traccc::host::clusterization_algorithm ca(mem);

    traccc::io::cell_reader_output readOut(&mem);

    bool deduplicate = true;

    // auto csv_cells = read_csv(event_file);
    // auto result_csv = read_deduplicated_cells(csv_cells);
    // read_cells(readOut, event_file, &surface_transforms, &digi_cfg, barcode_map.get(), deduplicate);

    auto cells = read_csv(event_file);
    read_cells(readOut, cells, &surface_transforms, &digi_cfg, barcode_map.get(), deduplicate);


    // traccc::io::read_cells(readOut, event_file);
    // traccc::io::read_cells(readOut, event_file,
    //                         traccc::data_format::csv,
    //                         &surface_transforms,
    //                         &digi_cfg, barcode_map.get());
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