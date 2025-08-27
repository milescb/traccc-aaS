#pragma once 

#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/definitions/common.hpp"

#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

#include <iostream>
#include <map>          
#include <unordered_map>
#include <string>       
#include <sstream>  

#include "DataStructures.hpp"

/// Structure for combined mapping CSV
// struct geometry_mapping {
//     int64_t ath_geoid;
//     uint64_t detray_id;

//     DFE_NAMEDTUPLE(geometry_mapping, ath_geoid, detray_id);
// };

// /// Create csv reader for combined mapping
// inline dfe::NamedTupleCsvReader<geometry_mapping> make_mapping_reader(
//     std::string_view filename) {
//     return {filename.data(), {"ath_geoid", "detray_id"}};
// }

// /// Read Athena-to-Detray geometry ID mapping from combined CSV file
// inline std::map<int64_t, uint64_t> read_athena_to_detray_mapping(
//     std::string_view filename) {

//     auto reader = make_mapping_reader(filename);
//     std::map<int64_t, uint64_t> result;
//     geometry_mapping mapping;

//     while (reader.read(mapping)) {
//         // athena_id -> detray_id
//         result.insert({mapping.ath_geoid, mapping.detray_id});
//     }
    
//     std::cout << "Loaded " << result.size() << " athena->detray mappings from: " << filename << std::endl;
//     return result;
// }

inline std::map<int64_t, uint64_t> read_athena_to_detray_mapping(
    std::string_view filename) {

    std::ifstream file(filename.data());
    std::map<int64_t, uint64_t> result;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return result;
    }

    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // Find the comma separator
        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) {
            std::cerr << "Warning: Invalid line format (no comma): " << line << std::endl;
            continue;
        }

        try {
            // Extract hex athena_id and convert to int64_t
            std::string hex_athena_str = line.substr(0, comma_pos);
            std::string detray_str = line.substr(comma_pos + 1);
            
            // Convert hex string to int64_t
            int64_t athena_id = std::stoll(hex_athena_str, nullptr, 16);
            
            // Convert detray_id string to uint64_t
            uint64_t detray_id = std::stoull(detray_str);
            
            result.insert({athena_id, detray_id});
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Error parsing line '" << line << "': " << e.what() << std::endl;
            continue;
        }
    }
    
    file.close();
    std::cout << "Loaded " << result.size() << " athena->detray mappings from: " << filename << std::endl;
    return result;
}

// Comparison / ordering operator for measurements
struct measurement_sort_comp {
    bool operator()(const traccc::measurement& lhs, const traccc::measurement& rhs){

        if (lhs.surface_link.value() != rhs.surface_link.value()) {
            return lhs.surface_link.value() < rhs.surface_link.value();
        } else if (lhs.local[0] != rhs.local[0]) {
            return lhs.local[0] < rhs.local[0];
        } else if (lhs.local[1] != rhs.local[1]) {
            return lhs.local[1] < rhs.local[1];
        } 
        return false;
    }
};

// Comparison / ordering operator for measurements
struct inputData_sort_comp {
    bool operator()(const InputData& lhs, const InputData& rhs){

        if (lhs.athena_id_1 != rhs.athena_id_1) {
            return lhs.athena_id_1 < rhs.athena_id_1;
        } else if (lhs.loc_eta_1 != rhs.loc_eta_1) {
            return lhs.loc_eta_1 < rhs.loc_eta_1;
        } else if (lhs.loc_phi_1 != rhs.loc_phi_1) {
            return lhs.loc_phi_1 < rhs.loc_phi_1;
        }
        return false;
    }
};

inline void read_input_data(
    traccc::measurement_collection_types::host& measurements,
    traccc::edm::spacepoint_collection::host& spacepoints,
    std::vector<InputData>& data, 
    const std::map<int64_t, uint64_t>& athena_to_detray_map,
    bool do_strip = false
) {

    for (size_t i = 0; i < data.size(); i++) {

        if (data.at(i).athena_id_2 != 0) {
            continue;
        }

        InputData cluster = data.at(i);

        auto it = athena_to_detray_map.find(data.at(i).athena_id_1);
        if (it == athena_to_detray_map.end()) {
            std::cerr << "Warning: athena_id_1 " << data.at(i).athena_id_1
                      << " not found in mapping at index " << i << std::endl;
            continue;
        }
        uint64_t geometry_id = it->second;

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<detray::dsize_type<traccc::default_algebra>, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (((cluster.local_key_1) & (1 << (ipar + 1))) != 0) {

                switch (ipar) {
                    case 0: {
                        meas.local[0] = cluster.loc_eta_1;
                        meas.variance[0] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case 1: {
                        meas.local[1] = cluster.loc_phi_1;
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

        // fill spacepoints
        spacepoints.push_back({
            static_cast<unsigned int>(i),
            traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX,
            {data.at(i).sp_x, data.at(i).sp_y, data.at(i).sp_z},
            0.f, // Variance in z
            0.f  // Variance in radius
        });

    }
}


inline void inputDataToTracccMeasurements(
    std::vector<InputData> data,
    traccc::edm::spacepoint_collection::host& spacepoints,
    traccc::measurement_collection_types::host& measurements,
    const std::map<int64_t, uint64_t>& athena_to_detray_map
) {

    // First create all measurements since we need them for spacepoint linking
    unsigned int measurement_idx = 0;
    for (size_t i = 0; i < data.size(); i++) {

        if (data.at(i).athena_id_2 != 0) {
            continue; // Skip entries with athena_id_2 != 0
        }

        try {
            traccc::measurement meas;
            meas.local = {data.at(i).loc_eta_1, data.at(i).loc_phi_1};
            meas.variance = {0.0025, 0.0025}; // TODO may need to adjust based on actual variance
            meas.measurement_id = measurement_idx++;
            meas.meas_dim = data.at(i).athena_id_2 == 0 ? 1 : 2;
            
            // Check if athena_id_1 exists in the mapping
            auto it = athena_to_detray_map.find(data.at(i).athena_id_1);
            if (it == athena_to_detray_map.end()) {
                std::cerr << "Warning: athena_id_1 " << data.at(i).athena_id_1 
                          << " not found in mapping at index " << i << std::endl;
                continue;
            }
            meas.surface_link = detray::geometry::barcode{it->second};
            
            measurements.push_back(meas);

            if (data.at(i).athena_id_2 > 0) {
                try {
                    // Add second measurement if available
                    traccc::measurement meas2;
                    meas2.local = {data.at(i).loc_eta_2, data.at(i).loc_phi_2};
                    meas2.variance = {0.0025, 0.0025}; // TODO may need to adjust based on actual variance
                    meas2.measurement_id = measurement_idx++;
                    meas2.meas_dim = 2; // Always 2D for this case
                    
                    // Check if athena_id_2 exists in the mapping
                    auto it2 = athena_to_detray_map.find(data.at(i).athena_id_2);
                    if (it2 == athena_to_detray_map.end()) {
                        std::cerr << "Warning: athena_id_2 " << data.at(i).athena_id_2 
                                  << " not found in mapping at index " << i << std::endl;
                        measurement_idx--; // Revert the increment since we're not adding this measurement
                    } else {
                        meas2.surface_link = detray::geometry::barcode{it2->second};
                        measurements.push_back(meas2);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error creating second measurement at index " << i 
                              << ": " << e.what() << std::endl;
                    measurement_idx--; // Revert the increment
                }
            }

            // each spacepoint has one or two measurements
            try {
                spacepoints.push_back({
                    data.at(i).athena_id_2 > 0 ? measurement_idx-1 : measurement_idx,
                    data.at(i).athena_id_2 > 0 ? measurement_idx :
                        traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX,
                    {data.at(i).sp_x, data.at(i).sp_y, data.at(i).sp_z},
                    0.f, // Variance in z
                    0.f  // Variance in radius
                });
            } catch (const std::exception& e) {
                std::cerr << "Error creating spacepoint at index " << i 
                          << ": " << e.what() << std::endl;
            }

        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range error at index " << i << ": " << e.what() << std::endl;
            continue;
        } catch (const std::exception& e) {
            std::cerr << "Error processing data at index " << i << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Sort the measurements
    //? do spacepoints need to be sorted? Think not because ids already match?
    std::sort(measurements.begin(), measurements.end(), measurement_sort_comp());

}
