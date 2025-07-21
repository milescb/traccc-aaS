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
struct geometry_mapping {
    int64_t ath_geoid;
    uint64_t detray_id;

    DFE_NAMEDTUPLE(geometry_mapping, ath_geoid, detray_id);
};

/// Create csv reader for combined mapping
inline dfe::NamedTupleCsvReader<geometry_mapping> make_mapping_reader(
    std::string_view filename) {
    return {filename.data(), {"ath_geoid", "detray_id"}};
}

/// Read Athena-to-Detray geometry ID mapping from combined CSV file
inline std::map<int64_t, uint64_t> read_athena_to_detray_mapping(
    std::string_view filename) {

    auto reader = make_mapping_reader(filename);
    std::map<int64_t, uint64_t> result;
    geometry_mapping mapping;

    while (reader.read(mapping)) {
        // athena_id -> detray_id
        result.insert({mapping.ath_geoid, mapping.detray_id});
    }
    
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

inline void inputDataToTracccMeasurements(
    std::vector<InputData> data,
    traccc::edm::spacepoint_collection::host& spacepoints,
    traccc::measurement_collection_types::host& measurements,
    const std::map<int64_t, uint64_t>& athena_to_detray_map)
{

    // First create all measurements since we need them for spacepoint linking
    unsigned int measurement_idx = 0;
    for (size_t i = 0; i < data.size(); i++) {
        traccc::measurement meas;
        meas.local = {data.at(i).loc_eta_1, data.at(i).loc_phi_1};
        meas.variance = {0.0025, 0.0025}; // TODO may need to adjust based on actual variance
        meas.measurement_id = measurement_idx++;
        meas.meas_dim = data.at(i).athena_id_2 == 0 ? 1 : 2;
        meas.surface_link = detray::geometry::barcode{athena_to_detray_map.at(data.at(i).athena_id_1)};
        
        measurements.push_back(meas);

        if (data.at(i).athena_id_2 >= 0) {
            // Add second measurement if available
            traccc::measurement meas2;
            meas2.local = {data.at(i).loc_eta_2, data.at(i).loc_phi_2};
            meas2.variance = {0.0025, 0.0025}; // TODO may need to adjust based on actual variance
            meas2.measurement_id = measurement_idx++;
            meas2.meas_dim = 2; // Always 2D for this case
            meas2.surface_link = detray::geometry::barcode{athena_to_detray_map.at(data.at(i).athena_id_2)};

            measurements.push_back(meas2);
        }

        // each spacepoint has one or two measurements
        spacepoints.push_back({
            data.at(i).athena_id_2 != 0 ? measurement_idx-1 : measurement_idx,
            data.at(i).athena_id_2 != 0 ? measurement_idx :
                traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX,
            {data.at(i).sp_x, data.at(i).sp_y, data.at(i).sp_z},
            0.f, // Variance in z
            0.f  // Variance in radius
        });
    }

    // Sort the measurements
    //? do spacepoints need to be sorted? Think not because ids already match?
    std::sort(measurements.begin(), measurements.end(), measurement_sort_comp());

    // print number of spacepoints / measurements
    // std::cout << "Number of measurements created: " << measurements.size() << std::endl;
    // std::cout << "Number of spacepoints created: " << spacepoints.size() << std::endl;
}
