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

/// Structure for combined mapping CSV
struct geometry_mapping {
    uint64_t ath_geoid;
    uint64_t detray_id;

    DFE_NAMEDTUPLE(geometry_mapping, ath_geoid, detray_id);
};

/// Create csv reader for combined mapping
inline dfe::NamedTupleCsvReader<geometry_mapping> make_mapping_reader(
    std::string_view filename) {
    return {filename.data(), {"ath_geoid", "detray_id"}};
}

/// Read Athena-to-Detray geometry ID mapping from combined CSV file
inline std::map<uint64_t, uint64_t> read_athena_to_detray_mapping(
    std::string_view filename) {

    auto reader = make_mapping_reader(filename);
    std::map<uint64_t, uint64_t> result;
    geometry_mapping mapping;

    while (reader.read(mapping)) {
        // athena_id -> detray_id
        result.insert({mapping.ath_geoid, mapping.detray_id});
    }
    
    std::cout << "Loaded " << result.size() << " athena->detray mappings from: " << filename << std::endl;
    return result;
}

// inline void inputDataToTracccMeasurements(
//     InputData gnnTracks,
//     traccc::edm::spacepoint_collection::host& spacepoints,
//     traccc::measurement_collection_types::host& measurements,
//     const std::map<std::uint64_t, detray::geometry::barcode>& acts_id_to_barcode_map,
//     const std::map<uint64_t, uint64_t>& athena_to_acts_map) 
// {
//     // First create all measurements since we need them for spacepoint linking
//     for (size_t i = 0; i < gnnTracks.cl_x.size(); i++) {
//         traccc::measurement meas;
//         // Set local coordinates (eta, phi)
//         meas.local = {gnnTracks.cl_loc_eta[i], gnnTracks.cl_loc_phi[i]};
//         // Set variance
//         meas.variance = {gnnTracks.cl_cov_00[i], gnnTracks.cl_cov_11[i]};
//         // Set measurement ID (needed for linking)
//         meas.measurement_id = i;
//         // Set measurement dimension
//         meas.meas_dim = gnnTracks.sp_cl2_index[i] >= 0 ? 2 : 1;
//         // Set geometry ID
//         // Output from GNN is Athena ID, first need to convert to ACTS ID
//         auto it_athena_to_acts = athena_to_acts_map.find(gnnTracks.cl_module_id[i]);
//         uint64_t acts_id = it_athena_to_acts->second;
//         // Traccc needs detray::geometry::barcode, so we convert with final map
//         auto it_acts_to_barcode = acts_id_to_barcode_map.find(acts_id);
//         meas.surface_link = it_acts_to_barcode->second;
        
//         measurements.push_back(meas);
//     }

//     // Now create spacepoints with measurement links
//     for (size_t i = 0; i < gnnTracks.sp_x.size(); i++) {
//         // Get measurement indices for this spacepoint
//         unsigned int meas_idx1 = gnnTracks.sp_cl1_index.at(i);
//         unsigned int meas_idx2 = gnnTracks.sp_cl2_index[i] >= 0 ? 
//             gnnTracks.sp_cl2_index[i] : 
//             traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX;

//         // Create spacepoint using same format as read_spacepoints
//         spacepoints.push_back({
//             meas_idx1,                              // First measurement index
//             meas_idx2,                              // Second measurement index (or INVALID)
//             {gnnTracks.sp_x[i],                     // Global position
//              gnnTracks.sp_y[i], 
//              gnnTracks.sp_z[i]},
//             0.f,                                    // Variance in z
//             0.f                                     // Variance in radius
//         });
//     }

//     // print number of spacepoints / measurements
//     std::cout << "Number of measurements created: " << measurements.size() << std::endl;
//     std::cout << "Number of spacepoints created: " << spacepoints.size() << std::endl;
// }
