#pragma once 

// #include "traccc/edm/spacepoint_collection.hpp"
// #include "traccc/edm/measurement_collection.hpp"
// #include "traccc/definitions/common.hpp"

// #include <dfe/dfe_io_dsv.hpp>
// #include <dfe/dfe_namedtuple.hpp>

#include <iostream>
#include <map>          
#include <unordered_map>
#include <string>       
#include <sstream>  
#include <fstream>

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
