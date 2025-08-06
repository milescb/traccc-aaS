#include "TracccGpuStandalone.hpp"

std::vector<InputData> read_clusters_from_csv(
    const std::string& filename)
{
    std::vector<InputData> input_data;
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
        InputData data;
        int col_index = 0;

        try {
            // Read each field separated by a comma
            while (std::getline(ss, field, ',')) {
                // Trim leading/trailing whitespace
                field.erase(0, field.find_first_not_of(" \t\n\r\f\v"));
                field.erase(field.find_last_not_of(" \t\n\r\f\v") + 1);

                switch (col_index) {
                    case 0: // geoid_1
                        data.athena_id_1 = std::stoll(field);
                        break;
                    case 1: // geoid_2
                        data.athena_id_2 = std::stoll(field);
                        break;
                    case 2: // local_key_1
                        data.local_key_1 = std::stoi(field);
                        break;
                    case 3: // local_key_2
                        data.local_key_2 = std::stoi(field);
                        break;
                    case 4: // x
                        data.sp_x = std::stof(field);
                        break;
                    case 5: // y
                        data.sp_y = std::stof(field);
                        break;
                    case 6: // z
                        data.sp_z = std::stof(field);
                        break;
                    case 7: // loc_eta_1
                        data.loc_eta_1 = std::stof(field);
                        break;
                    case 8: // loc_phi_1
                        data.loc_phi_1 = std::stof(field);
                        break;
                    case 9: // loc_eta_2
                        data.loc_eta_2 = std::stof(field);
                        break;
                    case 10: // loc_phi_2
                        data.loc_phi_2 = std::stof(field);
                        break;
                    case 11: // r
                        data.r = std::stof(field);
                        break;
                    case 12: // phi
                        data.phi = std::stof(field);
                        break;
                    case 13: // eta
                        data.eta = std::stof(field);
                        break;
                    case 14: // cluster_x_1
                        data.cluster_x_1 = std::stof(field);
                        break;
                    case 15: // cluster_y_1
                        data.cluster_y_1 = std::stof(field);
                        break;
                    case 16: // cluster_z_1
                        data.cluster_z_1 = std::stof(field);
                        break;
                    case 17: // cluster_x_2
                        data.cluster_x_2 = std::stof(field);
                        break;
                    case 18: // cluster_y_2
                        data.cluster_y_2 = std::stof(field);
                        break;
                    case 19: // cluster_z_2
                        data.cluster_z_2 = std::stof(field);
                        break;
                    case 20: // count_1
                        data.count_1 = std::stof(field);
                        break;
                    case 21: // charge_count_1
                        data.charge_count_1 = std::stof(field);
                        break;
                    case 22: // localDir0_1
                        data.localDir0_1 = std::stof(field);
                        break;
                    case 23: // localDir1_1
                        data.localDir1_1 = std::stof(field);
                        break;
                    case 24: // localDir2_1
                        data.localDir2_1 = std::stof(field);
                        break;
                    case 25: // lengthDir0_1
                        data.lengthDir0_1 = std::stof(field);
                        break;
                    case 26: // lengthDir1_1
                        data.lengthDir1_1 = std::stof(field);
                        break;
                    case 27: // lengthDir2_1
                        data.lengthDir2_1 = std::stof(field);
                        break;
                    case 28: // glob_eta_1
                        data.glob_eta_1 = std::stof(field);
                        break;
                    case 29: // glob_phi_1
                        data.glob_phi_1 = std::stof(field);
                        break;
                    case 30: // eta_angle_1
                        data.eta_angle_1 = std::stof(field);
                        break;
                    case 31: // phi_angle_1
                        data.phi_angle_1 = std::stof(field);
                        break;
                    case 32: // count_2
                        data.count_2 = std::stof(field);
                        break;
                    case 33: // charge_count_2
                        data.charge_count_2 = std::stof(field);
                        break;
                    case 34: // localDir0_2
                        data.localDir0_2 = std::stof(field);
                        break;
                    case 35: // localDir1_2
                        data.localDir1_2 = std::stof(field);
                        break;
                    case 36: // localDir2_2
                        data.localDir2_2 = std::stof(field);
                        break;
                    case 37: // lengthDir0_2
                        data.lengthDir0_2 = std::stof(field);
                        break;
                    case 38: // lengthDir1_2
                        data.lengthDir1_2 = std::stof(field);
                        break;
                    case 39: // lengthDir2_2
                        data.lengthDir2_2 = std::stof(field);
                        break;
                    case 40: // glob_eta_2
                        data.glob_eta_2 = std::stof(field);
                        break;
                    case 41: // glob_phi_2
                        data.glob_phi_2 = std::stof(field);
                        break;
                    case 42: // eta_angle_2
                        data.eta_angle_2 = std::stof(field);
                        break;
                    case 43: // phi_angle_2
                        data.phi_angle_2 = std::stof(field);
                        break;
                    case 44: // cluster_r_1
                        data.cluster_r_1 = std::stof(field);
                        break;
                    case 45: // cluster_phi_1
                        data.cluster_phi_1 = std::stof(field);
                        break;
                    case 46: // cluster_eta_1
                        data.cluster_eta_1 = std::stof(field);
                        break;
                    case 47: // cluster_r_2
                        data.cluster_r_2 = std::stof(field);
                        break;
                    case 48: // cluster_phi_2
                        data.cluster_phi_2 = std::stof(field);
                        break;
                    case 49: // cluster_eta_2
                        data.cluster_eta_2 = std::stof(field);
                        break;
                    case 50: // cluster
                        data.cluster = std::stof(field);
                        break;
                    default:
                        // Handle unexpected extra columns if necessary
                        break;
                }
                col_index++;
            }
            // Check if we read the expected number of columns
            if (col_index != 51) {
                 std::cerr << "Warning: Row has " << col_index << " columns, expected 51. Line: " << line << std::endl;
                 continue;
            }

            input_data.push_back(data);

        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid argument during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Out of range during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            continue;
        }
    }

    file.close();

    // print first 5 entries to ensure they are read correctly
    std::cout << "Read " << input_data.size() << " entries from file." << std::endl;
    for (size_t i = 0; i < std::min(input_data.size(), size_t(5)); ++i) {
        std::cout << "Entry " << i << ": "
                  << "geoid_1: " << input_data[i].athena_id_1
                  << ", geoid_2: " << input_data[i].athena_id_2
                  << ", spacepoint: (" << input_data[i].sp_x << ", " << input_data[i].sp_y << ", " << input_data[i].sp_z << ")"
                  << ", loc_eta_1: " << input_data[i].loc_eta_1
                  << ", loc_phi_1: " << input_data[i].loc_phi_1
                  << std::endl;
    }

    return input_data;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceID);
    
    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceID);
   
    traccc::edm::spacepoint_collection::host spacepoints(host_mr);
    traccc::measurement_collection_types::host measurements(&host_mr);

    std::vector<InputData> input_data = read_clusters_from_csv(event_file);
    std::sort(input_data.begin(), input_data.end(), inputData_sort_comp());
    
    // inputDataToTracccMeasurements(
    //     input_data, spacepoints, measurements, 
    //     traccc_gpu.getAthenaToDetrayMap());

    read_input_data(measurements, spacepoints, input_data, traccc_gpu.getAthenaToDetrayMap());

    // run the traccc algorithm
    auto traccc_result = traccc_gpu.run(spacepoints, measurements);

    // print out results
    size_t printed_tracks = 0;
    size_t failed_tracks = 0;
    for (size_t i = 0; i < traccc_result.size() && printed_tracks < 5; ++i)
    {
        const auto& [fit_res, state] = traccc_result.at(i);

        // Only process and print tracks with a valid fit (ndf > 0)
        if (fit_res.trk_quality.ndf <= 0) {
            failed_tracks++;
            continue;
        }
        printed_tracks++;

        std::cout << "Track " << i << ": chi2 = " << fit_res.trk_quality.chi2
                  << ", ndf = " << fit_res.trk_quality.ndf
                  << std::endl;

        // Get the fitted track parameters from the fitting result
        const auto& fitted_params = fit_res.fit_params;
        
        // Extract the track parameters
        traccc::scalar phi = fitted_params.phi();
        traccc::scalar theta = fitted_params.theta();
        traccc::scalar qop = fitted_params.qop();
        
        // Calculate eta from theta
        traccc::scalar eta = -std::log(std::tan(theta / 2.0));
        
        std::cout << "Track " << i << ": chi2 = " << fit_res.trk_quality.chi2
                  << ", ndf = " << fit_res.trk_quality.ndf
                  << ", phi = " << phi
                  << ", eta = " << eta  
                  << ", q/p = " << qop
                  << std::endl;

        for (auto const& st : state) 
        {
            const traccc::measurement& measurement = st.get_measurement();

            const std::array<float, 2> localPosition = measurement.local;
            const std::array<float, 2> localCovariance = measurement.variance;
            uint64_t detray_id = measurement.surface_link.value();
            float time = measurement.time;
            size_t measurement_id = measurement.measurement_id;
            unsigned int measDim = measurement.meas_dim;

            std::cout << "  Measurement ID: " << measurement_id
                      << ", Detected at detray ID: " << detray_id
                      << ", Local Position: (" << localPosition[0] << ", " 
                      << localPosition[1] << ")"
                      << ", Local Covariance: (" << localCovariance[0] << ", "
                      << localCovariance[1] << ")"
                      << ", Time: " << time
                      << ", Measurement Dimension: " << measDim
                      << std::endl;
        }
    }

    std::cout << "Total tracks processed: " << traccc_result.size() 
              << ", Printed tracks: " << printed_tracks 
              << ", Failed tracks (no valid fit): " << failed_tracks 
              << std::endl;

    return 0;
}