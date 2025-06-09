#include "TracccGpuStandalone.hpp"

std::vector<clusterInfo> read_clusters_from_csv(
    const std::string& filename)
{
    std::vector<clusterInfo> clusters;
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
        clusterInfo cluster;
        int col_index = 0;
        double global_x, global_y, global_z;
        double local_x, local_y;
        int pixel_val;

        try {
            // Read each field separated by a comma
            while (std::getline(ss, field, ',')) {
                // Trim leading/trailing whitespace
                field.erase(0, field.find_first_not_of(" \t\n\r\f\v"));
                field.erase(field.find_last_not_of(" \t\n\r\f\v") + 1);

                switch (col_index) {
                    case 0: // atlas_id (ignored)
                        break;
                    case 1: // detray_id
                        cluster.detray_id = std::stoull(field);
                        break;
                    case 2: // measurement_id (ignored)
                        break;
                    case 3: // local_key
                        cluster.local_key = std::stoul(field);
                        break;
                    case 4: // local_x
                        local_x = std::stod(field);
                        break;
                    case 5: // local_y
                        local_y = std::stod(field);
                        break;
                    case 6: // global_x
                        global_x = std::stod(field);
                        break;
                    case 7: // global_y
                        global_y = std::stod(field);
                        break;
                    case 8: // global_z
                        global_z = std::stod(field);
                        break;
                    case 9: // pixel
                        pixel_val = std::stoi(field);
                        cluster.pixel = (pixel_val != 0);
                        break;
                    default:
                        // Handle unexpected extra columns if necessary
                        break;
                }
                col_index++;
            }
             // Check if we read the expected number of columns
            if (col_index != 10) {
                 std::cerr << "Warning: Row has " << col_index << " columns, expected 10. Line: " << line << std::endl;
                 continue; // Skip this row or handle error as appropriate
            }

            // Assign Eigen vectors after reading all components
            cluster.globalPosition = Eigen::Vector3d(global_x, global_y, global_z);
            cluster.localPosition = Eigen::Vector2d(local_x, local_y);

            clusters.push_back(cluster);

        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid argument during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Out of range during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        }
    }

    file.close();

    // print first 5 clusters to ensure they are read correctly
    std::cout << "Read " << clusters.size() << " clusters from file." << std::endl;
    for (size_t i = 0; i < std::min(clusters.size(), size_t(5)); ++i) {
        std::cout << "Cluster " << i << ": "
                  << "detray_id: " << clusters[i].detray_id
                  << ", local_key: " << clusters[i].local_key
                  << ", globalPosition: (" << clusters[i].globalPosition.transpose() << ")"
                  << ", localPosition: (" << clusters[i].localPosition.transpose() << ")"
                  << ", pixel: " << clusters[i].pixel
                  << std::endl;
    }

    return clusters;
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

    std::vector<clusterInfo> detray_clusters = read_clusters_from_csv(event_file);
    std::sort(detray_clusters.begin(), detray_clusters.end(), measurement_sort_comp());
    
    traccc_gpu.read_measurements(measurements, detray_clusters, false);
    traccc_gpu.read_spacepoints(spacepoints, detray_clusters, false);

    // run the traccc algorithm
    auto traccc_result = traccc_gpu.run(spacepoints, measurements);

    // print out results
    for (size_t i = 0; i < 5; ++i)
    {
        const auto& [fit_res, state] = traccc_result.at(i);

        std::cout << "Track " << i << ": chi2 = " << fit_res.trk_quality.chi2
                  << ", ndf = " << fit_res.trk_quality.ndf
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

    return 0;
}