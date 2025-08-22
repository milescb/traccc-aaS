#include "TracccGpuStandalone.hpp"

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    std::cout << "Reading cells from " << filename << std::endl;

    while (reader.read(iocell))
    {
        if (iocell.geometry_id == 0)
        {   
            std::cout << "Warning: Found cell with geometry_id = 0, "
                      << "this may indicate an issue with the input data." 
                      << std::endl;
            continue;
        }
        cells.push_back(iocell);
    }

    return cells;
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
   
    std::vector<traccc::io::csv::cell> cells = read_csv(event_file);

    // run the traccc algorithm
    auto traccc_result = traccc_gpu.run(cells);

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