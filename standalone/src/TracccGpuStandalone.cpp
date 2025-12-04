#include "TracccGpuStandalone.hpp"

std::vector<traccc::io::csv::cell> read_csv(
    const std::string &filename,
    const std::map<int64_t, uint64_t> athenaToDetrayMap,
    bool athenaIDs = false
) {
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

        if (athenaIDs) {
            iocell.geometry_id = athenaToDetrayMap.at(iocell.geometry_id);
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
   
    std::vector<traccc::io::csv::cell> cells = read_csv(
        event_file, traccc_gpu.getAthenaToDetrayMap(), true);

    auto traccc_result = traccc_gpu.run(cells, true);

    int total_tracks = traccc_result.tracks_and_states.tracks.size();
    int excluded_non_positive_ndf = 0;
    int excluded_not_all_smoothed = 0;
    int excluded_unknown = 0;
    int excluded_no_state = 0;
    int printed_tracks = 0;

    for (size_t i = 0; i < traccc_result.tracks_and_states.tracks.size() && printed_tracks < 5; ++i)
    {
        const auto& track = traccc_result.tracks_and_states.tracks.at(i);

        auto track_fit_outcome = track.fit_outcome();
        if (track_fit_outcome ==
            traccc::track_fit_outcome::FAILURE_NON_POSITIVE_NDF) {
            ++excluded_non_positive_ndf;
            continue;
        } else if (track_fit_outcome ==
                   traccc::track_fit_outcome::FAILURE_NOT_ALL_SMOOTHED) {
            ++excluded_not_all_smoothed;
            continue;
        } else if (track_fit_outcome == traccc::track_fit_outcome::UNKNOWN) {
            ++excluded_unknown;
            continue;
        }

        if (track.constituent_links().size() < 1) {
            excluded_no_state += 1;
            continue;
        }

        const auto& fitted_params = track.params();
        traccc::scalar phi = fitted_params.phi();
        traccc::scalar theta = fitted_params.theta();
        traccc::scalar qop = fitted_params.qop();
        
        std::cout << "Track " << i << ": chi2 = " << track.chi2()
                  << ", ndf = " << track.ndf()
                  << ", phi = " << phi
                  << ", theta = " << theta  
                  << ", q/p = " << qop
                  << std::endl;

        const auto& constituent_links = track.constituent_links();
        for (size_t j = 0; j < constituent_links.size(); ++j)
        {
            const auto& link = constituent_links[j];
            
            if (link.type != traccc::edm::track_constituent_link::track_state) {
                continue;
            }

            const auto& state = traccc_result.tracks_and_states.states.at(link.index);
            size_t meas_idx = state.measurement_index();

            std::cout << "  Smoothed parameters: " << state.smoothed_params() << std::endl;

            const auto& measurement = traccc_result.measurements.at(meas_idx);

            std::cout << "  Measurement ID: " << measurement.identifier()
                      << ", Detected at detray ID: " << measurement.surface_link().value()
                      << ", Local Position: (" << measurement.local_position()[0] << ", " 
                      << measurement.local_position()[1] << ")"
                      << ", Local Variance: (" << measurement.local_variance()[0] << ", "
                      << measurement.local_variance()[1] << ")"
                      << ", Time: " << measurement.time()
                      << ", Measurement Dimension: " << measurement.dimensions()
                      << std::endl;
        }
        ++printed_tracks;
    }

    // Print final exclusion statistics
    std::cout << "\n=== Track Exclusion Summary ===" << std::endl;
    std::cout << "Total tracks processed: " << total_tracks << std::endl;
    std::cout << "Excluded (non-positive NDF): " << excluded_non_positive_ndf << std::endl;
    std::cout << "Excluded (not all smoothed): " << excluded_not_all_smoothed << std::endl;
    std::cout << "Excluded (unknown outcome): " << excluded_unknown << std::endl;
    std::cout << "Excluded (no state): " << excluded_no_state << std::endl;
    std::cout << "Total excluded: " << (excluded_non_positive_ndf + excluded_not_all_smoothed + 
                                         excluded_unknown + excluded_no_state) << std::endl;
    std::cout << "Tracks printed: " << printed_tracks << std::endl;

    return 0;
}