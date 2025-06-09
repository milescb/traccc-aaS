#include "TracccGpuStandalone.hpp"

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

    std::vector<clusterInfo> detray_clusters = traccc_gpu.read_clusters_from_csv(event_file);
    std::sort(detray_clusters.begin(), detray_clusters.end(), measurement_sort_comp());
    
    traccc_gpu.read_measurements(measurements, detray_clusters, false);
    traccc_gpu.read_spacepoints(spacepoints, detray_clusters, false);

    // run the traccc algorithm
    auto traccc_result = traccc_gpu.run(spacepoints, measurements);

    return 0;
}