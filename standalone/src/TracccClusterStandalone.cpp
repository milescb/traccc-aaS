#include "TracccClusterStandalone.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << argv[0] << " <event_file>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    TracccClusterStandalone clusterStandalone;
    auto cells = read_csv(event_file);
    clusterStandalone.runPipeline(cells);

    return 0;
}