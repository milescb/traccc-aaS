#include "TracccCpuStandalone.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << argv[0] << " <event_file>" << std::endl;
        return -1;
    }

    std::string path = std::string(argv[1]);
    std::cout << "Running " << argv[0] << " on " << path << std::endl;

    TracccClusterStandalone clusterStandalone;

    if (std::filesystem::is_directory(path))
    {
        for (const auto &entry : std::filesystem::directory_iterator(path))
        {
            if (entry.path().extension() == ".csv" && entry.path().filename().string().find("cells.csv") != std::string::npos)
            {
                std::cout << "Processing file: " << entry.path() << std::endl;
                auto cells = clusterStandalone.read_csv(entry.path().string());
                clusterStandalone.runPipeline(cells);
            }
        }
    }
    else
    {
        std::cout << "The provided path is neither a file nor a directory" << std::endl;
        return -1;
    }

    return 0;
}