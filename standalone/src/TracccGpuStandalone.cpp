#include "TracccGpuStandalone.hpp"

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

    TracccGpuStandalone standalone(2);
    auto cells = read_csv(event_file);
    standalone.run(cells);

    return 0;
}