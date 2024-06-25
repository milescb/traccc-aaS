cmake_minimum_required(VERSION 3.17)

project(TracccStandalone LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Include CUDA headers
include_directories(/usr/local/cuda/include/)

find_package(actsvg REQUIRED)
find_package(covfie REQUIRED)
find_package(traccc REQUIRED)

message(STATUS "Printing all traccc related variables:")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if (_variableName MATCHES "traccc")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

message(STATUS "Printing all detray related variables:")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if (_variableName MATCHES "detray")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

add_executable(TracccClusterStandalone src/TracccClusterStandalone.cpp)

target_include_directories(TracccClusterStandalone 
PUBLIC 
${traccc_INCLUDE_DIR}
)
add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)

target_link_libraries(TracccClusterStandalone PUBLIC 
vecmem::core traccc::core traccc::io detray::core)
# vecmem::core traccc::core traccc::io traccc::performance traccc::options traccc::algebra traccc::device_common traccc::cuda
# traccc::eigen
# detray::core detray::utils detray::io detray::algebra_eigen)
