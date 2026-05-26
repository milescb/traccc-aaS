cmake_minimum_required(VERSION 3.22)

# set project settings
project(TracccStandalone LANGUAGES C CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

include(FetchContent)

# find required traccc packages
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

# Select vecmem backend library and propagate the alpaka backend define so
# that inline template functions (e.g. buffer_from_host_detector) are compiled
# with the same type definitions as the pre-built traccc library.
set(EXTRA_LIBS)
set(ALPAKA_BACKEND_DEFINES)
if(alpaka_ACC_GPU_CUDA_ENABLE)
    list(APPEND EXTRA_LIBS vecmem::cuda)
    list(APPEND ALPAKA_BACKEND_DEFINES ALPAKA_ACC_GPU_CUDA_ENABLED)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
    list(APPEND EXTRA_LIBS vecmem::hip)
    list(APPEND ALPAKA_BACKEND_DEFINES ALPAKA_ACC_GPU_HIP_ENABLED)
elseif(alpaka_ACC_SYCL_ENABLE)
    list(APPEND EXTRA_LIBS vecmem::sycl)
    list(APPEND ALPAKA_BACKEND_DEFINES ALPAKA_ACC_SYCL_ENABLED)
endif()

add_executable(TracccGpuStandalone
    src/TracccGpuStandalone.cpp)

# Compile the source with the HIP compiler so it shares visibility / typeinfo
# behavior with the pre-built traccc library (which was built with HIP).
set_source_files_properties(src/TracccGpuStandalone.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(TracccGpuStandalone
    PUBLIC src/${traccc_INCLUDE_DIR}
)

add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)

target_compile_definitions(TracccGpuStandalone PUBLIC ${ALPAKA_BACKEND_DEFINES})

target_link_libraries(TracccGpuStandalone PUBLIC
    vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka
    detray::io detray::core traccc::options ${EXTRA_LIBS})
