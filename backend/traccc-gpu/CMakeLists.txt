# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.22)

project(TracccBackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shadow FindGTest so detray's leaked test dependency doesn't abort our build
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# cmake's generation-time check validates INTERFACE_INCLUDE_DIRECTORIES of all
# imported targets transitively linked by any SHARED library in the build. For
# external packages (traccc, detray) it evaluates the install-tree view, which
# resolves to CMAKE_INSTALL_PREFIX/include. Create that directory at configure
# time so the existence check passes. The directory itself is never used.
file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/include")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../standalone/src)

#
# Options
#
option(TRITON_ENABLE_ROCM   "Enable AMD ROCm support" ON)
option(TRITON_ENABLE_ALPAKA "Use Alpaka portability layer (requires TRITON_ENABLE_ROCM)" ON)
option(TRITON_ENABLE_STATS  "Include statistics collections in backend" ON)

set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/backend repo")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 20 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# ROCm/HIP Configuration
#
if(TRITON_ENABLE_ROCM)
  set(TRITON_ROCM_HOME ${ROCM_PATH})

  set(ENV{ROCM_PATH} ${TRITON_ROCM_HOME})
  set(ENV{HIP_PATH}  ${TRITON_ROCM_HOME})

  list(APPEND CMAKE_PREFIX_PATH
    ${TRITON_ROCM_HOME}
    ${TRITON_ROCM_HOME}/hip
    ${TRITON_ROCM_HOME}/llvm
    ${TRITON_ROCM_HOME}/lib/cmake
    $ENV{ROCM_PATH}
    $ENV{HIP_PATH}
  )

  file(GLOB rocm_cmake_components ${TRITON_ROCM_HOME}/lib/cmake/*)
  list(APPEND CMAKE_PREFIX_PATH ${rocm_cmake_components})

  if(NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${TRITON_ROCM_HOME}/llvm/bin/clang++")
  endif()

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx940;gfx941;gfx942" CACHE STRING "HIP architectures")
  endif()

  enable_language(HIP)
  find_package(hip REQUIRED CONFIG PATHS ${TRITON_ROCM_HOME})

  include_directories(
    ${TRITON_ROCM_HOME}/include
    ${TRITON_ROCM_HOME}/hip/include
    ${HIP_INCLUDE_DIRS}
  )
  link_directories(
    ${TRITON_ROCM_HOME}/lib
    ${TRITON_ROCM_HOME}/hip/lib
  )

  add_compile_definitions(TRITON_ENABLE_ROCM __HIP_PLATFORM_AMD__)
  set(HIP_PLATFORM "amd" CACHE STRING "HIP platform")

  if(EXISTS "${TRITON_ROCM_HOME}/.info/version-dev")
    file(READ "${TRITON_ROCM_HOME}/.info/version-dev" ROCM_VERSION_DEV_RAW)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-.*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})
    if(ROCM_VERSION_DEV_MATCH)
      set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
      set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
      set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
      set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
      math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
    endif()
  endif()

  message(STATUS "ROCm support enabled")
  message(STATUS "ROCm path: ${TRITON_ROCM_HOME}")
  message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
  message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
  if(DEFINED ROCM_VERSION_DEV)
    message(STATUS "ROCm version: ${ROCM_VERSION_DEV} (${ROCM_VERSION_DEV_INT})")
  endif()
endif()

#
# Alpaka Configuration
#
if(TRITON_ENABLE_ALPAKA)
  if(NOT TRITON_ENABLE_ROCM)
    message(FATAL_ERROR "TRITON_ENABLE_ALPAKA requires TRITON_ENABLE_ROCM")
  endif()
  find_package(alpaka REQUIRED HINTS /usr/local/lib/cmake/alpaka)
  add_compile_definitions(TRITON_ENABLE_ALPAKA ALPAKA_ACC_GPU_HIP_ENABLED)
  message(STATUS "Alpaka portability layer enabled (HIP backend)")
endif()

#
# Dependencies
#
# repo-backend enables CUDA unconditionally when TRITON_ENABLE_GPU=ON (its default).
# Disable it since we are using ROCm/Alpaka instead.
set(BUILD_TESTING OFF CACHE BOOL "Disable all testing" FORCE)
set(TRITON_ENABLE_TESTS OFF CACHE BOOL "Disable Triton tests" FORCE)
set(TRITON_ENABLE_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(TRITON_ENABLE_GPU OFF CACHE BOOL "Disable GPU support in repo-backend" FORCE)

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Traccc dependencies
#
find_package(covfie REQUIRED)
find_package(traccc REQUIRED)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_traccc.ldscript libtriton_traccc.ldscript COPYONLY)

add_library(
  triton-traccc-backend SHARED
  src/traccc.cc
)

# Force HIP compiler (clang++) for .cc files when ROCm is enabled,
# since CMake doesn't recognize .cc as a HIP source by default.
# Also explicitly propagate Triton include dirs to the HIP compiler,
# as CMake does not always forward INTERFACE_INCLUDE_DIRECTORIES from
# CXX targets to HIP-language source files.
if(TRITON_ENABLE_ROCM)
  set_source_files_properties(
    src/traccc.cc
    PROPERTIES LANGUAGE HIP
  )
  target_include_directories(
    triton-traccc-backend
    PRIVATE
      $<TARGET_PROPERTY:triton-backend-utils,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-backendapi,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-serverapi,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-serverstub,INTERFACE_INCLUDE_DIRECTORIES>
      ${TRITON_ROCM_HOME}/hip/include
  )
endif()

add_library(
  TraccBackend::triton-traccc-backend ALIAS triton-traccc-backend
)

target_include_directories(
  triton-traccc-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${traccc_INCLUDE_DIR}
)

target_compile_features(triton-traccc-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-traccc-backend PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX,HIP>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

# Base Triton libraries
target_link_libraries(
  triton-traccc-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

# Traccc / VecMem libraries (common)
target_link_libraries(
  triton-traccc-backend
  PRIVATE
    vecmem::core
    traccc::io
    traccc::performance
    traccc::core
    traccc::device_common
    traccc::alpaka
    traccc::options
    detray::io
)

# ROCm/HIP runtime libraries
if(TRITON_ENABLE_ROCM)
  target_link_libraries(
    triton-traccc-backend
    PRIVATE
      hip::host
      hip::device
      vecmem::hip
      ${ROCM_PATH}/lib/libamdhip64.so
  )

  if(TRITON_ENABLE_ALPAKA)
    target_link_libraries(triton-traccc-backend PRIVATE alpaka::alpaka)
    target_include_directories(triton-traccc-backend PRIVATE
      $<TARGET_PROPERTY:alpaka::alpaka,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
endif()

# must set algebra plugin include array
add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)

if(WIN32)
  set_target_properties(
    triton-traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_traccc
  )
else()
  set_target_properties(
    triton-traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_traccc
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_traccc.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_traccc.ldscript"
  )
endif()

#
# Install
#
# cmake's generation-time check fires on imported targets' INTERFACE_INCLUDE_DIRECTORIES
# whenever any install(TARGETS/FILES <target-ref> ...) command exists — it evaluates the
# install-tree view of transitive deps and looks for CMAKE_INSTALL_PREFIX/include.
# Avoid the check entirely by using install(CODE ...) which runs cmake script at
# install time without doing target-interface analysis.
# The .so path is computed at configure time from the known output name.
#
set(_SO "${CMAKE_CURRENT_BINARY_DIR}/libtriton_traccc.so")
set(_DEST_BACKEND  "${CMAKE_INSTALL_PREFIX}/backends/traccc")
# Anchor the model-repo destination to the source tree so it is independent
# of CMAKE_INSTALL_PREFIX.  CMAKE_SOURCE_DIR == backend/traccc-gpu/, so
# one level up is the project root that contains the models/ directory.
set(_DEST_MODELS   "${CMAKE_SOURCE_DIR}/../models/traccc-gpu")
set(_DEST_MODELDIR "${CMAKE_INSTALL_PREFIX}")

install(CODE "
  file(MAKE_DIRECTORY \"${_DEST_BACKEND}\")
  file(COPY \"${_SO}\" DESTINATION \"${_DEST_BACKEND}\")
  file(MAKE_DIRECTORY \"${_DEST_MODELS}\")
  file(COPY \"${_SO}\" DESTINATION \"${_DEST_MODELS}\")
")

# Copy the model repository into the install tree (plain DIRECTORY, no target refs)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/../models"
        DESTINATION "${_DEST_MODELDIR}"
        USE_SOURCE_PERMISSIONS)
