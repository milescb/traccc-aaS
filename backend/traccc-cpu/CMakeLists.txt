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

# Modifications made by Haoran Zhao, haoran.zhao [at] cern.ch, June 2024.

cmake_minimum_required(VERSION 3.17)

project(TracccBackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/usr/local/cuda/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../standalone/src)

# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because Traccc backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

# change "main" to "r22.02"
set(TRITON_TAG "r22.02" CACHE STRING "Tag for triton-inference-server repositories")
set(TRITON_COMMON_REPO_TAG ${TRITON_TAG} CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG ${TRITON_TAG} CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG ${TRITON_TAG} CACHE STRING "Tag for triton-inference-server/backend repo")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Setup up Traccc

find_package(actsvg REQUIRED)
find_package(covfie REQUIRED)
find_package(traccc REQUIRED)

# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_Traccc.ldscript libtriton_Traccc.ldscript COPYONLY)

add_library(
  triton-Traccc-backend SHARED
  src/Traccc.cc
)

add_library(
  TracccBackend::triton-Traccc-backend ALIAS triton-Traccc-backend
)

target_include_directories(
  triton-Traccc-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${traccc_INCLUDE_DIR}
)

target_compile_features(triton-Traccc-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-Traccc-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

target_link_libraries(
  triton-Traccc-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    vecmem::core 
    traccc::core 
    traccc::io 
    detray::core
    traccc::performance 
    traccc::options 
    detray::utils 
    detray::io
)

add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)


if(WIN32)
  set_target_properties(
    triton-Traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_Traccc
  )
else()
  set_target_properties(
    triton-Traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_Traccc
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_Traccc.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_Traccc.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
# set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TracccBackend)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_PREFIX}/lib/cmake/TracccBackend)

install(
  TARGETS
    triton-Traccc-backend
  EXPORT
    triton-Traccc-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/Traccc
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/Traccc
)

install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/../../models
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        USE_SOURCE_PERMISSIONS
)

install(
  TARGETS
    triton-Traccc-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/traccc-cpu
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/traccc-cpu
)

install(
  EXPORT
    triton-Traccc-backend-targets
  FILE
    TracccBackendTargets.cmake
  NAMESPACE
    TracccBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TracccBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-Traccc-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendTargets.cmake
  NAMESPACE TracccBackend::
)

export(PACKAGE TracccBackend)