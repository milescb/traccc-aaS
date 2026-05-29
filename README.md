# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) 
as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of 
   the code included
2. a custom backend using the standalone version above to launch the Triton server
3. a client to send data to the server

**This branch serves an AMD-supported backend only**

A minimal description of how to build a working version is detailed below. In each subdirectory of 
this project, a README containing more information can be found. For instructions on running the 
pipeline with an Athena client on `lxplus` and the CUDA version, consult 
[this CodiMD page](https://codimd.web.cern.ch/1FcLmapORpeBtAVL_M6h4A?view). The last part of this 
README describes the necessary steps for maintaining and updating `traccc-aaS` when a new release 
of `traccc` is available. 

## Obtaining the `ITk` geometry files

You will need access to the `ITk` geometry files to run this repository, which can be found 
at `/eos/project/a/atlas-eftracking/GPU/ITk_data/ATLAS-P2-RUN4-03-00-01/` on CERN's `lxplus` 
computing cluster. Note that to access these, you must be a part of the e-group 
`atlas-tdaq-phase2-EFTracking-developers`. If you are outside of ATLAS, please refer to the 
`odd_legacy` branch to run the ODD version of the code.

### Get the code

Simply clone the repository with 

```
git clone --recurse-submodules git@github.com:milescb/traccc-aaS.git

# checkout this branch
cd traccc-aaS && git checkout amd-alpaka
```

### Docker

The easiest way to build the custom backend is with the docker at 
`docker.io/milescb/triton-server:25.02-py3_gcc13.3_alpaka2.1.1`. Run this interactively with

```
shifterimg pull milescb/triton-server:25.02-py3_gcc13.3
shifter --module=gpu --image=milescb/triton-server:25.02-py3_gcc13.3
```

or use your favorite docker application and mount the appropriate directories. 

### Building from source

Follow the instructions on the [traccc page](https://github.com/acts-project/traccc/tree/main) to 
build, or run this configure command for AMD support:

```
cmake -S <path_to_traccc> -B <build_dir> -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_ALPAKA=ON \
    -DTRACCC_BUILD_HIP=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DTRACCC_USE_ROOT=FALSE \
    -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -Dalpaka_ACC_GPU_HIP_ENABLE=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -Dalpaka_DISABLE_VENDOR_RNG=ON \
    -DTRACCC_SETUP_ROCTHRUST=ON \
    -DHIP_COMPILER=/opt/rocm/bin/hipcc \
    -DHIP_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DCMAKE_HIP_COMPILE_FEATURES=hip_std_20 \
    -DTRACCC_SUPPORTED_DETECTORS="itk_detector" \
    -DDETRAY_GENERATE_METADATA=itk_metadata



cmake --build <build_dir> -j 20
cmake --install <install_dir> 
```

Once this is built, set the following environement variables:

```
export INSTALLDIR=<install_dir>
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH
```

Note: if building on the ATLAS EF testbed, you will need to set the following environment variables:

```
export HTTP_PROXY=http://np04-web-proxy.cern.ch:3128
export HTTPS_PROXY=http://np04-web-proxy.cern.ch:3128
export NO_PROXY=".cern.ch"
export http_proxy=http://np04-web-proxy.cern.ch:3128
export https_proxy=http://np04-web-proxy.cern.ch:3128
export no_proxy=".cern.ch"
```

## Building the backend

First, enter the docker and set environment variables as documented above. Then run

```
cd backend/traccc-gpu && mkdir build install && cd build
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --target install -- -j20
```

The server can then be launched as above:

```
tritonserver --model-repository=../../models
```

## Running the simple `python` client

To test the server, the python client can be used. In the above docker image, run:

```
python TracccTritonClient.py
```

## Code Maintenance

Ideally this repository should be up-to-date with the latest release of `traccc`. In a perfect 
world where `traccc` has no major API changes, all that is required to keep this repository in sync 
is rebuilding `traccc` and then rebuilding the backend with the commands outlined above. However, 
when major API changes are introduced, this is not possible and parts of the code may need to be 
edited and refactored. 

### Updating `traccc`

The first step in updating this repository is to pull the most recent changes of `traccc`, and then 
build using the commands outlined above. From here, you should first attempt to build the standalone 
wrapper in the `standalone` repository. 

### Updating standalone wrapper

If this doesn't work, compare `standalone/src/TracccGpuStandalone.hpp` to the examples in 
`traccc/examples/run/cuda/`. In particular, `seq_example_cuda.cpp` is a good place to start for a 
template of how running the current version of `traccc` looks like. For proper variable 
initialization in the class `TracccGpuStandalone`, both `full_chain_algorithm.hpp` and the 
corresponding implementation file `full_chain_algorithm.cpp` are quite helpful. This step can take 
time and requires at least a cursory understanding of how `traccc` algorithms are scheduled. 

In the worst case, the output tracking containers may be changed. In this case, a more thorough 
investigation of the code may be necessary. First, find out what new output container(s) look like 
by browsing `traccc/core/include/traccc/edm` for something like `track_state_collection` or 
`track_fit_collection`. Then, identify how to access all the parameters necessary to execute the 
print statements in `standalone/src/TracccGpuStandalone.cpp`. 

Once the standalone compiles and runs, you can move on to the next step!

### Updating backend

If you did not have to make any changes to `standalone/src/TracccGpuStandalone.cpp` but only to the 
header file, then you should be able to build as before. If you did find yourself in the worst-case 
scenario above, then you will need to make similar changes to `backend/traccc-gpu/traccc.cc:676-836` 
as before to properly deal with output parsing. 

When complete, test by building and running the simple python client as detailed above. 

### Building production image

Once all the above steps are completed, you can build a container with your new developments by 
first creating a merge request with `main` of this repository (or change the tag of tracc-aaS in 
`backend/Dockerfile`), and then build an image from the `backend` repository with:

```
podman-hpc build -t name:tag .
```
or use your favorite docker application. 

If you would like to build a container without making a MR, edit `backend/Dockerfile:22` with your 
fork of this repository. 
