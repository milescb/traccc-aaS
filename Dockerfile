# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/tritonserver:26.06-py3
# Base image: Ubuntu 24.04, CUDA 13.3 (nvcc supports host compilers up to gcc-15)

LABEL description="Triton Server backend with other dependencies for traccc-as-a-Service"
LABEL version="1.0"

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
ENV PYTHONNOUSERSITE=True
ENV TORCH_CUDA_ARCH_LIST="80"

# System dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential curl git git-lfs freeglut3-dev libfreetype6-dev libpcre3-dev \
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
    libtbb-dev ninja-build time tree \
    python3 python3-dev python3-pip python3-numpy \
    zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig \
    rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev libb64-dev libxxhash-dev \
    libsuitesparse-dev libhwloc-dev libsuperlu-dev \
    software-properties-common \
  && git lfs install \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install gcc-15 (latest toolchain supported by CUDA 13.3) and make it the default
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test \
  && apt-get update -y && apt-get install -y gcc-15 g++-15 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 150 \
     --slave /usr/bin/g++ g++ /usr/bin/g++-15 \
     --slave /usr/bin/gcov gcov /usr/bin/gcov-15 \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
  && gcc --version

# Python packages: build tooling (cmake kept current via pip instead of a manual
# source build), triton client, and everything needed for acorn
RUN pip3 install -U \
    cmake pandas matplotlib seaborn \
    pyyaml astunparse "expecttest!=0.2.0" hypothesis numpy psutil requests setuptools \
    types-dataclasses "typing-extensions>=4.8.0" sympy filelock networkx jinja2 fsspec \
    lintrunner ninja packaging "optree>=0.11.0" \
    "tritonclient[all]" \
    click pytest pytest-cov class-resolver scipy uproot tqdm ipykernel \
    atlasify wandb mplhep \
  && pip3 install -U git+https://github.com/LAL/trackml-library.git
