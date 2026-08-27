FROM docker.io/milescb/triton-server:26.06-py3_gcc15_cuda13.3

# install traccc
WORKDIR /traccc
RUN mkdir build install itk-geometry
RUN git clone https://github.com/acts-project/traccc.git 
RUN cd traccc && git checkout v1.6.0
WORKDIR /traccc/build
RUN cmake -DCMAKE_INSTALL_PREFIX=../install ../traccc \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRACCC_BUILD_CUDA=ON \
        -DTRACCC_BUILD_EXAMPLES=ON \
        -DTRACCC_USE_ROOT=FALSE \
        -DTRACCC_SUPPORTED_DETECTORS="itk_detector;odd_detector"

RUN cmake --build . --target install -- -j20

# add variables to environment
ENV PATH=/traccc/install/bin:$PATH
ENV LD_LIBRARY_PATH=/traccc/install/lib:$LD_LIBRARY_PATH

# install custom backend
WORKDIR /traccc-aaS
RUN git clone https://github.com/milescb/traccc-aaS.git && cd traccc-aaS/ \
    && git checkout return-track-buffers

RUN cd /traccc-aaS/traccc-aaS/backend/traccc-gpu && mkdir build install && cd build && \
    cmake -B . -S ../ -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-Wno-template-body" && \
    cmake --build . --target install -- -j20
