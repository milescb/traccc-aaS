FROM docker.io/milescb/tritonserver:latest

# install traccc
WORKDIR /traccc
RUN mkdir build install
RUN git clone https://github.com/acts-project/traccc.git 
RUN cd traccc && git checkout e7a03e9d
WORKDIR /traccc/build
RUN cmake -DCMAKE_INSTALL_PREFIX=../install ../traccc \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRACCC_BUILD_CUDA=ON \
        -DTRACCC_BUILD_EXAMPLES=ON \
        -DTRACCC_USE_ROOT=FALSE 
RUN cmake --build . --target install -- -j20

# add variables to environment
ENV PATH=/traccc/install/bin:$PATH
ENV LD_LIBRARY_PATH=/traccc/install/lib:$LD_LIBRARY_PATH

# install custom backend
WORKDIR /traccc-aaS
RUN git clone https://github.com/milescb/traccc-aaS.git
RUN cd traccc-aaS/backend/traccc-gpu && mkdir build install && cd build && \
    cmake -B . -S ../ -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install -- -j20

ENV MODEL_REPO=/traccc-aaS/traccc-aaS/backend/models