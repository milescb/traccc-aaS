FROM docker.io/milescb/triton-server:latest

# install traccc
WORKDIR /traccc
# copy geometry files to container
COPY odd_configuration ./odd/
# build and install traccc
RUN mkdir build install
RUN git clone https://github.com/acts-project/traccc.git 
RUN cd traccc && git checkout v0.20.0
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

# Replace hard-coded paths in TracccGpuStandalone.hpp with local paths in docker
RUN sed -i 's|/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_geometry_detray.json|/traccc/odd/odd-detray_geometry_detray.json|g' \
                /traccc-aaS/traccc-aaS/standalone/src/TracccGpuStandalone.hpp
RUN sed -i 's|/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-digi-geometric-config.json|/traccc/odd/odd-digi-geometric-config.json|g' \
                /traccc-aaS/traccc-aaS/standalone/src/TracccGpuStandalone.hpp
RUN sed -i 's|/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_surface_grids_detray.json|/traccc/odd/odd-detray_surface_grids_detray.json|g' \
                /traccc-aaS/traccc-aaS/standalone/src/TracccGpuStandalone.hpp
RUN sed -i 's|/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_material_detray.json|/traccc/odd/odd-detray_material_detray.json|g' \
                /traccc-aaS/traccc-aaS/standalone/src/TracccGpuStandalone.hpp

RUN cd traccc-aaS/backend/traccc-gpu && mkdir build install && cd build && \
    cmake -B . -S ../ -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install -- -j20

RUN cp -r /traccc-aaS/traccc-aaS/backend/models /traccc-aaS/traccc-aaS/backend/models_multi_gpu 
RUN sed -i "/gpus:/c\    gpus: [ 0, 1, 2, 3 ]" /traccc-aaS/traccc-aaS/backend/models_multi_gpu/traccc-gpu/config.pbtxt

ENV MODEL_REPO=/traccc-aaS/traccc-aaS/backend/models
ENV MODEL_REPO_MULTI_GPU=/traccc-aaS/traccc-aaS/backend/models_multi_gpu

# make multiple versions with up to 8 model instances per GPU
RUN for i in {1..8}; do \
        cp -r /traccc-aaS/traccc-aaS/backend/models /traccc-aaS/traccc-aaS/backend/nmodels_$i \
        && sed -i "s/count: 1/count: $i/" /traccc-aaS/traccc-aaS/backend/nmodels_$i/traccc-gpu/config.pbtxt; done