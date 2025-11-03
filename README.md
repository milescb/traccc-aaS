# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of the code included
2. a custom backend using the standalone version above to launch the Triton server
3. a client to send data to the server

A minimal description of how to build a working version is detailed below. In each subdirectory of this project, a README containing more information can be found. For instructions on running the pipeline with an Athena client on `lxplus`, consult [this CodiMD page](https://codimd.web.cern.ch/1FcLmapORpeBtAVL_M6h4A?view). The last part of this README describes the necessary steps for maintaining and updating `traccc-aaS` when a new release of `traccc` is available. 

## Obtaining the `ITk` geometry files

You will need access to the `ITk` geometry files to run this repository, which can be found at `/eos/project/a/atlas-eftracking/GPU/ITk_data/ATLAS-P2-RUN4-03-00-01/` on CERN's `lxplus` computing cluster. Note that to access these, you must be a part of the e-group `atlas-tdaq-phase2-EFTracking-developers`. If you are outside of ATLAS, please refer to the `main` branch to run the ODD version of the code.

## Running out of the box

The easiest way to run `traccc` as-a-Service is with our container. Pull the image at `docker.io/milescb/traccc-aas:v1.2_athena_client` and run the image interactively. To do this, you need access to the ITk geometry files, obtained by following the above instructions, and these need to be mounted to `/traccc/itk-geometry`. 

Then, server can be launched with:

```
tritonserver --model-repository=$MODEL_REPO
```

To test this with the client, open another window in the docker (using tmux, for instance), then run:

```
cd traccc-aas/client
python TracccTritonClient.py
```

### Get the code

Simply clone the repository with 

```
git clone --recurse-submodules git@github.com:milescb/traccc-aaS.git

# checkout this branch
cd traccc-aaS && git checkout itk-hits-to-tracks
```

### Docker

The easiest way to build the custom backend is with the docker at `docker.io/milescb/triton-server:25.02-py3_gcc13.3`. Run this interactively with

```
shifterimg pull milescb/triton-server:25.02-py3_gcc13.3
shifter --module=gpu --image=milescb/triton-server:25.02-py3_gcc13.3
```

or use your favorite docker application and mount the appropriate directories. 

### Shared Library on `nersc`

To run out of the box at `nersc`, an installation of `traccc` and the the backend can be found at `/global/cfs/projectdirs/m3443/data/traccc-aaS/prod/ver_082625_g200/install`. To set up the environment, enter the image, then set the following environment variables

```
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/prod/ver_082625_g200/install
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH
```

Then, the server can be launched with 

```
tritonserver --model-repository=$INSTALLDIR/models
```

Once the server is launched, run the model (on the same node to avoid networking problems) via:

```
cd client && python TracccTritionClient.py 
```
More info in the client directory. 

### Building from source

If you don't have access to `nersc`, you'll have to build `traccc` yourself. Follow the instructions on the [traccc page](https://github.com/acts-project/traccc/tree/main) to build or run this configure command:

```
cmake <path_to_cmake> \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_CUDA=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DTRACCC_USE_ROOT=FALSE \
    -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
make -j20 install
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

## Deploy on K8s cluster using SuperSONIC

For server-side large-scale deployment we are using the [SuperSONIC](https://github.com/fastmachinelearning/SuperSONIC) 
framework. 


### To deploy the server on NRP Nautilus

```
source deploy-nautilus-atlas.sh
```

The settings are defined in `helm/values-nautilus-atlas.yaml` files. 
You can update the setting simply by sourcing the deployment script again. 
 
You can find the server URL in the same configs. It will take a few seconds to start a server, depending on the specs of the GPUs requested.

### Running the client

In order for the client to interface with the server, the location of the server needs to be specified. First, ensure the server is running

```
kubectl get pods -n atlas-sonic
```
which has output something like:

```
NAME                            READY   STATUS    RESTARTS   AGE
envoy-atlas-7f6d99df88-667jd    1/1     Running   0          86m
triton-atlas-594f595dbf-n4sk7   1/1     Running   0          86m
```

or, use the [k9s](https://k9scli.io) tool to manage your pods. You can then check everything is healthy with

```
curl -kv https://atlas.nrp-nautilus.io/v2/health/ready
```

which should produce somewhere in the output the lines:

```
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

Then, the client can be run with, for instance:

```
python TracccTritonClient.py -u atlas.nrp-nautilus.io --ssl
```

To see what's going on from the server side, run

```
kubectl logs triton-atlas-594f595dbf-n4sk7
```

where `triton-atlas-594f595dbf-n4sk7` is the name of the server found when running the `get pods` command above. 

### !!! Important !!!

Make sure to `uninstall` once the server is not needed anymore. 

```
helm uninstall atlas-sonic -n atlas-sonic
```

Make sure to read the [Policies](https://docs.nationalresearchplatform.org/userdocs/start/policies/) before using Nautilus. 

## Code Maintenance

Ideally this repository should be up-to-date with the latest release of `traccc`. In a perfect world where `traccc` has no major API changes, all that is required to keep this repository in sync is rebuilding `traccc` and then rebuilding the backend with the commands outlined above. However, when major API changes are introduced, this is not possible and parts of the code may need to be edited and refactored. 

### Updating `traccc`

The first step in updating this repository is to pull the most recent changes of `traccc`, and then build using the commands outlined above. From here, you should first attempt to build the standalone wrapper in the `standalone` repository. 

### Updating standalone wrapper

If this doesn't work, compare `standalone/src/TracccGpuStandalone.hpp` to the examples in `traccc/examples/run/cuda/`. In particular, `seq_example_cuda.cpp` is a good place to start for a template of how running the current version of `traccc` looks like. For proper variable initialization in the class `TracccGpuStandalone`, both `full_chain_algorithm.hpp` and the corresponding implementation file `full_chain_algorithm.cpp` are quite helpful. This step can take time and requires at least a cursory understanding of how `traccc` algorithms are scheduled. 

In the worst case, the output tracking containers may be changed. In this case, a more thorough investigation of the code may be necessary. First, find out what new output container(s) look like by browsing `traccc/core/include/traccc/edm` for something like `track_state_collection` or `track_fit_collection`. Then, identify how to access all the parameters necessary to execute the print statements in `standalone/src/TracccGpuStandalone.cpp`. 

Once the standalone compiles and runs, you can move on to the next step!

### Updating backend

If you did not have to make any changes to `standalone/src/TracccGpuStandalone.cpp` but only to the header file, then you should be able to build as before. If you did find yourself in the worst-case scenario above, then you will need to make similar changes to `backend/traccc-gpu/traccc.cc:676-836` as before to properly deal with output parsing. 

When complete, test by building and running the simple python client as detailed above. 

### Building production image

Once all the above steps are completed, you can build a container with your new developments by first creating a merge request with `main` of this repository, and then build an image from the `backend` repository with:

```
podman-hpc build -t name:tag .
```
or use your favorite docker application. 

If you would like to build a container without making a MR, edit `backend/Dockerfile:22` with your fork of this repository. 