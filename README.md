# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of the code included
2. a custom backend using the standalone version above to launch the trition server
3. a client to send data to the server

A minimal description of how to build a working version is detailed below. In each subdirectory of this project, a README containing more information can be found. For instructions on running the pipeline with an Athena client on `lxplus`, consult [this CodiMD page](https://codimd.web.cern.ch/1FcLmapORpeBtAVL_M6h4A?view). 

## Obtaining the `ITk` geometry files

You will need access to the `ITk` geometry files to run this repository, which can be found at `/eos/project/a/atlas-eftracking/GPU/ITk_data/ATLAS-P2-RUN4-03-00-01/` on CERN's `lxplus` computing cluster. Note that to access these, you must be a part of the e-group `atlas-tdaq-phase2-EFTracking-developers`. If you are outside of ATLAS, please refer to the `main` branch to run the ODD version of the code.

## Running out of the box

The easiest way to run `traccc` as-a-Service is with our container. Pull the image at `docker.io/milescb/traccc-aas:v1.0_athena_client` and run the image interactively. To do this, you need access to the ITk geometry files, obtained by following the above instructions, and these need to be mounted to `/traccc/itk-geometry`. 

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
shifter --module=gpu --image=milescb/tritonserver:triton-server:25.02-py3_gcc13.3
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
    -DCMAKE_INSTALL_PREFIX=../install/

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