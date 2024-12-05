# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of the code included
2. a custom backend using the standalone version above to launch the trition server
3. a client to send data to the server

A minimal description of how to build a working version is detailed below. In each subdirectory of this project, a README containing more information can be found. 

## Previous work

The beginnings of this work is based on a CPU version developed by Haoran Zhao. The original repo can be found [here](https://github.com/hrzhao76/traccc-aaS). This CPU version has been incorporated into other branches of this work such as `odd_traccc_v0.10.0` but is omitted here for clarity. 

## Running out of the box

### Get the code

Simply clone the repository with 

```
git clone --recurse-submodules git@github.com:milescb/traccc-aaS.git
```

### Docker

A docker built for the triton server can be found at `docker.io/milescb/tritonserver:latest`. To run this do

```
shifter --module=gpu --image=milescb/tritonserver:latest
```

or use your favorite docker application and mount the appropriate directories. 

Finally, an image has been built with the custom backend pre-installed at `docker.io/milescb/traccc-aas:latest`. To run this, open the image, then run the server with

```
tritonserver --model-repository=$MODEL_REPO
```

This corresponds to the `Dockerfile` in this repository. 

### Shared Library 

To run out of the box, an installation of `traccc` and the the backend can be found at `/global/cfs/projectdirs/m3443/data/traccc-aaS/software/prod/ver_09152024/install`. To set up the environment, run the docker then set the following environment variables

```
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/software/prod/ver_09152024/install
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH
```

Then the server can be launched with 

```
tritonserver --model-repository=$INSTALLDIR/models
```

Once the server is launched, run the model via:

```
cd client && python TracccTritionClient.py 
```
More info in the client directory. 

## Building the backend

First, enter the docker and set environment variables as documented above. Then run

```
cd backend/traccc-gpu && mkdir build install && cd build
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_INSTALL_PREFIX=../install/

cmake --build . --target install -- -j20
```

Then, the server can be launched as above:

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

Then, forward the appropriate port:

```
kubectl port-forward service/triton-atlas 8000:8000 -n atlas-sonic
```

You can then check everything is healthy with

```
curl -v localhost:8000/v2/health/ready
```

which should produce the output

```
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

Then, the client can be run as before. To see what's going on on the server, run

```
kubectl logs triton-atlas-594f595dbf-n4sk7
```

where `triton-atlas-594f595dbf-n4sk7` is the name of the server found when running the `get pods` command above. 

### !!! Important !!!

Make sure to `uninstall` once the server is not needed anymore. 

```
helm uninstall super-sonic -n atlas-sonic
```

Make sure to read the [Policies](https://docs.nationalresearchplatform.org/userdocs/start/policies/) before using Nautilus. 