# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of the code included
2. a custom backend using the standalone version above to launch the trition server
3. a client to send data to the server

A minimal description of how to build a working version is detailed below. In each subdirectory of this project, a README containing more information can be found. 

## Running out of the box

### Docker

A docker built for the triton server can be found at `docexoty/tritonserver:latest`. To run this do

```
shifter --module=gpu --image=docexoty/tritonserver:latest
```

or use your favorite docker application and mount the appropriate directories. 

### Shared Library 

To run out of the box, an installation of `traccc` and the the backend can be found at `/global/cfs/projectdirs/m3443/data/traccc-aaS/prod/ver_07032024/install`. To set up the environment, run the docker then set the following environment variables

```
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/prod/ver_07032024/install
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
cd backend/traccc && mkdir build install && cd build
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_INSTALL_PREFIX=../install/

cmake --build . --target install -- -j20
```

The relevant shared library for triton can then be found in `install/backends/traccc/libtriton_traccc.so`. Copy this file into the `models/traccc` directory. Then the server can be started via (assuming the user is still in the build directory above)

```
tritonserver --model-repository=../models
```