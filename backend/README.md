
Taken in large part from [triton-inference-server/backend](https://github.com/triton-inference-server/backend/tree/main)

## Build model

Both a cpu and gpu model are available. Both must be built separately as shown below:

```
cmake -B <build_dir> -S <source_dir> \
    -DCMAKE_INSTALL_PREFIX=<install_dir> \
    -DCMAKE_INSTALL_PREFIX=<install_dir>

cmake --build <build_dir> --target install -- -j20
```
Assuming building from `<build_dir>` this would look like

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_INSTALL_PREFIX=../install/

cmake --build . --target install -- -j20
````

## Run model

```
export INSTALLDIR=/pscratch/sd/m/milescb/traccc-aaS/dev/install
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH

tritonserver --model-repository=$INSTALLDIR/models
```

This will launch both the cpu and gpu model for deployment. 