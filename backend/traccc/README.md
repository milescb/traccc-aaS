
Taken in large part from [triton-inference-server/backend](https://github.com/triton-inference-server/backend/tree/main)

## Build model

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
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/ACTS-aaS/sw/prod/ver_02012024
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH

tritonserver --model-repository=<model_repo>
```