## Build the standalone model for testing

Run within the standalone directory, add the appropriate variables to `PATH` as discussed in the base README, then run

```
mkdir build && cd build
cmake ../
cmake --build .
```

## Run the standalone model

Run within the build directory

```
./TracccGpuStandalone $DATADIR/odd/geant4_10muon_10GeV/event000000000-cells.csv 0
```

## Compare to official example

Running the following is ~equivalent to what the standalone version does:

```
rm -rf traccc/data && ln -s $DATADIR traccc/data # necessary to avoid bug with ODD detector
$INSTALLDIR/bin/traccc_seq_example_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json
    --digitization-file=geometries/odd/odd-digi-geometric-config.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --input-directory=$DATADIR/odd/geant4_10muon_100GeV/
```

The CPU version of this executable is `$INSTALLDIR/bin/traccc_seq_example` and can be run exactly the same as above. 