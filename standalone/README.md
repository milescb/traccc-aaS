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
./TracccGpuStandalone $DATADIR/odd/geant4_10muon_100GeV/event000000000-cells.csv
```

The output should look something like:

```
Measurement ID: 16
Local coordinates: [-6.775, 32.725]
Measurement ID: 17
Local coordinates: [8.525, -7.075]
Measurement ID: 18
Local coordinates: [5.425, 15.775]
Measurement ID: 19
Local coordinates: [-12.675, 21.875]
Measurement ID: 20
Local coordinates: [-5.825, 9.025]
Measurement ID: 21
Local coordinates: [4.775, -6.70172]
Measurement ID: 22
Local coordinates: [4.675, -5.90643]
Measurement ID: 23
Local coordinates: [-10.825, -1.66034]
Measurement ID: 24
Local coordinates: [13.175, -2.08204]
Measurement ID: 25
Local coordinates: [-5.075, -10.725]
```

or to run the CPU version:

```
./TracccCpuStandalone $DATADIR/tml_pixels/event000000000-cells.csv
```

## Compare to official example

Running the following is ~equivalent to what the standalone version does:

```
$INSTALLDIR/bin/traccc_seq_example_cuda \
    --detector-file=$DATADIR/tml_detector/trackml-detector.csv \
    --digitization-file=$DATADIR/tml_detector/default-geometric-config-generic.json 
    --input-directory=$DATADIR/tml_pixels/
```

Running without the option `--use-detray-detector` does not run track fitting or track finding. The CPU version of this executable is `$INSTALLDIR/bin/traccc_seq_example` and can be run exactly the same as above. 