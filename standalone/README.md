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
./TracccGpuStandalone $DATADIR/tml_pixels/event000000000-cells.csv
```

The output should look something like:

```
Measurement ID: 416
Local coordinates: [-0.575, -30.425]
Measurement ID: 417
Local coordinates: [-6.825, -20.775]
Measurement ID: 418
Local coordinates: [5.375, -33.775]
Measurement ID: 419
Local coordinates: [6.475, -20.525]
Measurement ID: 420
Local coordinates: [7.425, -34.775]
Measurement ID: 421
Local coordinates: [-1.575, -17.075]
Measurement ID: 422
Local coordinates: [-1.175, -34.625]
Measurement ID: 423
Local coordinates: [4.125, -33.475]
Measurement ID: 424
Local coordinates: [-3.675, -1.875]
Measurement ID: 425
Local coordinates: [6.375, -1.375]
```

## Compare to official example

Running the following is ~equivalent to what the standalone version does:

```
$INSTALLDIR/bin/traccc_seq_example_cuda --detector-file=$DATADIR/tml_detector/trackml-detector.csv --digitization-file=$DATADIR/tml_detector/default-geometric-config-generic.json --input-directory=$DATADIR/tml_pixels/
```

Running without the option `--use-detray-detector` does not run track fitting or track finding. 