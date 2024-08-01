#!/usr/bin/bash

for i in 200 300
do
    echo Submitting job for mu = $i
    $INSTALLDIR/bin/traccc_seq_example \
        --use-detray-detector \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=$DATADIR/../data_odd_ttbar_large/geant4_ttbar_mu$i \
        --input-events=10 \
        &> data/logs_odd/out_cpu_mu$i.log
    echo CPU job completed for mu = $i
    $INSTALLDIR/bin/traccc_seq_example_cuda \
        --use-detray-detector \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=$DATADIR/../data_odd_ttbar_large/geant4_ttbar_mu$i \
        --input-events=10 \
        &> data/logs_odd/out_gpu_mu$i.log
    echo GPU job completed for mu = $i
done