#!/usr/bin/bash

for i in 20 40 60 80 100 140 200 300
do
    echo Submitting job for mu = $i
    $INSTALLDIR/bin/traccc_throughput_mt \
        --use-detray-detector \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=$DATADIR/../data_odd_ttbar_large/geant4_ttbar_mu$i \
        &> data/logs_odd/out_cpu_mu$i.log
    echo CPU job completed for mu = $i
    $INSTALLDIR/bin/traccc_throughput_mt_cuda \
        --use-detray-detector \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=$DATADIR/../data_odd_ttbar_large/geant4_ttbar_mu$i \
        &> data/logs_odd/out_gpu_mu$i.log
    echo GPU job completed for mu = $i
done