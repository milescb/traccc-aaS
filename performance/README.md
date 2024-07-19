# Run Performance Studies

## Create input 

Create input json file with 

```
python generate_json.py -i <input_file> -o <output_file>
```

## Perf Analyzer

Run `perf_analyzer` with the following:

```
perf_analyzer -m traccc-gpu --input-data $DATADIR/../test_data/test_perf_data.json
```

## Performance Plots

### Throughput as a function of concurrency

Run `perf_analyzer` with the CPU and GPU configuration

```
perf_analyzer -m traccc-gpu --input-data $DATADIR/../test_data/test_perf_data.json \
    --concurrency-range <start>:<end>:<step> \
    --verbose-csv --collect-metrics \
    --measurement-interval 10000 -f out_gpu.csv 
```

then run

```
python plot_concurrency_studies.py
```

### Compare CPU / GPU performance on `traccc` examples

First, make `.log` files containing the output of running the `traccc` examples via (for instance)

```
$INSTALLDIR/bin/traccc_seq_example_cuda \
    --use-detray-detector \
    --detector-file=$DATADIR/tml_detector/trackml-detector.csv \
    --digitization-file=$DATADIR/tml_detector/default-geometric-config-generic.json \
    --input-directory=$DATADIR/tml_full/ttbar_mu100 &> data/logs/gpu_mu100.log
```

then run the following to create the plot:

```
python compare_gpu_cpu_traccc.py
```