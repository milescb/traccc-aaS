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