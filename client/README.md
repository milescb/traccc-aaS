## Run the client

First, consult the README in the `backend` folder for instructions on running the server. 

To run the model, install required packages (if not already installed) via conda with:

```
conda env create -f env.yml
conda activate triton-client
```

then run the model from the client directory:

```
python TracccTritonClient.py
```

or to run using the remote configuration:

```
python TracccTritonClint.py --ssl -u <url_to_server>
```

This uses the example file `event000000000-cells.csv` from the ODD detector. More example files can be found in `$DATADIR` and passed to the client via the `--filename` argument. Additionally, if the server is not running on `localhost:8000`, then a url may be provided through the `--url` argument. Finally, the architecture (CPU vs GPU) can be changed with the flag `-a cpu`. 
