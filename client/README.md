## Run the client

First, consult the README in the `backend` folder for instructions on running the server. 

To run the model, install required packages (if not already installed):

```
pip install numpy pandas tritonclient
```

then run the model from the client directory

```
python TracccTritionClient.py
```

This uses the example file `event000000000-cells.csv` from the tml detector. More example files can be found in `$DATADIR` and passed to the client via the `--filename` argument. Additionally, if the server is not running on `localhost:8000`, then a url may be provided through the `--url` argument. 