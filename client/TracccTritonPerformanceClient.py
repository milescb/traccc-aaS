import argparse
import sys
import os

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ATLAS)

def get_data_files(directory):
    directory = os.path.expandvars(directory)
    files = []
    for root, dirs, filenames in os.walk(directory):
        if '300' in root:
            continue
        for filename in filenames:
            if filename.endswith("cells.csv"):
                files.append(os.path.join(root, filename))
    if len(files) == 0:
        print(f"No data files found in {directory}")
        sys.exit(1)
    return files

def run_inference(filename):
    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 1
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count, ssl=FLAGS.ssl
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    print("\n=========")
    
    # Read input data
    input_data = pd.read_csv(filename)
    input0_data = input_data['geometry_id'].to_numpy(dtype=np.uint64)
    input1_data = input_data.drop('geometry_id', axis=1).to_numpy(dtype=np.float64)

    # Prepare inputs
    inputs = [
        httpclient.InferInput("GEOMETRY_ID", input0_data.shape, "UINT64"),
        httpclient.InferInput("FEATURES", input1_data.shape, "FP64")
    ]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    # Specify outputs
    output_names = ["chi2", "ndf", "local_positions", "local_positions_lengths", "variances"]
    outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

    # Send inference request synchronously
    result = triton_client.infer(
        model_name="traccc-gpu",
        inputs=inputs,
        outputs=outputs
    )

    # Retrieve and process outputs
    chi2 = result.as_numpy("chi2")  # Should be shape (N,)
    ndf = result.as_numpy("ndf")    # Should be shape (N,)
    local_positions_buffer = result.as_numpy("local_positions")  # Shape (N, 2)
    local_positions_lengths = result.as_numpy("local_positions_lengths")  # Shape (N,)
    variances_buffer = result.as_numpy("variances")          # Shape (N, 2)
    
    # chi2/ndf protect possible division by zero
    NUM_TRACKS.append(chi2.shape[0])
    for chi2, ndf in zip(chi2, ndf):
        CHI2.append(chi2)
        NDF.append(ndf)
        if ndf != 0:
            CHI2_NDF.append(chi2/ndf)
    
def main():
    
    data_files = get_data_files(FLAGS.directory)
    
    for file in data_files:
        print(f"Running inference on {file}")
        run_inference(file)
    
    plt.figure(figsize=(6,6))
    plt.hist(CHI2_NDF, bins=40, range=(0, 40))
    plt.xlabel(r"$\chi^2/ndf$", loc="right")
    plt.ylabel("Fitted Track Results", loc="top")
    plt.savefig("plots/chi2.png", bbox_inches="tight", dpi=300)
    
    # plt.figure(figsize=(6,6))
    # plt.hist(NDF, bins=25, range=(0, 25))
    # plt.xlabel("NDF", loc="right")
    # plt.ylabel("Fitted Track Results", loc="top")
    # plt.savefig("plots/ndf.png", bbox_inches="tight", dpi=300)
    
    plt.figure(figsize=(6,6))
    plt.hist(NUM_TRACKS, bins=25, range=(0, 40000))
    plt.xlabel("Number of Tracks", loc="right")
    plt.ylabel("Fitted Track Results", loc="top")
    plt.savefig("plots/num_tracks.png", bbox_inches="tight", dpi=300)
    
    np.save("plots/chi2.npy", CHI2)
    np.save("plots/ndf.npy", NDF)
    np.save("plots/chi2_ndf.npy", CHI2_NDF)
    np.save("plots/num_tracks.npy", NUM_TRACKS)
    
    return 0;

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable encrypted link to the server.",
    )
    parser.add_argument(
        "-dir",
        "--directory",
        type=str,
        required=False,
        default="$DATADIR/../data_odd_ttbar_large/",
        help="Input file directory. Default is $DATADIR/../data_odd_ttbar_large/",
    )
    FLAGS = parser.parse_args()
    
    CHI2 = []
    NDF = []
    CHI2_NDF = []
    NUM_TRACKS = []

    main()