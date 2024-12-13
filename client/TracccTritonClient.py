import argparse
import sys

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ROOT)

def main():
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

    # Read input data
    input_data = pd.read_csv(FLAGS.filename)
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
    chi2 = result.as_numpy("chi2")  # Should be shape (1,)
    ndf = result.as_numpy("ndf")    # Should be shape (1,)
    local_positions_buffer = result.as_numpy("local_positions")  # Shape (N, 2)
    local_positions_lengths = result.as_numpy("local_positions_lengths")  # Shape (N,)
    variances_buffer = result.as_numpy("variances")          # Shape (N, 2)
    
    print("local_positions_buffer:", local_positions_buffer)
    print("local_positions_buffer shape:", local_positions_buffer.shape)
    print("local_positions_lengths:", local_positions_lengths)
    print("local_positions_lengths shape:", local_positions_lengths.shape)
    print("variances_buffer:", variances_buffer)
    print("variances_buffer shape:", variances_buffer.shape)
    
    idx = 0
    reconstructed_positions = []
    reconstructed_variances = []
    # position and variance lengths are the same by construction
    for length in local_positions_lengths:
        track_positions = []
        track_variances = []
        for _ in range(length):
            pos = local_positions_buffer[idx]
            track_positions.append(pos)
            var = variances_buffer[idx]
            track_variances.append(var)
            idx += 1
        reconstructed_positions.append(track_positions)
        reconstructed_variances.append(track_variances)

    # Print the outputs
    print("chi2:", chi2)
    print("chi2 shape:", chi2.shape)
    print("ndf:", ndf)
    print("reconstructed_positions:", reconstructed_positions)
    print("reconstructed_variances:", reconstructed_variances)
    
    plt.figure(figsize=(6,6))
    plt.hist(chi2/ndf, bins=25, range=(0, 5))
    plt.xlabel(r"$\chi^2/ndf$", loc="right")
    plt.ylabel("Fitted Track Results", loc="top")
    plt.savefig("plots/chi2.png", bbox_inches="tight", dpi=300)

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
        "-f",
        "--filename",
        type=str,
        required=False,
        default="event000000000-cells.csv",
        help="Input file name. Default is event000000000-cells.csv.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        default="gpu",
        help="Model architecture. Default is gpu.",
    )
    FLAGS = parser.parse_args()

    main()