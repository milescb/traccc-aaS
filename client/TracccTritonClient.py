import argparse
import sys

import numpy as np
import pandas as pd
# import tritonclient.http as httpclient

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ROOT)

import tritonclient.grpc as grpcclient

def plot_histogram(data, name, xlabel, bins=50):
    """Plots a histogram for the given data on a new canvas."""
    if data is None or data.size == 0:
        print(f"No data to plot for {name}")
        return
    plt.figure(figsize=(8, 8))
    plt.hist(data, bins=bins, histtype='step', label=name)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f"plots/{name.replace(" ", "_")}.png")

def main():
    # For the gRPC client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 1
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, ssl=FLAGS.ssl
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # Read input data
    input_data = pd.read_csv(FLAGS.filename)
    input0_data = input_data['detray_id'].to_numpy(dtype=np.uint64)
    feature_columns = ['local_key', 'local_x', 'local_y', 'global_x', 'global_y', 'global_z', 'is_pixel']
    input1_data = input_data[feature_columns].to_numpy(dtype=np.float64)

    # Prepare inputs
    inputs = [
        grpcclient.InferInput("GEOMETRY_ID", input0_data.shape, "UINT64"),
        grpcclient.InferInput("FEATURES", input1_data.shape, "FP64")
    ]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    # Specify outputs
    output_names = ["chi2", "ndf", "local_positions", "local_positions_lengths", "variances"]
    output_names = [
        "chi2", "ndf", 
        "local_positions", 
        "variances",
        "detray_ids",
        "measurement_ids",
        "measurement_dims",
        "times"
    ]
    outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

    # Send inference request synchronously
    result = triton_client.infer(
        model_name="traccc-gpu",
        inputs=inputs,
        outputs=outputs
    )

    # Retrieve and process outputs
    chi2 = result.as_numpy("chi2")
    ndf = result.as_numpy("ndf")
    local_positions = result.as_numpy("local_positions") 
    variances = result.as_numpy("variances")             
    detray_ids = result.as_numpy("detray_ids")            
    measurement_ids = result.as_numpy("measurement_ids")  
    measurement_dims = result.as_numpy("measurement_dims")
    times = result.as_numpy("times")    
    
    # filter data with measurement_dims = 2
    mask = measurement_dims == 2
    local_positions = local_positions[mask]
    variances = variances[mask]
    detray_ids = detray_ids[mask]
    measurement_ids = measurement_ids[mask]
    measurement_dims = measurement_dims[mask]
    times = times[mask] 
    
    plot_histogram(chi2, "Chi2", "Chi2")
    plot_histogram(ndf, "NDF", "NDF")    
    plot_histogram(local_positions[:, 0], "Local Position X", "Local Position X")
    plot_histogram(local_positions[:, 1], "Local Position Y", "Local Position Y")    
    plot_histogram(measurement_dims, "Measurement Dims", "Measurement Dimensions")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
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
        default="clusters.csv",
        help="Input file name. Default is clusters.csv.",
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