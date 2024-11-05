import argparse
import sys

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

def main():
    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count, ssl=False
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    print("\n=========")
    async_requests = []

    input_data = pd.read_csv(FLAGS.filename)
    input0_data = input_data['geometry_id'].to_numpy(dtype=np.uint64)
    input1_data = input_data.drop('geometry_id', axis=1).to_numpy(dtype=np.float64)
    inputs = [httpclient.InferInput("GEOMETRY_ID", input0_data.shape, "UINT64"),
              httpclient.InferInput("FEATURES", input1_data.shape, "FP64")]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)
    async_requests.append(triton_client.async_infer(f"traccc-{FLAGS.architecture}", inputs))

    # Define the outputs to retrieve
    output_names = ["chi2", "ndf", "local_positions", "covariances"]
    outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

    # Send the inference request
    async_request = triton_client.async_infer(
        model_name=f"traccc-{FLAGS.architecture}",
        inputs=inputs,
        outputs=outputs
    )

    # Collect the result
    result = async_request.get_result()

    # Retrieve and process outputs
    chi2 = result.as_numpy("chi2")  # Should be shape (1,)
    ndf = result.as_numpy("ndf")    # Should be shape (1,)
    local_positions = result.as_numpy("local_positions")  # Shape (N, 2)
    covariances = result.as_numpy("covariances")          # Shape (N, 2, 2)

    # Print the outputs
    print("chi2:", chi2)
    print("ndf:", ndf)
    print("local_positions shape:", local_positions.shape)
    print("local_positions:", local_positions)
    print("covariances shape:", covariances.shape)
    print("covariances:", covariances)

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