import argparse
import sys

import numpy as np
import pandas as pd
import awkward as ak

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
    plt.ylabel("Events")
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

    input_data = pd.read_csv(FLAGS.filename)
    
    cell_positions_columns = ['geometry_id', 'measurement_id', 'channel0', 'channel1']
    cell_properties_columns = ['timestamp', 'value']

    input0_data = input_data[cell_positions_columns].to_numpy(dtype=np.int64)
    input1_data = input_data[cell_properties_columns].to_numpy(dtype=np.float32)

    inputs = [
        grpcclient.InferInput("CELL_POSITIONS", input0_data.shape, "INT64"),
        grpcclient.InferInput("CELL_PROPERTIES", input1_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    # Specify outputs
    output_names = [
        "TRK_PARAMS",      # [n_tracks, 2] - chi2, ndf
        "MEASUREMENTS",    # [total_meas_with_seps, 6] - localx, localy, phi, theta, qop, time
        "COVARIANCES",     # [total_meas_with_seps, 25] - 5x5 covariance matrix flattened
        "GEOMETRY_IDS"     # [total_meas_with_seps] - geometry IDs with 0 separators
    ]
    outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

    # Send inference request synchronously
    result = triton_client.infer(
        model_name="traccc-gpu",
        inputs=inputs,
        outputs=outputs
    )

    # Retrieve and process outputs
    trk_params = result.as_numpy("TRK_PARAMS")       # [n_tracks, 2]
    measurements = result.as_numpy("MEASUREMENTS")   # [total_meas_with_seps, 6]
    covariances = result.as_numpy("COVARIANCES")     # [total_meas_with_seps, 25]
    geometry_ids = result.as_numpy("GEOMETRY_IDS")   # [total_meas_with_seps]
    
    # Extract track parameters
    chi2 = trk_params[:, 0]
    ndf = trk_params[:, 1]
    
    # Reconstruct tracks using geometry_id separators (gid == 0)
    track_measurements = []
    track_covariances = []
    track_geometry_ids = []

    cur_meas = []
    cur_cov = []
    cur_geo = []
    meas_idx = 0
    for gid in geometry_ids:
        if gid == 0:
            if cur_meas:
                track_measurements.append(np.stack(cur_meas))
                track_covariances.append(np.stack(cur_cov))
                track_geometry_ids.append(np.stack(cur_geo))
                cur_meas, cur_cov, cur_geo = [], [], []
            continue
        if meas_idx >= len(measurements):
            break
        cur_meas.append(measurements[meas_idx])
        cur_cov.append(covariances[meas_idx])
        cur_geo.append(gid)
        meas_idx += 1
    if cur_meas:
        track_measurements.append(np.stack(cur_meas))
        track_covariances.append(np.stack(cur_cov))
        track_geometry_ids.append(np.stack(cur_geo))

    ak_measurements = ak.Array(track_measurements)
    ak_covariances = ak.Array(track_covariances)
    ak_geometry_ids = ak.Array(track_geometry_ids)

    print(f"Number of tracks: {len(trk_params)}")
    print(f"Number of measurements per track: {ak.num(ak_measurements, axis=1)}")
    print(f"Total measurements: {ak.sum(ak.num(ak_measurements, axis=1))}")
    print(f"Track parameters shape: {trk_params.shape}")
    print(f"Awkward measurements shape: {ak_measurements.type}")
    print(f"Awkward covariances shape: {ak_covariances.type}")
    print(f"Awkward geometry IDs shape: {ak_geometry_ids.type}")

    # Extract variances from covariances (diag elements 0 and 7)
    ak_var_x = ak.Array([[cov[0] for cov in covs] for covs in track_covariances])
    ak_var_y = ak.Array([[cov[6] for cov in covs] for covs in track_covariances])

    # Example: Access measurements for first tracks
    if len(ak_measurements) > 0:
        print(f"\nTrack 0 has {len(ak_measurements[0])} measurements:")
        print(f"  Local positions: {ak_measurements[0]}")
        print(f"  Local variances (from diagonal): x={ak_var_x[0]}, y={ak_var_y[0]}")
        print(f"  Geometry IDs: {ak_geometry_ids[0]}")
    if len(ak_measurements) > 1:
        print(f"\nTrack 1 has {len(ak_measurements[1])} measurements:")
        print(f"  Local positions: {ak_measurements[1]}")
        print(f"  Local variances (from diagonal): x={ak_var_x[1]}, y={ak_var_y[1]}")
        print(f"  Geometry IDs: {ak_geometry_ids[1]}")

    # Plot histograms
    plot_histogram(chi2, "Chi2", "Chi2")
    plot_histogram(ndf, "NDF", "NDF")
    
    # Plot variances extracted from covariance matrices
    plot_histogram(ak.flatten(ak_var_x).to_numpy(), "Var_X", "Local X Variance")
    plot_histogram(ak.flatten(ak_var_y).to_numpy(), "Var_Y", "Local Y Variance")
    
    # Additional awkward array operations
    print(f"\nAwkward array operations:")
    print(f"Average measurements per track: {ak.mean(ak.num(ak_measurements, axis=1)):.2f}")
    print(f"Max measurements in any track: {ak.max(ak.num(ak_measurements, axis=1))}")
    print(f"Min measurements in any track: {ak.min(ak.num(ak_measurements, axis=1))}")

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
        default="event000000000-cells.csv",
        help="Input file name. Default is event000000000-cells.csv",
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