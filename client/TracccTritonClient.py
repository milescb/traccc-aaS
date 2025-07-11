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
    output_names = [
        "TRK_PARAMS",      # [n_tracks, 5] - chi2, ndf, phi, theta, qop
        "MEASUREMENTS",    # [total_meas_with_seps, 4] - localx, localy, varx, vary with -1 separators
        "GEOMETRY_IDS"     # [total_meas_with_seps] - geometry IDs with -1 separators
    ]
    outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

    # Send inference request synchronously
    result = triton_client.infer(
        model_name="traccc-gpu",
        inputs=inputs,
        outputs=outputs
    )

    # Retrieve and process outputs
    trk_params = result.as_numpy("TRK_PARAMS")       # [n_tracks, 5]
    measurements = result.as_numpy("MEASUREMENTS")   # [total_meas_with_seps, 4]
    geometry_ids = result.as_numpy("GEOMETRY_IDS")   # [total_meas_with_seps]
    
    # Extract track parameters
    chi2 = trk_params[:, 0]
    ndf = trk_params[:, 1]
    phi = trk_params[:, 2]
    theta = trk_params[:, 3]
    qop = trk_params[:, 4]
    
    # Parse flattened measurements and geometry IDs into awkward arrays
    # Find separator indices (where all measurement values are -1)
    separator_mask = (measurements[:, 0] == -1) & (measurements[:, 1] == -1) & \
                     (measurements[:, 2] == -1) & (measurements[:, 3] == -1)
    
    # Get separator positions
    separator_indices = np.where(separator_mask)[0]
    
    # Split measurements into tracks using separators
    track_measurements = []
    track_geometry_ids = []
    
    start_idx = 0
    for sep_idx in separator_indices:
        # Extract measurements for this track
        track_meas = measurements[start_idx:sep_idx]
        track_geo = geometry_ids[start_idx:sep_idx]
        
        track_measurements.append(track_meas)
        track_geometry_ids.append(track_geo)
        
        start_idx = sep_idx + 1
    
    # Don't forget the last track (after the last separator)
    if start_idx < len(measurements):
        track_measurements.append(measurements[start_idx:])
        track_geometry_ids.append(geometry_ids[start_idx:])
    
    # Create awkward arrays for ragged data
    ak_measurements = ak.Array(track_measurements)  # [n_tracks][variable_meas, 4]
    ak_geometry_ids = ak.Array(track_geometry_ids)  # [n_tracks][variable_meas]
    
    # Extract components as awkward arrays
    ak_local_x = ak_measurements[:, :, 0]    # [n_tracks][variable_meas]
    ak_local_y = ak_measurements[:, :, 1]    # [n_tracks][variable_meas]
    ak_var_x = ak_measurements[:, :, 2]      # [n_tracks][variable_meas]
    ak_var_y = ak_measurements[:, :, 3]      # [n_tracks][variable_meas]
    
    # For printing measurement dimensions
    local_x = ak.flatten(ak_local_x)
    measurement_dims = np.full(len(local_x), 2, dtype=np.uint32)
    
    print(f"Number of tracks: {len(trk_params)}")
    print(f"Number of measurements per track: {ak.num(ak_measurements, axis=1)}")
    print(f"Total measurements: {ak.sum(ak.num(ak_measurements, axis=1))}")
    print(f"Track parameters shape: {trk_params.shape}")
    print(f"Awkward measurements shape: {ak_measurements.type}")
    print(f"Awkward geometry IDs shape: {ak_geometry_ids.type}")
    
    # Example: Access measurements for specific tracks
    print(f"\nTrack 0 has {len(ak_measurements[0])} measurements:")
    print(f"  Local positions: {ak_measurements[0][:, :2]}")
    print(f"  Geometry IDs: {ak_geometry_ids[0]}")
    
    if len(ak_measurements) > 1:
        print(f"\nTrack 1 has {len(ak_measurements[1])} measurements:")
        print(f"  Local positions: {ak_measurements[1][:, :2]}")
        print(f"  Geometry IDs: {ak_geometry_ids[1]}")
    
    # Plot histograms using flattened data
    plot_histogram(chi2, "Chi2", "Chi2")
    plot_histogram(ndf, "NDF", "NDF")
    plot_histogram(phi, "Phi", "Phi [rad]")
    plot_histogram(theta, "Theta", "Theta [rad]")
    plot_histogram(qop, "Q_over_P", "Charge/Momentum [1/GeV]")
    plot_histogram(measurement_dims, "Measurement Dims", "Measurement Dimensions")
    
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