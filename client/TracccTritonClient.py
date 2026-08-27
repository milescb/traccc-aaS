import tritonclient.grpc as grpcclient

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import awkward as ak

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ROOT)

def plot_histogram(data, name, xlabel, bins=50, xlims=None, logy=False):
    """Plots a histogram for the given data on a new canvas."""
    if data is None or data.size == 0:
        print(f"No data to plot for {name}")
        return
    plt.figure(figsize=(8, 8))
    plt.hist(data, bins=bins, histtype='step', label=name)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.tight_layout()
    if xlims is not None:
        plt.xlim(xlims)
    if logy:
        plt.yscale('log')
    plt.savefig(f"plots/{name.replace(" ", "_")}.png")

def load_athena_to_detray_map(path):
    """Parse the same "<hex athena id>,<decimal detray id>" file the server
    loads (read_athena_to_detray_mapping in TracccEdmConversion.hpp)."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hex_athena, detray = line.split(",")
            mapping[int(hex_athena, 16)] = int(detray)
    return mapping

def load_geom_id_map(path):
    """Load the geometry_id (detray) -> module_index map dumped once by the
    server (dump_geom_id_map in TracccGpuStandalone.cpp). This ordering comes
    from traccc/detray's detector construction and can't be recomputed
    client-side, so it's shipped as a static file instead."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def build_cell_buffer(input_data, athena_to_detray, geom_id_map):
    """Pack CSV cells into the raw byte layout expected by
    TracccGpuStandalone::cells_from_buffer: an 8-byte cell-count header
    followed by 5 SoA column blocks of length N (channel0, channel1,
    activation, time, module_index)."""
    input_data = input_data[input_data["geometry_id"] != 0]

    detray_ids = np.array(
        [athena_to_detray[int(gid)] for gid in input_data["geometry_id"]],
        dtype=np.uint64,
    )
    module_index = np.array(
        [geom_id_map[int(did)] for did in detray_ids], dtype=np.uint32
    )
    channel0 = input_data["channel0"].to_numpy(dtype=np.uint32)
    channel1 = input_data["channel1"].to_numpy(dtype=np.uint32)
    activation = input_data["value"].to_numpy(dtype=np.float32)
    time = input_data["timestamp"].to_numpy(dtype=np.float32)

    # Sort cells per-module by channel1/channel0, mirroring the ordering the
    # clusterization algorithm expects. Sorting happens on the client side.
    order = np.lexsort((channel0, channel1, module_index))

    num_cells = np.uint64(len(channel0))
    buffer = (
        num_cells.tobytes()
        + channel0[order].tobytes()
        + channel1[order].tobytes()
        + activation[order].tobytes()
        + time[order].tobytes()
        + module_index[order].tobytes()
    )
    return np.frombuffer(buffer, dtype=np.uint8)

# traccc::track_fit_outcome::SUCCESS
TRACK_FIT_OUTCOME_SUCCESS = 1

# detray::bound_track_parameters as it sits in memory: the 6-element bound
# vector (loc0, loc1, phi, theta, qop, time), the 6x6 covariance, then the
# surface identifier. 176 bytes with a float scalar.
BOUND_PARAMS_DTYPE = np.dtype(
    {
        "names": ["vector", "covariance", "surface_link"],
        "formats": [(np.float32, 6), (np.float32, (6, 6)), np.uint64],
        "offsets": [0, 24, 168],
        "itemsize": 176,
    }
)

# struct track_constituent_link { unsigned short type; unsigned int index; }
LINK_DTYPE = np.dtype(
    {
        "names": ["type", "index"],
        "formats": [np.uint16, np.uint32],
        "offsets": [0, 4],
        "itemsize": 8,
    }
)

SCALAR = np.float32
SURFACE_LINK = np.uint64

# The columns of the three traccc containers, in the order
# visit_track_columns writes them (TracccGpuStandalone.hpp). Each entry is
# (container, name, dtype).
TRACKS_COLUMNS = [
    ("measurements", "local_position", (np.float32, 2)),
    ("measurements", "local_variance", (np.float32, 2)),
    ("measurements", "dimensions", np.uint32),
    ("measurements", "time", np.float32),
    ("measurements", "diameter", np.float32),
    ("measurements", "identifier", np.uint32),
    ("measurements", "surface_link", SURFACE_LINK),
    ("measurements", "subspace", (np.uint8, 2)),
    ("measurements", "cluster_index", np.uint32),
    ("tracks", "fit_outcome", np.uint16),
    ("tracks", "params", BOUND_PARAMS_DTYPE),
    ("tracks", "ndf", SCALAR),
    ("tracks", "chi2", SCALAR),
    ("tracks", "pval", SCALAR),
    ("tracks", "nholes", np.uint32),
    ("states", "state", np.uint8),
    ("states", "filtered_chi2", SCALAR),
    ("states", "smoothed_chi2", SCALAR),
    ("states", "backward_chi2", SCALAR),
    ("states", "filtered_params", BOUND_PARAMS_DTYPE),
    ("states", "smoothed_params", BOUND_PARAMS_DTYPE),
    ("states", "measurement_index", np.uint32),
]

def parse_tracks_buffer(buf):
    """Unpack the raw byte layout produced by
    TracccGpuStandalone::tracks_to_buffer, which is the traccc SoA containers
    dumped column by column: four uint64 element counts, then every column back
    to back, then the flattened jagged constituent_links.

    A client that links traccc reads this straight back into the containers (see
    tracks_from_buffer in TracccGpuStandalone.cpp). This is the numpy
    equivalent, so plotting works without a traccc build. Since the counts fix
    the total length, the check at the end catches any layout drift between the
    server's traccc build and the sizes assumed above.
    """
    raw = buf.tobytes()

    if len(raw) < 32:
        raise ValueError("TRACKS buffer is too small to contain the counts")

    n_meas, n_tracks, n_states, n_links = (
        int(v) for v in np.frombuffer(raw, dtype=np.uint64, count=4)
    )
    counts = {
        "measurements": n_meas,
        "tracks": n_tracks,
        "states": n_states,
    }

    offset = 32

    def take(dtype, count):
        """Read one column."""
        nonlocal offset
        dtype = np.dtype(dtype)
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(raw):
            raise ValueError(
                f"TRACKS buffer is truncated: needed {nbytes} more bytes at "
                f"offset {offset} of {len(raw)}"
            )
        column = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
        offset += nbytes
        return column

    out = {"measurements": {}, "tracks": {}, "states": {}}
    for container, name, dtype in TRACKS_COLUMNS:
        out[container][name] = take(dtype, counts[container])

    # Re-inflate the jagged constituent_links column.
    link_counts = take(np.uint32, n_tracks)
    links = take(LINK_DTYPE, n_links)
    out["tracks"]["constituent_links"] = np.split(
        links, np.cumsum(link_counts)[:-1]
    )

    if offset != len(raw):
        raise ValueError(
            f"TRACKS buffer size mismatch: consumed {offset} of {len(raw)} "
            f"bytes; the server's traccc build does not match the column sizes "
            f"assumed by this parser"
        )

    return out

def print_inference_statistics(triton_client, model_name):
    """Fetch and print Triton's per-model statistics (queue/compute time,
    request counts) via the gRPC client's built-in stats API."""
    stats = triton_client.get_inference_statistics(model_name=model_name, as_json=True)
    for model_stat in stats.get("model_stats", []):
        inference_stats = model_stat.get("inference_stats", {})
        print(f"\n=== Triton stats for '{model_stat['name']}' (version {model_stat['version']}) ===")
        print(f"Inference count:  {model_stat.get('inference_count', 0)}")
        print(f"Execution count:  {model_stat.get('execution_count', 0)}")
        for field in ("success", "fail", "queue", "compute_input", "compute_infer", "compute_output"):
            duration = inference_stats.get(field, {})
            count = int(duration.get("count", 0))
            ns = int(duration.get("ns", 0))
            avg_ms = (ns / count / 1e6) if count else 0.0
            print(f"  {field:<15} count={count:<8} total={ns/1e6:.3f} ms  avg={avg_ms:.3f} ms")
    print()

def main():
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, ssl=FLAGS.ssl
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    input_data = pd.read_csv(FLAGS.filename)

    athena_to_detray = load_athena_to_detray_map(FLAGS.athena_map)

    geom_map_path = FLAGS.geom_map
    if geom_map_path is None:
        geom_map_path = os.path.join(
            os.path.dirname(os.path.abspath(FLAGS.filename)),
            "geom_id_to_module_index.json",
        )
    geom_id_map = load_geom_id_map(geom_map_path)

    cell_buffer = build_cell_buffer(input_data, athena_to_detray, geom_id_map)

    inputs = [
        grpcclient.InferInput("CELLS", cell_buffer.shape, "UINT8")
    ]
    inputs[0].set_data_from_numpy(cell_buffer)

    # Single raw output buffer, unpacked by parse_tracks_buffer
    outputs = [grpcclient.InferRequestedOutput("TRACKS")]

    model_name = "traccc-gpu"

    # Send inference request synchronously
    result = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    if FLAGS.print_stats:
        print_inference_statistics(triton_client, model_name)

    # Retrieve and process outputs
    decoded = parse_tracks_buffer(result.as_numpy("TRACKS"))
    tracks = decoded["tracks"]
    measurements = decoded["measurements"]

    # The server ships the containers unfiltered, so the track quality cuts are
    # the client's to make.
    n_links = np.array([len(links) for links in tracks["constituent_links"]])
    accepted = (
        (tracks["fit_outcome"] == TRACK_FIT_OUTCOME_SUCCESS)
        & (tracks["ndf"] >= 0)
        & (n_links >= 3)
    )

    print(
        f"Recieved {len(accepted)} tracks "
        f"({int(accepted.sum())} passing the quality cuts), "
        f"{len(measurements['surface_link'])} measurements. "
        f"Plotting parameters..."
    )

    # The bound vector is (loc0, loc1, phi, theta, qop, time).
    bound = tracks["params"]["vector"][accepted]
    chi2 = tracks["chi2"][accepted]
    ndf = tracks["ndf"][accepted]
    l0, l1, phi, theta, qop = (bound[:, i] for i in range(5))

    # Measurement surface links are raw detray IDs. A consumer wanting Athena
    # identifiers inverts the map it already loaded:
    #   {v: k for k, v in athena_to_detray.items()}

    plot_histogram(chi2, "Chi2", "Chi2", logy=True)
    plot_histogram(ndf, "NDF", "NDF")
    plot_histogram(l0, "L0", "L0")
    plot_histogram(l1, "L1", "L1")
    plot_histogram(phi, "Phi", "Phi (radians)")
    plot_histogram(theta, "Theta", "Theta (radians)")
    plot_histogram(qop, "Qop", "Q/P (1/GeV)")
    
    print("Inference complete!")

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
    parser.add_argument(
        "--athena-map",
        type=str,
        required=False,
        default="/traccc/itk-geometry/athenaIdentifierToDetrayMap.txt",
        help="Path to the athena->detray geometry ID mapping file used by "
             "the server. Default is /traccc/itk-geometry/"
             "athenaIdentifierToDetrayMap.txt.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        required=False,
        default=False,
        help="Print Triton's per-model inference statistics (queue/compute "
             "time, request counts) after the inference call.",
    )
    parser.add_argument(
        "--geom-map",
        type=str,
        required=False,
        default=None,
        help="Path to the geom_id_to_module_index.json file dumped by the "
             "server (TracccGpuStandalone::dump_geom_id_map). Default is "
             "geom_id_to_module_index.json next to --filename.",
    )
    FLAGS = parser.parse_args()

    main()