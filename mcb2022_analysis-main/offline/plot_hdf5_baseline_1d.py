"""Visualize baseline frequency for XY scans"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

import h5py
import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    util.require_processing_stage(h5file, 2)
    run_id = h5file["meta"].attrs["run_id"]
    scan_type = h5file["meta"].attrs["scan_type"]
    assert (
        scan_type == "X" or scan_type == "Y"
    ), "Provided file does not contain 1D (X or Y) scan data"

    if scan_type == "X":
        const_axis = "Y"
        top_group = "by_y"
        top_attr = "y_lsb"
        scan_attr = "x_lsb"
    if scan_type == "Y":
        const_axis = "X"
        top_group = "by_x"
        top_attr = "x_lsb"
        scan_attr = "y_lsb"

    sdr_data_by_sweep = list(h5file["sdr_data"][top_group].values())[0]
    const_coord = sdr_data_by_sweep.attrs[top_attr]

    scan_values = np.array(
        sorted(
            [scan_group.attrs[scan_attr] for scan_group in sdr_data_by_sweep.values()]
        ),
        dtype=int,
    )
    print(f"Scan dimension: {len(scan_values)} px")

    baseline_freqs = np.zeros(len(scan_values))
    baseline_stds = np.zeros(len(scan_values))
    baseline_ns = np.zeros(len(scan_values))

    outlier_threshold = 3.5

    for pos_group in sdr_data_by_sweep.values():
        pos_lsb = pos_group.attrs[scan_attr]
        print(f"Processing {scan_type} = {pos_lsb}...")
        scan_idx = np.argmax(scan_values == pos_lsb)
        transients = [
            tran
            for tran in pos_group.values()
            if tran.attrs["baseline_outlier_score"] < outlier_threshold
        ]
        baseline_freq = np.mean(
            [tran.attrs["baseline_freq_mean_hz"] for tran in transients]
        )
        baseline_std = np.mean(
            [tran.attrs["baseline_freq_std_hz"] for tran in transients]
        )
        baseline_freqs[scan_idx] = baseline_freq
        baseline_stds[scan_idx] = baseline_std
        baseline_ns[scan_idx] = len(transients)

    if scan_type == "X":
        scan_values_um = scan_values / h5file["meta"].attrs["scan_x_lsb_per_um"]
        const_coord_um = const_coord / h5file["meta"].attrs["scan_y_lsb_per_um"]
    if scan_type == "Y":
        scan_values_um = scan_values / h5file["meta"].attrs["scan_y_lsb_per_um"]
        const_coord_um = const_coord / h5file["meta"].attrs["scan_x_lsb_per_um"]

    plt.title(
        f"Mean baseline frequency.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        baseline_freqs,
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("Baseline frequency (Hz)")
    plt.grid()
    plt.show()

    plt.title(
        f"Mean baseline standard deviation.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        baseline_stds,
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("Baseline standard deviation (Hz)")
    plt.grid()
    plt.show()

    plt.title(
        f"Number of transients after outlier rejection.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        baseline_ns,
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("Number of transients")
    plt.grid()
    plt.show()
