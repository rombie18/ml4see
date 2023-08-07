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
    sdr_cf = h5file["sdr_data"].attrs["sdr_info_cf"]
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

    seft_flattop_dfs_ppm = np.zeros(len(scan_values))
    seft_flattop_df_stds_ppm = np.zeros(len(scan_values))
    seft_exp_n0s_ppm = np.zeros(len(scan_values))
    seft_exp_lambdas = np.zeros(len(scan_values))

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
        baseline_freq_std = np.mean(
            [tran.attrs["baseline_freq_mean_std_hz"] for tran in transients]
        ) / np.sqrt(len(transients))
        seft_exp_n0 = np.mean([tran.attrs["seft_exp_n0"] for tran in transients])
        seft_exp_lambda = np.mean(
            [tran.attrs["seft_exp_lambda"] for tran in transients]
        )
        seft_flattop_mean = np.mean(
            [tran.attrs["seft_flattop_mean"] for tran in transients]
        )
        seft_flattop_std = np.mean(
            [tran.attrs["seft_flattop_mean_std"] for tran in transients]
        ) / np.sqrt(len(transients))

        f0 = sdr_cf + baseline_freq

        seft_flattop_dfs_ppm[scan_idx] = (seft_flattop_mean - baseline_freq) / f0 * 1e6
        seft_flattop_df_stds_ppm[scan_idx] = (
            np.sqrt(baseline_freq_std**2 + seft_flattop_std**2) / f0 * 1e6
        )
        seft_exp_n0s_ppm[scan_idx] = seft_exp_n0 / f0 * 1e6
        seft_exp_lambdas[scan_idx] = seft_exp_lambda

    if scan_type == "X":
        scan_values_um = scan_values / h5file["meta"].attrs["scan_x_lsb_per_um"]
        const_coord_um = const_coord / h5file["meta"].attrs["scan_y_lsb_per_um"]
    if scan_type == "Y":
        scan_values_um = scan_values / h5file["meta"].attrs["scan_y_lsb_per_um"]
        const_coord_um = const_coord / h5file["meta"].attrs["scan_x_lsb_per_um"]

    plt.title(
        f"SEFT peak frequency.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        seft_flattop_dfs_ppm,
        ".",
    )
    plt.fill_between(
        scan_values_um,
        seft_flattop_dfs_ppm - 2 * seft_flattop_df_stds_ppm,
        seft_flattop_dfs_ppm + 2 * seft_flattop_df_stds_ppm,
        alpha=0.3,
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("SEFT peak deviation (ppm)")
    plt.grid()
    plt.show()

    plt.title(
        f"SEFT exponential fit $N_0$.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        seft_exp_n0s_ppm,
        ".",
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("SEFT exponential fit $N_0$ (ppm)")
    plt.grid()
    plt.show()

    plt.title(
        f"SEFT exponential fit $t_{1/2}$.\nRun {run_id}; {const_axis} = {const_coord_um:.02f} µm."
    )
    plt.plot(
        scan_values_um,
        np.log(2) / seft_exp_lambdas,
        ".",
    )
    plt.xlabel(f"{scan_type} position (µm)")
    plt.ylabel("SEFT $t_{1/2}$ (s)")
    plt.grid()
    plt.show()
