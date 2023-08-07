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
    assert scan_type == "XY", "Provided file does not contain XY scan data"

    sdr_data_by_x = h5file["sdr_data"]["by_x"]
    x_values = np.array(
        sorted([x_group.attrs["x_lsb"] for x_group in sdr_data_by_x.values()]),
        dtype=int,
    )
    y_values = np.array(
        sorted(
            [
                y_group.attrs["y_lsb"]
                for y_group in list(sdr_data_by_x.values())[0].values()
            ]
        ),
        dtype=int,
    )
    print(f"Scan dimensions: {len(x_values)}x{len(y_values)} px")

    # convention for 2D arrays: indexed as (row,col), referring to (Y,X)
    # with imshow(origin="lower") convention, this keeps X/Y axes oriented correctly

    seft_flattop_dfs_ppm = np.zeros((len(y_values), len(x_values)))
    seft_flattop_df_stds_ppm = np.zeros((len(y_values), len(x_values)))
    seft_exp_n0s_ppm = np.zeros((len(y_values), len(x_values)))
    seft_exp_lambdas = np.zeros((len(y_values), len(x_values)))

    outlier_threshold = 3.5

    for x_group in sdr_data_by_x.values():
        x_lsb = x_group.attrs["x_lsb"]
        print(f"Processing X = {x_lsb}...")
        col_idx = np.argmax(x_values == x_lsb)
        for y_group in x_group.values():
            y_lsb = y_group.attrs["y_lsb"]
            row_idx = np.argmax(y_values == y_lsb)
            transients = [
                tran
                for tran in y_group.values()
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

            seft_flattop_dfs_ppm[row_idx, col_idx] = (
                (seft_flattop_mean - baseline_freq) / f0 * 1e6
            )
            seft_flattop_df_stds_ppm[row_idx, col_idx] = (
                np.sqrt(baseline_freq_std**2 + seft_flattop_std**2) / f0 * 1e6
            )
            seft_exp_n0s_ppm[row_idx, col_idx] = seft_exp_n0 / f0 * 1e6
            seft_exp_lambdas[row_idx, col_idx] = seft_exp_lambda

    x_values_um = x_values / h5file["meta"].attrs["scan_x_lsb_per_um"]
    y_values_um = y_values / h5file["meta"].attrs["scan_y_lsb_per_um"]
    im_extent = (
        np.min(x_values_um),
        np.max(x_values_um),
        np.min(y_values_um),
        np.max(y_values_um),
    )

    plt.title(f"SEFT peak frequency (ppm).\nRun {run_id}.")
    pos = plt.imshow(
        seft_flattop_dfs_ppm,
        origin="lower",
        extent=im_extent,
        cmap="hot",
        interpolation="nearest",
    )
    cbar = plt.colorbar(pos)
    plt.xlabel("X position (µm)")
    plt.ylabel("Y position (µm)")
    cbar.set_label("SEFT peak frequency (ppm)")
    plt.show()

    plt.title(f"SEFT exponential fit $N_0$ (ppm).\nRun {run_id}.")
    pos = plt.imshow(
        seft_exp_n0s_ppm,
        origin="lower",
        extent=im_extent,
        cmap="hot",
        interpolation="nearest",
    )
    cbar = plt.colorbar(pos)
    plt.xlabel("X position (µm)")
    plt.ylabel("Y position (µm)")
    cbar.set_label("SEFT exponential fit $N_0$ (ppm)")
    plt.show()

    plt.title(f"SEFT exponential fit $t_{{1/2}}$ (s).\nRun {run_id}.")
    pos = plt.imshow(
        np.log(2) / seft_exp_lambdas,
        origin="lower",
        extent=im_extent,
        cmap="hot",
        interpolation="nearest",
    )
    cbar = plt.colorbar(pos)
    plt.xlabel("X position (µm)")
    plt.ylabel("Y position (µm)")
    cbar.set_label("SEFT exponential fit $t_{{1/2}}$ (s)")
    plt.show()
