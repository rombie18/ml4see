"""Visualize baseline frequency for XY scans"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

import h5py
import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--writenpy", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    util.require_processing_stage(h5file, 2)
    run_id = h5file["meta"].attrs["run_id"]
    scan_type = h5file["meta"].attrs["scan_type"]
    sdr_cf = h5file["sdr_data"].attrs["sdr_info_cf"]
    assert scan_type == "S", "Provided file does not contain stationary scan data"

    transients = h5file["sdr_data"]["all"]

    outlier_threshold = 3.5
    transients = [
        tran
        for tran in transients.values()
        if tran.attrs["baseline_outlier_score"] < outlier_threshold
    ]
    tran_nums = np.array(
        [transient.attrs["tran_num"] for transient in transients],
        dtype=int,
    )
    baseline_freqs = np.array(
        [tran.attrs["baseline_freq_mean_hz"] for tran in transients]
    )
    baseline_freq_stds = np.array(
        [tran.attrs["baseline_freq_mean_std_hz"] for tran in transients]
    )
    seft_exp_n0s = np.array([tran.attrs["seft_exp_n0"] for tran in transients])
    seft_exp_lambdas = np.array([tran.attrs["seft_exp_lambda"] for tran in transients])
    seft_flattop_dfs = np.array(
        [tran.attrs["seft_flattop_mean"] for tran in transients]
    )
    seft_flattop_stds = np.array(
        [tran.attrs["seft_flattop_mean_std"] for tran in transients]
    )

    f0 = sdr_cf + baseline_freqs

    seft_flattop_dfs_ppm = (seft_flattop_dfs - baseline_freqs) / f0 * 1e6
    seft_flattop_df_stds_ppm = (
        np.sqrt(baseline_freq_stds**2 + seft_flattop_stds**2) / f0 * 1e6
    )
    seft_exp_n0s_ppm = seft_exp_n0s / f0 * 1e6

    sort_idx = np.argsort(tran_nums)
    tran_nums = np.take_along_axis(tran_nums, sort_idx, axis=0)
    seft_flattop_dfs_ppm = np.take_along_axis(seft_flattop_dfs_ppm, sort_idx, axis=0)
    seft_flattop_df_stds_ppm = np.take_along_axis(
        seft_flattop_df_stds_ppm, sort_idx, axis=0
    )
    seft_exp_n0s_ppm = np.take_along_axis(seft_exp_n0s_ppm, sort_idx, axis=0)
    seft_exp_lambdas = np.take_along_axis(seft_exp_lambdas, sort_idx, axis=0)

    if args.writenpy is not None:
        np.save(
            args.writenpy,
            np.stack((tran_nums, seft_flattop_dfs_ppm, seft_exp_n0s_ppm, np.log(2) / seft_exp_lambdas))
        )

    plt.title(f"SEFT peak deviation.\nRun {run_id}.")
    plt.plot(tran_nums, seft_flattop_dfs_ppm, ".")
    plt.xlabel(f"Transient Number")
    plt.ylabel("SEFT peak deviation (ppm)")
    plt.grid()
    plt.show()

    plt.title(f"SEFT exponential fit $N_0$.\nRun {run_id}.")
    plt.plot(tran_nums, seft_exp_n0s_ppm, ".")
    plt.xlabel(f"Transient Number")
    plt.ylabel("SEFT exponential fit $N_0$ (ppm)")
    plt.grid()
    plt.show()

    plt.title(f"SEFT exponential fit $t_{{1/2}}$.\nRun {run_id}.")
    plt.plot(tran_nums, np.log(2) / seft_exp_lambdas, ".")
    plt.xlabel(f"Transient Number")
    plt.ylabel("SEFT exponential fit $t_{{1/2}}$ (s)")
    plt.grid()
    plt.show()
