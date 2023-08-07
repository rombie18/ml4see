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
        [transient.attrs["baseline_freq_mean_hz"] for transient in transients]
    )
    baseline_stds = np.array(
        [transient.attrs["baseline_freq_std_hz"] for transient in transients]
    )

    sort_idx = np.argsort(tran_nums)
    tran_nums = np.take_along_axis(tran_nums, sort_idx, axis=0)
    baseline_freqs = np.take_along_axis(baseline_freqs, sort_idx, axis=0)
    baseline_stds = np.take_along_axis(baseline_stds, sort_idx, axis=0)

    plt.title(f"Baseline frequency.\nRun {run_id}.")
    plt.plot(
        tran_nums,
        baseline_freqs,
    )
    plt.xlabel(f"Transient Number")
    plt.ylabel("Baseline frequency (Hz)")
    plt.grid()
    plt.show()

    plt.title(f"Baseline standard deviation.\nRun {run_id}.")
    plt.plot(
        tran_nums,
        baseline_stds,
    )
    plt.xlabel(f"Transient Number")
    plt.ylabel("Baseline standard deviation (Hz)")
    plt.grid()
    plt.show()
