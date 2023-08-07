"""Plot SDR transients from an HDF5 file in batches"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import h5py

import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--downsample", type=int, default=1)
parser.add_argument("--min-outlier-score", type=float)
parser.add_argument("--max-outlier-score", type=float)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]

    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    event_len = len_pretrig + len_posttrig - dsp_ntaps
    t = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs

    t = t[:: args.downsample]

    batch_len = 10
    batch_num = 0
    batch_count = 0

    if args.min_outlier_score is not None:
        util.require_processing_stage(h5file, 2)
    if args.max_outlier_score is not None:
        util.require_processing_stage(h5file, 2)

    for dsref in transients.values():
        if args.min_outlier_score is not None:
            if dsref.attrs["baseline_outlier_score"] < args.min_outlier_score:
                continue
        if args.max_outlier_score is not None:
            if dsref.attrs["baseline_outlier_score"] > args.max_outlier_score:
                continue
        data = np.array(dsref)[:: args.downsample]
        tran_num = dsref.attrs["tran_num"]
        plt.plot(t, np.array(data), label=f"Transient {tran_num}")

        batch_count += 1
        if batch_count == batch_len:
            batch_num += 1
            batch_count = 0
            plt.title(f"Run ID {run_num}; Batch {batch_num}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.legend()
            plt.grid()
            plt.show()
