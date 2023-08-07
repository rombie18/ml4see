"""Plot histograms from timestamp differences"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    run_num = h5file["meta"].attrs["run_id"]
    fpga_data = h5file["fpga_hit_data"]
    transients = h5file["sdr_data"]["all"]

    fpga_hw_ts = fpga_data["hw_ts_sec"]
    fpga_hw_ts_deltas_ms = np.diff(fpga_hw_ts) * 1e3

    sdr_hw_ts = []

    for dsref in transients.values():
        sdr_hw_ts.append(dsref.attrs["hw_ts_sec"])

    sdr_hw_ts_deltas_ms = np.diff(np.array(sdr_hw_ts)) * 1e3

    BIN_SIZE_MS = 1
    max_delta_ms = max(np.max(sdr_hw_ts_deltas_ms), np.max(fpga_hw_ts_deltas_ms))
    num_bins = int(max_delta_ms / BIN_SIZE_MS)
    hist_bins = np.linspace(0, max_delta_ms, num_bins)

    plt.title(
        f"Event separation histogram\nRun {run_num}, N(SDR) = {len(sdr_hw_ts)}, N(FPGA) = {len(fpga_hw_ts)}, bin size {BIN_SIZE_MS} ms"
    )
    plt.hist(sdr_hw_ts_deltas_ms, bins=hist_bins, label="SDR timestamps")
    plt.hist(fpga_hw_ts_deltas_ms, bins=hist_bins, label="FPGA timestamps", alpha=0.75)
    plt.yscale("log")
    plt.xlim([0, max_delta_ms * 1.05])
    plt.xlabel("Time between adjacent hits (ms)")
    plt.ylabel("Population")
    plt.grid()
    plt.legend()
    plt.show()
