import os
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt

from config import (
    DATA_STRUCTURED_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
)
from utils import generatePlotTitle, require_processing_stage

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
parser.add_argument("tran_number", type=int)
args = parser.parse_args()
run_number = args.run_number
tran_number = args.tran_number

# TODO cleanup code

h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
with h5py.File(h5_path, "r") as h5file:
    tran_name = f"tran_{tran_number:06d}"
    
    # Check if file is up to required processing stage
    require_processing_stage(h5file, 2, strict=True)

    # Get additional meta data
    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    # Calculate real time from meta data
    event_len = len_pretrig + len_posttrig - dsp_ntaps
    time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
    tran_data = np.array(h5file["sdr_data"]["all"][tran_name])

    # Subtract mean baseline frequency from each sample to get delta frequency
    baseline_freq = h5file["sdr_data"]["all"][tran_name].attrs["baseline_freq_mean_hz"]
    tran_data = np.subtract(tran_data, baseline_freq)
    
    # Apply moving average filter
    WINDOW_SIZE = 1000
    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    tran_data = np.convolve(tran_data, window, mode="valid")

    # Adjust time data to match length of convoluted output
    time_data = time_data[WINDOW_SIZE - 1 :]

    # Downsample time and frequency data
    time_data = time_data[::DOWNSAMPLE_FACTOR]
    tran_data = tran_data[::DOWNSAMPLE_FACTOR]
    
    # Calculate derivative
    time_data_dt = time_data[1:]
    tran_data_dt = np.diff(tran_data)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time_data, tran_data)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Delta frequency (Hz)")
    
    ax[1].plot(time_data_dt, tran_data_dt)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Delta frequency (Hz)")

    ax[0].set_title(f"Transient {tran_name} - Derivative")
    plt.savefig(f"plots/run_dt_{run_number:03d}_{tran_name}.png", bbox_inches="tight")