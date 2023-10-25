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
    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    tran_data_ds = np.convolve(tran_data, window, mode="valid")

    # Adjust time data to match length of convoluted output
    time_data_ds = time_data[WINDOW_SIZE - 1 :]

    # Downsample time and frequency data
    time_data_ds = time_data_ds[::DOWNSAMPLE_FACTOR]
    tran_data_ds = tran_data_ds[::DOWNSAMPLE_FACTOR]
    
    fig, ax = plt.subplots()
    ## FFT
    # Calculate time intervals
    time_intervals = np.diff(time_data_ds)  # Differences between consecutive timestamps

    # Perform Fourier transformation
    fft_values = np.fft.fft(tran_data_ds)
    frequencies = np.fft.fftfreq(len(fft_values), d=time_intervals[0])  # Use the first time interval

    # Keep only positive frequencies (assuming real input)
    positive_frequencies = frequencies[:len(frequencies) // 2]
    positive_fft_values = 2.0 / len(fft_values) * np.abs(fft_values[:len(fft_values) // 2])

    # Plot the Fourier-transformed value vs. frequency
    ax.plot(positive_frequencies[25:], positive_fft_values[25:])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    generatePlotTitle(ax, f"Transient {tran_name} - FFT", run_number)
    plt.savefig(f"plots/run_fft_{run_number:03d}_{tran_name}.png", bbox_inches="tight")