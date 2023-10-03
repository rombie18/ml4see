import pandas as pd
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

from config import DATA_STRUCTURED_DIRECTORY

RUN_NUMBER = 29
TRAN_NUMBER = 1

def moving_average(tran_data, time_data, downsample_factor, window_size):
    # Apply moving average filter
    window = np.ones(window_size) / window_size
    tran_data = np.convolve(tran_data, window, mode='valid')
    
    # Adjust time data to match length of convoluted output
    time_data = time_data[(len(window)-1):]
    
    # Downsample time and frequency data
    time_data = time_data[::downsample_factor]
    tran_data = tran_data[::downsample_factor]
    
    return tran_data, time_data

def exponential_decay(t, N, λ, c):
    return N * np.exp(-λ * t) + c

def double_exponential_decay(t, Nf, Ns, λf, λs, c):
        return Nf * np.exp(-λf * t) + Ns * np.exp(-λs * t) + c

h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{RUN_NUMBER:03d}.h5")
with h5py.File(h5_path, "r") as h5file:
    # Get run number and transients from file
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]
    transient_data = np.array(transients[f"tran_{TRAN_NUMBER:06d}"])

    
    # Get additional meta data
    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    # Calculate real time from meta data
    event_len = len_pretrig + len_posttrig - dsp_ntaps
    time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
    
    # Downsample data
    window_size = 500
    downsample_factor = 250
    tran_data_processed, time_data_processed = moving_average(transient_data, time_data, downsample_factor, window_size)
    
    # Try to fit exponential decay
    max_index = np.argmax(tran_data_processed)
    time_data_processed_exp = time_data_processed[max_index:]
    tran_data_processed_exp = tran_data_processed[max_index:]
    params, _ = curve_fit(
        exponential_decay, time_data_processed_exp, tran_data_processed_exp, p0=(35000, 10000, 212500), bounds=([0, 10, 0], [1e8, 6.5e5, 1e6])
    )
    params_double, _ = curve_fit(
        double_exponential_decay, time_data_processed_exp, tran_data_processed_exp, p0=(35000, 35000, 10000, 100, 20000), bounds=([0, 0, 1000, 0, 0], [1e8, 1e8, 1e5, 1e4, 1e6])
    )
        
    # Plot results
    figure, axis = plt.subplots(1, 1)
    
    axis.plot(time_data, transient_data, '.')
    axis.plot(time_data_processed, tran_data_processed, '.-')
    axis.plot(time_data_processed_exp, exponential_decay(time_data_processed_exp, *params), 'c-', linewidth=3)
    axis.plot(time_data_processed_exp, double_exponential_decay(time_data_processed_exp, *params_double), 'm-', linewidth=3)
    axis.set_title(f"Moving Average (tran_{TRAN_NUMBER:06d})")
    
    min = np.min(tran_data_processed)
    min = min / 1.05
    max = np.max(tran_data_processed)
    max = max * 1.05
    axis.set_ylim(min, max)
    
    # Visualize window size
    # for point in time_data[::window_size]:
    #     axis.axvline(x = point, color = 'gray', label = 'axvline - full height', zorder=-1)

    # Set figure size and save
    figure.set_size_inches(18.5, 10.5)
    plt.savefig(f"test_tran_{TRAN_NUMBER:06d}.png")