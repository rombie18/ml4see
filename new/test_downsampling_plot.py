import pandas as pd
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import signal

DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"
RUN_NUMBER = 18
TRAN_NUMBER = 112

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
    
    # Plot results
    figure, axis = plt.subplots(1, 1)
    
    axis.plot(time_data, transient_data, '.')
    axis.plot(time_data_processed, tran_data_processed, '.-')
    axis.set_title(f"Moving Average (tran_{TRAN_NUMBER:06d})")
    
    # Visualize window size
    # for point in time_data[::window_size]:
    #     axis.axvline(x = point, color = 'gray', label = 'axvline - full height', zorder=-1)

    # Set figure size and save
    figure.set_size_inches(18.5, 10.5)
    plt.savefig("test.png")