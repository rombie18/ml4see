import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
from scipy.optimize import curve_fit

from config import DATA_STRUCTURED_DIRECTORY, WINDOW_SIZE, DOWNSAMPLE_FACTOR


def moving_average(tran_data, time_data, downsample_factor, window_size):
    # Apply moving average filter
    window = np.ones(window_size) / window_size
    tran_data = np.convolve(tran_data, window, mode="valid")

    # Adjust time data to match length of convoluted output
    time_data = time_data[(len(window) - 1) :]

    # Downsample time and frequency data
    time_data = time_data[::downsample_factor]
    tran_data = tran_data[::downsample_factor]

    return tran_data, time_data


def exponential_decay(t, N, λ, c):
    return (N - c) * np.exp(-λ * t) + c


# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
parser.add_argument("tran_number", type=int)
args = parser.parse_args()
run_number = args.run_number
tran_number = args.tran_number

h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
with h5py.File(h5_path, "r") as h5file:
    tran_name = f"tran_{tran_number:06d}"

    # Get run number and transients from file
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]
    tran_data = np.array(transients[f"tran_{tran_number:06d}"])

    # Get additional meta data
    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    # Calculate real time from meta data
    event_len = len_pretrig + len_posttrig - dsp_ntaps
    time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs

    # Subtract mean baseline frequency from each sample to get delta frequency
    baseline_freq = h5file["sdr_data"]["all"][tran_name].attrs["baseline_freq_mean_hz"]
    baseline_freq_var = h5file["sdr_data"]["all"][tran_name].attrs[
        "baseline_freq_mean_hz"
    ]
    tran_data = np.subtract(tran_data, baseline_freq)

    # Downsample data
    tran_data_processed, time_data_processed = moving_average(
        tran_data, time_data, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )

    # Try to fit exponential decay

    # TODO find better way to select zero index
    zero_timestamp = np.where(np.isclose(time_data_processed, 0, atol=1e-5))[0][0]
    time_data_processed_exp = time_data_processed[zero_timestamp:]
    tran_data_processed_exp = tran_data_processed[zero_timestamp:]

    print(f"std: {np.std(tran_data_processed[:zero_timestamp])}")

    try:
        # params: N, λ, c
        # model: (N - c) * np.exp(-λ * t) + c
        params, _ = curve_fit(
            exponential_decay,
            time_data_processed_exp,
            tran_data_processed_exp,
            p0=(
                np.max(tran_data_processed_exp),
                10000,
                np.mean(tran_data_processed[:zero_timestamp]),
            ),
            bounds=(
                [
                    np.max(tran_data_processed[:zero_timestamp])
                    + np.abs(np.max(tran_data_processed[:zero_timestamp])),
                    0,
                    -1e5,
                ],
                [1e6, 1e6, 1e5],
            ),
        )
        print(params)
    except:
        print("Exponential fit failed")

    # Plot results
    figure, axis = plt.subplots(1, 1)

    axis.plot(time_data, tran_data, ".")
    axis.plot(time_data_processed, tran_data_processed, ".-")

    try:
        axis.plot(
            time_data_processed_exp,
            exponential_decay(time_data_processed_exp, *params),
            "c-",
            linewidth=3,
        )
    except:
        pass

    axis.set_title(f"Moving Average ({tran_name})")

    min = np.min(tran_data_processed)
    min = min / 1.05
    max = np.max(tran_data_processed)
    max = max * 1.05
    axis.set_ylim(min, max)

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Delta frequency (Hz)")

    # Visualize window size
    # for point in time_data[::window_size]:
    #     axis.axvline(x = point, color = 'gray', label = 'axvline - full height', zorder=-1)

    # Set figure size and save
    figure.set_size_inches(13.5, 7.5)
    plt.savefig(f"plots/downsampling_run_{run_num:03d}_{tran_name}.png")
