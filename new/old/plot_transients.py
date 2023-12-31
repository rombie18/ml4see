import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
from scipy.optimize import curve_fit

from config import (
    DATA_STRUCTURED_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
    PRETRIG_GUARD_SAMPLES,
    R2_THRESHOLD
)
from utils import moving_average, exponential_decay

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
parser.add_argument("tran_number", type=int)
args = parser.parse_args()
run_number = args.run_number
tran_number = args.tran_number

h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
with h5py.File(h5_path, "r") as h5file:
    run_name = f"run_{run_number:03d}"
    tran_name = f"tran_{tran_number:06d}"

    # Get transient data from file
    transient = h5file["sdr_data"]["all"][tran_name]

    # Get additional meta data
    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]
    event_len = len_pretrig + len_posttrig - dsp_ntaps

    # Subtract mean baseline frequency from each sample to get delta frequency
    baseline_freq = h5file["sdr_data"]["all"][tran_name].attrs["baseline_freq_mean_hz"]
    baseline_freq_var = h5file["sdr_data"]["all"][tran_name].attrs[
        "baseline_freq_mean_hz"
    ]

    # Construct time and frequency arrays
    tran_time = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
    tran_freq = np.subtract(np.array(transient), baseline_freq) / 2.5e9 * 1e6
    # tran_freq = np.array(transient)

    # Construct pre-trigger baseline arrays
    tran_pretrig_time = tran_time[: len_pretrig - PRETRIG_GUARD_SAMPLES]
    tran_pretrig_freq = tran_freq[: len_pretrig - PRETRIG_GUARD_SAMPLES]

    # Construct post-trigger arrays
    tran_posttrig_time = tran_time[len_pretrig:]
    tran_posttrig_freq = tran_freq[len_pretrig:]

    # Downsample data
    tran_freq_ds, tran_time_ds = moving_average(
        tran_freq, tran_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )
    tran_pretrig_freq_ds, tran_pretrig_time_ds = moving_average(
        tran_pretrig_freq, tran_pretrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )
    tran_posttrig_freq_ds, tran_posttrig_time_ds = moving_average(
        tran_posttrig_freq, tran_posttrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )

    # Calculate features
    features = {}
    features["transient"] = tran_name
    features["pretrig_std"] = np.std(tran_pretrig_freq_ds)
    features["posttrig_std"] = np.std(tran_posttrig_freq_ds)

    # Try to fit exponential decay
    initial_guess = (
        np.max(tran_posttrig_freq_ds),
        1000,
        np.mean(tran_pretrig_freq_ds),
    )
    boundaries = (
        [
            0,
            0,
            -np.inf,
        ],
        [np.inf, np.inf, np.inf],
    )

    try:
        # params: N, λ, c
        # model: (N - c) * np.exp(-λ * t) + c
        params, _ = curve_fit(
            exponential_decay,
            tran_posttrig_time_ds,
            tran_posttrig_freq_ds,
            p0=initial_guess,
            bounds=boundaries
        )

        # Caluculate coefficient of determination (R²)
        residuals = tran_posttrig_freq_ds - exponential_decay(
            tran_posttrig_time_ds, *params
        )
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tran_posttrig_freq_ds - np.mean(tran_posttrig_freq_ds)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        features["posttrig_exp_fit_R2"] = r_squared

        if r_squared > R2_THRESHOLD:
            # Assign fitted parameters and R² to resulting feature set
            features["posttrig_exp_fit_N"] = params[0]
            features["posttrig_exp_fit_λ"] = params[1]
            features["posttrig_exp_fit_c"] = params[2]
        else:
            # Insufficient fit, set parameters to zero
            features["posttrig_exp_fit_N"] = 0
            features["posttrig_exp_fit_λ"] = 0
            features["posttrig_exp_fit_c"] = 0
            params = None

    except:
        # If exponential fit fails, assign parameters to zero
        features["posttrig_exp_fit_N"] = 0
        features["posttrig_exp_fit_λ"] = 0
        features["posttrig_exp_fit_c"] = 0
        features["posttrig_exp_fit_R2"] = 0

    print(features)

    # Plot results
    figure, axis = plt.subplots(1, 1)

    axis.plot(tran_time_ds, tran_freq_ds, ".-")

    axis.axvline(
        x=tran_pretrig_time[-1],
        color="lime",
        linestyle="--",
    )

    try:
        axis.plot(
            tran_posttrig_time_ds,
            exponential_decay(tran_posttrig_time_ds, *params),
            "c-",
            linewidth=3,
        )
    except:
        pass

    axis.set_title(f"Moving Average ({tran_name})")

    min = np.min(tran_freq_ds)
    min = min * 0.9
    max = np.max(tran_freq_ds)
    max = max * 1.1
    axis.set_ylim(min, max)

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Delta frequency (Hz)")

    # Visualize window size
    # for point in time_data[::window_size]:
    #     axis.axvline(x = point, color = 'gray', label = 'axvline - full height', zorder=-1)

    # Set figure size and save
    figure.set_size_inches(13.5, 7.5)
    plt.savefig(f"plots/tran_{run_name}_{tran_name}.png")
