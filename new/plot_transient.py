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

    # Get run meta data
    scan_x_lsb_per_um = h5file["meta"].attrs["scan_x_lsb_per_um"]
    scan_y_lsb_per_um = h5file["meta"].attrs["scan_y_lsb_per_um"]

    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    sdr_cf = h5file["sdr_data"].attrs["sdr_info_cf"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]
    event_len = len_pretrig + len_posttrig - dsp_ntaps

    # Get additional transient meta data
    baseline_freq = transient.attrs["baseline_freq_mean_hz"]
    x_lsb = transient.attrs["x_lsb"]
    y_lsb = transient.attrs["y_lsb"]
    
    # Calculate true oscillator frequency for ppm error
    f0 = sdr_cf + baseline_freq

    # Construct time and frequency arrays, calculate the frequency error in ppm
    tran_time = (
        np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
    )
    tran_freq = (
        np.subtract(np.array(transient), baseline_freq) / f0 * 1e6
    )

    # Construct pre-trigger baseline arrays
    tran_pretrig_time = tran_time[: len_pretrig - PRETRIG_GUARD_SAMPLES]
    tran_pretrig_freq = tran_freq[: len_pretrig - PRETRIG_GUARD_SAMPLES]

    # Construct post-trigger arrays
    tran_posttrig_time = tran_time[len_pretrig:]
    tran_posttrig_freq = tran_freq[len_pretrig:]

    # Downsample data
    tran_pretrig_freq_ds, tran_pretrig_time_ds = moving_average(
        tran_pretrig_freq, tran_pretrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )
    tran_posttrig_freq_ds, tran_posttrig_time_ds = moving_average(
        tran_posttrig_freq, tran_posttrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
    )

    # Calculate features
    features = {}
    features["transient"] = tran_name
    features["x_lsb"] = x_lsb
    features["y_lsb"] = y_lsb
    features["x_um"] = x_lsb / scan_x_lsb_per_um
    features["y_um"] = y_lsb / scan_y_lsb_per_um
    features["trig_val"] = tran_posttrig_freq_ds[0]
    features["pretrig_min"] = np.min(tran_pretrig_freq_ds)
    features["posttrig_min"] = np.min(tran_posttrig_freq_ds)
    features["pretrig_max"] = np.max(tran_pretrig_freq_ds)
    features["posttrig_max"] = np.max(tran_posttrig_freq_ds)
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
            -np.inf,
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
            bounds=boundaries,
        )

        # Caluculate coefficient of determination (R²)
        residuals = tran_posttrig_freq_ds - exponential_decay(
            tran_posttrig_time_ds, *params
        )
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum(
            (tran_posttrig_freq_ds - np.mean(tran_posttrig_freq_ds)) ** 2
        )
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

    axis.plot(tran_time, tran_freq, ".")
    axis.plot(tran_pretrig_time_ds, tran_pretrig_freq_ds, color="orange", linestyle="-")
    axis.plot(tran_posttrig_time_ds, tran_posttrig_freq_ds, color="orange", linestyle="-")

    axis.axvline(
        x=tran_pretrig_time[-1],
        color="lime",
        linestyle="--",
    )

    try:
        axis.plot(
            tran_posttrig_time_ds,
            exponential_decay(tran_posttrig_time_ds, *params),
            color="cyan",
        )
        
        axis.axhline(
            y=tran_posttrig_freq_ds[0],
            color="red",
            linestyle=":",
        )
        
    except:
        pass

    axis.set_title(f"Moving Average ({tran_name})")

    min = np.min([np.min(tran_pretrig_freq_ds), np.min(tran_posttrig_freq_ds)])
    min = min * 1.1
    max = np.max([np.max(tran_pretrig_freq_ds), np.max(tran_posttrig_freq_ds)])
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
