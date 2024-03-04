import multiprocessing
import os
import logging
import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import math
from functools import partial

from config import DATA_SYNTHETIC_DIRECTORY
from utils import exponential_decay, chunker

META_PROCESSING_STAGE = 1
META_STAGE_1_VERSION = "2.0"

TIME_START = -1e-3
TIME_STOP = 10e-3

SCAN_X_START = -180
SCAN_X_STOP = 180
SCAN_X_STEPS = 181
SCAN_Y_START = -180
SCAN_Y_STOP = 180
SCAN_Y_STEPS = 181
SCAN_HITS_PER_STEP = 1
SCAN_X_LSB_PER_UM = 99
SCAN_Y_LSB_PER_UM = 93.46

SAMPLING_FREQUENCY = 20e6
PRETRIG_SAMPLES = 20000
POSTTRIG_SAMPLES = 200000


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("metrics_augmentation.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting augmentation process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_number",
        type=int,
        metavar="ID_NUMBER",
        help="The identification number for this synthetically generated run",
    )
    parser.add_argument(
        "--amplitude",
        default=10,
        type=int,
        metavar="VALUE",
        help="The base peak value amplitude at the trigger of the exponential decay. (default: %(default)s ppm)",
    )
    parser.add_argument(
        "--decay",
        default=1000,
        type=int,
        metavar="VALUE",
        help="The base exponential decay constant. (default: %(default)s 1/s)",
    )
    parser.add_argument(
        "--offset",
        default=0,
        type=int,
        metavar="VALUE",
        help="The base asymptotic y-axis offset to which the decay evolves. (default: %(default)s ppm)",
    )
    parser.add_argument(
        "--noise_std",
        default=10,
        type=float,
        metavar="VALUE",
        help="The standard deviation of the added white noise. (default: %(default)s ppm)",
    )
    args = parser.parse_args()

    # Setting parameters
    run_number = args.run_number
    amplitude = args.amplitude
    decay = args.decay
    offset = args.offset
    noise_std = args.noise_std

    # Initialize profiles
    global amplitude_profile_transient_N
    global amplitude_profile_transient_λ
    global amplitude_profile_transient_c
    global noise_profile_transient_N
    global noise_profile_transient_λ
    global noise_profile_transient_c
    amplitude_profile_transient_N = partial(profile_sigmoid_plateau, a=50, b=55)
    amplitude_profile_transient_λ = partial(profile_sigmoid_plateau, a=50, b=55)
    amplitude_profile_transient_c = partial(profile_constant, value=0)
    noise_profile_transient_N = partial(profile_noise, noise_std=0.05)
    noise_profile_transient_λ = partial(profile_noise, noise_std=0.05)
    noise_profile_transient_c = partial(profile_noise, noise_std=0.1)

    # Check if directories exist
    if not os.path.exists(DATA_SYNTHETIC_DIRECTORY):
        logging.error(
            f"The data SYNTHETIC directory does not exist at {DATA_SYNTHETIC_DIRECTORY}."
        )
        exit()

    # Convert µm constant parameters to testgrid
    x_step = (SCAN_X_STOP - SCAN_X_START + 2) * SCAN_X_LSB_PER_UM / SCAN_X_STEPS
    X = np.arange(
        SCAN_X_START * SCAN_X_LSB_PER_UM,
        SCAN_X_STOP * SCAN_X_LSB_PER_UM,
        x_step,
    )

    y_step = (SCAN_Y_STOP - SCAN_Y_START + 2) * SCAN_Y_LSB_PER_UM / SCAN_Y_STEPS
    Y = np.arange(
        SCAN_Y_START * SCAN_Y_LSB_PER_UM,
        SCAN_Y_STOP * SCAN_Y_LSB_PER_UM,
        y_step,
    )
    X = np.ceil(X).astype("int")
    Y = np.ceil(Y).astype("int")

    # Plot amplitude/noise profiles
    profiles = [
        profile_constant,
        profile_gaussian,
        profile_hexagon,
        profile_noise,
        profile_sigmoid_plateau,
    ]
    # [plot_profile(profile) for profile in profiles]
    # exit(0)

    x, y = 0, 0
    noisy_signal = generate_point(x, y, amplitude, decay, offset, noise_std)
    # Plot
    time = np.linspace(TIME_START, TIME_STOP, PRETRIG_SAMPLES + POSTTRIG_SAMPLES)
    plt.plot(time, noisy_signal)
    plt.savefig(f"plots/test/x_{x}--y_{y}.png", bbox_inches="tight")
    exit(0)

    # Initialize .h5 file for transient storage
    init_file(run_number)

    pool = multiprocessing.Pool()
    time_start = time.time()

    tran_count = 0
    file_dicts = []
    for x in X:
        for y in Y:
            for _ in range(SCAN_HITS_PER_STEP):
                file_dicts.append(
                    {
                        "x": x,
                        "y": y,
                        "id": tran_count,
                    }
                )
                tran_count = tran_count + 1

    chunk_size = 1000
    num_chunks = int(np.ceil(float(len(file_dicts)) / chunk_size))

    for chunk_id, chunk in enumerate(chunker(file_dicts, chunk_size)):
        logging.info(f"  Processing chunk {chunk_id + 1}/{num_chunks}...")
        args = [
            (entry["x"], entry["y"], amplitude, decay, offset, noise_std)
            for entry in chunk
        ]
        freqs = pool.map(generate_point_args, args)

        transients_data = []
        for tran_dict, freq in zip(chunk, freqs):
            transients_data.append(
                {
                    "number": tran_dict["id"],
                    "position": (tran_dict["x"], tran_dict["y"]),
                    "data": freq,
                }
            )

        save_transients(run_number, transients_data)

        time_elapsed = time.time() - time_start
        time_per_chunk = time_elapsed / (chunk_id + 1)
        time_remaining = time_per_chunk * (num_chunks - chunk_id - 1)
        logging.info(
            f"  Time per chunk: {time_per_chunk:.02f} s; Remaining: {time_remaining:.02f} s"
        )

    tps = len(file_dicts) / time_elapsed
    logging.info(
        f"  Processing done. Elapsed time: {time_elapsed:.02f} s. Performance; {tps:.02f} files/s"
    )


def init_file(run_number):
    h5_path = os.path.join(DATA_SYNTHETIC_DIRECTORY, f"syn_{run_number:03d}.h5")
    with h5py.File(h5_path, "w") as h5file:
        # add run metadata
        logging.debug("Adding metadata...")
        meta_ds = h5file.create_dataset("meta", dtype="f")
        meta_ds.attrs.create("scan_x_lsb_per_um", SCAN_X_LSB_PER_UM)
        meta_ds.attrs.create("scan_y_lsb_per_um", SCAN_Y_LSB_PER_UM)

        sdr_group = h5file.create_group("sdr_data")
        # add SDR-related metadata
        sdr_group.attrs.create("sdr_info_fs", int(SAMPLING_FREQUENCY))
        sdr_group.attrs.create("sdr_info_len_pretrig", int(PRETRIG_SAMPLES))
        sdr_group.attrs.create("sdr_info_len_posttrig", int(POSTTRIG_SAMPLES))
        sdr_group.create_group("all")
        sdr_group.create_group("by_x")
        sdr_group.create_group("by_y")

        # Set processing stage at end of processing
        meta_ds.attrs.create("processing_stage", META_PROCESSING_STAGE)
        meta_ds.attrs.create("processing_stage_1_version", META_STAGE_1_VERSION)


def save_transients(run_number, transients_data):

    h5_path = os.path.join(DATA_SYNTHETIC_DIRECTORY, f"syn_{run_number:03d}.h5")
    with h5py.File(h5_path, "a") as h5file:

        sdr_group = h5file["sdr_data"]
        all_group = sdr_group["all"]
        by_x_group = sdr_group["by_x"]
        by_y_group = sdr_group["by_y"]

        for transient_data in transients_data:

            tran_number = transient_data["number"]
            tran_x = transient_data["position"][0]
            tran_y = transient_data["position"][1]
            transient = transient_data["data"]

            # store everything to a dataset
            tran_ds = all_group.create_dataset(
                f"tran_{tran_number:06d}", data=transient
            )
            tran_ds.attrs.create("tran_num", tran_number)
            tran_ds.attrs.create("x_lsb", tran_x)
            tran_ds.attrs.create("y_lsb", tran_y)
            tran_ds.attrs.create("dataset_unit", "Hz")

            # append dataset to by-x hierarchy
            by_x_x_group = by_x_group.require_group(f"x_{tran_x:06d}")
            if "x_lsb" not in by_x_x_group.attrs:
                by_x_x_group.attrs.create("x_lsb", tran_x)
            by_x_y_group = by_x_x_group.require_group(f"y_{tran_y:06d}")
            if "y_lsb" not in by_x_y_group.attrs:
                by_x_y_group.attrs.create("y_lsb", tran_y)
            by_x_y_group[f"tran_{tran_number:06d}"] = tran_ds

            # append dataset to by-y hierarchy
            by_y_y_group = by_y_group.require_group(f"y_{tran_y:06d}")
            if "y_lsb" not in by_y_y_group.attrs:
                by_y_y_group.attrs.create("y_lsb", tran_y)
            by_y_x_group = by_y_y_group.require_group(f"x_{tran_x:06d}")
            if "x_lsb" not in by_y_x_group.attrs:
                by_y_x_group.attrs.create("x_lsb", tran_x)
            by_y_x_group[f"tran_{tran_number:06d}"] = tran_ds


def generate_point_args(args):
    x, y, N, λ, c, noise_std = args
    return generate_point(x, y, N, λ, c, noise_std)


def generate_point(x, y, N, λ, c, noise_std):
    # Use probability to select different outlier types
    random_number = random.uniform(0, 1)
    if random_number < 0.05:
        N = N * 2

    random_number = random.uniform(0, 1)
    if random_number < 0.05:
        λ = λ * 2

    # Calculate adjusted randomised parameters based on profiles
    N_adj = (
        N
        * map_profile(amplitude_profile_transient_N, x, y)
        * map_profile(noise_profile_transient_N, x, y)
    )
    λ_adj = (
        λ
        * map_profile(amplitude_profile_transient_λ, x, y)
        * map_profile(noise_profile_transient_λ, x, y)
    )
    c_adj = (
        c
        * map_profile(amplitude_profile_transient_c, x, y)
        * map_profile(noise_profile_transient_c, x, y)
    )

    # Generate transient or constant based on probability profile
    signal = generate_transient(N_adj, λ_adj, c_adj)

    # Add noise to signal
    noisy_signal = noisify_signal(signal, noise_std)

    return noisy_signal


def generate_transient(N, λ, c):
    time = np.linspace(TIME_START, TIME_STOP, PRETRIG_SAMPLES + POSTTRIG_SAMPLES)
    time_posttrig = time[PRETRIG_SAMPLES:]

    signal = np.concatenate(
        (
            np.zeros(PRETRIG_SAMPLES),
            exponential_decay(time_posttrig, N, λ, c),
        )
    )

    return signal


def plot_profile(profile_function):

    X = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(X, X)
    Z = profile_function(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.savefig(
        f"plots/{profile_function.__name__}.png",
        bbox_inches="tight",
    )
    plt.close()


@np.vectorize
def map_profile(profile_function, x, y):
    if x < 0:
        mapped_x = x / (SCAN_X_START * SCAN_X_LSB_PER_UM)
    if x > 0:
        mapped_x = x / (SCAN_X_STOP * SCAN_X_LSB_PER_UM)
    if y < 0:
        mapped_y = y / (SCAN_Y_START * SCAN_Y_LSB_PER_UM)
    if y > 0:
        mapped_y = y / (SCAN_Y_STOP * SCAN_Y_LSB_PER_UM)
    if x == 0:
        mapped_x = 0
    if y == 0:
        mapped_y = 0

    return profile_function(mapped_x, mapped_y)


@np.vectorize
def profile_gaussian(x, y, covariance_value=0.125):
    mean = [0, 0]
    covariance = [[covariance_value, 0], [0, covariance_value]]
    return gaussian_2d(x, y, mean, covariance)


@np.vectorize
def profile_hexagon(x, y):
    def hexagon(pos):
        x, y = map(abs, pos)
        return y < 3**0.5 * min(1 - x, 1 / 2)

    if hexagon((x, y)):
        return 1
    else:
        return 0


@np.vectorize
def profile_sigmoid_plateau(x, y, a=50, b=55):
    return 1 / (1 + np.exp(b * np.sqrt(x**2 + y**2) - a))


@np.vectorize
def profile_constant(x, y, value=0):
    return value


@np.vectorize
def profile_noise(x, y, noise_std=0.1):
    return np.random.normal(1, noise_std, 1)


def noisify_signal(signal, noise_std):
    noise = np.random.normal(0, noise_std, len(signal))
    noisy_signal = signal + noise
    return noisy_signal


def gaussian_2d(x, y, mean, covariance):
    """
    Compute the value of a 2D Gaussian distribution at given (x, y) coordinates.

    Parameters:
    - x: X position
    - y: Y position
    - mean: Mean vector [mean_x, mean_y]
    - covariance: Covariance matrix [[cov_xx, cov_xy], [cov_yx, cov_yy]]

    Returns:
    - Value of the 2D Gaussian distribution at (x, y)
    """
    mean_vector = np.array([mean[0], mean[1]])
    position_vector = np.array([x, y])

    # Calculate the exponent term in the Gaussian formula
    exponent_term = -0.5 * np.dot(
        np.dot((position_vector - mean_vector).T, np.linalg.inv(covariance)),
        (position_vector - mean_vector),
    )

    # Calculate the normalization constant
    normalization = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

    # Calculate the Gaussian value
    gaussian_value = normalization * np.exp(exponent_term)

    # Calculate the maximum value of the Gaussian distribution
    max_value = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

    # Normalize the Gaussian value
    normalized_value = gaussian_value / max_value

    return normalized_value


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
