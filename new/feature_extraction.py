"""
feature_extraction.py

This Python script is designed to perform feature extraction from raw data files and save the extracted features to CSV files. It utilizes the `tsfresh` library for feature extraction, Dask for parallel processing, and configuration constants imported from an external module for flexibility.

Usage:
    python feature_extraction.py [run_numbers [run_numbers ...]]

Arguments:
    run_numbers (optional): A list of integers representing specific run numbers to process and extract features from.

Configuration (imported from 'config.py'):
    - DATA_STRUCTURED_DIRECTORY: The directory where raw data files (in HDF5 format) are located.
    - DATA_FEATURES_DIRECTORY: The directory where extracted features will be saved.
    - FC_PARAMETERS: Feature extraction parameters (configured based on your requirements).
    - WINDOW_SIZE: Size of the moving average window for filtering transient data.
    - DOWNSAMPLE_FACTOR: Factor for downsampling time and frequency data.

The script performs the following steps:
1. Initializes logging to record feature extraction progress and errors.
2. Parses command-line arguments to optionally specify which runs to process and extract features from.
3. Sets Pandas options for better readability of output.
4. Checks if the specified data directories exist; exits if not.
5. Iterates through the provided run numbers, processing each run individually.
6. Reads data from HDF5 files, applies filtering, downsampling, and prepares the data for feature extraction.
7. Extracts features using the `tsfresh` library and saves them to CSV files.
8. Closes the Dask client and logs any fatal exceptions.

Note: The script assumes that it is executed using a Dask cluster for parallel processing.

Example Usage:
- Extract features from all available runs:
    python feature_extraction.py

- Extract features from specific runs (e.g., run numbers 1 and 2):
    python feature_extraction.py 1 2
"""

import os
import logging
import h5py
import argparse
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from scipy.optimize import curve_fit

from config import (
    DATA_STRUCTURED_DIRECTORY,
    DATA_FEATURES_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
    PRETRIG_GUARD_SAMPLES,
)
from utils import require_processing_stage, moving_average, exponential_decay


# TODO add beam position info to transient


def process_transient(h5_path, tran_name):
    # TODO try to find way to speed up reading transients from disk
    with h5py.File(h5_path, "r") as h5file:
        # Get transient data from file
        transient = h5file["sdr_data"]["all"][tran_name]

        # Get run meta data
        fs = h5file["sdr_data"].attrs["sdr_info_fs"]
        len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
        len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
        dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]
        event_len = len_pretrig + len_posttrig - dsp_ntaps
        baseline_freq = transient.attrs["baseline_freq_mean_hz"]

        # Get additional transient meta data
        # TODO subtract baseline variance from this transients to get a delta, use that as feature instead of pretrig std
        baseline_freq = transient.attrs["baseline_freq_mean_hz"]
        baseline_freq_var = transient.attrs["baseline_freq_mean_hz"]
        x_lsb = transient.attrs["x_lsb"]
        y_lsb = transient.attrs["y_lsb"]

        # Construct time and frequency arrays, subtract mean baseline frequency from each sample to get delta frequency
        tran_time = (
            np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
        )
        tran_freq = np.subtract(np.array(transient), baseline_freq)

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
        features["pretrig_std"] = np.std(tran_pretrig_freq_ds)

        # Try to fit exponential decay
        initial_guess = (
            np.max(tran_posttrig_freq_ds),
            1000,
            np.mean(tran_pretrig_freq_ds),
        )

        minimum_exp_height = np.mean(tran_pretrig_freq_ds) + 2 * (
            np.abs(np.max(tran_pretrig_freq_ds)) + np.abs(np.min(tran_pretrig_freq_ds))
        )
        boundaries = (
            [
                minimum_exp_height,
                0,
                -1e6,
            ],
            [1e6, 1e6, 1e6],
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

            # Assign fitted parameters and R² to resulting feature set
            features["posttrig_exp_fit_N"] = params[0]
            features["posttrig_exp_fit_λ"] = params[1]
            features["posttrig_exp_fit_c"] = params[2]
            features["posttrig_exp_fit_R2"] = r_squared

        except:
            # If exponential fit fails, assign parameters to zero
            features["posttrig_exp_fit_N"] = 0
            features["posttrig_exp_fit_λ"] = 0
            features["posttrig_exp_fit_c"] = 0
            features["posttrig_exp_fit_R2"] = 0

        # Convert feature set to Pandas dataframe
        df = pd.DataFrame(features, index=[0])

        return df


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("feature_extraction.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting feature extraction process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Set Pandas options to increase readability
    pd.set_option("display.float_format", lambda x: "%.9f" % x)
    pd.options.display.max_rows = 1000

    # Check if directories exist
    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.error(
            f"The structured data directory does not exist at {DATA_STRUCTURED_DIRECTORY}."
        )
        exit()
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(
            f"The features data directory does not exist at {DATA_FEATURES_DIRECTORY}."
        )
        exit()

    run_numbers = []
    if len(args.run_numbers) == 0:
        for file in os.listdir(DATA_STRUCTURED_DIRECTORY):
            if file.endswith(".h5"):
                run_numbers.append(int(file[4:7]))
        print(f"No runs specified, running on all available runs: {run_numbers}")
    else:
        run_numbers = args.run_numbers

    for run_number in run_numbers:
        logging.info(f"Processing run {run_number:03d}")
        # Combine data directory with provided run number to open .h5 file in read mode
        h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
        with h5py.File(h5_path, "r") as h5file:
            # TODO when opening h5 check for various things such as below and move into unified file eg utils
            # If run has no transients, skip
            if "sdr_data" not in h5file:
                logging.warning(
                    f"Skipping run_{run_number:03d} since it has no transients (sdr_data)."
                )
                continue

            # Check if file is up to required processing stage
            require_processing_stage(h5file, 2, strict=True)

            # Get transients from h5 file
            transients = h5file["sdr_data"]["all"]

            # Set up tasks to convert transients to Pandas dataframes
            transient_tasks = []
            for tran_name in transients.keys():
                transient_tasks.append(
                    dask.delayed(process_transient)(h5_path, tran_name)
                )

            # Set up task to merge all transients into single Dask dataframe
            features_task = dd.from_delayed(transient_tasks)

            # Execute above tasks
            features: pd.DataFrame = features_task.compute()

            # Save extracted features to csv file
            features.to_csv(
                os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"),
                index=False,
            )


if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()
