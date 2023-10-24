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


from config import (
    DATA_STRUCTURED_DIRECTORY,
    DATA_FEATURES_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
)
from utils import require_processing_stage


# TODO add beam position info to transient


def load_transient(h5_path, tran_name, time_data):
    # TODO try to find way to speed up reading transients from disk
    with h5py.File(h5_path, "r") as h5file:
        # Get transient samples
        tran_data = np.array(h5file["sdr_data"]["all"][tran_name])

        # Subtract mean baseline frequency from each sample to get delta frequency
        baseline_freq = h5file["sdr_data"]["all"][tran_name].attrs[
            "baseline_freq_mean_hz"
        ]
        tran_data = np.subtract(tran_data, baseline_freq)

        # Apply moving average filter
        window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
        tran_data = np.convolve(tran_data, window, mode="valid")

        # Adjust time data to match length of convoluted output
        time_data = time_data[WINDOW_SIZE - 1 :]

        # Downsample time and frequency data
        time_data = time_data[::DOWNSAMPLE_FACTOR]
        tran_data = tran_data[::DOWNSAMPLE_FACTOR]

        # Convert transient data to Pandas dataframe
        df = pd.DataFrame.from_dict(
            {"transient": tran_name, "time": time_data, "frequency": tran_data}
        )
        return df


def process_transient(df):
    # Remove NaN 'probe' data since it is causing issues with feature extraction
    df = df[df["transient"].notna()]

    # Set tranient column as index
    feature_df = pd.DataFrame()
    feature_df = feature_df.rename_axis("transient").reset_index(drop=False)
    feature_df.transient = feature_df.transient.astype("category")
    feature_df.set_index("transient")

    feature_df["transient"] = [df['transient'].iloc[0]]

    if (len(df["time"].to_numpy()) > 1):
        bin_maxs, last_timestamps = max_values_in_bins(df["time"].to_numpy(), df["frequency"].to_numpy(), 100)
        for param_name, param_value in enumerate(bin_maxs):
            feature_df["max_bin_" + str(param_name)] = [param_value]
    else:
        bin_maxs = np.repeat(1, 100)
        for param_name, param_value in enumerate(bin_maxs):
            feature_df["max_bin_" + str(param_name)] = [param_value]

    return feature_df


def max_values_in_bins(timestamps, values, num_bins):
    # Calculate the bin size
    bin_size = (timestamps[-1] - timestamps[0]) / num_bins

    # Calculate the bin edges
    bin_edges = [timestamps[0] + i * bin_size for i in range(num_bins + 1)]

    # Compute the histogram
    bin_indices = np.digitize(timestamps, bin_edges)

    # Initialize variables to store sums and last timestamps
    bin_maxs = []
    last_timestamps = []

    # Iterate through the bins
    for bin_num in range(1, num_bins + 1):
        mask = bin_indices == bin_num
        bin_max = np.max(values[mask])
        last_timestamp = timestamps[mask][-1] if np.any(mask) else None
        bin_maxs.append(bin_max)
        last_timestamps.append(last_timestamp)

    # Normalize the bin sums to the range [0, 1]
    # max_sum = np.max(bin_maxs)
    # min_sum = np.min(bin_maxs)
    # normalized_bin_maxs = [(x - min_sum) / (max_sum - min_sum) for x in bin_maxs]

    return bin_maxs, last_timestamps


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
            # If run has no transients, skip
            if "sdr_data" not in h5file:
                logging.warning(
                    f"Skipping run_{run_number:03d} since it has no transients (sdr_data)."
                )
                continue

            # Check if file is up to required processing stage
            require_processing_stage(h5file, 2, strict=True)

            # Get run number and transients from file
            run_num = h5file["meta"].attrs["run_id"]
            transients = h5file["sdr_data"]["all"]

            # Get additional meta data
            fs = h5file["sdr_data"].attrs["sdr_info_fs"]
            len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
            len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
            dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

            # Calculate real time from meta data
            event_len = len_pretrig + len_posttrig - dsp_ntaps
            time_data = (
                np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
            )

            # TODO write comment
            time_data_future = client.scatter(time_data)

            # Set up tasks to convert transients to Pandas dataframes
            transient_tasks = []
            for tran_name in transients.keys():
                transient_tasks.append(
                    dask.delayed(load_transient)(h5_path, tran_name, time_data_future)
                )
            # transient_tasks.append(dask.delayed(load_transient)(h5_path, "tran_000000", time_data_future))

            # Set up task to merge all transients into single Dask dataframe
            transients_task = dd.from_delayed(transient_tasks)

            # Set up task to extract features from each transient and merge then into one fDask dataframe
            features_task = transients_task.map_partitions(process_transient)

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