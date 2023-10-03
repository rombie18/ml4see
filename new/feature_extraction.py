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
import traceback
import h5py
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from tsfresh.feature_extraction import extract_features, feature_calculators
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters

from config import DATA_STRUCTURED_DIRECTORY, DATA_FEATURES_DIRECTORY

FC_PARAMETERS = ComprehensiveFCParameters()
WINDOW_SIZE = 500
DOWNSAMPLE_FACTOR = 250

#TODO add beam position info to transient
def load_transient(h5_path, tran_name, time_data):
    #TODO try to find way to speed up reading transients from disk
    with h5py.File(h5_path, "r") as h5file:
        # Get transient samples
        tran_data = np.array(h5file["sdr_data"]["all"][tran_name])
        
        # Apply moving average filter
        window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
        tran_data = np.convolve(tran_data, window, mode='valid')
                
        # Adjust time data to match length of convoluted output
        time_data = time_data[WINDOW_SIZE-1:]

        # Downsample time and frequency data
        time_data = time_data[::DOWNSAMPLE_FACTOR]
        tran_data = tran_data[::DOWNSAMPLE_FACTOR]

        # Convert transient data to Pandas dataframe
        df = pd.DataFrame.from_dict({'transient': tran_name, 'time': time_data, 'frequency': tran_data})
        return df
        
def process_transient(df):
    
    # Remove NaN 'probe' data since it is causing issues with feature extraction
    df = df[df['transient'].notna()]
        
    # Extract features of single transient
    # TODO use multiprocessing here dependant on how many thread this worker has available
    feature_df = extract_features(
        df, 
        column_id="transient",
        column_sort="time",
        n_jobs=0,
        default_fc_parameters=FC_PARAMETERS,
        disable_progressbar=True,
        show_warnings=False
    )
    
    # Set tranient column as index
    feature_df = feature_df.rename_axis("transient").reset_index(drop=False)
    feature_df.transient = feature_df.transient.astype('category')
    feature_df.set_index('transient')
    
    # Extract single exponential decay parameters and add to dataframe
    params = fit_single_exponential_decay(df['frequency'], df['time'])
    for param_name, param_value in params.items():
        feature_df[param_name] = [param_value]
    
    # Extract double exponential decay parameters and add to dataframe
    params = fit_double_exponential_decay(df['frequency'], df['time'])
    for param_name, param_value in params.items():
        feature_df[param_name] = [param_value]
        
    return feature_df


def fit_double_exponential_decay(freq_data, time_data):
    def double_exponential_decay(t, Nf, Ns, λf, λs, c):
        return Nf * np.exp(-λf * t) + Ns * np.exp(-λs * t) + c
       
    start_index = np.argmax(freq_data)
    time_data = time_data[start_index:]
    freq_data = freq_data[start_index:]
    
    #TODO limit boundries to fast and slow 
    params, _ = curve_fit(
        double_exponential_decay, time_data, freq_data, p0=(35000, 35000, 10000, 100, 20000), bounds=([0, 0, 1000, 0, 0], [1e8, 1e8, 1e5, 1e4, 1e6])
    )
    
    return {
        "fit_double_exponential_decay__Nf": params[0],
        "fit_double_exponential_decay__Ns": params[1],
        "fit_double_exponential_decay__λf": params[2],
        "fit_double_exponential_decay__λs": params[3],
        "fit_double_exponential_decay__c": params[4],
    }
    
def fit_single_exponential_decay(freq_data, time_data):
    def single_exponential_decay(t, N, λ, c):
        return N * np.exp(-λ * t) + c
       
    start_index = np.argmax(freq_data)
    time_data = time_data[start_index:]
    freq_data = freq_data[start_index:]
    
    params, _ = curve_fit(
        single_exponential_decay, time_data, freq_data, p0=(35000, 5000, 20000), bounds=([0, 0, 0], [1e8, 1e5, 1e6])
    )
    
    return {
        "fit_single_exponential_decay__N": params[0],
        "fit_single_exponential_decay__λ": params[1],
        "fit_single_exponential_decay__c": params[2],
    }
    
        
def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("feature_extraction.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting feature extraction process...")
    
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('run_numbers', metavar='run_number', nargs='*', type=int)
    args = parser.parse_args()
    
    # Set Pandas options to increase readability
    pd.set_option('display.float_format', lambda x: '%.9f' % x)
    pd.options.display.max_rows = 1000

    # Check if directories exist
    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.error(f"The structured data directory does not exist at {DATA_STRUCTURED_DIRECTORY}.")
        exit()
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(f"The features data directory does not exist at {DATA_FEATURES_DIRECTORY}.")
        exit()
    
    for run_number in args.run_numbers:
        logging.info(f"Processing run {run_number:03d}")
        # Combine data directory with provided run number to open .h5 file in read mode
        h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
        with h5py.File(h5_path, "r") as h5file:
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
            time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
            
            #TODO write comment
            time_data_future = client.scatter(time_data)
            
            # Set up tasks to convert transients to Pandas dataframes
            transient_tasks = []
            for tran_name in transients.keys():
                transient_tasks.append(dask.delayed(load_transient)(h5_path, tran_name, time_data_future))
            # transient_tasks.append(dask.delayed(load_transient)(h5_path, "tran_000000", time_data_future))

            # Set up task to merge all transients into single Dask dataframe
            transients_task = dd.from_delayed(transient_tasks)
                        
            # Set up task to extract features from each transient and merge then into one fDask dataframe
            features_task = transients_task.map_partitions(process_transient)
            
            # Execute above tasks
            features: pd.DataFrame = features_task.compute()
            
            # Add valid column for feature significance testing
            features.insert(1, "valid", True)

            # Save extracted features to csv file
            features.to_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"), index=False)
            
                        
if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()