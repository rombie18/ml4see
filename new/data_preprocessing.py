import os
import logging
import traceback
import h5py
import argparse
import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters

# TODO set these variables in single external file
DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"
DATA_FEATURES_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/features"

FC_PARAMETERS = EfficientFCParameters()

def load_transient(h5_path, tran_name, time_data):
    try:
        with h5py.File(h5_path, "r") as h5file:
            # Get transient samples
            tran_data = h5file["sdr_data"]["all"][tran_name]
            
            # Downsampling, TODO make downsaple factor global constant
            tran_data = tran_data[::100]
        
            # Convert transient data to Pandas dataframe
            df = pd.DataFrame.from_dict({'transient': tran_name, 'time': time_data, 'frequency': np.array(tran_data)})
            
            return df
        
    except Exception as e:
        logging.error(f"Error processing transient {tran_name}: {str(e)}")
        traceback.print_exc()
        
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
        disable_progressbar=True
    )
    
    # Set tranient column as index and mark as category type
    feature_df = feature_df.rename_axis("transient").reset_index(drop=False)
    feature_df.transient = feature_df.transient.astype('category')
    
    return feature_df

        
def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_preprocessing.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data preprocessing process...")
    
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
            
            # Downsampling
            time_data = time_data[::100]
            
            #TODO write comment
            time_data_future = client.scatter(time_data)
            
            # Set up tasks to convert transients to Pandas dataframes
            transient_tasks = []
            for tran_name in transients.keys():
                transient_tasks.append(dask.delayed(load_transient)(h5_path, tran_name, time_data_future))

            # Set up task to merge all transients into single Dask dataframe
            transients_task = dd.from_delayed(transient_tasks)
                        
            # Set up task to extract features from each transient and merge then into one fDask dataframe
            features_task = transients_task.map_partitions(process_transient)
            
            # Execute above tasks
            features = features_task.compute()
            # print(features)

            # Save extracted features to csv file
            features.to_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
            
                        
if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()