import os
import logging
import h5py
import argparse
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client

# DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"
DATA_STRUCTURED_DIRECTORY = "../data/hdf5"

def process_transient(h5_path, run_num, tran_name, time_data):
    with h5py.File(h5_path, "r") as h5file:
        tran_data = h5file["sdr_data"]["all"][tran_name]
        df = pd.DataFrame.from_dict({'run': run_num, 'transient': tran_name, 'time': time_data, 'frequency': np.array(tran_data)})
        return df

def main():
    logging.basicConfig(filename='data_preprocessing.log', filemode="w", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info("Starting data preprocessing process...")
    
    client = Client()

    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.warning("The structured data directory does not exist at {}.".format(DATA_STRUCTURED_DIRECTORY))
        
    parser = argparse.ArgumentParser()
    parser.add_argument("run_number", type=int)
    args = parser.parse_args()
    
    h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{args.run_number:03d}.h5")
    with h5py.File(h5_path, "r") as h5file:            
            
        run_num = h5file["meta"].attrs["run_id"]
        transients = h5file["sdr_data"]["all"]

        fs = h5file["sdr_data"].attrs["sdr_info_fs"]
        len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
        len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
        dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

        event_len = len_pretrig + len_posttrig - dsp_ntaps
        time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
        
        dfs = []
        for tran_name in transients.keys():
            dfs.append(dask.delayed(process_transient)(h5_path, run_num, tran_name, time_data))

        df = dd.from_delayed(dfs)
        df = df.compute()
        client.close()
        
        print(df)

if __name__ == "__main__":
    main()