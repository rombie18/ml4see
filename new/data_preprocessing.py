import os
import logging
import traceback
import h5py
import argparse
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
from tsfresh.utilities.distribution import LocalDaskDistributor
from tsfresh import extract_features

DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"
#DATA_STRUCTURED_DIRECTORY = "../data/hdf5"

def process_transient(h5_path, run_num, tran_name):
    try:
        with h5py.File(h5_path, "r") as h5file:
            tran_data = h5file["sdr_data"]["all"][tran_name]
            
            fs = h5file["sdr_data"].attrs["sdr_info_fs"]
            len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
            len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
            dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

            event_len = len_pretrig + len_posttrig - dsp_ntaps
            time_data = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
        
            #time_data[:10000]
            df = pd.DataFrame.from_dict({'run': run_num, 'transient': tran_name, 'time': np.arange(0, 1000, 1, dtype=np.float64), 'frequency': np.array(tran_data)[:1000]})
            return df
        
    except Exception as e:
        logging.error(f"Error processing transient {tran_name}: {str(e)}")
        
def main(cluster):
    logging.basicConfig(filename='data_preprocessing.log', filemode="w", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info("Starting data preprocessing process...")
    logging.getLogger("distributed.scheduler").setLevel(logging.DEBUG)
    
    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.warning("The structured data directory does not exist at {}.".format(DATA_STRUCTURED_DIRECTORY))
        
    parser = argparse.ArgumentParser()
    parser.add_argument("run_number", type=int)
    args = parser.parse_args()
    
    h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{args.run_number:03d}.h5")
    with h5py.File(h5_path, "r") as h5file:
                
        run_num = h5file["meta"].attrs["run_id"]
        transients = h5file["sdr_data"]["all"]
        
        dfs = []
        # for tran_name in transients.keys():
        #     dfs.append(dask.delayed(process_transient)(h5_path, run_num, tran_name))
        dfs.append(dask.delayed(process_transient)(h5_path, run_num, "tran_000000"))

        df = dd.from_delayed(dfs)
        df = df.compute()
        
        # pd.set_option('display.float_format', lambda x: '%.9f' % x)
        # print(df)
        
        #FIXME features extraction doesn't start properly and gets stuck
        task = extract_features(df,
                     column_id="run",
                     column_sort="time",
                     column_kind="transient",
                     column_value="frequency",
                     pivot=False,
                     distributor=cluster)

        features = task.compute()
        
        print("test")

        print(features)

if __name__ == "__main__":
    try:
        cluster = LocalDaskDistributor(n_workers=2)
        client = Client(cluster)
        main(client)
    except:
        logging.exception("Fatal exception in main")
        traceback.print_exc()
    finally:
        client.close()