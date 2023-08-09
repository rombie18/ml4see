import argparse

import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd
import dask.array as da
import h5py

import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]

    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    event_len = len_pretrig + len_posttrig - dsp_ntaps
    t = np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
    
    util.require_processing_stage(h5file, 2)

    data = np.array(transients.values())
    tran_nums = np.array([np.repeat(transient.attrs["tran_num"], t.size) for transient in transients.values()])
    data = np.column_stack((tran_nums, np.repeat(t, data.size), data))
    data = da.from_array(data, chunks=1000, name="test")
    df = dd.from_dask_array(data, columns=['tran_num', 'time', 'frequency'])
    print(df.head())