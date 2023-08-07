"""Processing stage 2: annotate baseline data"""

import multiprocessing
import argparse
import math
import time
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

import h5py
import util


META_PROCESSING_STAGE = 3  # processing stage after completing this stage
META_STAGE_3_VERSION = "1.0"  # version string for this stage (stage 2)
TRIGGER_DELAY = 3.75e-6  # seconds
FIT_T_START = 25e-6
FIT_T_STOP = 0.99e-3
FT_T_START = 4e-6
FT_T_STOP = 15e-6

# main changes
# v1.0: simple exponential decay model

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "a") as h5file:
    util.require_processing_stage(h5file, 2, strict=False)
    meta_ds = h5file["meta"]
    meta_ds.attrs.modify("processing_stage", META_PROCESSING_STAGE)
    meta_ds.attrs.create("processing_stage_3_version", META_STAGE_3_VERSION)

    # model fit related stuff
    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
    dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]

    event_len = len_pretrig + len_posttrig - dsp_ntaps
    t = (
        np.arange(start=0, stop=event_len / fs, step=1 / fs)
        - len_pretrig / fs
        + TRIGGER_DELAY
    )
    fit_start_idx = np.argmax(t >= FIT_T_START)
    fit_stop_idx = np.argmax(t >= FIT_T_STOP)
    t_fit = t[fit_start_idx : fit_stop_idx + 1]

    ft_start_idx = np.argmax(t >= FT_T_START)
    ft_stop_idx = np.argmax(t >= FT_T_STOP)

    def exp_model(t, exp_n, exp_lambda):
        return exp_n * np.exp(-exp_lambda * t)

    print("Getting list of datasets in run...")
    transients = h5file["sdr_data"]["all"]
    transient_names = list(transients)

    time_start = time.time()

    def fit_transient(transient, baseline_freq, baseline_std):
        f_fit = transient[fit_start_idx : fit_stop_idx + 1] - baseline_freq
        params, _ = spo.curve_fit(
            exp_model, t_fit, f_fit, p0=(0, 10000), bounds=([-1e8, 10], [1e8, 6.5e5])
        )
        fit_exp_n, fit_exp_lambda = params
        tran_ft = np.mean(transient[ft_start_idx : ft_stop_idx + 1])
        return {
            "seft_exp_n0": fit_exp_n,
            "seft_exp_lambda": fit_exp_lambda,
            "seft_flattop_mean": tran_ft,
            "seft_flattop_mean_std": baseline_std / np.sqrt(ft_stop_idx - ft_start_idx),
        }

    pool = multiprocessing.Pool()

    CHUNK_SIZE = 1024
    num_chunks = int(np.ceil(float(len(transient_names)) / CHUNK_SIZE))
    for chunk_id, chunk in enumerate(util.chunker(list(transients), CHUNK_SIZE)):
        print(f"  Processing chunk {chunk_id + 1}/{num_chunks}...")
        tran_data = [np.array(transients[tran]) for tran in chunk]
        tran_baselines = [
            transients[tran].attrs["baseline_freq_mean_hz"] for tran in chunk
        ]
        tran_stds = [transients[tran].attrs["baseline_freq_std_hz"] for tran in chunk]
        tran_results = pool.starmap(
            fit_transient, zip(tran_data, tran_baselines, tran_stds)
        )
        for tran, tran_result in zip(chunk, tran_results):
            for key, value in tran_result.items():
                transients[tran].attrs.create(key, value)

        time_elapsed = time.time() - time_start
        time_per_chunk = time_elapsed / (chunk_id + 1)
        time_remaining = time_per_chunk * (num_chunks - chunk_id - 1)
        print(
            f"  Time per chunk: {time_per_chunk:.02f} s; Remaining: {time_remaining:.02f} s"
        )

    tps = len(transient_names) / time_elapsed
    print(
        f"  Processing done. Elapsed time: {time_elapsed:.02f} s. Performance; {tps:.02f} files/s"
    )
