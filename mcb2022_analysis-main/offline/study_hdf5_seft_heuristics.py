"""Calculate & visualize heuristics useful for outlier classification during baseline period"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

TRIGGER_DELAY = 3.75e-6  # seconds
FIT_T_START = 25e-6
FIT_T_STOP = 1e-3
FT_T_START = 4e-6
FT_T_STOP = 15e-6

with h5py.File(args.filename, "r") as h5file:
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]

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

    transients = [
        x for x in transients.values() if x.attrs["baseline_outlier_score"] < 3.5
    ]

    tran_nums = np.zeros(len(transients))
    fit_ns = np.zeros(len(transients))
    fit_lambdas = np.zeros(len(transients))
    tran_fts = np.zeros(len(transients))

    for tran_idx, dsref in enumerate(transients):
        tran_num = dsref.attrs["tran_num"]
        print(f"Processing transient {tran_num}...")
        baseline_freq = dsref.attrs["baseline_freq_mean_hz"]
        tran_freq = np.array(dsref) - baseline_freq
        f_fit = tran_freq[fit_start_idx : fit_stop_idx + 1]
        params, _ = spo.curve_fit(
            exp_model, t_fit, f_fit, p0=(1e5, 10000), bounds=([0, 10], [1e8, 6.5e5])
        )
        fit_exp_n, fit_exp_lambda = params
        tran_nums[tran_idx] = tran_num
        fit_ns[tran_idx] = fit_exp_n
        fit_lambdas[tran_idx] = fit_exp_lambda
        tran_fts[tran_idx] = np.mean(tran_freq[ft_start_idx : ft_stop_idx + 1])

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    ax1.plot(tran_nums, fit_ns, ".", alpha=0.2, label="N")
    ax1.plot(tran_nums, tran_fts, ".", alpha=0.2, label="FT estimate")
    ax1.set_xlabel("Transient Number")
    ax1.set_ylabel("Exponential fit $N_0$ (Hz)")
    ax1.grid()
    ax2.plot(tran_nums, np.log(2) / np.array(fit_lambdas), ".", alpha=0.2)
    ax2.set_xlabel("Transient Number")
    ax2.set_ylabel("Exponential fit $t_{1/2}$ (s)")
    ax2.grid()
    plt.show()

    plt.plot(fit_ns, np.log(2) / np.array(fit_lambdas), ".", alpha=0.2)
    plt.xlabel("Exponential fit $N_0$ (Hz)")
    plt.ylabel("Exponential fit $t_{1/2}$ (s)")
    plt.grid()
    plt.show()

    plt.plot(fit_ns, tran_fts, ".", alpha=0.2)
    plt.xlabel("Exponential fit $N_0$ (Hz)")
    plt.ylabel("Flat top estimate (Hz)")
    plt.grid()
    plt.show()
