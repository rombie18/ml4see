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

PRETRIG_GUARD_SAMPLES = 100

with h5py.File(args.filename, "r") as h5file:
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]

    fs = h5file["sdr_data"].attrs["sdr_info_fs"]
    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]

    ev_ids = []
    ev_freq_means = []
    ev_freq_stds = []

    for dsref in transients.values():
        tran_num = dsref.attrs["tran_num"]
        print(f"Processing transient {tran_num}...")
        tran_baseline_freq = np.array(dsref)[: len_pretrig - PRETRIG_GUARD_SAMPLES]
        tran_baseline_freq_mean = np.mean(tran_baseline_freq)
        tran_baseline_freq_std = np.std(tran_baseline_freq)

        ev_ids.append(tran_num)
        ev_freq_means.append(tran_baseline_freq_mean)
        ev_freq_stds.append(tran_baseline_freq_std)

    ev_ids = np.array(ev_ids)
    ev_freq_means = np.array(ev_freq_means)
    ev_freq_stds = np.array(ev_freq_stds)

    std_hist, std_hist_bin_edges = np.histogram(
        ev_freq_stds, bins=5000, range=(0e3, 20e3)
    )
    std_hist_bin_centers = (std_hist_bin_edges[:-1] + std_hist_bin_edges[1:]) / 2
    std_hist_max_idx = np.argmax(std_hist)
    std_hist_max = std_hist[std_hist_max_idx]
    std_hist_thresh = std_hist_max * 0.25  # fit down to 1.65 sigma
    fit_start_idx = -1
    fit_stop_idx = -1
    for i in range(100):
        if std_hist[std_hist_max_idx - i] < std_hist_thresh:
            fit_start_idx = std_hist_max_idx - i
            break
    for i in range(100):
        if std_hist[std_hist_max_idx + i] < std_hist_thresh:
            fit_stop_idx = std_hist_max_idx + i
            break
    assert fit_start_idx > 0, "Threshold search did not converge..."
    assert fit_stop_idx > 0, "Threshold search did not converge..."

    def gaussian_pdf(x, mu, sigma):
        return (
            std_hist_max
            * sps.norm.pdf(x, loc=mu, scale=sigma)
            / sps.norm.pdf(0, loc=0, scale=sigma)
        )

    fit_params, _ = spo.curve_fit(
        f=gaussian_pdf,
        xdata=std_hist_bin_centers[fit_start_idx : fit_stop_idx + 1],
        ydata=std_hist[fit_start_idx : fit_stop_idx + 1],
        p0=(
            std_hist_bin_centers[std_hist_max_idx],
            (std_hist_bin_centers[fit_stop_idx] - std_hist_bin_centers[fit_start_idx])
            / 3.3,
        ),
    )
    fit_mu = fit_params[0]
    fit_sigma = fit_params[1]
    print(f"Gaussian fit: mu {fit_mu} Hz; sigma {fit_sigma} Hz")

    outlier_score = np.abs(ev_freq_stds - fit_mu) / fit_sigma

    plt.title(f"Baseline heuristics for run ID {run_num}")
    plt.subplot(2, 2, 1)
    plt.plot(ev_ids, ev_freq_means, ".")
    plt.xlabel("Event ID")
    plt.ylabel("Baseline Mean Frequency (Hz)")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(ev_ids, ev_freq_stds, ".")
    plt.xlabel("Event ID")
    plt.ylabel("Baseline Standard Deviation (Hz)")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.hist(ev_freq_stds, 1024, range=(2e3, 10e3))
    plt.xlabel("Baseline Standard Deviation (Hz)")
    plt.ylabel("Frequency of occurrence")
    plt.yscale("log")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(ev_ids, outlier_score, ".")
    plt.xlabel("Event ID")
    plt.ylabel("Outlier score")
    plt.grid()

    plt.show()
