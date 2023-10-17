"""Processing stage 2: annotate baseline data"""

import argparse
import math
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

import h5py
import util

META_PROCESSING_STAGE = 2  # processing stage after completing this stage
META_STAGE_2_VERSION = "2.0"  # version string for this stage (stage 2)

#TODO move this script into processing pipeline

# main changes
# v1.0: baseline heuristics annotation (mean, std, std-of-mean, std-of-std) and outlier scoring
# v2.0: fix baseline stats misattribution, compatibility with stage 1 v2.0

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

PRETRIG_GUARD_SAMPLES = 100

with h5py.File(args.filename, "a") as h5file:
    util.require_processing_stage(h5file, 1, strict=True)
    meta_ds = h5file["meta"]
    meta_ds.attrs.modify("processing_stage", META_PROCESSING_STAGE)
    meta_ds.attrs.create("processing_stage_2_version", META_STAGE_2_VERSION)

    len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
    transients = h5file["sdr_data"]["all"]

    print("Getting list of datasets in run...")
    run_datasets = list(transients.keys())
    num_datasets = len(run_datasets)
    # choose a chunk size such that at least a few thousand events land in each chunk
    # this is important to obtain a good estimate for the standard deviation of the mean
    # which is used in outlier scoring
    MIN_CHUNK_PROP = 0.95
    for chunk_size in range(4000, 5000):
        num_chunks = num_datasets / chunk_size
        if num_chunks - int(num_chunks) > MIN_CHUNK_PROP:
            break
    if num_chunks > 1 and num_chunks - int(num_chunks) < MIN_CHUNK_PROP:
        raise RuntimeEror("Last chunk will be too small...")

    for chunk_idx, chunk in enumerate(util.chunker(run_datasets, chunk_size)):
        print(f"  Processing chunk {chunk_idx+1}/{math.ceil(num_chunks)}.")

        tran_freq_means = []
        tran_freq_stds = []

        for tran in [transients[name] for name in chunk]:
            tran_baseline_freq = np.array(tran)[: len_pretrig - PRETRIG_GUARD_SAMPLES]
            tran_baseline_freq_mean = np.mean(tran_baseline_freq)
            tran_baseline_freq_std = np.std(tran_baseline_freq)

            tran_freq_means.append(tran_baseline_freq_mean)
            tran_freq_stds.append(tran_baseline_freq_std)

        tran_freq_means = np.array(tran_freq_means)
        tran_freq_stds = np.array(tran_freq_stds)

        std_hist, std_hist_bin_edges = np.histogram(
            tran_freq_stds, bins=5000, range=(0e3, 20e3)
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
                (
                    std_hist_bin_centers[fit_stop_idx]
                    - std_hist_bin_centers[fit_start_idx]
                )
                / 3.3,
            ),
        )
        fit_mu = fit_params[0]
        fit_sigma = fit_params[1]
        print(f"  Gaussian fit: mu {fit_mu:.03f} Hz; sigma {fit_sigma:.03f} Hz")

        outlier_scores = np.abs(tran_freq_stds - fit_mu) / fit_sigma

        for tran, tran_mean, tran_std, tran_outlier_score in zip(
            [transients[name] for name in chunk], tran_freq_means, tran_freq_stds, outlier_scores
        ):
            tran.attrs.create("baseline_freq_mean_hz", tran_mean)
            tran.attrs.create("baseline_freq_mean_std_hz", tran_std / np.sqrt(len_pretrig - PRETRIG_GUARD_SAMPLES))
            tran.attrs.create("baseline_freq_std_hz", tran_std)
            tran.attrs.create("baseline_freq_std_std_hz", fit_sigma)
            tran.attrs.create("baseline_outlier_score", tran_outlier_score)
