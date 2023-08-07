"""Print mean of SEFT characteristics for stationary runs"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

import h5py
import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

with h5py.File(args.filename, "r") as h5file:
    util.require_processing_stage(h5file, 2)
    run_id = h5file["meta"].attrs["run_id"]
    scan_type = h5file["meta"].attrs["scan_type"]
    sdr_cf = h5file["sdr_data"].attrs["sdr_info_cf"]
    assert scan_type == "S", "Provided file does not contain stationary scan data"

    transients = h5file["sdr_data"]["all"]

    outlier_threshold = 3.5
    transients = [
        tran
        for tran in transients.values()
        if tran.attrs["baseline_outlier_score"] < outlier_threshold
    ]
    baseline_freq = np.mean(
        [tran.attrs["baseline_freq_mean_hz"] for tran in transients]
    )
    baseline_freq_std = np.mean(
        [tran.attrs["baseline_freq_mean_std_hz"] for tran in transients]
    ) / np.sqrt(len(transients))
    seft_exp_n0 = np.mean([tran.attrs["seft_exp_n0"] for tran in transients])
    seft_exp_lambda = np.mean([tran.attrs["seft_exp_lambda"] for tran in transients])
    seft_flattop_df = np.mean([tran.attrs["seft_flattop_mean"] for tran in transients])
    seft_flattop_std = np.mean(
        [tran.attrs["seft_flattop_mean_std"] for tran in transients]
    ) / np.sqrt(len(transients))

    f0 = (
        sdr_cf + baseline_freq + transients[0].attrs["dsp_info_pre_demod_bb_lo_freq_hz"]
    )

    seft_flattop_df_ppm = (seft_flattop_df - baseline_freq) / f0 * 1e6
    seft_flattop_df_std_ppm = (
        np.sqrt(baseline_freq_std**2 + seft_flattop_std**2) / f0 * 1e6
    )
    seft_exp_n0_ppm = seft_exp_n0 / f0 * 1e6
    seft_exp_t12 = (np.log(2) / seft_exp_lambda,)

    print(f"Run {run_id} SEFT summary.")
    print(f"  f0: {f0:.03e} Hz")
    print(
        f"  Flattop delta-F: {seft_flattop_df_ppm:.02f} ppm +- {seft_flattop_df_std_ppm:.02f} ppm"
    )
