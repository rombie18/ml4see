"""Give statistics on number of classified outliers within a run"""

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
    run_num = h5file["meta"].attrs["run_id"]
    transients = h5file["sdr_data"]["all"]

    num_trans = len(transients)
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    nums_trans_above = []

    for threshold in thresholds:
        nums_trans_above.append(
            len([x for x in transients.values() if x.attrs["baseline_outlier_score"] > threshold])
        )

    print(f"Statistics for run {run_num}")
    print("Threshold\t| Outliers (%)")
    for threshold, num_trans_above in zip(thresholds, nums_trans_above):
        print(f"{threshold:.02f}\t\t| {num_trans_above/num_trans*100:.02f}")
