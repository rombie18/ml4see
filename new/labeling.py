import os
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import (
    DATA_STRUCTURED_DIRECTORY,
    DATA_LABELED_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
)
from utils import generatePlotTitle

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# TODO cleanup code
try:
    df_labeled = pd.read_csv(
        os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv")
    )
    labeled_transients = df_labeled["transient"].tolist()
    print(
        f"Detected already labeled transients. Skipping {len(labeled_transients)+1} transients."
    )
except:
    df_labeled = pd.DataFrame()
    labeled_transients = []

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

    labels = []
    for i, tran_name in enumerate(transients.keys()):
        if tran_name in labeled_transients:
            continue

        tran_data = np.array(h5file["sdr_data"]["all"][tran_name])

        # Convert transient data to Pandas dataframe
        df = pd.DataFrame.from_dict(
            {"transient": tran_name, "time": time_data, "frequency": tran_data}
        )

        # Apply moving average filter
        window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
        tran_data_ds = np.convolve(tran_data, window, mode="valid")

        # Adjust time data to match length of convoluted output
        time_data_ds = time_data[WINDOW_SIZE - 1 :]

        # Downsample time and frequency data
        time_data_ds = time_data_ds[::DOWNSAMPLE_FACTOR]
        tran_data_ds = tran_data_ds[::DOWNSAMPLE_FACTOR]

        # Plot result
        fig, ax = plt.subplots()
        ax.plot(time_data, tran_data, ".", label="Original raw data")
        ax.axvline(x=0, color="lime")
        ax.plot(
            time_data_ds,
            tran_data_ds,
            "-",
            label="Filtered and downsampled",
            linewidth=2,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        generatePlotTitle(ax, f"Transient {tran_name} - downsampled", run_number)

        plt.savefig(f"plots/labeling.png", bbox_inches="tight")

        prompt = input(
            f"Please choose a type for this transient. To stop labeling, enter STOP. To scrap last label, enter UNDO. ({i+1}/{len(transients)}) > "
        )
        if prompt == "STOP":
            break
        elif prompt == "UNDO":
            labels.pop()
        else:
            labels.append({"transient": tran_name, "type": prompt})

        plt.close()

    df = pd.DataFrame(labels)
    df = pd.concat([df_labeled, df])
    df = df.sort_values("transient")
    df.to_csv(
        os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"), index=False
    )
