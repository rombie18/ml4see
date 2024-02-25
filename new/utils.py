import os
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config import RUN_LOGBOOK_PATH, RUN_DUTPICS_DIRECTORY


# TODO use h5 run meta data instead of combining with logbook.xlsx
def generatePlotTitle(ax: matplotlib.axes.Axes, title, run_number):
    plt.suptitle(title, y=1)

    try:
        # Try to add descriptive fields to plot subtitle such as: scan_type (stationary, xy), etc
        df = pd.read_excel(RUN_LOGBOOK_PATH)
        scan_type = df.loc[df["run_id"] == run_number, "scan_type"].iloc[0]
        run_comment = df.loc[df["run_id"] == run_number, "run_comment"].iloc[0]
        dut_position = df.loc[df["run_id"] == run_number, "dut_position"].iloc[0]
        dut_name = df.loc[df["run_id"] == run_number, "dut_name"].iloc[0]

        if scan_type == "S":
            scan_type = "S (stationary)"
        elif scan_type == "X":
            scan_type = "X (horizontal scan)"
        elif scan_type == "Y":
            scan_type = "Y (vertical scan)"
        elif scan_type == "XY":
            scan_type = "XY (2D scan)"

        ax.set_title(
            f"run_{run_number:03d}, {scan_type}, DUT={dut_name} \n {run_comment}",
            fontsize="small",
        )

        imagebox_full = OffsetImage(
            mpimg.imread(os.path.join(RUN_DUTPICS_DIRECTORY, f"{dut_position}_5x.tif")),
            zoom=0.2,
        )
        ab_full = AnnotationBbox(
            imagebox_full,
            (1.25, 0.70),
            xycoords="axes fraction",
            boxcoords="offset points",
        )
        ax.add_artist(ab_full)

        try:
            imagebox_detail = OffsetImage(
                mpimg.imread(
                    os.path.join(RUN_DUTPICS_DIRECTORY, f"{dut_position}_50x.tif")
                ),
                zoom=0.2,
            )
            ab_detail = AnnotationBbox(
                imagebox_detail,
                (1.25, 0.30),
                xycoords="axes fraction",
                boxcoords="offset points",
            )
            ax.add_artist(ab_detail)
        except:
            pass

    except:
        # Fall back to run number only on fail
        ax.set_title(f"run_{run_number:03d}", fontsize="small")


def require_processing_stage(h5file, stage_req, strict=False):
    """Raises an exception when a file does not undergone the required level of processing."""
    file_stage = int(h5file["meta"].attrs["processing_stage"])

    if not strict:
        if file_stage < stage_req:
            raise ValueError(
                f"HDF file is not at required level of processing. Found: {file_stage}; Required: >={stage_req}."
            )
    else:
        if file_stage != stage_req:
            raise ValueError(
                f"HDF file is not at required level of processing. Found: {file_stage}; Required: =={stage_req}."
            )


def moving_average(tran_data, time_data, downsample_factor, window_size):
    # Apply moving average filter
    window = np.ones(window_size) / window_size
    tran_data = np.convolve(tran_data, window, mode="valid")

    # Adjust time data to match length of convoluted output
    time_data = time_data[(len(window) - 1) :]

    # Downsample time and frequency data
    time_data = time_data[::downsample_factor]
    tran_data = tran_data[::downsample_factor]

    return tran_data, time_data


def exponential_decay(t, N, λ, c):
    return (N - c) * np.exp(-λ * t) + c


def chunker(seq, size):
    """Helper method for iterating over chunks of a list"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))
