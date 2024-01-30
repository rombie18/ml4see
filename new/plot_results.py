import logging
import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import seaborn as sns

from config import (
    DATA_PROCESSED_DIRECTORY,
)


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("plot_results.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting plotting process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_PROCESSED_DIRECTORY):
        logging.error(
            f"The processed data directory does not exist at {DATA_PROCESSED_DIRECTORY}."
        )
        exit()

    # If runs are provided as arguments, only verify the specified runs
    run_numbers = []
    if len(args.run_numbers) > 0:
        run_numbers = args.run_numbers
        logging.info(f"Runs argument present, only generating plots for: {run_numbers}")
    else:
        for file in os.listdir(DATA_PROCESSED_DIRECTORY):
            if file.endswith(".csv"):
                run_numbers.append(int(file[4:7]))
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    # TODO cleanup
    runs_data = []

    for run_number in run_numbers:
        logging.info(f"Generating plots for run {run_number:03d}")

        # Combine labeled data with unlabeled extracted features
        logging.debug("Reading processed data from csv")
        df = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIRECTORY, f"run_{run_number:03d}.csv")
        )

        # Only retain specified area of interest
        # TODO make this cleaner and maybe seperate function?
        x_range = (-215, 215)
        y_range = (-215, 215)
        if x_range != None:
            logging.debug(f"x_range present, limiting plot to {y_range}")
            df = df[(df["x_um"] >= x_range[0]) & (df["x_um"] <= x_range[1])]

        if y_range != None:
            logging.debug(f"y_range present, limiting plot to {y_range}")
            df = df[(df["y_um"] >= y_range[0]) & (df["y_um"] <= y_range[1])]

        # Split data into in/outliers
        df_inliers = df.loc[df["type"] == "inlier"]
        df_outliers = df.loc[df["type"] == "outlier"]

        runs_data.append((df, df_inliers, run_number))

        # Plot heatmap
        logging.info("Plotting heatmap")
        if not os.path.exists(f"plots/{run_number:03d}"):
            os.mkdir(f"plots/{run_number:03d}")

        plot(df, df_inliers, run_number)
        plot_λ(df, df_inliers, run_number)

        # TODO manual parameter selections based on run numbers, improve this
        if run_number in [11, 12]:
            plot_3(df, df_inliers, run_number, None)
            plot_3_λ(df, df_inliers, run_number, None)
            plot_4(df, df_inliers, run_number, None)
            plot_4_λ(df, df_inliers, run_number, None)
        else:
            plot_3(df, df_inliers, run_number, 0)
            plot_3_λ(df, df_inliers, run_number, 0)
            plot_4(df, df_inliers, run_number, 0)
            plot_4_λ(df, df_inliers, run_number, 0)

        # Display elementary metrics
        outlier_percentage = (
            len(df_outliers) / (len(df_inliers) + len(df_outliers)) * 100
        )
        print(f"----- RUN {run_number:03d} -----")
        print(f"Number of outliers: {len(df_outliers)}")
        print(f"Number of inliers: {len(df_inliers)}")
        print(f"Outlier percentage: {outlier_percentage:.2f}%")
        print("-------------------")

        logging.info(f"Successfully processed run {run_number:03d}")

    # Combined plots if multiple runs specified
    if len(run_numbers) > 1:
        if run_number in [11, 12]:
            plot_3_multiple(runs_data, None)
            plot_3_λ_multiple(runs_data, None)
            plot_4_multiple(runs_data, None)
            plot_4_λ_multiple(runs_data, None)
        else:
            plot_3_multiple(runs_data, 0)
            plot_3_λ_multiple(runs_data, 0)
            plot_4_multiple(runs_data, 0)
            plot_4_λ_multiple(runs_data, 0)

    logging.info("Done!")


def plot(df, df_filtered, run_number):
    """Heatmap of fitted maximum frequency deviation (trig_val) in function to X and Y position"""

    df_filtered_grouped = (
        df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()
    )
    heatmap_filtered = df_filtered_grouped.pivot(
        index="x_um", columns="y_um", values="trig_val"
    ).transpose()

    df_grouped = df.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()
    heatmap = df_grouped.pivot(
        index="x_um", columns="y_um", values="trig_val"
    ).transpose()

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.tight_layout(w_pad=15)

    im_extent = (
        np.min(df_filtered["x_um"]),
        np.max(df_filtered["x_um"]),
        np.min(df_filtered["y_um"]),
        np.max(df_filtered["y_um"]),
    )

    h1 = axs[0].imshow(
        heatmap_filtered,
        origin="lower",
        extent=im_extent,
        cmap="jet",
    )
    cbar = plt.colorbar(h1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label("|SEFT peak deviation| (ppm)")

    h2 = axs[1].imshow(
        heatmap,
        origin="lower",
        extent=im_extent,
        cmap="jet",
    )
    cbar = plt.colorbar(h2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("|SEFT peak deviation| (ppm)")

    axs[0].set_title(f"With outliers filtered (run_{run_number:03d})")
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Y position (µm)")

    axs[1].set_title(f"No filtering (run_{run_number:03d})")
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Y position (µm)")

    # Use color scale of filtered heatmap for unfiltered to prevent extreme color changes
    h2.set_clim(h1.get_clim())

    plt.savefig(
        f"plots/{run_number:03d}/heatmap__frequency_deviation.png", bbox_inches="tight"
    )
    plt.close()


def plot_λ(df: pd.DataFrame, df_filtered: pd.DataFrame, run_number: int):
    """Heatmap of exponential decay constant (λ) in function to X and Y position"""

    df_filtered_grouped = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()
    )
    heatmap_filtered = df_filtered_grouped.pivot(
        index="x_um", columns="y_um", values="posttrig_exp_fit_λ"
    ).transpose()

    df_grouped = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()
    heatmap = df_grouped.pivot(
        index="x_um", columns="y_um", values="posttrig_exp_fit_λ"
    ).transpose()

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.tight_layout(w_pad=15)

    im_extent = (
        np.min(df_filtered["x_um"]),
        np.max(df_filtered["x_um"]),
        np.min(df_filtered["y_um"]),
        np.max(df_filtered["y_um"]),
    )

    h1 = axs[0].imshow(
        heatmap_filtered,
        origin="lower",
        extent=im_extent,
        cmap="jet",
    )
    cbar = plt.colorbar(h1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label("Exponential decay constant (1/s)")

    h2 = axs[1].imshow(
        heatmap,
        origin="lower",
        extent=im_extent,
        cmap="jet",
    )
    cbar = plt.colorbar(h2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Exponential decay constant (1/s)")

    axs[0].set_title(f"With outliers filtered (run_{run_number:03d})")
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Y position (µm)")

    axs[1].set_title(f"No filtering (run_{run_number:03d})")
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Y position (µm)")

    # Use color scale of filtered heatmap for unfiltered to prevent extreme color changes
    # TODO manual adjustment
    if run_number == 26:
        h1.set_clim(h1.get_clim()[0], 750)

    h2.set_clim(h1.get_clim())

    plt.savefig(
        f"plots/{run_number:03d}/heatmap__exponential_decay_constant.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_2(df, df_filtered, run_number):
    """3D visual of SEFT deviation"""

    df_heatmap = df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

    x, y, z = df_heatmap["x_um"], df_heatmap["y_um"], df_heatmap["trig_val"]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf = ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False, cmap="jet")

    plt.savefig(
        f"plots/{run_number:03d}/3Dmap__frequency_deviation.png", bbox_inches="tight"
    )
    plt.close()


def plot_3(df, df_filtered, run_number, slice_y=None):
    """Cross section in X direction"""

    if slice_y != None:
        df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
        df = df[df["y_um"].round(0) == slice_y]
    else:
        slice_y = "auto"

    df_filtered = df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

    df = df.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["x_um"], df_filtered["trig_val"], marker=".")
    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["x_um"], df["trig_val"], marker=".")
    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(
        f"plots/{run_number:03d}/cross_section__Y={slice_y}__frequency_deviation.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_4(df, df_filtered, run_number, slice_x=None):
    """Cross section in Y direction"""

    if slice_x != None:
        df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
        df = df[df["x_um"].round(0) == slice_x]
    else:
        slice_x = "auto"

    df_filtered = df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

    df = df.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["y_um"], df_filtered["trig_val"], marker=".")
    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[0].set_xlabel("Y position (µm)")
    axs[0].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["y_um"], df["trig_val"], marker=".")
    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("Y position (µm)")
    axs[1].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(
        f"plots/{run_number:03d}/cross_section__X={slice_x}__frequency_deviation.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_3_λ(df, df_filtered, run_number, slice_y=None):
    """Cross section in X direction"""

    if slice_y != None:
        df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
        df = df[df["y_um"].round(0) == slice_y]
    else:
        slice_y = "auto"

    df_filtered = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()
    )

    df = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["x_um"], df_filtered["posttrig_exp_fit_λ"], marker=".")
    axs[0].set_title(
        f"SEFT exponential decay constant, with outliers filtered \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Exponential decay constant (1/s)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["x_um"], df["posttrig_exp_fit_λ"], marker=".")
    axs[1].set_title(
        f"SEFT exponential decay constant, no filtering \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    # TODO temp manual override
    if run_number == 26:
        axs[0].set_ylim(top=750)
        axs[1].set_ylim(top=750)

    plt.savefig(
        f"plots/{run_number:03d}/cross_section__Y={slice_y}__exponential_decay_constant.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_4_λ(df, df_filtered, run_number, slice_x=None):
    """Cross section in Y direction"""

    if slice_x != None:
        df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
        df = df[df["x_um"].round(0) == slice_x]
    else:
        slice_x = "auto"

    df_filtered = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()
    )

    df = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["y_um"], df_filtered["posttrig_exp_fit_λ"], marker=".")
    axs[0].set_title(
        f"SEFT exponential decay constant, with outliers filtered \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[0].set_xlabel("Y position (µm)")
    axs[0].set_ylabel("Exponential decay constant (1/s)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["y_um"], df["posttrig_exp_fit_λ"], marker=".")
    axs[1].set_title(
        f"SEFT exponential decay constant, no filtering \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("Y position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(
        f"plots/{run_number:03d}/cross_section__X={slice_x}__exponential_decay_constant.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_3_multiple(runs_data, slice_y_glob=None):
    """Cross section in X direction"""

    run_numbers = list(map(lambda x: x[2], runs_data))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    run_name = ["Inductor A", "Inductor B"]
    run_name_i = 0

    for run_data in runs_data:
        df, df_filtered, run_number = run_data

        slice_y = slice_y_glob
        if slice_y != None:
            df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
            df = df[df["y_um"].round(0) == slice_y]
        else:
            slice_y = "auto"

        df_filtered = (
            df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()
        )

        df = df.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

        axs[0].scatter(
            df_filtered["x_um"],
            df_filtered["trig_val"],
            marker=".",
            label=run_name[run_name_i],
        )
        axs[1].scatter(
            df["x_um"],
            df["trig_val"],
            marker=".",
            label=run_name[run_name_i],
        )

        run_name_i = run_name_i + 1

    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {', '.join([str(run_number) for run_number in run_numbers])}; Y = {slice_y} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")
    axs[0].legend(loc="lower center")

    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Runs {', '.join([str(run_number) for run_number in run_numbers])}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")
    axs[1].legend(loc="lower center")

    plt.savefig(
        f"plots/{'-'.join([str(run_number) for run_number in run_numbers])}__cross_section__Y={slice_y}__frequency_deviation.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_4_multiple(runs_data, slice_x_glob=None):
    """Cross section in Y direction"""

    run_numbers = list(map(lambda x: x[2], runs_data))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    for run_data in runs_data:
        df, df_filtered, run_number = run_data

        slice_x = slice_x_glob
        if slice_x != None:
            df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
            df = df[df["x_um"].round(0) == slice_x]
        else:
            slice_x = "auto"

        df_filtered = (
            df_filtered.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()
        )

        df = df.groupby(["x_um", "y_um"])["trig_val"].mean().reset_index()

        axs[0].scatter(
            df_filtered["y_um"],
            df_filtered["trig_val"],
            marker=".",
            label=f"run {run_number:03d}",
        )
        axs[1].scatter(
            df["y_um"],
            df["trig_val"],
            marker=".",
            label=f"run {run_number:03d}",
        )

    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {', '.join([str(run_number) for run_number in run_numbers])}; X = {slice_x} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")
    axs[0].legend()

    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Runs {', '.join([str(run_number) for run_number in run_numbers])}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("|SEFT peak deviation| (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")
    axs[1].legend()

    plt.savefig(
        f"plots/{'-'.join([str(run_number) for run_number in run_numbers])}__cross_section__X={slice_x}__frequency_deviation.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_3_λ_multiple(runs_data, slice_y_glob=None):
    """Cross section in X direction"""

    run_numbers = list(map(lambda x: x[2], runs_data))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    run_name = ["Inductor A", "Inductor B"]
    run_name_i = 0

    for run_data in runs_data:
        df, df_filtered, run_number = run_data

        slice_y = slice_y_glob
        if slice_y != None:
            df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
            df = df[df["y_um"].round(0) == slice_y]
        else:
            slice_y = "auto"

        df_filtered = (
            df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"]
            .mean()
            .reset_index()
        )

        df = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()

        axs[0].scatter(
            df_filtered["x_um"],
            df_filtered["posttrig_exp_fit_λ"],
            marker=".",
            label=run_name[run_name_i],
        )
        axs[1].scatter(
            df["x_um"],
            df["posttrig_exp_fit_λ"],
            marker=".",
            label=run_name[run_name_i],
        )

        run_name_i = run_name_i + 1

    axs[0].set_title(
        f"SEFT exponential decay constant, with outliers filtered \n Run {', '.join([str(run_number) for run_number in run_numbers])}; Y = {slice_y} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Exponential decay constant (1/s)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")
    axs[0].legend(loc="lower center")

    axs[1].set_title(
        f"SEFT exponential decay constant, no filtering \n Runs {', '.join([str(run_number) for run_number in run_numbers])}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")
    axs[1].legend(loc="lower center")

    axs[0].set_ylim(top=790, bottom=-50)
    axs[1].set_ylim(top=790, bottom=-50)

    plt.savefig(
        f"plots/{'-'.join([str(run_number) for run_number in run_numbers])}__cross_section__Y={slice_y}__exponential_decay_constant.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_4_λ_multiple(runs_data, slice_x_glob=None):
    """Cross section in Y direction"""

    run_numbers = list(map(lambda x: x[2], runs_data))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    for run_data in runs_data:
        df, df_filtered, run_number = run_data

        slice_x = slice_x_glob
        if slice_x != None:
            df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
            df = df[df["x_um"].round(0) == slice_x]
        else:
            slice_x = "auto"

        df_filtered = (
            df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"]
            .mean()
            .reset_index()
        )

        df = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_λ"].mean().reset_index()

        axs[0].scatter(
            df_filtered["y_um"],
            df_filtered["posttrig_exp_fit_λ"],
            marker=".",
            label=f"run {run_number:03d}",
        )
        axs[1].scatter(
            df["y_um"],
            df["posttrig_exp_fit_λ"],
            marker=".",
            label=f"run {run_number:03d}",
        )

    axs[0].set_title(
        f"SEFT exponential decay constant, with outliers filtered \n Run {', '.join([str(run_number) for run_number in run_numbers])}; X = {slice_x} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Exponential decay constant (1/s)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")
    axs[0].legend()

    axs[1].set_title(
        f"SEFT exponential decay constant, no filtering \n Runs {', '.join([str(run_number) for run_number in run_numbers])}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")
    axs[1].legend()

    plt.savefig(
        f"plots/{'-'.join([str(run_number) for run_number in run_numbers])}__cross_section__X={slice_x}__exponential_decay_constant.png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
