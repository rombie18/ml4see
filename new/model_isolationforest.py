import logging
import os
from random import randint
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import warnings
import seaborn as sns
from isotree import IsolationForest
from multiprocessing import Pool

from config import (
    DATA_FEATURES_DIRECTORY,
    BLOCK_SIZE_X,
    BLOCK_SIZE_Y,
    BLOCK_OVERLAP,
)

# Decision boundary for outlier score classification (outlier if score >= OUTLIER_BOUNDARY, inlier if score < OUTLIER_BOUNDARY)
OUTLIER_BOUNDARY = 0.4

# Features to be used for outlier detection, any extreme or deviating values in these will likely result in outlier
FEATURES = [
    "pretrig_std",
    "posttrig_exp_fit_R2",
    "posttrig_exp_fit_N",
    "posttrig_exp_fit_λ",
    "posttrig_std",
]


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("model_isolationforest.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting Isolation Forest process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_number", type=int)
    args = parser.parse_args()
    run_number = args.run_number

    # Combine labeled data with unlabeled extracted features
    logging.debug("Reading features from csv")
    df = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))

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

    # Segment dataframe into blocks and apply processing to each block
    logging.info("Segmenting run into blocks")
    # TODO make block size dynamic based on run resolution
    blocks = segment_dataframe(df, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_OVERLAP)
    logging.info("Starting processing pipeline")
    blocks_filtered = blocks.apply(processing_pipeline)

    # Get transient ids of outliers and inliers
    outliers, inliers = [], []
    for block in blocks_filtered:
        local_outliers, local_inliers = block
        outliers.extend(local_outliers)
        inliers.extend(local_inliers)

    # Filter dataframe based on inliers-only
    df_filtered = df[df["transient"].isin(inliers)]

    # If all transients get rejected at one position, interpolate lost data from neighboring positions
    logging.info("Interpolating missing data points from neighbors")
    df_filtered = interpolate_lost_data(inliers, df_filtered, df)

    # TODO temp save processed transients to csv
    # df_filtered = df_filtered.sort_values(by=["x_um"])
    # df_filtered.to_csv(
    #     f"run_processed_{run_number:03d}.csv",
    #     index=False,
    # )

    # Plot heatmap
    logging.info("Plotting heatmap")
    plot(df, df_filtered, run_number)
    plot_λ(df, df_filtered, run_number)
    plot_3(df, df_filtered, run_number, 0)
    plot_4(df, df_filtered, run_number, 0)
    plot_3_λ(df, df_filtered, run_number, 0)
    plot_4_λ(df, df_filtered, run_number, 0)
    # plot_5(df, df_filtered, run_number)


def processing_pipeline(df):
    # Determine current block
    block_x, block_y = df["block_x"].iloc[0], df["block_y"].iloc[0]
    logging.debug(f"Processing block_x: {block_x}, block_y: {block_y}")

    # Inject manual outliers to prevent no-outlier situation
    dfi = inject_points(df)

    # Apply Isolation Forest model on selected features
    y_pred = isolation_forest(dfi[FEATURES])

    # Undo effect of manual outlier injection
    y_pred = y_pred[:-2]

    # Decide outliers and inliers based on score
    outlier_indices = [i for i, point in enumerate(y_pred) if point >= OUTLIER_BOUNDARY]
    inlier_indices = [i for i, point in enumerate(y_pred) if point < OUTLIER_BOUNDARY]

    # Get outlier ids in block
    outliers = df.iloc[outlier_indices]["transient"].tolist()
    inliers = df.iloc[inlier_indices]["transient"].tolist()

    return outliers, inliers


def segment_dataframe(df, block_size_x, block_size_y, overlap_percentage):
    # Calculate overlap in terms of absolute values
    overlap_x = (block_size_x * overlap_percentage) / 100
    overlap_y = (block_size_y * overlap_percentage) / 100

    # Create new columns for block_x and block_y with overlap
    df["block_x"] = df["x_um"].apply(
        lambda x: x // (block_size_x - overlap_x) * (block_size_x - overlap_x)
    )
    df["block_y"] = df["y_um"].apply(
        lambda y: y // (block_size_y - overlap_y) * (block_size_y - overlap_y)
    )

    # Group the DataFrame by the block_x and block_y columns
    blocks = df.groupby(["block_x", "block_y"])

    return blocks


def inject_points(df):
    manual_points = [
        {
            "transient": "manu_000000",
            "x_lsb": -1,
            "y_lsb": -1,
            "x_um": -1,
            "y_um": -1,
            "pretrig_max": 0,
            "posttrig_max": 0,
            "pretrig_std": 0,
            "posttrig_std": 0,
            "posttrig_exp_fit_N": 0,
            "posttrig_exp_fit_λ": 0,
            "posttrig_exp_fit_c": 0,
            "posttrig_exp_fit_R2": 0,
        },
        {
            "transient": "manu_000001",
            "x_lsb": -1,
            "y_lsb": -1,
            "x_um": -1,
            "y_um": -1,
            "pretrig_max": 0,
            "posttrig_max": 10,
            "pretrig_std": 0,
            "posttrig_std": 3000,
            "posttrig_exp_fit_N": 10,
            "posttrig_exp_fit_λ": 3000,
            "posttrig_exp_fit_c": 0,
            "posttrig_exp_fit_R2": 1,
        },
    ]

    for manual_point in manual_points:
        df = pd.concat([df, pd.DataFrame([manual_point])], ignore_index=True)

    return df


def isolation_forest(df: pd.DataFrame):
    # Ignore annoying warnings
    warnings.filterwarnings("ignore")

    # Set up Isolation Forest model
    clf = IsolationForest(
        sample_size=None,
        ntrees=100,
        ndim=2,
        missing_action="fail",
        scoring_metric="adj_depth",
        n_jobs=-1,
    )

    # Fit data on model and predict
    y_pred = clf.fit_predict(df)

    # Reactivate warnings
    warnings.filterwarnings("default")

    return y_pred


def interpolate_lost_data(inliers, df, df_original):
    # When all points on same position are rejected as outlier, no data is available
    # --> Interpolate lost data points from neighboring points

    positions = df_original.groupby(["x_um", "y_um"])["transient"]

    # Extract unique x and y values from the grouped object
    xy_groups = [position for (position, _) in positions]
    x_groups, y_groups = zip(*xy_groups)
    x_values = np.unique(x_groups)
    y_values = np.unique(y_groups)

    # Calculate step size for x and y
    step_x = np.diff(x_values)[0] * 1.5 if len(x_values) > 1 else 0
    step_y = np.diff(y_values)[0] * 1.5 if len(y_values) > 1 else 0

    # Interpolate missing data from neighbors in parallel
    with Pool() as pool:
        args = [
            (inliers, df, position, group, step_x, step_y)
            for (position, group) in positions
        ]
        points_list = pool.map(do_interpolate_args, args)
        points_list = [x for x in points_list if x is not None]

        df = pd.concat([df, pd.DataFrame(points_list)], ignore_index=True)

    return df


def do_interpolate_args(args):
    inliers, df, position, group, step_x, step_y = args
    return do_interpolate(inliers, df, position, group, step_x, step_y)


def do_interpolate(inliers, df, position, group, step_x, step_y):
    logging.debug(f"Interpolating {position}")

    x_um, y_um = position
    transients = group.values
    # If no transient at position is inlier i.e. if all transients are outliers, do interpolation
    if not any(transient in inliers for transient in transients):
        select_neighbors = (
            (df["x_um"] < x_um + step_x)
            & (df["x_um"] > x_um - step_x)
            & (df["y_um"] < y_um + step_y)
            & (df["y_um"] > y_um - step_y)
        )

        neighbors = df.loc[select_neighbors]

        point = {
            "transient": f"intr_{randint(0, 999999):06d}",
            "x_lsb": None,
            "y_lsb": None,
            "x_um": x_um,
            "y_um": y_um,
            "pretrig_max": neighbors["pretrig_max"].mean(),
            "posttrig_max": neighbors["posttrig_max"].mean(),
            "pretrig_std": neighbors["pretrig_std"].mean(),
            "posttrig_std": neighbors["posttrig_std"].mean(),
            "posttrig_exp_fit_N": neighbors["posttrig_exp_fit_N"].mean(),
            "posttrig_exp_fit_λ": neighbors["posttrig_exp_fit_λ"].mean(),
            "posttrig_exp_fit_c": neighbors["posttrig_exp_fit_c"].mean(),
            "posttrig_exp_fit_R2": neighbors["posttrig_exp_fit_R2"].mean(),
        }

        return point


def plot(df, df_filtered, run_number):
    """Heatmap of fitted maximum frequency deviation (N) in function to X and Y position"""
    # TODO replace fitted frequency deviation by actual maximum deviation

    df_filtered_grouped = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()
    )
    heatmap_filtered = df_filtered_grouped.pivot(
        index="x_um", columns="y_um", values="posttrig_max"
    ).transpose()

    df_grouped = df.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()
    heatmap = df_grouped.pivot(
        index="x_um", columns="y_um", values="posttrig_max"
    ).transpose()

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.tight_layout(w_pad=15)

    h1 = sns.heatmap(
        heatmap_filtered,
        square=True,
        cbar_kws={"label": "SEFT peak deviation (ppm)", "fraction": 0.046, "pad": 0.04},
        cmap="jet",
        ax=axs[0],
    )
    h2 = sns.heatmap(
        heatmap,
        square=True,
        cbar_kws={"label": "SEFT peak deviation (ppm)", "fraction": 0.046, "pad": 0.04},
        cmap="jet",
        ax=axs[1],
    )

    # TODO ticks on axis in steps of 15 µm
    axs[0].set_title(f"With outliers filtered (run_{run_number:03d})")
    axs[0].set_xlabel("Beam position X (µm)")
    axs[0].set_ylabel("Beam position Y (µm)")
    axs[0].set_xticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[0].get_xticklabels()]
    )
    axs[0].set_yticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[0].get_yticklabels()]
    )
    axs[0].invert_yaxis()

    axs[1].set_title(f"No filtering (run_{run_number:03d})")
    axs[1].set_xlabel("Beam position X (µm)")
    axs[1].set_ylabel("Beam position Y (µm)")
    axs[1].set_xticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[1].get_xticklabels()]
    )
    axs[1].set_yticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[1].get_yticklabels()]
    )
    axs[1].invert_yaxis()

    # Use color scale of filtered heatmap for unfiltered to prevent extreme color changes
    h2.collections[0].set_clim(h1.collections[0].get_clim())

    plt.savefig(f"plots/heatmap_1.png", bbox_inches="tight")
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

    h1 = sns.heatmap(
        heatmap_filtered,
        square=True,
        cbar_kws={
            "label": "Exponential decay constant (1/s)",
            "fraction": 0.046,
            "pad": 0.04,
        },
        cmap="jet",
        ax=axs[0],
        vmax=1000
    )
    h2 = sns.heatmap(
        heatmap,
        square=True,
        cbar_kws={
            "label": "Exponential decay constant (1/s)",
            "fraction": 0.046,
            "pad": 0.04,
        },
        cmap="jet",
        ax=axs[1],
    )

    axs[0].set_title(f"With outliers filtered (run_{run_number:03d})")
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("Y position (µm)")
    axs[0].set_xticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[0].get_xticklabels()]
    )
    axs[0].set_yticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[0].get_yticklabels()]
    )
    axs[0].invert_yaxis()

    axs[1].set_title(f"No filtering (run_{run_number:03d})")
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Y position (µm)")
    axs[1].set_xticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[1].get_xticklabels()]
    )
    axs[1].set_yticklabels(
        ["{:.0f}".format(float(t.get_text())) for t in axs[1].get_yticklabels()]
    )
    axs[1].invert_yaxis()

    # Use color scale of filtered heatmap for unfiltered to prevent extreme color changes
    h2.collections[0].set_clim(h1.collections[0].get_clim())

    plt.savefig(f"plots/heatmap_1_λ.png", bbox_inches="tight")
    plt.close()


def plot_2(df, df_filtered, run_number):
    """3D visual of SEFT deviation"""

    df_heatmap = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()
    )

    x, y, z = df_heatmap["x_um"], df_heatmap["y_um"], df_heatmap["posttrig_max"]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf = ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False, cmap="jet")

    plt.savefig(f"plots/heatmap_2.png", bbox_inches="tight")
    plt.close()


def plot_3(df, df_filtered, run_number, slice_y=None):
    """Cross section in X direction"""

    if slice_y != None:
        df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
        df = df[df["y_um"].round(0) == slice_y]

    df_filtered = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()
    )

    df = df.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["x_um"], df_filtered["posttrig_max"], marker=".")
    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[0].set_xlabel("X position (µm)")
    axs[0].set_ylabel("SEFT peak deviation (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["x_um"], df["posttrig_max"], marker=".")
    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("SEFT peak deviation (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(f"plots/heatmap_3.png", bbox_inches="tight")
    plt.close()


def plot_4(df, df_filtered, run_number, slice_x=None):
    """Cross section in Y direction"""

    if slice_x != None:
        df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
        df = df[df["x_um"].round(0) == slice_x]

    df_filtered = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()
    )

    df = df.groupby(["x_um", "y_um"])["posttrig_max"].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["y_um"], df_filtered["posttrig_max"], marker=".")
    axs[0].set_title(
        f"SEFT peak frequency, with outliers filtered \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[0].set_xlabel("Y position (µm)")
    axs[0].set_ylabel("SEFT peak deviation (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["y_um"], df["posttrig_max"], marker=".")
    axs[1].set_title(
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("Y position (µm)")
    axs[1].set_ylabel("SEFT peak deviation (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(f"plots/heatmap_4.png", bbox_inches="tight")
    plt.close()


def plot_3_λ(df, df_filtered, run_number, slice_y=None):
    """Cross section in X direction"""

    if slice_y != None:
        df_filtered = df_filtered[df_filtered["y_um"].round(0) == slice_y]
        df = df[df["y_um"].round(0) == slice_y]

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
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; Y = {slice_y} µm"
    )
    axs[1].set_xlabel("X position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(f"plots/heatmap_3_λ.png", bbox_inches="tight")
    plt.close()


def plot_4_λ(df, df_filtered, run_number, slice_x=None):
    """Cross section in Y direction"""

    if slice_x != None:
        df_filtered = df_filtered[df_filtered["x_um"].round(0) == slice_x]
        df = df[df["x_um"].round(0) == slice_x]

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
        f"SEFT peak frequency, no filtering \n Run {run_number:03d}; X = {slice_x} µm"
    )
    axs[1].set_xlabel("Y position (µm)")
    axs[1].set_ylabel("Exponential decay constant (1/s)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(f"plots/heatmap_4_λ.png", bbox_inches="tight")
    plt.close()


def plot_5(df, df_filtered, run_number):
    """Sensitivity loss in function of time"""

    df_filtered = df_filtered.reset_index()
    df = df.reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=5)

    axs[0].scatter(df_filtered["transient"], df_filtered["posttrig_max"], marker=".")
    axs[0].set_title(f"With outliers filtered (run_{run_number:03d})")
    axs[0].set_xlabel("Transient ID")
    axs[0].set_ylabel("SEFT peak deviation (ppm)")
    axs[0].set_axisbelow(True)
    axs[0].grid(color="lightgray")

    axs[1].scatter(df["transient"], df["posttrig_max"], marker=".")
    axs[1].set_title(f"No filtering (run_{run_number:03d})")
    axs[1].set_xlabel("Transient ID")
    axs[1].set_ylabel("SEFT peak deviation (ppm)")
    axs[1].set_axisbelow(True)
    axs[1].grid(color="lightgray")

    plt.savefig(f"plots/heatmap_5.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
