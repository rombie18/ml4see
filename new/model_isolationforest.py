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

from config import DATA_FEATURES_DIRECTORY, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_OVERLAP

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

    # Segment dataframe into blocks and apply processing to each block
    logging.info("Segmenting run into blocks")
    #TODO make block size dynamic based on run resolution
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

    # Plot heatmap
    logging.info("Plotting heatmap")
    plot_3(df, df_filtered)


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
            "pretrig_std": 0,
            "posttrig_std": 3000,
            "posttrig_exp_fit_N": 20000,
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


def plot(df, df_filtered):
    df_heatmap = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_N"].mean().reset_index()
    )
    heatmap_data_filtered = df_heatmap.pivot(
        index="x_um", columns="y_um", values="posttrig_exp_fit_N"
    )

    df_heatmap = df.groupby(["x_um", "y_um"])["posttrig_exp_fit_N"].mean().reset_index()
    heatmap_data = df_heatmap.pivot(
        index="x_um", columns="y_um", values="posttrig_exp_fit_N"
    )

    # Mark missing data with vibrant color
    sns.set_style(rc={"axes.facecolor": "limegreen"})

    fig, axs = plt.subplots(1, 2, figsize=(30, 20))
    fig.tight_layout(pad=3.5, w_pad=10)
    h1 = sns.heatmap(heatmap_data_filtered, ax=axs[0])
    h2 = sns.heatmap(heatmap_data, ax=axs[1])
    axs[0].set_title("Outliers filtered")
    axs[1].set_title("No filtering")

    # Use color scale from filtered plot
    h2.collections[0].set_clim(h1.collections[0].get_clim())

    plt.savefig(f"plots/heatmap_v2.png", bbox_inches="tight")
    plt.close()

def plot_2(df, df_filtered):
    from matplotlib import cbook, cm
    from matplotlib.colors import LightSource

    df_heatmap = (
        df_filtered.groupby(["x_um", "y_um"])["posttrig_exp_fit_N"].mean().reset_index()
    )

    x, y, z = df_heatmap["x_um"], df_heatmap["y_um"], df_heatmap["posttrig_exp_fit_N"]
    
    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    surf = ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    plt.savefig(f"plots/heatmap_v3.png", bbox_inches="tight")
    plt.close()
    
def plot_3(df, df_filtered):
   

    df_heatmap = (
        df_filtered.groupby(["x_um"])["posttrig_exp_fit_N"].mean().reset_index()
    )

    plt.scatter(df_heatmap["x_um"], df_heatmap["posttrig_exp_fit_N"])

    plt.savefig(f"plots/heatmap_v4.png", bbox_inches="tight")
    plt.close()

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
    outlier_indices = [i for i, point in enumerate(y_pred) if point >= 0.5]
    inlier_indices = [i for i, point in enumerate(y_pred) if point < 0.5]

    # Get outlier ids in block
    outliers = df.iloc[outlier_indices]["transient"].tolist()
    inliers = df.iloc[inlier_indices]["transient"].tolist()

    return outliers, inliers


def interpolate_lost_data(inliers, df, df_original: pd.DataFrame):
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

    #FIXME pool is not using all available resources
    with Pool() as pool:
        args = [
            (inliers, df, position, group, step_x, step_y)
            for (position, group) in positions
        ]
        points_list = pool.map(do_interpolate, args)
        points_list = [x for x in points_list if x is not None]
        
        df = pd.concat([df, pd.DataFrame(points_list)], ignore_index=True)

    return df


def do_interpolate(args):
    (inliers, df, position, group, step_x, step_y) = args

    logging.debug(f"Interpolating {position}")

    x_um, y_um = position
    transients = group.values
    # If no transient at position is inlier i.e. if all transients are outliers, do interpolation
    if not any(transient in inliers for transient in transients):
        select_neighbors = (
            (df["x_um"] < x_um + step_x) & (df["x_um"] > x_um - step_x) &
            (df["y_um"] < y_um + step_y) & (df["y_um"] > y_um - step_y)
        )

        neighbors = df.loc[select_neighbors]
        
        point = {
            "transient": f"intr_{randint(0, 999999):03d}",
            "x_lsb": None,
            "y_lsb": None,
            "x_um": x_um,
            "y_um": y_um,
            "pretrig_std": neighbors["pretrig_std"].mean(),
            "posttrig_std": neighbors["posttrig_std"].mean(),
            "posttrig_exp_fit_N": neighbors["posttrig_exp_fit_N"].mean(),
            "posttrig_exp_fit_λ": neighbors["posttrig_exp_fit_λ"].mean(),
            "posttrig_exp_fit_c": neighbors["posttrig_exp_fit_c"].mean(),
            "posttrig_exp_fit_R2": neighbors["posttrig_exp_fit_R2"].mean(),
        }

        return point


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
