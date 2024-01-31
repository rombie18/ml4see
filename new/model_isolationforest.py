import logging
import os
from random import randint
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import warnings
from multiprocessing import Pool
from isotree import IsolationForest

from config import (
    DATA_FEATURES_DIRECTORY,
    DATA_PROCESSED_DIRECTORY,
    BLOCK_SIZE_X,
    BLOCK_SIZE_Y,
    BLOCK_OVERLAP,
)

# Decision boundary for outlier score classification (outlier if score >= OUTLIER_BOUNDARY, inlier if score < OUTLIER_BOUNDARY)
OUTLIER_BOUNDARY = 0.4

# Features to be used for outlier detection, any extreme or deviating values in these will likely result in outlier
FEATURES = [
    "pretrig_std",
    "trig_val",
    "posttrig_std",
    "posttrig_exp_fit_R2",
    "posttrig_exp_fit_λ",
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
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(
            f"The data features directory does not exist at {DATA_FEATURES_DIRECTORY}."
        )
        exit()
    if not os.path.exists(DATA_PROCESSED_DIRECTORY):
        logging.error(
            f"The processed data directory does not exist at {DATA_PROCESSED_DIRECTORY}."
        )
        exit()

    # If runs are provided as arguments, only verify the specified runs
    run_numbers = []
    if len(args.run_numbers) > 0:
        run_numbers = args.run_numbers
        logging.info(
            f"Runs argument present, only applying outlier rejection for: {run_numbers}"
        )
    else:
        for file in os.listdir(DATA_FEATURES_DIRECTORY):
            if file.endswith(".csv"):
                run_numbers.append(int(file[4:7]))
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    for run_number in run_numbers:
        logging.info(f"Processing run {run_number:03d}")

        # Combine labeled data with unlabeled extracted features
        logging.debug("Reading features from csv")
        df = pd.read_csv(
            os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
        )

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

        # Anotate dataframes with inliers/outlier type
        df_inliers = df[df["transient"].isin(inliers)]
        df_inliers.insert(loc=1, column="type", value="inlier")
        df_outliers = df[df["transient"].isin(outliers)]
        df_outliers.insert(loc=1, column="type", value="outlier")

        # If all transients get rejected at one position, interpolate lost data from neighboring positions
        logging.info("Interpolating missing data points from neighbors")
        df_inliers = interpolate_lost_data(inliers, df_inliers, df)

        # Merge inlier/outlier data into single frame
        df = pd.concat([df_inliers, df_outliers], ignore_index=True)
        df = df.sort_values(["transient"])

        # Save processed data to csv file
        logging.debug(f"Storing processed data in file run_{run_number:03d}.csv")
        df.to_csv(
            os.path.join(DATA_PROCESSED_DIRECTORY, f"run_{run_number:03d}.csv"),
            index=False,
        )

        logging.info(f"Successfully processed run {run_number:03d}")

    logging.info("Done!")


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
    block_x = df["x_um"].apply(
        lambda x: x // (block_size_x - overlap_x) * (block_size_x - overlap_x)
    )
    block_y = df["y_um"].apply(
        lambda y: y // (block_size_y - overlap_y) * (block_size_y - overlap_y)
    )

    df.insert(loc=5, column="block_x", value=block_x)
    df.insert(loc=6, column="block_y", value=block_y)

    # Group the DataFrame by the block_x and block_y columns
    blocks = df.groupby(["block_x", "block_y"])

    return blocks


def inject_points(df):
    manual_points = [
        {
            "transient": "manu_000000",
            "type": "",
            "x_lsb": "",
            "y_lsb": "",
            "x_um": "",
            "y_um": "",
            "block_x": "",
            "block_y": "",
            "trig_val": 0,
            "pretrig_min": 0,
            "posttrig_min": 0,
            "pretrig_max": 0,
            "posttrig_max": 0,
            "pretrig_std": 0,
            "posttrig_std": 0,
            "posttrig_exp_fit_R2": 0,
            "posttrig_exp_fit_N": 0,
            "posttrig_exp_fit_λ": 0,
            "posttrig_exp_fit_c": 0,
        },
        {
            "transient": "manu_000001",
            "type": "",
            "x_lsb": "",
            "y_lsb": "",
            "x_um": "",
            "y_um": "",
            "block_x": "",
            "block_y": "",
            "trig_val": 10,
            "pretrig_min": 0,
            "posttrig_min": 0,
            "pretrig_max": 0,
            "posttrig_max": 10,
            "pretrig_std": 0,
            "posttrig_std": 3000,
            "posttrig_exp_fit_R2": 1,
            "posttrig_exp_fit_N": 10,
            "posttrig_exp_fit_λ": 3000,
            "posttrig_exp_fit_c": 0,
        },
    ]

    for manual_point in manual_points:
        df = pd.concat([df, pd.DataFrame([manual_point])], ignore_index=True)

    return df


def isolation_forest(df: pd.DataFrame):
    # Ignore annoying warnings
    warnings.filterwarnings("ignore")

    # TODO move to pyod instead of isotree: https://pyod.readthedocs.io/
    # Set up Isolation Forest model
    clf = IsolationForest(
        ndim=1,
        sample_size=None,
        max_depth=8,
        ntrees=100,
        missing_action="fail",
        scoring_metric="adj_depth",
        n_jobs=-1,
        random_state=42,
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
            "type": "inlier",
            "x_lsb": "",
            "y_lsb": "",
            "x_um": x_um,
            "y_um": y_um,
            "block_x": "",
            "block_y": "",
            "trig_val": neighbors["trig_val"].mean(),
            "pretrig_min": neighbors["pretrig_min"].mean(),
            "posttrig_min": neighbors["posttrig_min"].mean(),
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


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
