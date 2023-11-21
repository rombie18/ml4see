import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import warnings
import seaborn as sns
from isotree import IsolationForest

from config import DATA_FEATURES_DIRECTORY

FEATURE_1 = "pretrig_std"
FEATURE_2 = "posttrig_exp_fit_R2"
FEATURE_3 = "posttrig_exp_fit_N"

BLOCK_SIZE_X = 100
BLOCK_SIZE_Y = 100
BLOCK_OVERLAP = 0


def segment_dataframe(df, block_size_x, block_size_y, overlap_percentage):
    # Calculate overlap in terms of absolute values
    overlap_x = (block_size_x * overlap_percentage) / 100
    overlap_y = (block_size_y * overlap_percentage) / 100

    # Create new columns for block_x and block_y with overlap
    df["block_x"] = df["x_lsb"].apply(
        lambda x: x // (block_size_x - overlap_x) * (block_size_x - overlap_x)
    )
    df["block_y"] = df["y_lsb"].apply(
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


def plot():
    pass


def processing_pipeline(df):
    # Determine current block
    block_x, block_y = df["block_x"].iloc[0], df["block_y"].iloc[0]
    print(f"Processing block_x: {block_x}, block_y: {block_y}")

    # Inject manual outliers to prevent no-outlier situation
    dfi = inject_points(df)

    # Apply Isolation Forest model on selected features
    features = ["pretrig_std", "posttrig_exp_fit_N", "posttrig_exp_fit_R2"]
    y_pred = isolation_forest(dfi[features])

    # Undo effect of manual outlier injection
    y_pred = y_pred[:-2]

    # Decide outliers and inliers based on score
    outlier_indices = [i for i, point in enumerate(y_pred) if point >= 0.5]
    inlier_indices = [i for i, point in enumerate(y_pred) if point < 0.5]

    # Get outlier ids in block
    outliers = df.iloc[outlier_indices]["transient"].to_numpy()
    inliers = df.iloc[inlier_indices]["transient"].to_numpy()

    return inliers


# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine labeled data with unlabeled extracted features
df = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))

# Segment dataframe into blocks and apply processing to each block
blocks = segment_dataframe(df, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_OVERLAP)
blocks_filtered = blocks.apply(processing_pipeline)

# Get list of inlier transient ids
inliers = blocks_filtered.explode().to_list()

df_filtered = df[df["transient"].isin(inliers)]
df_heatmap = df_filtered.groupby(["x_lsb", "y_lsb"])["posttrig_exp_fit_N"].mean().reset_index()
heatmap_data_filtered  = df_heatmap.pivot(index='x_lsb', columns='y_lsb', values='posttrig_exp_fit_N')

df_heatmap = df.groupby(["x_lsb", "y_lsb"])["posttrig_exp_fit_N"].mean().reset_index()
heatmap_data  = df_heatmap.pivot(index='x_lsb', columns='y_lsb', values='posttrig_exp_fit_N')

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout(pad=3.5, w_pad=10)
sns.heatmap(heatmap_data_filtered, ax=axs[0])
sns.heatmap(heatmap_data, ax=axs[1])
axs[0].set_title('Outliers filtered')
axs[1].set_title('No filtering')

plt.savefig(f"plots/heatmap_v1.png", bbox_inches="tight")
plt.close()