import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from isotree import IsolationForest

from config import DATA_FEATURES_DIRECTORY
from utils import generatePlotTitle

FEATURE_1 = "pretrig_std"
FEATURE_2 = "posttrig_exp_fit_λ"
FEATURE_3 = "posttrig_exp_fit_N"

BLOCK_SIZE_X = 1000
BLOCK_SIZE_Y = 1000
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


def prefit_reject(df):
    return df[df["pretrig_std"] < 1000]


def isolation_forest(dff, run_number):
    dfs = [
        dff[[FEATURE_1, FEATURE_2]],
        dff[[FEATURE_1, FEATURE_3]],
        # df[[FEATURE_2, FEATURE_3]],
    ]

    # Init classifiers
    models = {}

    models["iforest"] = IsolationForest(
        ndim=1, ntrees=100, missing_action="fail", n_jobs=-1
    )

    models["ext_iforest"] = IsolationForest(
        ndim=2, ntrees=100, missing_action="fail", n_jobs=-1
    )

    models["dens_iforest"] = IsolationForest(
        ndim=2,
        ntrees=100,
        missing_action="fail",
        scoring_metric="density",
        n_jobs=-1,
    )

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    ax = ax.T.flatten()
    fig.tight_layout(pad=5)

    for z, (name, clf) in enumerate(models.items()):
        for y, df in enumerate(dfs):
            clf.fit(df)
            y_pred_scores = clf.decision_function(df)
            threshold = 0.5
            y_pred = [0 if score < threshold else 1 for score in y_pred_scores]

            disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                df,
                response_method="decision_function",
                alpha=0.5,
                ax=ax[z + 3 * y],
            )
            disp.ax_.scatter(
                df[df.columns[0]], df[df.columns[1]], c=y_pred, s=20, edgecolor="k"
            )
            disp.ax_.set_title(name)
            fig.colorbar(disp.ax_.collections[1])

            # Add transient names to points
            for i, j in enumerate(list(df.index.values)):
                if y_pred[i] == 1:
                    ax[z + 3 * y].text(
                        df[df.columns[0]][j],
                        df[df.columns[1]][j],
                        dff["transient"][j],
                        fontsize="small",
                        clip_on=True,
                    )
                    
    fig.suptitle("Comparison of Forest models")
    plt.savefig(f"plots/test.png", bbox_inches="tight")
    plt.close()

    input()

    return y_pred


# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine labeled data with unlabeled extracted features
df = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))

# Manually reject transients with certain invalid characteristics
df = prefit_reject(df)

blocks = segment_dataframe(df, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_OVERLAP)
preds = blocks.apply(isolation_forest, run_number)