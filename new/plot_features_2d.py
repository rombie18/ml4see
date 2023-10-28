import os
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY
from utils import generatePlotTitle

FEATURE_1 = "posttrig_exp_fit_N"
FEATURE_2 = "posttrig_exp_fit_R2"

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine features with labeled data
df_features = pd.read_csv(
    os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
)
df_labeled = pd.read_csv(
    os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv")
)
df = pd.merge(df_features, df_labeled, on="transient")
df.type = df.type.astype("category")
df.valid = df.valid.astype("category")

# Plot result
fig, ax = plt.subplots()
sns.scatterplot(x=FEATURE_1, y=FEATURE_2, data=df, hue="valid", legend=True, ax=ax)


xdata = df[FEATURE_1]
ydata = df[FEATURE_2]
for i in range(len(xdata)):
    if df["valid"][i] == 0:
        ax.text(xdata[i], ydata[i], df["transient"][i], fontsize="small").set_clip_on(
            True
        )

ax.set_xlabel(FEATURE_1)
ax.set_ylabel(FEATURE_2)

ax.set_ylim(0.8, 1)

generatePlotTitle(ax, "2D plot", run_number)

plt.savefig(f"plots/2d.png", bbox_inches="tight")
