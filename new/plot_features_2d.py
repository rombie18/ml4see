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

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

df_features = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
df_labeled = pd.read_csv(os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"))
df = pd.merge(df_features, df_labeled, on='transient')
df.type = df.type.astype("category")

# Plot result
fig, ax = plt.subplots()
sns.scatterplot(
    x="zero_min_dis", y="num_peaks", data=df, hue="type", legend=True, ax=ax
)

ax.set_xlabel("zero_min_dis")
ax.set_ylabel("num_peaks")

generatePlotTitle(ax, "2D plot", run_number)

plt.savefig(f"plots/2d.png", bbox_inches="tight")
