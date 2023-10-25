import os
import pandas as pd
import numpy as np
import argparse
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY
from utils import generatePlotTitle

N_COMPONENTS = 3

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

df_features = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
df_labeled = pd.read_csv(os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"))
df = pd.merge(df_features, df_labeled, on='transient')
df.valid = df.valid.astype("category")

# Initialize the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=45., azim=60)

# Define scaled features as arrays
xdata = df["std"]
ydata = df["exp_fit_N"]
zdata = df["exp_fit_λ"]

# Plot 3D scatterplot of PCA
color_labels = df["valid"].unique()
col_values = sns.color_palette(n_colors=len(color_labels))
color_map = dict(zip(color_labels, col_values))
colors = [color_map[label] for label in df['valid'].values]

# Add transient names to plot
# for i in range(len(xdata)):
#     if random.random() < 0.1:
#         ax.text(
#             xdata[i], ydata[i], zdata[i], pca_df_scaled["transient"][i], fontsize='small'
#         )

ax.scatter(xdata, ydata, zdata, c=colors, alpha=0.5)

# Plot title of graph
generatePlotTitle(ax, f"3D plot, run_{run_number:03d}", run_number)

# Plot x, y, z labels
ax.set_xlabel("std")
ax.set_ylabel("exp_fit_N")
ax.set_zlabel("exp_fit_λ")

ax.set_xlim(0, 1000)
ax.set_ylim(-1000, 5e4)
ax.set_zlim(2500, 5000)

plt.savefig(f"plots/3d.png", bbox_inches="tight")
