import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
import seaborn as sns

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY
from utils import generatePlotTitle

FEATURE_1 = "pretrig_std"
FEATURE_2 = "posttrig_exp_fit_R2"
FEATURE_3 = "posttrig_exp_fit_λ"

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine labeled data with unlabeled extracted features
df = pd.read_csv(
    os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
)

# Only retain numeric columns with no NaN values
df_cleaned = df.dropna(axis=1)
df_cleaned = df_cleaned.select_dtypes(include="number")

# Assign target vectors
X = df_cleaned

df_cleaned = df_cleaned[[FEATURE_1, FEATURE_2, FEATURE_3]]

# Train classifier
clf = IsolationForest()
clf.fit(X)
y_pred_scores = clf.decision_function(X)

threshold = 0
y_pred = [0 if score < threshold else 1 for score in y_pred_scores]

# -------------------


df["valid"] = y_pred

# Initialize the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=10, azim=40)

# Define scaled features as arrays
xdata = df[FEATURE_1]
ydata = df[FEATURE_2]
zdata = df[FEATURE_3]

# Plot 3D scatterplot of PCA
color_labels = df["valid"].unique()
col_values = sns.color_palette(n_colors=len(color_labels))
color_map = dict(zip(color_labels, col_values))
colors = [color_map[label] for label in df["valid"].values]

# Add outlier scores to plot
# for i in range(len(xdata)):
#     if y_pred_scores[i] < threshold:
#         ax.text(
#             xdata[i], ydata[i], zdata[i], f"{y_pred_scores[i]:.3f}", fontsize="small"
#         ).set_clip_on(True)

ax.scatter(xdata, ydata, zdata, c=colors, alpha=0.5)

# Plot x, y, z labels
ax.set_xlabel(FEATURE_1)
ax.set_ylabel(FEATURE_2)
ax.set_zlabel(FEATURE_3)

ax.set_xlim(0, 500)
ax.set_ylim(0.95, 1)
ax.set_zlim(2.5e3, 5e3)

plt.savefig(f"plots/model_isolationforest.png")
