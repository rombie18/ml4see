import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY
from utils import generatePlotTitle

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine labeled data with unlabeled extracted features
df_features = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
df_labeled = pd.read_csv(os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"))
df = pd.merge(df_features, df_labeled, on='transient')
df.type = df.type.astype("category")
df.valid = df.valid.astype("category")

# Only retain numeric columns with no NaN values
df_cleaned = df.dropna(axis=1)
df_cleaned = df_cleaned.select_dtypes(include="number")

# Assign target vectors
X = df_cleaned
y = df['valid']

# Train classifier
clf = LocalOutlierFactor(n_neighbors=2)
y_pred = clf.fit_predict(X)
y_pred = [0 if x == -1 else 1 for x in y_pred]

print(classification_report(y, y_pred))


# -------------------


df['valid'] = y_pred

FEATURE_1 = "posttrig_exp_fit_N"
FEATURE_2 = "posttrig_exp_fit_R2"
FEATURE_3 = "posttrig_exp_fit_λ"

# Initialize the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=45., azim=60)

# Define scaled features as arrays
xdata = df[FEATURE_1]
ydata = df[FEATURE_2]
zdata = df[FEATURE_3]

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
ax.set_xlabel(FEATURE_1)
ax.set_ylabel(FEATURE_2)
ax.set_zlabel(FEATURE_3)

ax.set_xlim(0, 50e3)
ax.set_ylim(0.8, 1)
ax.set_zlim(1e3, 3e3)

plt.savefig(f"plots/model_localoutlierfactor.png", bbox_inches="tight")