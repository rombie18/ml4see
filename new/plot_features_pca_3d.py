import os
import pandas as pd
import numpy as np
import argparse
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY
from utils import generatePlotTitle

N_COMPONENTS = 3

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
df = pd.read_csv(csv_path)

# Only retain numeric columns with no NaN values
df_cleaned = df.dropna(axis=1)
df_cleaned = df_cleaned.select_dtypes(include="number")

# Feature names before PCA
feature_names = list(df_cleaned.columns)

# Scaling the data to keep the different attributes in same range.
df_scaled = StandardScaler().fit_transform(df_cleaned)

# Do PCA
pca = PCA(n_components=N_COMPONENTS)
pca_result = pca.fit_transform(df_scaled)
pca_components = [f"PC{i+1}" for i in range(N_COMPONENTS)]
df_pca = pd.DataFrame(data=pca_result, columns=pca_components)
loadings = pca.components_
print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")
print(
    f"Cumulative variance explained by {N_COMPONENTS} principal components: {np.sum(pca.explained_variance_ratio_):.2%}"
)

# Map target names to PCA features
df_pca["valid"] = df["valid"]
df_pca["transient"] = df["transient"]

# Scale PCS into a DataFrame
pca_df_scaled = df_pca.copy()

scaler_df = df_pca[pca_components]
scaler = 1 / (scaler_df.max() - scaler_df.min())

for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]

# Initialize the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=45., azim=60)

# Define scaled features as arrays
xdata = pca_df_scaled["PC1"]
ydata = pca_df_scaled["PC2"]
zdata = pca_df_scaled["PC3"]

# Plot 3D scatterplot of PCA
cmap = []
valid_points = [valid_point for valid_point in pca_df_scaled["valid"]]
if valid_points.count(False) > 0:
    for valid_point in valid_points:
        if valid_point == True:
            cmap.append("green")
        else:
            cmap.append("red")
else:
    cmap.extend(np.repeat("royalblue", len(valid_points)))

# Add transient names to plot
# for i in range(len(xdata)):
#     if random.random() < 0.1:
#         ax.text(
#             xdata[i], ydata[i], zdata[i], pca_df_scaled["transient"][i], fontsize='small'
#         )

ax.scatter(xdata, ydata, zdata, c=cmap, alpha=0.5)

# Define the x, y, z variables
loadings = pca.components_
xs = loadings[0]
ys = loadings[1]
zs = loadings[2]

# Plot the arrows
x_arr = np.zeros(len(loadings[0]))
y_arr = z_arr = x_arr
ax.quiver(x_arr, y_arr, z_arr, xs, ys, zs, color="gray", alpha=0.3)

# Plot title of graph
generatePlotTitle(ax, f"3D Biplot, run_{run_number:03d}", run_number)

xticks = np.linspace(-1, 1, num=5)
yticks = np.linspace(-1, 1, num=5)
zticks = np.linspace(-1, 1, num=5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_zticks(zticks)

# Plot x, y, z labels
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.savefig(f"plots/pca_3d.png", bbox_inches="tight")
