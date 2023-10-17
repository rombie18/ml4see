import os
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY

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
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
loadings = pca.components_
print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")
print(
    f"Cumulative variance explained by 2 principal components: {np.sum(pca.explained_variance_ratio_):.2%}"
)

# Map target names to PCA features
df_pca["valid"] = df["valid"]

# Scale PCS into a DataFrame
pca_df_scaled = df_pca.copy()

scaler_df = df_pca[["PC1", "PC2"]]
scaler = 1 / (scaler_df.max() - scaler_df.min())

for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]

# Plot the loadings on a Scatter plot
xs = loadings[0]
ys = loadings[1]

# Plot result
sns.lmplot(
    x="PC1",
    y="PC2",
    data=pca_df_scaled,
    hue='valid', 
    fit_reg=False, 
    legend=True
)

for i, varnames in enumerate(feature_names):
    plt.arrow(
        0,
        0,  # coordinates of arrow base
        xs[i],  # length of the arrow along x
        ys[i],  # length of the arrow along y
        color="gray",
        head_width=0.01,
        alpha=0.3,
    )
plt.text(xs[i], ys[i], varnames)

xticks = np.linspace(-0.8, 0.8, num=5)
yticks = np.linspace(-0.8, 0.8, num=5)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(f"2D Biplot, run_{run_number:03d}")

plt.savefig(f"plots/pca.png", bbox_inches='tight')
