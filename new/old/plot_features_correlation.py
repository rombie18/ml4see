import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

# Combine features with labeled data
df_features = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
df_labeled = pd.read_csv(os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"))
df = pd.merge(df_features, df_labeled, on='transient')
df.transient = df.transient.astype("category")
df.type = df.type.astype("category")
df.valid = df.valid.astype("category")

# Only retain numeric columns with no NaN values
df = df.dropna(axis=1)
df = df.select_dtypes(include="number")

# Calculate the correlation matrix (using Pearson correlation in this example)
correlation_matrix = df.corr()

# Create a heatmap
color = plt.get_cmap('coolwarm')
color.set_bad('black')
sns.heatmap(correlation_matrix, annot=True, cmap=color, square=True)
plt.title("Feature Correlation Matrix")
plt.savefig(f"plots/run_{run_number:03d}_correlation.png")