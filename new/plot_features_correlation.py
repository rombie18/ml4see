import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY

RUN_NUMBER = 7

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{RUN_NUMBER:03d}.csv")
df = pd.read_csv(csv_path)

# Ignore non-feature columns
df = df.drop('transient', axis=1)
df = df.drop('valid', axis=1)

# Calculate the correlation matrix (using Pearson correlation in this example)
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(100, 100))
color = plt.get_cmap('coolwarm')
color.set_bad('black')
sns.heatmap(correlation_matrix, annot=True, cmap=color, square=True)
plt.title("Feature Correlation Matrix")
plt.savefig(f"plots/run_{RUN_NUMBER:03d}_correlation.png")