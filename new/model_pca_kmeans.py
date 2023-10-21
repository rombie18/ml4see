import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import DATA_FEATURES_DIRECTORY, DATA_LABELED_DIRECTORY

N_COMPONENTS = 10

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument("run_number", type=int)
args = parser.parse_args()
run_number = args.run_number

df_features = pd.read_csv(os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"))
df_labeled = pd.read_csv(os.path.join(DATA_LABELED_DIRECTORY, f"run_{run_number:03d}.csv"))
df = pd.merge(df_features, df_labeled, on='transient')
df.type = df.type.astype("category")

# Only retain numeric columns with no NaN values
df_cleaned = df.dropna(axis=1)
df_cleaned = df_cleaned.select_dtypes(include="number")

# Feature names before PCA
feature_names = list(df_cleaned.columns)
target_names = df['type'].unique()

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
df_pca["type"] = df["type"]
df_pca["transient"] = df["transient"]

# Scale PCS into a DataFrame
pca_df_scaled = df_pca.copy()

scaler_df = df_pca[pca_components]
scaler = 1 / (scaler_df.max() - scaler_df.min())

for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]
    
X = pca_df_scaled.drop(['transient', 'type'], axis=1)
y = df['type']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, labels=target_names, zero_division=0))