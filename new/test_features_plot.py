import pandas as pd
import os
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY

FEATURE_1 = 'fit_single_exponential_decay__N'
FEATURE_2 = 'fit_single_exponential_decay__c'

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, "run_029.csv")
df = pd.read_csv(csv_path)

ax = df.plot.scatter(x=FEATURE_1, y=FEATURE_2)

for i, txt in enumerate(df["transient"]):
    ax.annotate(txt, (df[FEATURE_1][i], df[FEATURE_2][i]))

plt.savefig(f"test_{FEATURE_1}___{FEATURE_2}.png")