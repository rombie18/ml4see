import pandas as pd
import os
import matplotlib.pyplot as plt

from config import DATA_FEATURES_DIRECTORY
from utils import generatePlotTitle

RUN_NUMBER = 18
MARK_TRANSIENTS = []
FEATURE_1 = 'pretrig_std'
FEATURE_2 = 'posttrig_exp_fit_R2'

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{RUN_NUMBER:03d}.csv")
df = pd.read_csv(csv_path)

ax = df.plot.scatter(x=FEATURE_1, y=FEATURE_2)

for i, txt in enumerate(df["transient"]):
    if len(MARK_TRANSIENTS) == 0 or txt in MARK_TRANSIENTS:
        ax.annotate(txt, (df[FEATURE_1][i], df[FEATURE_2][i]))

generatePlotTitle(ax, "Feature Comparsion Plot", RUN_NUMBER)
plt.savefig(f"plots/run_{RUN_NUMBER:03d}_{FEATURE_1}___{FEATURE_2}.png", bbox_inches="tight")