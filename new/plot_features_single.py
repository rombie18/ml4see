import pandas as pd
import os
import matplotlib.pyplot as plt
import random

from config import DATA_FEATURES_DIRECTORY
from utils import generatePlotTitle

RUN_NUMBER = 18
FEATURE = 'frequency__index_mass_quantile__q_0.1'

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{RUN_NUMBER:03d}.csv")
df = pd.read_csv(csv_path)

# df = df.loc[df['transient'].isin(["tran_000026", "tran_000054"])]

ax = df.plot.scatter(x='transient', y=FEATURE)

for i, txt in enumerate(df["transient"]):
    if random.random() < 0.05:
        ax.annotate(txt, (df["transient"][i], df[FEATURE][i]))

generatePlotTitle(ax, "Single Feature Plot", RUN_NUMBER)
plt.savefig(f"plots/run_{RUN_NUMBER:03d}_{FEATURE}.png", bbox_inches="tight")