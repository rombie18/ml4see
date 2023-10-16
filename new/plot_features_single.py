import pandas as pd
import os
import matplotlib.pyplot as plt
import random

from config import DATA_FEATURES_DIRECTORY

RUN_NUMBER = 7
FEATURE = 'fit_single_exponential_decay__N'

#TODO add beam moving or stationary to axis

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{RUN_NUMBER:03d}.csv")
df = pd.read_csv(csv_path)

# df = df.loc[df['transient'].isin(["tran_000026", "tran_000054"])]

ax = df.plot.scatter(x='transient', y=FEATURE)

for i, txt in enumerate(df["transient"]):
    if random.random() < 0.05:
        ax.annotate(txt, (df["transient"][i], df[FEATURE][i]))

plt.title(f"run_{RUN_NUMBER:03d}") 
plt.savefig(f"plots/run_{RUN_NUMBER:03d}_{FEATURE}.png")