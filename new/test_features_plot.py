import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_FEATURES_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/features"
FEATURE= 'frequency__index_mass_quantile__q_0.1'

csv_path = os.path.join(DATA_FEATURES_DIRECTORY, "run_029.csv")
df = pd.read_csv(csv_path)

ax = df.plot.scatter(x='transient', y=FEATURE)

for i, txt in enumerate(df["transient"]):
    ax.annotate(txt, (df["transient"][i], df[FEATURE][i]))

plt.savefig(f"test_{FEATURE}.png")