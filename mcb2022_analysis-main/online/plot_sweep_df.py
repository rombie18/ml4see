import argparse
import functools
import multiprocessing
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from dsp import EventProcessor

x_lsb_per_um = 99.001
y_lsb_per_um = 93.45

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str, help="Run folder")
parser.add_argument("--first-only", dest="first_only", action="store_true", help="Process only first file")
args = parser.parse_args()

x_folders = [os.path.basename(x) for x in glob.glob(os.path.join(args.folder, "*"))]
assert x_folders[0].startswith("x_"), "Provided the right folder?"
x_folders = [name for name in x_folders if name != "x_999999"]
x_folders = sorted(x_folders, key=lambda d: int(d[2:]))

y_folders = [os.path.basename(x) for x in glob.glob(os.path.join(args.folder, x_folders[0], "*"))]
assert y_folders[0].startswith("y_"), "Provided the right folder?"
y_folders = [name for name in y_folders if name != "y_999999"]
y_folders = sorted(y_folders, key=lambda d: int(d[2:]))

dims = 1
if len(x_folders) == 1:
    coords = [int(y[2:]) for y in y_folders]
    folders = [os.path.join(args.folder, x_folders[0], y_folder) for y_folder in y_folders]
elif len(y_folders) == 1:
    coords = [int(x[2:]) for x in x_folders]
    folders = [os.path.join(args.folder, x_folder, y_folders[0]) for x_folder in x_folders]
else:  # X and Y scan
    dims = 2
    folders = []
    coords = []
    for y_folder in y_folders:
        coords.extend([(int(x[2:]), int(y_folder[2:])) for x in x_folders])
    for y_folder in y_folders:
        folders.extend([os.path.join(args.folder, x_folder, y_folder) for x_folder in x_folders])

pool = multiprocessing.Pool(4)
df_avgs = np.zeros(len(coords), dtype=float)
dsp = EventProcessor(fs=20e6, bw=3e6, n_fir=129)

for idx, folder in enumerate(folders):
    print(f"Processing {folder}...")
    fnames = glob.glob(os.path.join(folder, "*"))
    if args.first_only:
        fnames = [fnames[0]]
    dfs = pool.map(functools.partial(dsp.file_to_df, t1=-200e-6, t2=20e-6), fnames)
    df_avgs[idx] = np.mean(dfs)
print(df_avgs)

if dims == 1:
    plt.plot(coords, df_avgs)
    plt.grid()
    plt.xlabel("Position (LSB)")
    plt.ylabel("df (Hz)")
if dims == 2:
    # reshape coords to XxY map
    x_coords = [coord[0] / x_lsb_per_um for coord in coords]
    y_coords = [coord[1] / y_lsb_per_um for coord in coords]
    extent = [min(x_coords), max(x_coords), min(y_coords), max(y_coords)]
    df_avgs = np.reshape(df_avgs, (len(y_folders), len(x_folders)))
    # reshape df_avgs to XxY map
    pos = plt.imshow(df_avgs, extent=extent, cmap='hot', interpolation="nearest", origin="lower")
    cbar = plt.colorbar(pos)
    plt.xlabel("X Position (LSB)")
    plt.ylabel("Y Position (LSB)")
    cbar.set_label("Frequency Delta (Hz)")
plt.show()
