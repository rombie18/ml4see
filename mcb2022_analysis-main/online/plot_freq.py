import argparse
import glob
import os

import matplotlib.pyplot as plt

from dsp import EventProcessor

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Name of the file to read from")
args = parser.parse_args()

if os.path.isdir(args.filename):
    fnames = glob.glob(os.path.join(args.filename, "*"))
else:
    fnames = [args.filename]


dsp = EventProcessor(fs=20e6, bw=3e6, n_fir=129)
for fname in fnames:
    t, freq = dsp.file_to_freq(fname)
    plt.plot(t * 1e3, freq)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (Hz)")
plt.grid()
plt.show()
