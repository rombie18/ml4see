import argparse

import numpy as np

from dsp import EventProcessor

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Name of the file to read from")
args = parser.parse_args()

dsp = EventProcessor(fs=20e6, bw=3e6, n_fir=129)
t, data_iq = dsp.file_to_iq(args.filename)

max_real = np.max(np.real(data_iq))
min_real = np.min(np.real(data_iq))
max_imag = np.max(np.imag(data_iq))
min_imag = np.min(np.imag(data_iq))

print(f"Real: min: {min_real} - max: {max_real} - pkpk: {max_real - min_real}")
print(f"Imag: min: {min_imag} - max: {max_imag} - pkpk: {max_imag - min_imag}")
