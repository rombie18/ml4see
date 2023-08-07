"""Display some information about a HDF5 data file."""
import argparse

import h5py
import numpy as np

import util

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()


with h5py.File(args.filename, "r") as h5file:
    util.require_processing_stage(h5file, 1)
    # get handles to the important datasets
    meta = h5file["meta"]
    fpga_hit_data = h5file["fpga_hit_data"]

    # display global run metadata
    print("Run metadata in file:")
    for attr_name, attr_value in meta.attrs.items():
        print(f"  {attr_name} : {attr_value}")
    print()

    print("First 3 hits captured by FPGA:")
    print("  " + str(fpga_hit_data.dtype))
    for fpga_hit in fpga_hit_data[:3]:
        print("  " + str(fpga_hit))
    print()

    if meta.attrs["sdr_data"]:
        sdr_data = h5file["sdr_data"]

        print("SDR metadata in file:")
        for attr_name, attr_value in sdr_data.attrs.items():
            print(f"  {attr_name} : {attr_value}")
        print()

        print(f"Number of transients stored in file: {len(sdr_data)}")
        print()
        print("First 3 transients:")
        for ds_idx, ds_tuple in enumerate(sdr_data["all"].items()):
            ds_name, ds = ds_tuple
            ds = np.array(ds)
            print(f"  Dataset name: {ds_name}")
            print(f"  Dataset length: {len(ds)} samples")
            print(f"  First 3 samples: {ds[:3]}")
            print()
            if ds_idx > 2:
                break
