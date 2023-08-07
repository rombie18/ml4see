"""List a directory together and show processing version for each file"""

import os
import glob
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str)
args = parser.parse_args()

fnames = glob.glob(os.path.join(args.directory, "*.h5"))

print(f"total {len(fnames)}")

for fname in sorted(fnames):
    try:
        with h5py.File(fname, "r") as h5file:
            meta = h5file["meta"].attrs
            processing_stage = meta["processing_stage"]
            versions = []
            for stage in range(processing_stage):
                try:
                    versions.append(meta[f"processing_stage_{stage+1}_version"])
                except KeyError:
                    versions.append("???")

            version_string = "(" + ", ".join(versions) + ")"
            print(f"{fname}   stage {processing_stage}   {version_string}")
    except BlockingIOError:
        print(f"{fname}   ---")
