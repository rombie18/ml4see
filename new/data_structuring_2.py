"""Processing stage 2: annotate baseline data"""

import argparse
import math
import numpy as np
import h5py
import os

from config import DATA_STRUCTURED_DIRECTORY
from utils import require_processing_stage

META_PROCESSING_STAGE = 2  # processing stage after completing this stage
META_STAGE_2_VERSION = "3.0"  # version string for this stage (stage 2)

# main changes
# v1.0: baseline heuristics annotation (mean, std, std-of-mean, std-of-std) and outlier scoring
# v2.0: fix baseline stats misattribution, compatibility with stage 1 v2.0
# v3.0: removed outlier score calculation from added stats

PRETRIG_GUARD_SAMPLES = 100

"""Various utility methods"""
def chunker(seq, size):
    """Helper method for iterating over chunks of a list"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    run_numbers = []
    if len(args.run_numbers) == 0:
        for file in os.listdir(DATA_STRUCTURED_DIRECTORY):
            if file.endswith(".h5"):
                run_numbers.append(int(file[4:7]))
        print(f"No runs specified, running on all available runs: {run_numbers}")
    else:
        run_numbers = args.run_numbers

    for run_number in run_numbers:
        print(f"** PROCESSING RUN {run_number:03d} **")

        h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
        with h5py.File(h5_path, "a") as h5file:
            require_processing_stage(h5file, 1, strict=True)
            meta_ds = h5file["meta"]
            meta_ds.attrs.modify("processing_stage", META_PROCESSING_STAGE)
            meta_ds.attrs.create("processing_stage_2_version", META_STAGE_2_VERSION)

            len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
            transients = h5file["sdr_data"]["all"]

            print("Getting list of datasets in run...")
            run_datasets = list(transients.keys())
            num_datasets = len(run_datasets)
            # choose a chunk size such that at least a few thousand events land in each chunk
            # this is important to obtain a good estimate for the standard deviation of the mean
            # which is used in outlier scoring
            MIN_CHUNK_PROP = 0.95
            for chunk_size in range(4000, 5000):
                num_chunks = num_datasets / chunk_size
                if num_chunks - int(num_chunks) > MIN_CHUNK_PROP:
                    break
            if num_chunks > 1 and num_chunks - int(num_chunks) < MIN_CHUNK_PROP:
                raise "Last chunk will be too small..."

            for chunk_idx, chunk in enumerate(chunker(run_datasets, chunk_size)):
                print(f"  Processing chunk {chunk_idx+1}/{math.ceil(num_chunks)}.")

                tran_freq_means = []
                tran_freq_stds = []

                for tran in [transients[name] for name in chunk]:
                    tran_baseline_freq = np.array(tran)[
                        : len_pretrig - PRETRIG_GUARD_SAMPLES
                    ]
                    tran_baseline_freq_mean = np.mean(tran_baseline_freq)
                    tran_baseline_freq_std = np.std(tran_baseline_freq)

                    tran_freq_means.append(tran_baseline_freq_mean)
                    tran_freq_stds.append(tran_baseline_freq_std)

                tran_freq_means = np.array(tran_freq_means)
                tran_freq_stds = np.array(tran_freq_stds)

                for tran, tran_mean, tran_std in zip(
                    [transients[name] for name in chunk],
                    tran_freq_means,
                    tran_freq_stds,
                ):
                    tran.attrs.create("baseline_freq_mean_hz", tran_mean)
                    tran.attrs.create(
                        "baseline_freq_mean_std_hz",
                        tran_std / np.sqrt(len_pretrig - PRETRIG_GUARD_SAMPLES),
                    )
                    tran.attrs.create("baseline_freq_std_hz", tran_std)

            print(f"** FINISHED RUN {run_number:03d} **")

    print("Done.")
    print()


if __name__ == "__main__":
    main()
