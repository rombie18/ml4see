"""Plot FPGA hit map from HDF5 dataset"""
import argparse
import matplotlib.pyplot as plt
import numpy as np

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()


def reject_first(fpga_hit_data):
    """Filter initial events at (0|0)"""
    for hit_idx, hit in enumerate(fpga_hit_data):
        if hit["x_lsb"] != 0 or hit["y_lsb"] != 0:
            return fpga_hit_data[hit_idx:]


def analyze_1d(hits):
    coords, counts = np.unique(hits, return_counts=True)
    step = np.min(np.diff(coords))
    all_coords = np.arange(np.min(coords), np.max(coords) + step + 1, step)
    all_counts = np.zeros_like(all_coords)

    for coord, count in zip(coords, counts):
        idx = np.argwhere(all_coords == coord)
        all_counts[idx] = count

    return all_coords, all_counts


def smooth(coords, counts, mavg_len=8):
    assert mavg_len % 2 == 0
    coords = coords[mavg_len // 2 : -mavg_len // 2 + 1]
    counts = np.convolve(counts, np.ones(mavg_len) / mavg_len, "valid")

    return coords, counts


with h5py.File(args.filename, "r") as h5file:
    # get handles to the important datasets
    meta = h5file["meta"].attrs
    fpga_hit_data = h5file["fpga_hit_data"]

    unit = meta["scan_unit"]
    if unit == "um":
        x_lsb_per_um = meta["scan_x_lsb_per_um"]
        y_lsb_per_um = meta["scan_y_lsb_per_um"]

    fpga_hit_data = reject_first(fpga_hit_data)
    hits_x = fpga_hit_data["x_lsb"]
    hits_y = fpga_hit_data["y_lsb"]

    if unit == "um":
        hits_x = [lsb / x_lsb_per_um for lsb in hits_x]
        hits_y = [lsb / y_lsb_per_um for lsb in hits_y]

    if meta["scan_type"] == "XY":
        plt.plot(hits_x, hits_y, "k.", alpha=0.15)
        plt.axis("equal")
        plt.xlabel(f"Beam X position ({unit})")
        plt.ylabel(f"Beam Y position ({unit})")
    if meta["scan_type"] == "X":
        assert len(np.unique(hits_y)) == 1
        hits_x, counts_x = analyze_1d(hits_x)
        hits_x_mavg, counts_x_mavg = smooth(hits_x, counts_x)

        plt.plot(hits_x, counts_x, "x", color="#aaaaaa", label="Raw counts")
        plt.plot(hits_x_mavg, counts_x_mavg, "k-", label="Smoothed")
        plt.xlabel(f"Beam X position ({unit})")
        plt.ylabel(f"Event count")
        plt.legend()
    if meta["scan_type"] == "Y":
        assert len(np.unique(hits_x)) == 1
        hits_y, counts_y = analyze_1d(hits_y)
        hits_y_mavg, counts_y_mavg = smooth(hits_y, counts_y)

        plt.plot(hits_y, counts_y, "x", color="#aaaaaa", label="Raw counts")
        plt.plot(hits_y_mavg, counts_y_mavg, "k-", label="Smoothed")
        plt.xlabel(f"Beam Y position ({unit})")
        plt.ylabel(f"Event count")
        plt.legend()

    plt.title(
        f"Run {meta['run_id']} @ {meta['run_date']} {meta['run_time_appx']}\nDUT: {meta['dut_type']} ({meta['dut_name']})"
    )
    plt.grid()
    plt.show()
