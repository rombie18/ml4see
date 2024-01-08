import os
import logging
import h5py
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool

from config import (
    DATA_STRUCTURED_DIRECTORY,
    DATA_FEATURES_DIRECTORY,
    WINDOW_SIZE,
    DOWNSAMPLE_FACTOR,
    PRETRIG_GUARD_SAMPLES,
    R2_THRESHOLD,
)
from utils import require_processing_stage, moving_average, exponential_decay


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("feature_extraction.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting feature extraction process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.error(
            f"The structured data directory does not exist at {DATA_STRUCTURED_DIRECTORY}."
        )
        exit()
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(
            f"The features data directory does not exist at {DATA_FEATURES_DIRECTORY}."
        )
        exit()

    # If runs are provided as arguments, only verify the specified runs
    run_numbers = []
    if len(args.run_numbers) > 0:
        run_numbers = args.run_numbers
        logging.info(
            f"Runs argument present, only extracting features of: {run_numbers}"
        )
    else:
        for file in os.listdir(DATA_STRUCTURED_DIRECTORY):
            if file.endswith(".h5"):
                run_numbers.append(int(file[4:7]))
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    with Pool() as pool:
        for run_number in run_numbers:
            logging.info(f"Processing run {run_number:03d}")

            # Combine data directory with provided run number to open .h5 file in read mode
            h5_path = os.path.join(
                DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5"
            )
            with h5py.File(h5_path, "r") as h5file:
                # Check if file is up to required processing stage
                require_processing_stage(h5file, 2, strict=True)

                # Get transients from h5 file
                transients = h5file["sdr_data"]["all"]

                transient_args = [
                    (h5_path, tran_name) for tran_name in transients.keys()
                ]
                features_list = pool.map(process_transient_args, transient_args)

                features = pd.concat(features_list, ignore_index=True)

                # Save extracted features to csv file
                logging.debug(
                    f"Storing extracted features in file run_{run_number:03d}.csv"
                )
                features.to_csv(
                    os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv"),
                    index=False,
                )

            logging.info(f"Successfully processed run {run_number:03d}")

    logging.info("Done!")


def process_transient_args(args):
    h5_path, tran_name = args
    return process_transient(h5_path, tran_name)


def process_transient(h5_path, tran_name):
    logging.debug(f"Processing transient {tran_name}")

    # TODO try to find way to speed up reading transients from disk
    with h5py.File(h5_path, "r") as h5file:
        # Get transient data from file
        transient = h5file["sdr_data"]["all"][tran_name]

        # Get run meta data
        scan_x_lsb_per_um = h5file["meta"].attrs["scan_x_lsb_per_um"]
        scan_y_lsb_per_um = h5file["meta"].attrs["scan_y_lsb_per_um"]

        fs = h5file["sdr_data"].attrs["sdr_info_fs"]
        sdr_cf = h5file["sdr_data"].attrs["sdr_info_cf"]
        len_pretrig = h5file["sdr_data"].attrs["sdr_info_len_pretrig"]
        len_posttrig = h5file["sdr_data"].attrs["sdr_info_len_posttrig"]
        dsp_ntaps = h5file["sdr_data"].attrs["dsp_info_pre_demod_lpf_taps"]
        event_len = len_pretrig + len_posttrig - dsp_ntaps

        # Get additional transient meta data
        baseline_freq = transient.attrs["baseline_freq_mean_hz"]
        x_lsb = transient.attrs["x_lsb"]
        y_lsb = transient.attrs["y_lsb"]

        # Calculate true oscillator frequency for ppm error
        f0 = sdr_cf + baseline_freq

        # Construct time and frequency arrays, calculate the frequency error in ppm
        tran_time = (
            np.arange(start=0, stop=event_len / fs, step=1 / fs) - len_pretrig / fs
        )
        tran_freq = np.subtract(np.array(transient), baseline_freq) / f0 * 1e6

        # Construct pre-trigger baseline arrays
        tran_pretrig_time = tran_time[: len_pretrig - PRETRIG_GUARD_SAMPLES]
        tran_pretrig_freq = tran_freq[: len_pretrig - PRETRIG_GUARD_SAMPLES]

        # Construct post-trigger arrays
        tran_posttrig_time = tran_time[len_pretrig:]
        tran_posttrig_freq = tran_freq[len_pretrig:]

        # Downsample data
        tran_pretrig_freq_ds, tran_pretrig_time_ds = moving_average(
            tran_pretrig_freq, tran_pretrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
        )
        tran_posttrig_freq_ds, tran_posttrig_time_ds = moving_average(
            tran_posttrig_freq, tran_posttrig_time, DOWNSAMPLE_FACTOR, WINDOW_SIZE
        )

        # Calculate features
        features = {}
        features["transient"] = tran_name
        features["x_lsb"] = x_lsb
        features["y_lsb"] = y_lsb
        features["x_um"] = x_lsb / scan_x_lsb_per_um
        features["y_um"] = y_lsb / scan_y_lsb_per_um
        features["trig_val"] = tran_posttrig_freq_ds[0]
        features["pretrig_min"] = np.min(tran_pretrig_freq_ds)
        features["posttrig_min"] = np.min(tran_posttrig_freq_ds)
        features["pretrig_max"] = np.max(tran_pretrig_freq_ds)
        features["posttrig_max"] = np.max(tran_posttrig_freq_ds)
        features["pretrig_std"] = np.std(tran_pretrig_freq_ds)
        features["posttrig_std"] = np.std(tran_posttrig_freq_ds)

        # Try to fit exponential decay
        initial_guess = (
            np.max(tran_posttrig_freq_ds),
            1000,
            np.mean(tran_pretrig_freq_ds),
        )
        boundaries = (
            [
                -np.inf,
                0,
                -np.inf,
            ],
            [np.inf, np.inf, np.inf],
        )

        try:
            # params: N, λ, c
            # model: (N - c) * np.exp(-λ * t) + c
            params, _ = curve_fit(
                exponential_decay,
                tran_posttrig_time_ds,
                tran_posttrig_freq_ds,
                p0=initial_guess,
                bounds=boundaries,
            )

            # Caluculate coefficient of determination (R²)
            residuals = tran_posttrig_freq_ds - exponential_decay(
                tran_posttrig_time_ds, *params
            )
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(
                (tran_posttrig_freq_ds - np.mean(tran_posttrig_freq_ds)) ** 2
            )
            r_squared = 1 - (ss_res / ss_tot)

            features["posttrig_exp_fit_R2"] = r_squared

            if r_squared > R2_THRESHOLD:
                # Assign fitted parameters and R² to resulting feature set
                features["posttrig_exp_fit_N"] = params[0]
                features["posttrig_exp_fit_λ"] = params[1]
                features["posttrig_exp_fit_c"] = params[2]
            else:
                # Insufficient fit, set parameters to zero
                features["posttrig_exp_fit_N"] = 0
                features["posttrig_exp_fit_λ"] = 0
                features["posttrig_exp_fit_c"] = 0
                params = None

        except:
            # If exponential fit fails, assign parameters to zero
            features["posttrig_exp_fit_N"] = 0
            features["posttrig_exp_fit_λ"] = 0
            features["posttrig_exp_fit_c"] = 0
            features["posttrig_exp_fit_R2"] = 0

        # Convert feature set to Pandas dataframe
        df = pd.DataFrame(features, index=[0])

        return df


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
