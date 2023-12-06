"""
data_retrieval_download.py

This script facilitates the download of data based on run numbers specified as command-line arguments. The script interacts with a data summary file to retrieve information about available runs and uses curl to download the selected runs.

Usage:
    python data_retrieval_download.py [run_numbers] [--keep]

Arguments:
    run_numbers (int, optional): Specify one or more run numbers to download. If no run numbers are provided, the script will prompt the user to select runs interactively.
    --keep: Optional flag to keep downloaded files even if the download process fails.

Dependencies:
    - Python 3.x
    - curl
    - inquirer
    - config.py (imported for configuration constants)

Example:
    python data_retrieval_download.py 1 2 3 --keep

"""

import csv
import os
import json
import logging
import subprocess
import argparse

from config import DATA_DOWNLOAD_DIRECTORY, DATA_SUMMARY_PATH, DOWNLOAD_RETRIES


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_download.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting data retrieval download process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete file if download process fails",
    )
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(
            f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}."
        )
        exit()
    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error(f"There is no data summary file found at {DATA_SUMMARY_PATH}.")
        exit()

    # Get run information from summary file
    runs = read_csv(DATA_SUMMARY_PATH)

    # If runs are provided as arguments, only download the specified runs
    if len(args.run_numbers) > 0:
        logging.info(f"Runs argument present, only downloading: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        runs = [run for run in runs if run["name"] in run_numbers]

    logging.info(
        "It can take some time before the downloading actually starts, this is due to server-side limitations."
    )
    command = (
        f"curl --parallel --parallel-immediate --remote-name-all --retry 10 --retry-connrefused --continue-at - "
        + " ".join([run["url"] for run in runs])
    )

    # Execute curl command
    for i in range(1, DOWNLOAD_RETRIES):
        try:
            # FIXME on curl command failure, doesn't catch error to except block
            subprocess.run(command.split(), cwd=DATA_DOWNLOAD_DIRECTORY, check=True)
            break
        except subprocess.CalledProcessError:
            if i == DOWNLOAD_RETRIES:
                logging.exception(f"Downloading failed after {DOWNLOAD_RETRIES} tries.")
            else:
                logging.warning(f"Downloading interrupted. Retrying... ({i}/{DOWNLOAD_RETRIES})")


def read_csv(file_path):
    result_array = []

    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Read the header to get column names
        headers = next(csv_reader)

        # Read the remaining rows and populate the result_array
        for row in csv_reader:
            line_dict = {}
            for i, value in enumerate(row):
                line_dict[headers[i]] = value
            result_array.append(line_dict)

    return result_array


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
