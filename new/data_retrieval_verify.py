"""
data_retrieval_verify.py

This Python script is designed to verify the integrity of downloaded data files by comparing their MD5 checksums with the values provided in a JSON summary file. It provides options to verify specific runs and handles verification failures, allowing for the deletion of corrupted files. Configuration constants are imported from an external module for flexibility.

Usage:
    python data_retrieval_verify.py [run_numbers [run_numbers ...]] [--keep]

Arguments:
    run_numbers (optional): A list of integers representing specific run numbers to verify. If provided, only the runs with matching numbers will be verified.
    --keep (optional): If this flag is set, files with failed verification will not be deleted.

Configuration (imported from 'config.py'):
    - DATA_DOWNLOAD_DIRECTORY: The directory where downloaded tar files are located.
    - DATA_SUMMARY_PATH: The path to the JSON summary file containing information about the data runs.

The script performs the following steps:
1. Initializes logging to record verification progress and errors.
2. Parses command-line arguments to optionally specify which runs to verify and whether to keep files with failed verification.
3. Checks if the specified data directories and summary file exist; exits if not.
4. Reads run information from the JSON summary file.
5. Filters the runs based on the provided run numbers, if any.
6. Defines a function to validate file integrity using MD5 checksums.
7. Sets up a Dask bag with the list of runs to parallelize the verification process.
8. Schedules the validate_file function for each run in the bag.
9. Executes the verification tasks in parallel.
10. Closes the Dask client and logs any fatal exceptions.

Note: The script assumes that it is executed using a Dask cluster for parallel processing.

Example Usage:
- Verify all downloaded runs:
    python data_retrieval_verify.py

- Verify specific runs (e.g., run numbers 1 and 2) and keep files with failed verification:
    python data_retrieval_verify.py 1 2 --keep
"""

import csv
import os
import json
import logging
import subprocess
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

from config import DATA_DOWNLOAD_DIRECTORY, DATA_SUMMARY_PATH


def main():
    # Initialise logging
    # FIXME logging not working in Dask workers (other processes)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_verify.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting data retrieval verify process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete file if verification process fails",
    )
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(
            f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}."
        )
        exit()

    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error(f"The data summary file does not exist at {DATA_SUMMARY_PATH}.")
        exit()

    # Get run information from summary file
    runs = read_csv(DATA_SUMMARY_PATH)

    # If runs are provided as arguments, only verify the specified runs
    if len(args.run_numbers) > 0:
        logging.info(f"Runs argument present, only verifying: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        runs = [run for run in runs if run["name"] in run_numbers]

    # Function that will verify file integrity using a md5sum
    def validate_file(run):
        # Compose full path where to save the downloaded run
        file_path = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["name"] + ".tar")

        # If run is not found on disk or has no md5sum available, skip
        if not os.path.exists(file_path):
            logging.warning(
                f"Skip validating {run['name']} since it was not found on disk!"
            )
            return

        if "md5sum" not in run:
            logging.warning(
                f"Skip validating {run['name']} since it has no md5 checksum available!"
            )
            return

        # Execute md5sum command and pipe output to this python script and select result
        result = subprocess.run(
            ["md5sum", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        md5_returned = result.stdout.split()[0]

        # Check if calculated md5sum matches the provided one in the data summary file
        if not md5_returned == run["md5sum"]:
            logging.error(f"Validation of {run['name']} failed!")
            logging.error(
                f"Got {md5_returned} as checksum but expected {run['md5sum']}."
            )
            # If the keep flag is not set, delete the file if verification fails
            if not args.keep:
                os.remove(file_path)
            return

        logging.info(f"Validation of {run['md5sum']} succeeded.")
        return

    # Set up bag with runs
    bag = db.from_sequence(runs)
    # For each run, schedule the validate file function
    tasks = bag.map(validate_file)

    # Execute task
    results = tasks.compute()


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
    # Set-up Dask local cluster for distributed processing
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()
