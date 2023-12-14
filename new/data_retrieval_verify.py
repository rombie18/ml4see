"""
data_retrieval_verify.py

This script performs the verification of file integrity by comparing calculated MD5 checksums
with the expected MD5 checksums obtained from a data summary file. It supports parallel validation
of multiple files, and users can specify run numbers as command-line arguments for selective validation.

Usage:
    python data_retrieval_verify.py [run_numbers ...] [--keep]

Parameters:
    - run_numbers (int, optional): Specify one or more run numbers to verify. If not provided,
                                   all runs from the data summary file will be verified.
    - --keep (optional): If present, the script will not delete files on verification failure.

The script uses the `config` module for data download directory and data summary file path.

Functions:
    - main(): The main entry point for the script.
    - validate_file(working_dir, file_name, expected_md5sum, keep_on_fail=True): Validates the integrity of a file.
    - calculate_md5(file_path, buffer_size=32768): Calculates the MD5 checksum of a file.
    - read_csv(file_path): Reads a CSV file and returns its content as a list of dictionaries.

Note: This script requires the `config` module, and the `validate_file`, `calculate_md5`, and `read_csv` functions
      assume the correct implementation of the `hashlib` module for MD5 checksum calculation.

Example Usage:
    python data_retrieval_verify.py 1 2 3 --keep
"""

import csv
import os
import logging
import argparse
import hashlib
from multiprocessing import Pool

from config import DATA_DOWNLOAD_DIRECTORY, DATA_SUMMARY_PATH


def main():
    # Initialise logging
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
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        runs = [run for run in runs if run["name"] in run_numbers]
        logging.info(f"Runs argument present, only verifying: {run_numbers}")
    else:
        run_numbers = [run["name"] for run in runs]
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    # Start calculating md5sums in parallel
    logging.info("Validating files")
    with Pool() as pool:
        args = [
            (
                DATA_DOWNLOAD_DIRECTORY,
                run["url"].split("/")[-1],
                run["md5sum"],
                args.keep,
            )
            for run in runs
        ]
        pool.map(validate_file_args, args)

    logging.info("Done!")


def validate_file_args(args):
    working_dir, file_name, expected_md5sum, keep_on_fail = args
    return validate_file(working_dir, file_name, expected_md5sum, keep_on_fail)


def validate_file(working_dir, file_name, expected_md5sum, keep_on_fail=True):
    """
    Validate the integrity of a file by comparing its calculated MD5 checksum
    with the expected MD5 checksum.

    Parameters:
    - working_dir (str): The directory where the file is located.
    - file_name (str): The name of the file to be validated.
    - expected_md5sum (str): The expected MD5 checksum to compare with.
    - keep_on_fail (bool): If True, keeps the file on validation failure; if False, deletes the file.

    The function performs the following steps:
    1. Composes the full path of the file using the working directory and file name.
    2. Logs an informational message about the validation process.
    3. Checks if the file exists on disk; if not, logs a warning and skips validation.
    4. Checks if the file has an MD5 checksum available; if not, logs a warning and skips validation.
    5. Calculates the MD5 checksum of the file using the `calculate_md5` function.
    6. Compares the calculated MD5 checksum with the expected MD5 checksum.
    7. If the checksums do not match, logs an error, and optionally deletes the file if the keep flag is not set.
    8. If the checksums match, logs a success message.

    Note: The `calculate_md5` function is assumed to be available and correctly implemented.

    Example Usage:
    ```python
    working_directory = '/path/to/working/directory'
    file_to_validate = 'example_file.txt'
    expected_checksum = 'e7d4a37e45a7a6c1c7f2b5b33fb14533'
    validate_file(working_directory, file_to_validate, expected_checksum)
    ```

    """

    logging.info(f"Validating {file_name}")

    # Compose full path where to save the downloaded run
    file_path = os.path.join(working_dir, file_name)

    # If run is not found on disk or has no md5sum available, skip
    if not os.path.exists(file_path):
        logging.warning(f"Skip validating {file_name} since it was not found on disk!")
        return

    if expected_md5sum == "":
        logging.warning(
            f"Skip validating {file_name} since it has no md5 checksum available!"
        )
        return

    # Calculate md5sum of file
    calculated_md5sum = calculate_md5(file_path)

    # Check if calculated md5sum matches the provided one in the data summary file
    if not calculated_md5sum == expected_md5sum:
        logging.error(f"Validation of {file_name} failed!")
        logging.debug(
            f"Got {calculated_md5sum} as checksum but expected {expected_md5sum}."
        )
        # If the keep flag is not set, delete the file if verification fails
        if not keep_on_fail:
            logging.info(f"Deleting file {file_name} since it is corrupt.")
            os.remove(file_path)
        return

    logging.info(f"Validation of {file_name} succeeded.")


def calculate_md5(file_path, buffer_size=32768):
    """
    Calculate the MD5 checksum of a file.

    Parameters:
    - file_path (str): The path to the file.
    - buffer_size (int): The size of the buffer for reading the file.

    Returns:
    - md5_checksum (str): The MD5 checksum of the file.
    """

    logging.debug(
        f"Calculating md5sum of {file_path} with buffer size of {buffer_size}"
    )

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Read the file in chunks and update the hash
        while chunk := file.read(buffer_size):
            md5_hash.update(chunk)

    # Get the hexadecimal representation of the hash
    md5_checksum = md5_hash.hexdigest()

    return md5_checksum


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
