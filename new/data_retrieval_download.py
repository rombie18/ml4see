"""
data_retrieval_download.py

This script facilitates the downloading of files in parallel, with the ability to resume interrupted downloads.
It reads run information from a data summary file, allowing users to download specific runs by providing
run numbers as command-line arguments.

Usage:
    python data_retrieval_download.py [run_numbers ...]

Parameters:
    - run_numbers (int, optional): Specify one or more run numbers to download. If not provided,
                                   all runs from the data summary file will be downloaded.

The script uses the `config` module for data download directory, data summary file path, and download attempts.

Functions:
    - main(): The main entry point for the script.
    - download_file(url, working_dir, file_name, attempts=5, chunk_size=1024): Download a file with the ability to resume if the download is interrupted.
    - read_csv(file_path): Reads a CSV file and returns its content as a list of dictionaries.

Note: This script requires the `config` module, and the `download_file` and `read_csv` functions assume the correct implementation
      of the `requests` module for handling HTTP requests and responses.

Example Usage:
    python data_retrieval_download.py 1 2 3
"""

import csv
import os
import logging
import argparse
import requests
from multiprocessing import Pool

from config import DATA_DOWNLOAD_DIRECTORY, DATA_SUMMARY_PATH, DOWNLOAD_ATTEMPTS

# TODO add support for partial file downloading


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
    else:
        run_numbers = [run["name"] for run in runs]
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    # Start downloading the files in parallel
    logging.info("Starting file downloads")
    with Pool() as pool:
        args = [
            (run["url"], DATA_DOWNLOAD_DIRECTORY, run["url"].split("/")[-1])
            for run in runs
        ]
        pool.map(download_file_args, args)

    logging.info("Done!")


def download_file_args(args):
    url, working_dir, file_name = args
    return download_file(url, working_dir, file_name, attempts=DOWNLOAD_ATTEMPTS)


def download_file(url, working_dir, file_name, attempts=5, chunk_size=1024):
    """
    Download a file with the ability to resume if the download is interrupted.

    Parameters:
    - url (str): The URL of the file to be downloaded.
    - working_dir (str): The local directory where the file should be saved.
    - file_name (str): The name of the file to be saved.
    - attempts (int): The number of attempts to download the file (default is 5).
    - chunk_size (int): The size of each chunk to download (default is 1 KB).
    """
    local_path = os.path.join(working_dir, file_name)

    for attempt in range(1, attempts):
        logging.info(f"Downloading file {file_name} (attempt {attempt}/{attempts})")

        try:
            # Get file size from server
            with requests.head(url) as head_response:
                server_content_length = int(
                    head_response.headers.get("Content-Length", "")
                )

            # Determine if file should be (partially) downloaded or not
            if os.path.exists(local_path):
                local_content_length = os.path.getsize(local_path)
                if local_content_length == server_content_length:
                    logging.info(
                        f"Full file {file_name} already exists locally, skipping download"
                    )
                    return
                else:
                    logging.debug(f"Partial file {file_name} exists locally")
                    resume_header = {"Range": f"bytes={local_content_length}-"}
            else:
                logging.debug(f"File {file_name} does not exist locally")
                local_content_length = 0
                resume_header = {}

            # Start file download
            logging.debug(f"Starting download for file {file_name}")
            with requests.get(url, stream=True, headers=resume_header) as response:
                response.raise_for_status()

                if (
                    "Accept-Ranges" not in head_response.headers
                    or head_response.headers["Accept-Ranges"] == "none"
                ):
                    if local_content_length != 0:
                        logging.warning(
                            "Server does not support partial content. Unable to resume download."
                        )

                    with open(local_path, "wb") as file:
                        logging.debug(f"Writing file {file_name} to disk")
                        for chunk in response.iter_content(chunk_size):
                            file.write(chunk)
                else:
                    with open(local_path, "ab") as file:
                        logging.debug(
                            f"Resuming file {file_name} from byte {local_content_length}"
                        )
                        for chunk in response.iter_content(chunk_size):
                            file.write(chunk)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading file {file_name} with message: {e}")
        else:
            logging.info(f"Successfully downloaded {file_name}")
            break


def read_csv(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.

    Parameters:
    - file_path (str): The path to the CSV file to be read.

    Returns:
    list of dict: A list where each element is a dictionary representing a row in the CSV file.
                  The keys of the dictionaries are the column names (headers), and the values are the
                  corresponding values from each row.

    Example:
    >>> file_path = "example.csv"
    >>> data = read_csv(file_path)
    >>> print(data)
    [{'column1': 'value1', 'column2': 'value2', ...}, {'column1': 'value3', 'column2': 'value4', ...}, ...]
    """
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
