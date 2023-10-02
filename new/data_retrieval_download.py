"""
data_retrieval_download.py

This Python script is designed to download data files specified in a JSON summary file, providing options to download specific runs and handling failed downloads. It leverages configuration constants from an external module for flexibility.

Usage:
    python data_retrieval_download.py [run_numbers [run_numbers ...]] [--keep]

Arguments:
    run_numbers (optional): A list of integers representing specific run numbers to download. If provided, only the runs with matching numbers will be downloaded.
    --keep (optional): If this flag is set, downloaded files will not be deleted in case of download failure.

Configuration (imported from 'config.py'):
    - DATA_DOWNLOAD_DIRECTORY: The directory where downloaded files will be saved.
    - DATA_SUMMARY_PATH: The path to the JSON summary file containing information about the data runs.
    - NETWORK_TIMEOUT: The maximum time (in seconds) to wait for a network response when downloading a file.

The script performs the following steps:
1. Initializes logging to record download progress and errors.
2. Parses command-line arguments to optionally specify which runs to download and whether to keep failed downloads.
3. Checks if the specified data directories and summary file exist; exits if not.
4. Reads run information from the JSON summary file.
5. Filters the runs based on the provided run numbers, if any.
6. Defines a function to download a file from a given URL and saves it to the download directory.
7. Sets up a Dask bag with the list of runs to parallelize the download process.
8. Schedules the download_file function for each run in the bag.
9. Executes the download tasks in parallel.
10. Closes the Dask client and logs any fatal exceptions.

Note: The script assumes that it is executed using a Dask cluster for parallel processing.

Example Usage:
- Download all runs:
    python data_retrieval_download.py

- Download specific runs (e.g., run numbers 1 and 2) and keep failed downloads:
    python data_retrieval_download.py 1 2 --keep
"""

import os
import requests
import json
import logging
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

from config import DATA_DOWNLOAD_DIRECTORY, DATA_SUMMARY_PATH, NETWORK_TIMEOUT

def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_download.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data retrieval download process...")
    
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('run_numbers', metavar='run_number', nargs='*', type=int)
    parser.add_argument("--keep", action="store_true", help="Do not delete file if download process fails")
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}.")
        exit()

    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error(f"There is no data summary file found at {DATA_SUMMARY_PATH}.")
        exit()
        
    # Get run information from summary file
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
        
    # If runs are provided as arguments, only download the specified runs
    if (len(args.run_numbers) > 0):
        logging.info(f"Runs argument present, only downloading: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        runs = [run for run in runs if run["name"] in run_numbers]
    
    # Function that will try to download a file to the downloads directory
    def download_file(run):
        try:
            # Compose full path where to save the downloaded run
            filename = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["url"].split('/')[-1])
            # Open connection to url, stream the data
            with requests.get(run["url"], timeout=NETWORK_TIMEOUT, stream=True) as response:
                response.raise_for_status()
                # Open a file on the full path in binary mode
                with open(filename, 'wb') as file:
                    # Push data downloaded from server to file in chunks so it fits in memory
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logging.info(f"Downloaded {run['name']}.")
        except Exception as e:
            # Delete partially downloaded file on failure if it exists and the keep flag is not set
            if not args.keep and os.path.exists(filename):
                os.remove(filename)
            logging.exception(f"Failed to download {run['name']}!")

    # Set up bag with runs
    bag = db.from_sequence(runs)
    # For each run, schedule the download file function
    tasks = bag.map(download_file)

    # Execute task
    results = tasks.compute()

    
if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()