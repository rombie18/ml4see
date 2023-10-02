"""
data_retrieval_extract.py

This Python script is designed to extract files from downloaded tar archives into a specified raw data directory. It provides options to extract files from specific runs and utilizes configuration constants from an external module for flexibility.

Usage:
    python data_retrieval_extract.py [run_numbers [run_numbers ...]]

Arguments:
    run_numbers (optional): A list of integers representing specific run numbers to extract. If provided, only the runs with matching numbers will be extracted.

Configuration (imported from 'config.py'):
    - DATA_DOWNLOAD_DIRECTORY: The directory where downloaded tar files are located.
    - DATA_RAW_DIRECTORY: The directory where extracted raw data will be saved.

The script performs the following steps:
1. Initializes logging to record extraction progress and errors.
2. Parses command-line arguments to optionally specify which runs to extract.
3. Checks if the specified data directories exist; exits if not.
4. Retrieves a list of downloaded tar files from the download directory.
5. Filters the tar files based on the provided run numbers, if any.
6. Defines a function to extract files from a tar archive and save them to the raw data directory.
7. Sets up a Dask bag with the list of tar files to parallelize the extraction process.
8. Schedules the untar_file function for each tar file in the bag.
9. Executes the extraction tasks in parallel.
10. Closes the Dask client and logs any fatal exceptions.

Note: The script assumes that it is executed using a Dask cluster for parallel processing.

Example Usage:
- Extract all downloaded runs:
    python data_retrieval_extract.py

- Extract specific runs (e.g., run numbers 1 and 2):
    python data_retrieval_extract.py 1 2
"""

import os
import logging
import tarfile
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

from config import DATA_DOWNLOAD_DIRECTORY, DATA_RAW_DIRECTORY

def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_extract.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data retrieval extract process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('run_numbers', metavar='run_number', nargs='*', type=int)
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}.")
        exit()
        
    if not os.path.exists(DATA_RAW_DIRECTORY):
        logging.error(f"The data raw directory does not exist at {DATA_RAW_DIRECTORY}.")
        exit()
        
    # Get all downloaded runs
    tar_files = []
    for filename in os.listdir(DATA_DOWNLOAD_DIRECTORY):
        if filename.endswith(".tar"):
            tar_files.append(os.path.join(DATA_DOWNLOAD_DIRECTORY, filename))
            
    # If runs are provided as arguments, only download the specified runs
    if (len(args.run_numbers) > 0):
        logging.info(f"Runs argument present, only extracting: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        tar_files = [tar_file for tar_file in tar_files if tar_file.split('/')[-1][:-4] in run_numbers]
    
    # Function that extracts a tarfile to the raw directory
    def untar_file(file_name):
        with tarfile.open(file_name, 'r') as tar:
            tar.extractall(path=DATA_RAW_DIRECTORY)

    # Set up bag with tar files
    bag = db.from_sequence(tar_files)
    # For tar file, schedule the untar function
    tasks = bag.map(untar_file)

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