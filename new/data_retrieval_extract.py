import os
import logging
import tarfile
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

# TODO set these variables in single external file
DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_RAW_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw"

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