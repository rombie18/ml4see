import os
import json
import logging
import subprocess
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

# TODO set these variables in single external file
DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/new/data_retrieval_download.json"

def main():
    # Initialise logging
    # FIXME logging not working in Dask workers (other processes)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_verify.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data retrieval verify process...")
    
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('run_numbers', metavar='N', nargs='+', type=int)
    parser.add_argument("--keep", action="store_false", help="Do not delete file if verification process fails")
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}.")
        exit()
        
    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error(f"The data summary file does not exist at {DATA_SUMMARY_PATH}.")
        exit()
    
    # Get run information from summary file
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
        
    # If runs are provided as arguments, only verify the specified runs
    if (len(args.run_numbers) > 0):
        logging.info(f"Runs argument present, only verifying: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        runs = [run for run in runs if run["name"] in run_numbers]
    
    # Function that will verify file integrity using a md5sum
    def validate_file(run):
        # Compose full path where to save the downloaded run
        file_path = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["name"] + ".tar")
        
        # If run is not found on disk or has no md5sum available, skip
        if not os.path.exists(file_path):
            logging.warning(f"Skip validating {run['name']} since it was not found on disk!")
            return
        
        if "md5sum" not in run:
            logging.warning(f"Skip validating {run['name']} since it has no md5 checksum available!")
            return
            
        # Execute md5sum command and pipe output to this python script and select result
        result = subprocess.run(['md5sum', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        md5_returned = result.stdout.split()[0]

        # Check if calculated md5sum matches the provided one in the data summary file
        if not md5_returned == run["md5sum"]:
            logging.error(f"Validation of {run['name']} failed!")
            logging.error(f"Got {md5_returned} as checksum but expected {run['md5sum']}.")
            # If the keep flag is not set, delete the file if verification fails
            if not args.keep:
                os.remove(file_path)
            return

        logging.info("Validation of {run['md5sum']} succeeded.")
        return

    # Set up bag with runs
    bag = db.from_sequence(runs)
    # For each run, schedule the validate file function
    tasks = bag.map(validate_file)

    # Execute task
    results = tasks.compute()


if __name__ == "__main__":
    # Set-up Dask local cluster for distributed processing
    cluster = LocalCluster(
        n_workers=20,
        threads_per_worker=1,
    )
    client = Client(cluster)
    
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
    finally:
        client.close()