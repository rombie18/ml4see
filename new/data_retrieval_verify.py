import os
import json
import logging
import subprocess

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

# TODO set these variables in single external file
DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/new/data_retrieval_download.json"

def main():
    # Initialiser logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_verify.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data retrieval verify process...")

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        exit()
        
    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error("The data summary file does not exist at {}.".format(DATA_SUMMARY_PATH))
        exit()
    
    # Get run information from summary file
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
    
    # Function that will verify file integrity using a md5sum
    def validate_file(run):
        # Compose full path where to save the downloaded run
        file_path = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["name"] + ".tar")
        
        # If run is not found on disk or has no md5sum available, skip
        if not os.path.exists(file_path):
            logging.warning("Skip validating {} since it was not found on disk!".format(run["name"]))
            return
        
        if "md5sum" not in run:
            logging.warning("Skip validating {} since it has no md5 checksum available!".format(run["name"]))
            return
            
        # Execute md5sum command and pipe output to this python script and select result
        result = subprocess.run(['md5sum', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        md5_returned = result.stdout.split()[0]

        # Check if calculated md5sum matches the provided one in the data summary file
        if not md5_returned == run["md5sum"]:
            logging.error("Validation of {} failed!".format(run["name"]))
            logging.error("Got {} as checksum but expected {}.".format(md5_returned, run["md5sum"]))
            return

        logging.info("Validation of {} succeeded.".format(run["md5sum"]))
        return

    # Set up bag with runs
    bag = db.from_sequence(runs)
    # For each run, schedule the validate file function
    tasks = bag.map(validate_file)

    # Execute task
    results = tasks.compute()


if __name__ == "__main__":
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