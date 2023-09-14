import os
import requests
import json
import logging
import argparse

import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster

# TODO set these variables in single external file
DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/new/data_retrieval_download.json"
NETWORK_TIMEOUT = 500

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
    parser.add_argument('runs', metavar='N', nargs='+', type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        exit()

    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.error("There is no data summary file found at {}.".format(DATA_SUMMARY_PATH))
        exit()
        
    # Get run information from summary file
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
        
    # If runs are provided as arguments, only download the specified runs
    if (len(args.runs) > 0):
        logging.info(f"Runs argument present, only downloading: {args.runs}")
        run_names = [f"run_{arg:03d}" for arg in args.runs]
        runs = [run for run in runs if run["name"] in run_names]
    
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
            logging.info("Downloaded {}.".format(run["name"]))
        except Exception as e:
            logging.fatal("Failed to download {}!".format(run["name"]), exc_info=True)

    # Set up bag with runs
    bag = db.from_sequence(runs)
    # For each run, schedule the download file function
    tasks = bag.map(download_file)

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