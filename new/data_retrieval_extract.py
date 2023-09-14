import os
import logging
import tarfile

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

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        exit()
        
    if not os.path.exists(DATA_RAW_DIRECTORY):
        logging.error("The data raw directory does not exist at {}.".format(DATA_RAW_DIRECTORY))
        exit()
        
    # Get all downloaded runs
    tar_files = []
    for filename in os.listdir(DATA_DOWNLOAD_DIRECTORY):
        if filename.endswith(".tar"):
            tar_files.append(os.path.join(DATA_DOWNLOAD_DIRECTORY, filename))
    
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