import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client
import os
import json
import logging
import subprocess

DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/data_retrieval_download.json"

def main():
    logging.basicConfig(filename='data_retrieval_verify.log', filemode="w", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info("Starting data retrieval verify process...")

    client = Client()

    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.warning("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        
    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.warning("The data summary file does not exist at {}.".format(DATA_SUMMARY_PATH))
        
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
        
    def validate_file(run):
        file_path = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["name"] + ".tar")
        if not os.path.exists(file_path):
            logging.warning("Skip validating {} since it was not found on disk!".format(run["name"]))
            return
        
        if "md5sum" not in run:
            logging.warning("Skip validating {} since it has no md5 checksum available!".format(run["name"]))
            return
            
        result = subprocess.run(['md5sum', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        md5_returned = result.stdout.split()[0]
            
        if not md5_returned == run["md5sum"]:
            logging.error("Validation of {} failed!".format(run["name"]))
            logging.error("Got {} as checksum but expected {}.".format(md5_returned, run["md5sum"]))
            return

        logging.info("Validation of {} succeeded.".format(md5_returned, run["md5sum"]))
        return

    bag = db.from_sequence(runs)
    tasks = bag.map(validate_file)

    #tasks.visualize()
    results = tasks.compute()
    client.close()


if __name__ == "__main__":
    main()