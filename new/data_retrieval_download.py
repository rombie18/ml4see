import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client
import os
import requests
import json
import logging

DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/data_retrieval_download.json"
NETWORK_TIMEOUT = 500

def main():
    logging.basicConfig(filename='data_retrieval_download.log', filemode="w", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info("Starting data retrieval download process...")

    client = Client()

    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.warning("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        
    if not os.path.exists(DATA_SUMMARY_PATH):
        logging.warning("There is no data summary file found at {}.".format(DATA_SUMMARY_PATH))
        
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)
        
    def download_file(run):
        try:
            filename = os.path.join(DATA_DOWNLOAD_DIRECTORY, run["url"].split('/')[-1])
            with requests.get(run["url"], timeout=NETWORK_TIMEOUT, stream=True) as response:
                response.raise_for_status()
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logging.info("Downloaded {}.".format(run["name"]))
        except Exception as e:
            logging.fatal("Failed to download {}!".format(run["name"]), exc_info=True)

    bag = db.from_sequence(runs)
    tasks = bag.map(download_file)

    #tasks.visualize()
    results = tasks.compute()
    client.close()
    
if __name__ == "__main__":
    main()