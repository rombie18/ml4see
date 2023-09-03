import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client
import os
import logging
import tarfile

DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_RAW_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw"

def main():
    logging.basicConfig(filename='data_retrieval_extract.log', filemode="w", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info("Starting data retrieval extract process...")

    client = Client()

    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.warning("The data download directory does not exist at {}.".format(DATA_DOWNLOAD_DIRECTORY))
        
    if not os.path.exists(DATA_RAW_DIRECTORY):
        logging.warning("The data raw directory does not exist at {}.".format(DATA_RAW_DIRECTORY))
        
    tar_files = []
    for filename in os.listdir(DATA_DOWNLOAD_DIRECTORY):
        if filename.endswith(".tar"):
            tar_files.append(os.path.join(DATA_DOWNLOAD_DIRECTORY, filename))
            
    def untar_file(file_name):
        with tarfile.open(file_name, 'r') as tar:
            tar.extractall(path=DATA_RAW_DIRECTORY)

    bag = db.from_sequence(tar_files)
    tasks = bag.map(untar_file)

    #tasks.visualize()
    results = tasks.compute()
    client.close()
    
if __name__ == "__main__":
    main()