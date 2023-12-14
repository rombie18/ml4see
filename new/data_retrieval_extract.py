"""
data_retrieval_extract.py

This script performs the extraction of downloaded tar files in parallel, using multiple processes.
It extracts the contents of tar files into a specified output directory. The script supports
extracting specific runs by providing run numbers as command-line arguments.

Usage:
    python data_retrieval_extract.py [run_numbers ...]

Parameters:
    - run_numbers (int, optional): Specify one or more run numbers to extract. If not provided,
                                   all runs from the data download directory will be extracted.

The script uses the `config` module for data download and raw data directories.

Functions:
    - main(): The main entry point for the script.
    - parallel_untar(tar_file, output_dir): Extracts a tar file in parallel using multiple processes.
    - untar_chunk(tar_file, tar_members, output_dir, chunk_start, chunk_end): Extracts a chunk of a tar file.

Note: This script requires the `config` module, and the `parallel_untar` and `untar_chunk` functions
      assume the correct implementation of the `tarfile` module and support for concurrent processing.

Example Usage:
    python data_retrieval_extract.py 1 2 3
"""

import os
import logging
import tarfile
import argparse
from multiprocessing import Pool

from config import DATA_DOWNLOAD_DIRECTORY, DATA_RAW_DIRECTORY


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_retrieval_extract.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting data retrieval extract process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(
            f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}."
        )
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
    if len(args.run_numbers) > 0:
        run_names = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        run_numbers = args.run_numbers
        tar_files = [
            tar_file
            for tar_file in tar_files
            if tar_file.split("/")[-1][:-4] in run_names
        ]
        logging.info(f"Runs argument present, only extracting: {run_numbers}")
    else:
        run_numbers = [int(tar_file.split("/")[-1][4:-4]) for tar_file in tar_files]
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    # Start extract each .tar file in parallel
    logging.info("Starting .tar extraction")
    for tar_file in tar_files:
        parallel_untar(tar_file, DATA_RAW_DIRECTORY)


def parallel_untar(tar_file, output_dir):
    """
    Extracts a tar file in parallel using multiple processes.

    Args:
        tar_file (str): Path to the tar file to be extracted.
        output_dir (str): Directory where the contents of the tar file should be extracted.
    """

    file_name = tar_file.split("/")[-1]

    with tarfile.open(tar_file, "r") as tar:
        logging.debug(f"Calculating tar members of {file_name}")
        tar_members = tar.getmembers()

        total_members = len(tar_members)
        num_threads = os.cpu_count()
        chunk_size = total_members // num_threads

        logging.info(
            f"Extracting {file_name} in {num_threads} chunks of size {chunk_size}"
        )

        with Pool() as pool:
            args = []
            for i in range(num_threads):
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size if i != num_threads - 1 else None
                args.append((tar_file, tar_members, output_dir, chunk_start, chunk_end))

            pool.map(untar_chunk_args, args)

    logging.info(f"Successfully extracted {file_name}")


def untar_chunk_args(args):
    tar_file, tar_members, output_dir, chunk_start, chunk_end = args
    return untar_chunk(tar_file, tar_members, output_dir, chunk_start, chunk_end)


def untar_chunk(tar_file, tar_members, output_dir, chunk_start, chunk_end):
    """
    Extracts a chunk of a tar file.

    Args:
        tar_file (str): Path to the tar file to be extracted.
        tar_members (list): List of tar members.
        output_dir (str): Directory where the contents of the tar file should be extracted.
        chunk_start (int): Index of the starting member in the chunk.
        chunk_end (int): Index of the ending member in the chunk.
    """

    logging.debug(f"Starting untar of chunk ({chunk_start}->{chunk_end})")
    with tarfile.open(tar_file, "r") as tar:
        members = tar_members[chunk_start:chunk_end]
        tar.extractall(path=output_dir, members=members)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
