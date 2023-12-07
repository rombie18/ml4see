"""
data_retrieval_extract.py

This script extracts data from downloaded tar files in parallel using multiple processes. It supports extracting specific runs if run numbers are provided as command-line arguments.

"""

import os
import logging
import tarfile
import argparse
import concurrent.futures

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
        logging.info(f"Runs argument present, only extracting: {args.run_numbers}")
        run_numbers = [f"run_{run_number:03d}" for run_number in args.run_numbers]
        tar_files = [
            tar_file
            for tar_file in tar_files
            if tar_file.split("/")[-1][:-4] in run_numbers
        ]

    # Start extract each .tar file in parallel
    # TODO find way to gracefully kill downloading on sigterm
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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_threads):
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size if i != num_threads - 1 else None
                future = executor.submit(
                    untar_chunk,
                    tar_file,
                    tar_members,
                    output_dir,
                    chunk_start,
                    chunk_end,
                )
                futures.append(future)

            # TODO find way to hide tracebacks and junk output on keyboard interrupt
            try:
                concurrent.futures.wait(futures)
            except KeyboardInterrupt:
                logging.warning("Ctrl+C pressed. Trying to cancel remaining tasks...")
                for future in futures:
                    future.cancel()
                exit()

    logging.info(f"Successfully extracted {file_name}")


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
