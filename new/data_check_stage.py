import os
import logging
import argparse
import h5py

from config import DATA_STRUCTURED_DIRECTORY


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data_check_stage.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting data stage check process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("run_numbers", metavar="run_number", nargs="*", type=int)
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_STRUCTURED_DIRECTORY):
        logging.error(
            f"The structured data directory does not exist at {DATA_STRUCTURED_DIRECTORY}."
        )
        exit()

    # If runs are provided as arguments, only verify the specified runs
    run_numbers = []
    if len(args.run_numbers) > 0:
        run_numbers = args.run_numbers
        logging.info(f"Runs argument present, only verifying: {run_numbers}")
    else:
        for file in os.listdir(DATA_STRUCTURED_DIRECTORY):
            if file.endswith(".h5"):
                run_numbers.append(int(file[4:7]))
        logging.info(f"No runs specified, running on all available runs: {run_numbers}")

    for run_number in run_numbers:
        h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")

        if not os.path.exists(h5_path):
            logging.error(f"run_{run_number:03d} does not exist on disk.")
            continue

        with h5py.File(h5_path, "r") as h5file:
            stage = h5file["meta"].attrs["processing_stage"]
            logging.info(f"run_{run_number:03d} has processing stage {stage}")


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
