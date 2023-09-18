import os
import logging
import traceback
import argparse
import numpy as np
import pandas as pd

from tsfresh.feature_selection.relevance import calculate_relevance_table

# TODO set these variables in single external file
DATA_FEATURES_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/features"
        
def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("feature_selection.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting data preprocessing process...")
    
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('run_numbers', metavar='run_number', nargs='*', type=int)
    args = parser.parse_args()
    
    # Set Pandas options to increase readability
    pd.set_option('display.float_format', lambda x: '%.9f' % x)
    pd.options.display.max_rows = 1000

    # Check if directories exist
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(f"The features data directory does not exist at {DATA_FEATURES_DIRECTORY}.")
        exit()
    
    for run_number in args.run_numbers:
        logging.info(f"Processing run {run_number:03d}")
        # Combine data directory with provided run number to open .h5 file in read mode
        csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
        df = pd.read_csv(csv_path)
        
        df = df.dropna(axis=1, how='all')
        
        y = pd.Series(data = df['valid'])
            
        relevance_table = calculate_relevance_table(df, y)
        relevance_table = relevance_table[relevance_table.relevant]
        relevance_table.sort_values("p_value", inplace=True)
        print(relevance_table[:10])
                        
if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")