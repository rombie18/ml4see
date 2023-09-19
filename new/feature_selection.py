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

    # Check if directories exist
    if not os.path.exists(DATA_FEATURES_DIRECTORY):
        logging.error(f"The features data directory does not exist at {DATA_FEATURES_DIRECTORY}.")
        exit()
    
    for run_number in args.run_numbers:
        logging.info(f"Processing run {run_number:03d}")
        # Combine data directory with provided run number to open .h5 file in read mode
        csv_path = os.path.join(DATA_FEATURES_DIRECTORY, f"run_{run_number:03d}.csv")
        df = pd.read_csv(csv_path, dtype={'transient': 'category'})
        
        # Drop transient column and columns that contain one or more NaN values
        df = df.drop('transient', axis=1)
        df = df.dropna(axis=1, how='all')

        # Target vector to predict is the valid column
        y = pd.Series(data = df['valid'])
            
        # Calculate relevance table
        relevance_table = calculate_relevance_table(df, y)
        relevance_table = relevance_table[relevance_table.relevant]
        relevance_table.sort_values("p_value", inplace=True)
        
        # Save and print relevance table
        relevance_table.to_csv(f"feature_selections_{run_number:03d}.csv")
        print(relevance_table)
        

if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")