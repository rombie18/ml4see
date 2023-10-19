"""
feature_selection.py

This Python script is designed to perform feature selection based on relevance analysis using the `tsfresh` library. It reads feature data from CSV files, drops irrelevant features, and saves the relevance table to a CSV file. Configuration constants are imported from an external module for flexibility.

Usage:
    python feature_selection.py [run_numbers [run_numbers ...]]

Arguments:
    run_numbers (optional): A list of integers representing specific run numbers to perform feature selection on.

Configuration (imported from 'config.py'):
    - DATA_FEATURES_DIRECTORY: The directory where feature CSV files are located.

The script performs the following steps:
1. Initializes logging to record the feature selection process's progress and errors.
2. Parses command-line arguments to optionally specify which runs to perform feature selection on.
3. Checks if the specified features data directory exists; exits if not.
4. Iterates through the provided run numbers, performing feature selection on each run individually.
5. Reads feature data from CSV files, drops irrelevant features, and calculates the relevance table.
6. Saves the relevance table to a CSV file and prints it.

Example Usage:
- Perform feature selection on all available runs:
    python feature_selection.py

- Perform feature selection on specific runs (e.g., run numbers 1 and 2):
    python feature_selection.py 1 2
"""

import os
import logging
import traceback
import argparse
import numpy as np
import pandas as pd
from tsfresh.feature_selection.relevance import calculate_relevance_table

from config import DATA_FEATURES_DIRECTORY
        
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

    #TODO maybe move directory and file checks to config file?
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