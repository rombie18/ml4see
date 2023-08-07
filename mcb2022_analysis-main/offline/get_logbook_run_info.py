"""Machine-readable campaign logbook query"""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_number", type=int)
    args = parser.parse_args()

    logbook = pd.read_excel("logbook/mcb2022_logbook.xlsx", index_col=0)
    run_data = logbook.loc[args.run_number]
    print("Raw Data:")
    print(run_data)
    print()


if __name__ == "__main__":
    main()
