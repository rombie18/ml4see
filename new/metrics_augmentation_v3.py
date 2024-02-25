import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
    
from config import DATA_DOWNLOAD_DIRECTORY
from utils import exponential_decay

# TODO add support for partial file handling, combine to one file if all parts valid checksum


def main():
    # Initialise logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("metrics_augmentation.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting augmentation process...")

    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', default=10, type=int, metavar="VALUE", help="The peak value amplitude at the trigger of the exponential decay. (default: %(default)s ppm)")
    parser.add_argument('--decay', default=1000, type=int, metavar="VALUE", help="The exponential decay constant. (default: %(default)s 1/s)")
    parser.add_argument('--offset', default=0, type=int, metavar="VALUE", help="The asymptotic y-axis offset to which the decay evolves. (default: %(default)s ppm)")
    parser.add_argument('--snr', default=15, type=int, metavar="VALUE", help="The signal-noise-ratio (SNR) of the constructed noisy signal. (default: %(default)s dB)")
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
        logging.error(
            f"The data download directory does not exist at {DATA_DOWNLOAD_DIRECTORY}."
        )
        exit()

    # Main code

    pretrig_time = np.linspace(-1e-3, 0, 100)
    posttrig_time = np.linspace(0, 10e-3, 1000)
    
    for i in range(1000):
        # Construct signal
        pretrig_signal = np.zeros(100)
        posttrig_signal = exponential_decay(posttrig_time, args.amplitude, args.decay, args.offset)
        time = np.concatenate((pretrig_time, posttrig_time))
        signal = np.concatenate((pretrig_signal, posttrig_signal))
        
        # Set a target SNR
        target_snr_db = args.snr
        # Calculate signal power and convert to dB 
        signal_watts = signal ** 2
        sig_avg_watts = np.mean(signal_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        
        # Generate an sample of white noise
        noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal_watts))
        # Noise up the original signal
        noisy_signal = signal + noise
        
        
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    plt.figure(figsize=(5,5))
    plt.plot(x, y, 'x')
    plt.axis('equal')

    plt.savefig(
        f"temp.png", bbox_inches="tight"
    )
    plt.close()

    logging.info("Done!")
    
    
    
    
if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
