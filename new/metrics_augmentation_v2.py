import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
    
from config import DATA_DOWNLOAD_DIRECTORY
from utils import exponential_decay

# TODO add support for partial file handling, combine to one file if all parts valid checksum
START = -1e-3
STOP = 10e-3
SAMPLES = 1100

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

    
    time = np.linspace(START, STOP, SAMPLES)
    signal = np.zeros(SAMPLES)
    
    for i in range(1):
        # Construct signal
        index = 50
        signal[index:] = np.add(signal[index:], exponential_decay(np.linspace(0, STOP, SAMPLES)[:-index], args.amplitude, args.decay, args.offset))
        
        index = 100
        signal[index:] = np.add(signal[index:], exponential_decay(np.linspace(0, STOP, SAMPLES)[:-index], args.amplitude, args.decay, args.offset))
        
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
        
        plt.figure(figsize=(6,10))
        plt.subplot(4, 1, 1)
        plt.plot(time, signal)
        
        plt.subplot(4, 1, 2)
        plt.plot(time, 20 * np.log10(signal))
        
        plt.subplot(4, 1, 3)
        plt.plot(time, noisy_signal)
        
        plt.subplot(4, 1, 4)
        plt.plot(time, 20 * np.log10(noisy_signal))
        
        plt.savefig(
            "temp.png",
            bbox_inches="tight",
        )
        plt.close()

    logging.info("Done!")
    
    
if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Fatal exception in main")
