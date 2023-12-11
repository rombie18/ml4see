# ML4SEE
Rejecting outliers from captured SEE-transients data @ CERN using ML

In this repository, a comprehensive data processing pipeline is designed that allows you to download, verify, extract, and analyze large datasets using Python scripts. The goal of the project is to gain insight into the radiation hardness of irradiated LC oscillator circuits, to improve existing processing scripts, generate relevant plots, and especially to reject measurement faults from the data.

The project involves a comprehensive data processing pipeline for handling experimental data from CERN. The process begins with the data_retrieval_download.py script, allowing users to parallelize file downloads with the ability to resume interrupted transfers. Following this, the integrity of the downloaded files is verified using data_retrieval_verify.py. The next step involves parallel extraction of downloaded tar files using data_retrieval_extract.py. Subsequently, two scripts, namely data_structuring_1.py and data_structuring_2.py, focus on structuring the data by creating consolidated HDF5 files, emphasizing modularity, and parallel processing. The first script processes Software-Defined Radio (SDR) data, while the second annotates baseline data for SDR transients. Finally, the feature_extraction.py script extracts features from transients stored in HDF5 files, supporting parallel processing and saving results as CSV files for further analysis. The overall processing flow encompasses downloading, verification, extraction, structuring, baseline annotation, and feature extraction, providing a comprehensive framework for handling experimental data efficiently.

## Summary of scripts
`data_retrieval_download.py`

The "data_retrieval_download.py" script enables parallel file downloads with the ability to resume interrupted transfers. Users can specify run numbers as command-line arguments to download specific runs. The script utilizes the config module for configuration details, including the data download directory and download attempts. It relies on the requests module for HTTP interactions. Key functions include main(), responsible for initializing the process, parsing arguments, and orchestrating downloads; download_file(), facilitating file downloads with resumption capabilities; and read_csv(), which reads a CSV file into a list of dictionaries. The script handles interruptions gracefully and supports concurrent downloads using a process pool.

`data_retrieval_verify.py`

The "data_retrieval_verify.py" script verifies file integrity by comparing calculated MD5 checksums with expected checksums obtained from a data summary file. Users can specify run numbers as command-line arguments for selective validation. The script supports parallel validation of multiple files and includes an optional flag (--keep) to retain files on verification failure. Utilizing the config module for data download directory and data summary file path, the script relies on functions like validate_file, calculate_md5, and read_csv. The validate_file function performs the validation process, checking file existence and MD5 checksum matching, logging outcomes, and optionally deleting corrupt files. The script handles parallel processing using process pools and logs information and errors.

`data_retrieval_extract.py`

The "data_retrieval_extract.py" script facilitates the parallel extraction of downloaded tar files, employing multiple processes for efficiency. Users can specify run numbers as command-line arguments to extract specific runs. The script utilizes the config module for data download and raw data directories. Key functions include main(), responsible for initializing the process, parsing arguments, and orchestrating the extraction; parallel_untar(), which extracts a tar file in parallel using multiple processes; and untar_chunk(), which extracts a chunk of a tar file. The script supports interruption handling and concurrent extraction using process pools.

`data_structuring_1.py`

The "data_structuring_1.py" script is designed to create consolidated HDF5 files for individual runs, with a primary focus on processing Software-Defined Radio (SDR) data. The script utilizes metadata attributes from a logbook, handles different types of runs (ADPLL and LJPLL) accordingly, and incorporates FPGA log and hit data into the HDF5 file. For runs with SDR data, the script processes it using the SDREventProcessor class, creating datasets and organizing them within the HDF5 file. The script supports command-line arguments for specifying run numbers and post-trigger sample counts. It employs logging for informative and error messages and includes exception handling to address potential errors during the processing. The overall design emphasizes modularity and parallel processing for efficient data structuring.

`data_structuring_2.py`

The "data_structuring_2.py" script represents the second stage of data processing, focusing on annotating baseline data for runs containing Software-Defined Radio (SDR) transients. The script utilizes metadata attributes, including the processing stage and version information. It defines a utility method for chunking datasets and calculates baseline statistics, such as mean, standard deviation, and standard deviation of the mean, for SDR transients in chunks. The script annotates the corresponding HDF5 files with these statistics and updates processing stage and version attributes. The main function parses command-line arguments, determines the runs to process, and performs the specified tasks. 

`feature_extraction.py`

The "feature_extraction.py" script extracts features from transients stored in HDF5 files and saves the results as CSV files. It supports parallel processing of transients using Dask and requires the config module for data directories and various parameters, as well as the utils module for some utility functions. Users can specify run numbers as command-line arguments for selective feature extraction, or if not provided, features will be extracted for all runs found in the structured data directory. The script utilizes the process_transient function to extract features for individual transients, and the main entry point, main(), orchestrates the feature extraction process for specified or all runs, saving the results as CSV files in the features data directory. The script handles parallel processing using Dask, and the logging system provides information about the extraction process and any encountered warnings or errors.

## Processing flow

1. First, download data from CERN servers using the `data_retrieval_download.py` script
2. Check the download .tar files for integrity using `data_retrieval_verify.py`
3. Then use the `data_retrieval_extract.py` script to unpack the files into plain directories and files
4. Next with `data_structuring_1.py` convert the raw unprocessed data to a consolidated .h5 file and add meta data
5. Then using `data_structuring_2.py` calculate some baseline statistics for every transient in the runs and add them to the .h5 file
6. Next use `feature_extraction.py` to calculate features for each transient and write them to a .csv file for further analysis