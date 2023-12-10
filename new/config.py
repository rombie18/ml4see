# Number of times a run download will be retried and attempted to continue before it fails
DOWNLOAD_ATTEMPTS = 20

# Number of guard samples before trigger
PRETRIG_GUARD_SAMPLES = 100
# Threshold to reject exponential fit under coefficient of determination
R2_THRESHOLD = 0.1

# Window size for moving average filter
WINDOW_SIZE = 500
# Downsampling factor from orginal data points to reduced
DOWNSAMPLE_FACTOR = 250

# Block sizes in µm to apply model on
BLOCK_SIZE_X = 20
BLOCK_SIZE_Y = 20
# Overlap percentage (0-100) of blocks overlapping
BLOCK_OVERLAP = 0

# Paths to misc directories
RUN_LOGBOOK_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/new/logbook/mcb2022_logbook.xlsx"
RUN_DUTPICS_DIRECTORY = "/home/r0835817/2023-WoutRombouts/ml4see/new/dut_position_pics"

# Paths to data storage directories, use absolute paths
DATA_SUMMARY_PATH = "/home/r0835817/2023-WoutRombouts/ml4see/new/runs_summary.csv"
DATA_DOWNLOAD_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/download"
DATA_RAW_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw"
DATA_RAW_GLASGOW_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw/mcb2022_glasgow"
DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"
DATA_FEATURES_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/features"
DATA_LABELED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/labeled"