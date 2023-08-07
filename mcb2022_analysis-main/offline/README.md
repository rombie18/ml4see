# Data Analysis Pipeline Documentation

## Pipeline
  * Stage 1: `create_run_hdf5.py`  - creates a single HDF5 file for each run, containing all demodulated data
  * Stage 2: `annotate_baseline_stats.py` - considers only baseline, estimates frequency, oscillator standard deviation, scores outliers
  * Stage 3: `annotate_seft_parameters.py` - extracts peak frequency excursion and fits exponential decay to each transient
