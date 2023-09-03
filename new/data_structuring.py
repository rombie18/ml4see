import dask.dataframe as dd
import dask.bag as db
import dask.array as da
from dask.distributed import Client, print
import os
import pathlib
import time
import struct
import traceback

import h5py
import pandas as pd
import numpy as np
import scipy.signal as sps

import util

DATA_RAW_SDR_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw"
DATA_RAW_GLASGOW_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/raw/mcb2022_glasgow"
DATA_STRUCTURED_DIRECTORY = "/home/r0835817/2023-WoutRombouts-NoCsBack/ml4see/structured"

META_PROCESSING_STAGE = 1  # processing stage of generated HDF files
META_STAGE_1_VERSION = "3.0"  # version string for this stage (stage 1)

# Major changes
# v1.0-alpha: initial structure
# v1.0: by_x, by_y groups for SDR data
# v2.0: sdr transients now in "all" group containing all transients (ancleard no other subgroups)
# v3.0: added support for parallelisation and clustering using Dask

META_STATIC_ATTR_LIST = [
    ("run_date", str),  # date of run
    ("run_time_appx", str),  # approximate start time
    ("run_comment", str),  # comment (string)
    ("run_beam_active", bool),  # beam was incident on DUT?
    ("scan_type", str),  # type of scan (S, X, Y, XY)
    ("scan_x_start", float),  # X start coordinate
    ("scan_x_stop", float),  # X stop coordinate
    ("scan_x_steps", int),  # number of X steps
    ("scan_y_start", float),  # Y start coordinate
    ("scan_y_stop", float),  # Y stop coordinate
    ("scan_y_steps", int),  # Y number of steps
    ("scan_hits_per_step", int),  # number of hits per position
    ("scan_step_timeout", float),  # timeout at each position (0 = no timeout)
    ("scan_unit", str),  # unit of scan parameters
    ("scan_x_lsb_per_um", float),  # X calibration coefficient (DAC LSB per micrometer)
    ("scan_y_lsb_per_um", float),  # Y calibration coefficient
    ("dut_type", str),  # type of DUT (ADPLL, LJPLL, CAL)
    ("dut_name", str),  # name of DUT (ADPLLn, LJPLLn, FIBn, CALs)
    ("dut_position", str),  # position over DUT (corresponds to images from kay)
    ("sdr_data", bool),  # presence of SDR data in this run
]

META_ADPLL_ATTR_LIST = [
    ("dut_info_dco_pvt", int),  # DCO PVT bank (0..79, higher value = lower freq)
    ("dut_info_dco_acq", int),  # DCO ACQ bank (0..63, higher value = higher freq)
    ("dut_info_dco_ibias", int),  # DCO bias current (0..23, 255=bypass)
]

META_LJPLL_ATTR_LIST = [
    ("dut_info_vco_capsel", int),  # VCO capacitor DAC setting (0..6)
    ("dut_into_vco_vtune", int),  # VCO external tune voltage
    ("dut_info_vco_ibias", int),  # VCO bias current (0..127, 255=bypass)
]

SDR_ATTR_LIST = [
    ("sdr_info_cf", float),
    ("sdr_info_fs", float),
]


class SDREventProcessor:
    """DSP for SDR transients"""

    def __init__(self, fs=20e6, bw=3e6, n_fir=129, tran_length=None):
        assert n_fir % 2 == 1, "Require odd FIR order."
        self._fs = fs
        self._bw = bw
        self._n_fir = n_fir
        self._taps = sps.remez(n_fir, [0, bw, bw + 200e3, fs / 2], [1, 0], fs=fs)
        self._tran_length = tran_length

    def file_to_iq(self, fname):
        if self._tran_length is not None:
            data = np.fromfile(
                fname, dtype=np.int16, offset=24, count=2 * self._tran_length
            )
        else:
            data = np.fromfile(fname, dtype=np.int16, offset=24)
        data = data.astype(float)
        data = da.from_array(data)
        data_iq = data[0::2] + 1.0j * data[1::2]
        return data_iq

    def _downconvert_and_filter(self, data_iq):
        # complex NCO for fs/4 downconversion
        dc_array = da.array([1 + 0j, 0 - 1j, -1 + 0j, 0 + 1j])
        dc_array = da.tile(dc_array, len(data_iq) // 4)
        dc_array = dc_array[: len(data_iq)]
        data_iq = data_iq * dc_array

        ## low pass filter (band select)
        data_iq_np = data_iq.compute()
        data_iq = np.convolve(data_iq_np, self._taps, "valid")

        return data_iq

    def _freq_demod(self, data_iq):
        def dask_unwrap(phase, discont=np.pi, axis=-1):
            phase_diff = da.diff(phase, axis=axis, prepend=0)
            jump = (phase_diff < -discont) | (phase_diff > discont)
            jump_cumsum = da.cumsum(jump, axis=axis)
            unwrapped_phase = phase + jump_cumsum * (2 * np.pi)
            return unwrapped_phase

        phase = dask_unwrap(da.arctan2(da.imag(data_iq), da.real(data_iq)))
        freq_hz = da.diff(phase) * self._fs / (2 * np.pi)
        return freq_hz

    def file_to_dc_iq(self, fname):
        data_iq = self.file_to_iq(fname)
        data_iq = self._downconvert_and_filter(data_iq)
        return data_iq

    def file_to_freq(self, fname):
        try:
            data_iq = self.file_to_iq(fname)
            data_iq = self._downconvert_and_filter(data_iq)
            freq_hz = self._freq_demod(data_iq)
            return freq_hz
        except Exception as e:
            print(e)
            traceback.print_exc()

            

def find_run_folder(candidate_list, run_id):
    for folder in candidate_list:
        candidate = os.path.join(folder, f"run_{run_id:03d}")
        if os.path.exists(candidate) and os.path.isdir(candidate):
            return candidate
    raise RuntimeError(f"Run folder not found in {str(candidate_list)}")

def create_log_attrs(attr_list, node, logbook):
    for attr_name, attr_type in attr_list:
        attr_val = logbook[attr_name]
        node.attrs.create(attr_name, attr_type(attr_val))


def create_sdr_datasets(run_folder, node, tran_length=None, chunk_size=512):
    all_group = node.create_group("all")
    by_x_group = node.create_group("by_x")
    by_y_group = node.create_group("by_y")

    sdr_fs = node.attrs.get("sdr_info_fs")
    DSP_PRE_DEMOD_LPF_BW_HZ = 3e6
    DSP_PRE_DEMOD_LPF_TAPS = 129

    dsp = SDREventProcessor(
        fs=sdr_fs,
        bw=DSP_PRE_DEMOD_LPF_BW_HZ,
        n_fir=DSP_PRE_DEMOD_LPF_TAPS,
        tran_length=tran_length,
    )
    node.attrs.create("dsp_info_pre_demod_lpf_bw_hz", DSP_PRE_DEMOD_LPF_BW_HZ)
    node.attrs.create("dsp_info_pre_demod_lpf_taps", DSP_PRE_DEMOD_LPF_TAPS)
    # TODO: better evaluate the downconversion strategy
    #   - currently: fixed downconversion frequency -> simplfies, might cause problems for DUT frequencies far from +fs/4
    #   - "adaptive" downconversion frequency -> optimizes BW for each transient, but might result in DC carrier issues

    # relative filenames are of the form: x_000000/y_-00093/trig_019740.dat
    # to avoid costly regex matches, we just use their fixed length to parse out the important bits
    len_prefix = len(run_folder) + 1
    print(f"  Getting list of files...")
    fnames = [str(fname) for fname in pathlib.Path(run_folder).rglob("*.dat")]

    file_dicts = [
        {
            "fname": fname,
            "x": int(fname[len_prefix + 2 : len_prefix + 8]),
            "y": int(fname[len_prefix + 11 : len_prefix + 17]),
            "id": int(fname[len_prefix + 23 : len_prefix + 29]),
        }
        for fname in fnames
    ]

    
    time_start = time.time()
    # We iterate over the data in chunks. This is required since:
    #  - we want to parallelize the DSP
    #  - parallel writes to the HDF datafile are not thread-safe
    #  - we only have finite amounts of RAM, so we can't process all transients before storing them
    
    # extract frequency information
    bag = db.from_sequence([entry["fname"] for entry in file_dicts])
    tasks = bag.map(dsp.file_to_freq)
    freqs = tasks.compute()
    
    for tran_dict, freq in zip(file_dicts, freqs):
        if tran_dict["x"] == 999999 or tran_dict["y"] == 999999:
            continue
        # recover timestamp information
        with open(tran_dict["fname"], "rb") as tranfile:
            ts_data = struct.unpack("<QQQ", tranfile.read(24))
        sys_ts_sec = float(ts_data[1]) + float(ts_data[2]) / 1e6
        hw_ts_sec = float(ts_data[0]) / sdr_fs

        # store everything to a dataset
        tran_ds = all_group.create_dataset(f"tran_{tran_dict['id']:06d}", data=freq)
        tran_ds.attrs.create("tran_num", tran_dict["id"])
        tran_ds.attrs.create("x_lsb", tran_dict["x"])
        tran_ds.attrs.create("y_lsb", tran_dict["y"])
        tran_ds.attrs.create("hw_ts_sec", hw_ts_sec)
        tran_ds.attrs.create("sys_ts_sec", sys_ts_sec)
        tran_ds.attrs.create("dsp_info_pre_demod_bb_lo_freq_hz", sdr_fs / 4)
        tran_ds.attrs.create("dataset_unit", "Hz")

        # append dataset to by-x hierarchy
        by_x_x_group = by_x_group.require_group(f"x_{tran_dict['x']:06d}")
        if "x_lsb" not in by_x_x_group.attrs:
            by_x_x_group.attrs.create("x_lsb", tran_dict["x"])
        by_x_y_group = by_x_x_group.require_group(f"y_{tran_dict['y']:06d}")
        if "y_lsb" not in by_x_y_group.attrs:
            by_x_y_group.attrs.create("y_lsb", tran_dict["y"])
        by_x_y_group[f"tran_{tran_dict['id']:06d}"] = tran_ds

        # append dataset to by-y hierarchy
        by_y_y_group = by_y_group.require_group(f"y_{tran_dict['y']:06d}")
        if "y_lsb" not in by_y_y_group.attrs:
            by_y_y_group.attrs.create("y_lsb", tran_dict["y"])
        by_y_x_group = by_y_y_group.require_group(f"x_{tran_dict['x']:06d}")
        if "x_lsb" not in by_y_x_group.attrs:
            by_y_x_group.attrs.create("x_lsb", tran_dict["x"])
        by_y_x_group[f"tran_{tran_dict['id']:06d}"] = tran_ds

    time_elapsed = time.time() - time_start
    tps = len(file_dicts) / time_elapsed
    print(
        f"  Processing done. Elapsed time: {time_elapsed:.02f} s. Performance; {tps:.02f} files/s"
    )

def process_run(run_number):

    print(f"Processing run {run_number:03d}")
    
    sdr_data_folders = DATA_RAW_SDR_DIRECTORY.split(":")
    glasgow_data_folders = DATA_RAW_GLASGOW_DIRECTORY.split(":")

    # get run information from logbook
    logbook_data = pd.read_excel("logbook/mcb2022_logbook.xlsx", index_col=0).loc[
        run_number
    ]

    pretrig_samples = logbook_data["sdr_info_len_pretrig"]
    posttrig_samples = logbook_data["sdr_info_len_posttrig"]

    run_folder_glasgow = find_run_folder(glasgow_data_folders, run_number)
    glasgow_txt_log_path = os.path.join(run_folder_glasgow, "run_log.txt")
    glasgow_csv_log_path = os.path.join(run_folder_glasgow, "hit_log.csv")

    assert os.path.exists(glasgow_txt_log_path), "Glasgow text log file not found"
    assert os.path.exists(glasgow_txt_log_path), "Glasgow CSV log file not found"

    h5_path = os.path.join(DATA_STRUCTURED_DIRECTORY, f"run_{run_number:03d}.h5")
    with h5py.File(h5_path, "w") as h5file:
        # add run metadata
        print("Adding metadata...")
        meta_ds = h5file.create_dataset("meta", dtype="f")
        meta_ds.attrs.create("processing_stage", META_PROCESSING_STAGE)
        meta_ds.attrs.create("processing_stage_1_version", META_STAGE_1_VERSION)
        meta_ds.attrs.create("run_id", run_number)
        create_log_attrs(
            attr_list=META_STATIC_ATTR_LIST, node=meta_ds, logbook=logbook_data
        )

        if logbook_data["dut_type"] == "ADPLL":
            create_log_attrs(
                attr_list=META_ADPLL_ATTR_LIST, node=meta_ds, logbook=logbook_data
            )
        if logbook_data["dut_type"] == "LJPLL":
            create_log_attrs(
                attr_list=META_LJPLL_ATTR_LIST, node=meta_ds, logbook=logbook_data
            )

        # add FPGA text log
        print("Adding FPGA log text file...")
        with open(glasgow_txt_log_path, "r") as glasgow_log:
            h5file.create_dataset("fpga_log", data=glasgow_log.read())

        # add FPGA hit data
        print("Adding FPGA hit log...")
        fpga_hit_data = np.genfromtxt(glasgow_csv_log_path, delimiter=",", names=True)
        fpga_hit_data["hw_ts_10us"] /= 1e5
        fpga_hit_data.dtype.names = ("hw_ts_sec",) + fpga_hit_data.dtype.names[1:]
        h5file.create_dataset("fpga_hit_data", data=fpga_hit_data)

        # process SDR data if available for this run
        if logbook_data["sdr_data"]:
            print("SDR data available for this run - processing SDR data.")
            sdr_group = h5file.create_group("sdr_data")
            # add SDR-related metadata
            create_log_attrs(
                attr_list=SDR_ATTR_LIST, node=sdr_group, logbook=logbook_data
            )
            sdr_group.attrs.create("sdr_info_len_pretrig", int(pretrig_samples))
            sdr_group.attrs.create("sdr_info_len_posttrig", int(posttrig_samples))
            run_folder_sdr = find_run_folder(sdr_data_folders, run_number)
            create_sdr_datasets(
                run_folder=run_folder_sdr,
                node=sdr_group,
                tran_length=int(pretrig_samples + posttrig_samples),
            )

    print("Done.")
    print()
    
    
def main():
    client = Client()
    
    entries = os.listdir(DATA_RAW_SDR_DIRECTORY)
    prefix = "run_"
    run_numbers = [int(entry.replace(prefix, "")) for entry in entries if entry.startswith(prefix)]
    
    try:
        process_run(6)
    except:
        print("FAILED TO PROCESS RUN")

    # for run_number in run_numbers:
    #     try:
    #         process_run(run_number)
    #     except:
    #         print("FAILED TO PROCESS RUN {}".format(run_number))

    client.close()
    
    
if __name__ == "__main__":
    main()