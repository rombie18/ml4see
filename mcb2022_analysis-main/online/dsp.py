import numpy as np
import scipy.signal as sps

class EventProcessor:
    def __init__(self, fs=20e6, bw=3e6, n_fir=129):
        assert n_fir % 2 == 1, "Require odd FIR order."
        self._fs = fs
        self._bw = bw
        self._n_fir = n_fir
        self._taps = sps.remez(n_fir, [0, bw, bw+200e3, fs/2], [1, 0], Hz=fs)

    def file_to_iq(self, fname):
        data = np.fromfile(fname, dtype=np.int16, offset=24)
        data = data.astype(float)
        data_iq = data[0::2] + 1.0j * data[1::2]
        t = 1.0 / self._fs * np.arange(len(data_iq)) - 1e-3
        return t, data_iq

    def _downconvert_and_filter(self, t, data_iq):
        # complex NCO for fs/4 downconversion
        dc_array = np.array([ 1 + 0j,
                              0 - 1j, 
                             -1 + 0j,
                              0 + 1j])
        dc_array = np.tile(dc_array, len(data_iq)//4)
        dc_array = dc_array[:len(data_iq)]
        data_iq = data_iq * dc_array

        ## low pass filter (band select)
        data_iq = np.convolve(data_iq, self._taps, 'valid')
        t = t[self._n_fir // 2 : -self._n_fir // 2 + 1]
        return t, data_iq

    def _freq_demod(self, t, data_iq, detrend=False, t_int=None):
        if t_int is not None:
            assert isinstance(t_int, list)
            t_start_idx = np.argmax(t >= t_int[0])
            t_stop_idx = np.argmax(t > t_int[1])
            t = t[t_start_idx:t_stop_idx]
            data_iq = data_iq[t_start_idx:t_stop_idx]
        phase = np.unwrap(np.arctan2(np.imag(data_iq), np.real(data_iq)))
        if detrend:
            phase = phase - np.linspace(phase[0], phase[-1], len(phase))
        freq_hz = np.diff(phase) * self._fs / (2 * np.pi)
        t = t[:-1]
        assert len(freq_hz) == len(t)
        return t, freq_hz

    def file_to_dc_iq(self, fname):
        t, data_iq = self.file_to_iq(fname)
        t, data_iq = self._downconvert_and_filter(t, data_iq)
        return t, data_iq

    def file_to_freq(self, fname):
        t, data_iq = self.file_to_iq(fname)
        t, data_iq = self._downconvert_and_filter(t, data_iq)
        t, freq_hz = self._freq_demod(t, data_iq)
        return t, freq_hz

    def file_to_df(self, fname, t1, t2, win=10e-6):
        t, data_iq = self.file_to_iq(fname)
        t, data_iq = self._downconvert_and_filter(t, data_iq)
        _, f1 = self._freq_demod(t, data_iq, t_int=[t1, t1 + win])
        _, f2 = self._freq_demod(t, data_iq, t_int=[t2, t2 + win])
        return np.mean(f2) - np.mean(f1)


