"""
signal_processing.py
Helper functions for EEG bandpower, filtering, and spectrogram using scipy.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt, spectrogram

def bandpower(data, fs, band, window_sec=2, relative=False):
    """
    Compute the average power of the signal x in a specific frequency band.
    Uses Welch's method.
    data: 1D numpy array
    fs: sampling frequency (Hz)
    band: tuple/list (low, high) in Hz
    window_sec: length of Welch window in seconds
    relative: if True, return bandpower / total_power
    """
    low, high = band
    nperseg = int(window_sec * fs)
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    # find index of band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    if relative:
        total_power = np.trapz(psd, freqs)
        return band_power / total_power if total_power > 0 else 0
    else:
        return band_power

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def compute_spectrogram(data, fs, nperseg=256):
    """Return f, t, Sxx for plotting a spectrogram"""
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg)
    return f, t, Sxx
