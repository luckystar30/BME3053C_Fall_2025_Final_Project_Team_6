import numpy as np
from scipy import signal

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)

def bandpower(data, fs, band, window_sec=2.0):
    low, high = band
    f, Pxx = signal.welch(data, fs=fs, nperseg=int(window_sec * fs))
    idx = np.logical_and(f >= low, f <= high)
    return np.trapz(Pxx[idx], f[idx])

def compute_spectrogram(data, fs, nperseg=256):
    f, t_spec, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg)
    return f, t_spec, Sxx
