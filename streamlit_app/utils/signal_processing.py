import numpy as np
from scipy import signal

# Generated with Copilot prompt: "Create a Python function for bandpass filtering EEG signals"
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to the input data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)

# Generated with Copilot prompt: "Compute the bandpower of a signal using Welch's method"
def bandpower(data, fs, band, window_sec=2.0):
    """Compute absolute power in a specific frequency band."""
    low, high = band
    f, Pxx = signal.welch(data, fs=fs, nperseg=int(window_sec * fs))
    idx = np.logical_and(f >= low, f <= high)
    return np.trapz(Pxx[idx], f[idx])
    
# Generated with Copilot prompt: "Return a spectrogram for a signal using scipy"
def compute_spectrogram(data, fs, nperseg=256):
    """Compute spectrogram of the input data."""
    f, t_spec, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg)
    return f, t_spec, Sxx
