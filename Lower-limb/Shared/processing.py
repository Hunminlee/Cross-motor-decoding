import numpy as np
import pandas as pd


# --- Sliding window ---
def sliding_window(data, window_size, step_size):
    num_samples, num_channels = data.shape
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end, :])  # (window_size, num_channels)
    return np.array(windows)  # (num_windows, window_size, num_channels)

# --- Feature extraction ---
def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x**2))
        var = np.var(x)
        wl = np.sum(np.abs(np.diff(x)))
        fft_vals = np.fft.rfft(x)
        fft_freq = np.fft.rfftfreq(len(x), d=1.0)
        psd = np.abs(fft_vals)**2
        mnf = np.sum(fft_freq * psd) / np.sum(psd)
        feats.extend([mav, rms, var, wl, mnf])
    return np.array(feats)  # (num_channels * 5,)