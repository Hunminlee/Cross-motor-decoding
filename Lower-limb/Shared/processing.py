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
def extract_features_five(window):
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


import numpy as np
import pywt

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]

        # --- Time-domain features ---
        mav = np.mean(np.abs(x))                          # Mean Absolute Value
        wl = np.sum(np.abs(np.diff(x)))                   # Waveform Length
        thr = 0.05 * np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 0  # threshold for WAMP/SSC
        wamp = np.sum(np.abs(np.diff(x)) > thr)           # Willison Amplitude
        mavs = np.mean(np.diff(np.abs(x)))                # MAV Slope
        rms = np.sqrt(np.mean(x**2))                      # Root Mean Square
        ssc = np.sum(((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:])) > 0)  # Slope Sign Changes

        # --- Statistical features ---
        msq = np.mean(x**2)                               # Mean Square
        v3 = np.mean(np.abs(x)**3)                        # v-order 3
        ld = np.exp(np.mean(np.log(np.abs(x)+1e-8)))      # Log Detector
        dabs = np.std(np.abs(np.diff(x)))                 # Diff Absolute Std Dev

        # --- Complexity measures ---
        mfl = np.sum(np.sqrt(1 + np.diff(x)**2))          # Max Fractal Length
        mpr = np.mean(np.abs(x) > thr)                    # Myopulse Percentage Rate

        # --- Frequency-domain features ---
        fft_vals = np.fft.rfft(x)
        fft_freq = np.fft.rfftfreq(len(x), d=1.0)
        psd = np.abs(fft_vals)**2
        mnf = np.sum(fft_freq * psd) / (np.sum(psd) + 1e-8)   # Mean Frequency
        # Power Spectrum Ratio: low (0–30 Hz) vs total
        psr = (np.sum(psd[(fft_freq >= 0) & (fft_freq <= 30)]) /
               (np.sum(psd) + 1e-8))

        # --- Model-based features (Autoregressive coefficients, order=4) ---
        # Using Yule-Walker estimation
        try:
            from statsmodels.regression.linear_model import yule_walker
            rho, sigma = yule_walker(x, order=4)
            arc = rho.tolist()
        except:
            arc = [0, 0, 0, 0]

        # --- Cepstrum features ---
        spectrum = np.abs(np.fft.fft(x))**2
        log_spectrum = np.log(spectrum + 1e-8)
        cepstrum = np.fft.ifft(log_spectrum).real
        cc = cepstrum[1:4]                  # first 3 coefficients
        cca = np.mean(cc)                   # average cepstrum coeff

        # --- Time-frequency features (Wavelet) ---
        coeffs = pywt.wavedec(x, 'db4', level=3)
        dwtc = coeffs[1][:2] if len(coeffs[1]) >= 2 else [0, 0]  # 2 coeffs from D1
        dwtpc = coeffs[2][:3] if len(coeffs[2]) >= 3 else [0, 0, 0]  # 3 coeffs from D2

        # --- Collect all ---
        feats.extend([
            mav, wl, wamp, mavs, rms, ssc,         # time-domain
            msq, v3, ld, dabs,                     # statistical
            mfl, mpr,                              # complexity
            mnf, psr                               # freq-domain
        ])
        feats.extend(arc)                          # AR coeffs (4)
        feats.extend(cc)                           # 3 cepstrum
        feats.append(cca)                          # avg cepstrum
        feats.extend(dwtc)                         # 2 DWT coeffs
        feats.extend(dwtpc)                        # 3 DWTP coeffs

    return np.array(feats)
