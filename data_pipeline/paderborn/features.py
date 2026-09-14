"""
Paderborn vibration feature extraction.

Computes time-domain and frequency-domain features over fixed-size windows
for 64 kHz vibration signals.

RESTRICTION: Vibration features must NOT be mapped to engine telemetry channels.
Never downsample high-frequency vibration signals to engine telemetry rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


DEFAULT_WINDOW_SIZE = 4096   # 64.0 ms window at 64 kHz
DEFAULT_OVERLAP = 2048       # 50% overlap (32.0 ms step)
DEFAULT_FS = 64000.0         # 64 kHz sampling rate


def extract_paderborn_window_features(
    signal: np.ndarray,
    meta: Dict[str, Any],
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    sampling_rate: float = DEFAULT_FS,
) -> pd.DataFrame:
    """
    Extract comprehensive time and frequency features across sliding windows.

    Time-domain features:
      mean, std, variance, rms, peak, peak_to_peak, crest_factor, skewness, kurtosis

    Frequency-domain features:
      dominant_frequency, spectral_energy, spectral_centroid,
      band_energy_0_5khz, band_energy_5_15khz, band_energy_15_32khz

    Traceability:
      dataset_name, source_file, source_id, record_id, bearing_id, measurement_id,
      window_id, fault_label, operating_condition, speed_rpm, torque_nm, radial_force_n
    """
    if signal is None or len(signal) < window_size:
        return pd.DataFrame()

    sig = np.asarray(signal, dtype=np.float64)
    # Check finiteness
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    step = window_size - overlap
    n_windows = (len(sig) - window_size) // step + 1
    if n_windows <= 0:
        return pd.DataFrame()

    # Slice array into sliding windows
    # Shape: (n_windows, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(
        sig[:(n_windows - 1) * step + window_size],
        window_size
    )[::step]

    # --- TIME-DOMAIN FEATURES ---
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    variances = np.var(windows, axis=1)
    rms_vals = np.sqrt(np.mean(windows ** 2, axis=1))
    peaks = np.max(np.abs(windows), axis=1)
    peak_to_peaks = np.max(windows, axis=1) - np.min(windows, axis=1)
    
    # Crest factor
    safe_rms = np.where(rms_vals > 1e-12, rms_vals, 1e-12)
    crest_factors = np.where(rms_vals > 1e-12, peaks / safe_rms, 0.0)

    # Skewness and kurtosis
    safe_stds = np.where(stds > 1e-12, stds, 1e-12)
    norm_diffs = (windows - means[:, None]) / safe_stds[:, None]
    skewnesses = np.where(stds > 1e-12, np.mean(norm_diffs ** 3, axis=1), 0.0)
    kurtoses = np.where(stds > 1e-12, np.mean(norm_diffs ** 4, axis=1) - 3.0, 0.0)

    # --- FREQUENCY-DOMAIN FEATURES ---
    rfft = np.fft.rfft(windows, axis=1)
    freqs = np.fft.rfftfreq(window_size, 1.0 / sampling_rate)
    psd = np.abs(rfft) ** 2

    # Dominant frequency
    dom_freq_idx = np.argmax(psd, axis=1)
    dominant_frequencies = freqs[dom_freq_idx]

    # Spectral energy & centroid
    spectral_energies = np.sum(psd, axis=1)
    safe_energies = np.where(spectral_energies > 1e-12, spectral_energies, 1e-12)
    spectral_centroids = np.sum(psd * freqs[None, :], axis=1) / safe_energies

    # Band energies
    mask_0_5k = (freqs >= 0.0) & (freqs < 5000.0)
    mask_5_15k = (freqs >= 5000.0) & (freqs < 15000.0)
    mask_15_32k = (freqs >= 15000.0) & (freqs <= 32000.0)

    band_energy_0_5khz = np.sum(psd[:, mask_0_5k], axis=1)
    band_energy_5_15khz = np.sum(psd[:, mask_5_15k], axis=1)
    band_energy_15_32khz = np.sum(psd[:, mask_15_32k], axis=1)

    # --- ASSEMBLE DATAFRAME ---
    meas_id = meta.get("measurement_id", "meas")
    bearing_id = meta.get("bearing_id", "unknown")
    source_file = meta.get("source_file", "")
    source_id = f"paderborn/{bearing_id}/{source_file}"

    window_ids = np.arange(n_windows, dtype=np.int32)
    record_ids = [f"{meas_id}_win_{w:04d}" for w in window_ids]

    df = pd.DataFrame({
        "dataset_name": "paderborn",
        "source_file": source_file,
        "source_id": source_id,
        "record_id": record_ids,
        "bearing_id": bearing_id,
        "measurement_id": meas_id,
        "window_id": window_ids,
        "fault_label": meta.get("fault_label", "unknown"),
        "operating_condition": meta.get("operating_condition", "UNKNOWN"),
        "speed_rpm": float(meta.get("speed_rpm", np.nan)),
        "torque_nm": float(meta.get("torque_nm", np.nan)),
        "radial_force_n": float(meta.get("radial_force_n", np.nan)),
        "window_size": int(window_size),
        "overlap": int(overlap),
        "sampling_frequency": float(sampling_rate),
        # Time-domain metrics
        "mean": means,
        "std": stds,
        "variance": variances,
        "rms": rms_vals,
        "peak": peaks,
        "peak_to_peak": peak_to_peaks,
        "crest_factor": crest_factors,
        "skewness": skewnesses,
        "kurtosis": kurtoses,
        # Frequency-domain metrics
        "dominant_frequency": dominant_frequencies,
        "spectral_energy": spectral_energies,
        "spectral_centroid": spectral_centroids,
        "band_energy_0_5khz": band_energy_0_5khz,
        "band_energy_5_15khz": band_energy_5_15khz,
        "band_energy_15_32khz": band_energy_15_32khz,
    })

    return df
