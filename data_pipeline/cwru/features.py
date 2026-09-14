"""
CWRU vibration feature extraction.

Computes 14 distinct mathematical time-domain features and 4 frequency-domain
features over fixed-length sliding windows for high-frequency CWRU vibration signals.

Time-domain features (14 distinct mathematical metrics):
  1. mean: Arithmetic mean of window
  2. std: Standard deviation
  3. variance: Signal variance
  4. RMS: Root-mean-square amplitude (with 'rms' as documented backward-compatibility alias)
  5. minimum: Minimum sample value
  6. maximum: Maximum sample value
  7. peak_to_peak: Difference between maximum and minimum
  8. absolute_mean: Mean of absolute signal values
  9. skewness: Fisher skewness (3rd standardized moment)
  10. kurtosis: Fisher excess kurtosis (4th standardized moment - 3.0)
  11. crest_factor: Ratio of peak absolute amplitude to RMS
  12. shape_factor: Ratio of RMS to absolute mean
  13. impulse_factor: Ratio of peak absolute amplitude to absolute mean
  14. clearance_factor: Peak amplitude divided by square of mean square-root amplitude

Auxiliary metrics:
  - peak: Maximum absolute amplitude
  - energy: Total squared window energy

Frequency-domain features:
  - dominant_frequency: Frequency corresponding to maximum spectral peak
  - spectral_energy: Total energy across the power spectrum
  - spectral_centroid: Spectral center of mass / power-weighted mean frequency
  - frequency_band_energy: Dominant sub-band energy across 4 equal 1.5 kHz bands (0-1.5k, 1.5-3k, 3-4.5k, 4.5-6k Hz)
  - band_energy_0_1500hz: Spectral energy in [0, 1500 Hz)
  - band_energy_1500_3000hz: Spectral energy in [1500, 3000 Hz)
  - band_energy_3000_4500hz: Spectral energy in [3000, 4500 Hz)
  - band_energy_4500_6000hz: Spectral energy in [4500, 6000 Hz]

Traceability & Condition:
  dataset_name, source_file, source_id, record_id, bearing_id, window_id,
  sensor_location, sampling_frequency, RPM, load_condition, fault_label, fault_size
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

DEFAULT_WINDOW_SIZE = 2048  # ~170.7 ms at 12 kHz
DEFAULT_OVERLAP = 1024      # 50% overlap (~85.3 ms step)
DEFAULT_FS = 12000.0        # 12 kHz standard sampling frequency


def extract_window_features(
    signal: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    sampling_rate: float = DEFAULT_FS,
    metadata_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Extract time-domain and frequency-domain features from windowed signal segments.

    Args:
        signal: 1D vibration signal array.
        window_size: Number of samples per window.
        overlap: Number of overlapping samples between consecutive windows.
        sampling_rate: Original sampling rate in Hz.
        metadata_context: Optional dictionary of provenance/condition metadata.

    Returns:
        pd.DataFrame containing feature columns, provenance, and condition metadata.
    """
    if signal is None or len(signal) < window_size:
        return pd.DataFrame()

    sig = np.asarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    step = window_size - overlap
    if step <= 0:
        step = window_size // 2

    n_windows = (len(sig) - window_size) // step + 1
    if n_windows <= 0:
        return pd.DataFrame()

    # Slice array into sliding windows: shape (n_windows, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(
        sig[:(n_windows - 1) * step + window_size],
        window_size
    )[::step]

    # --- 14 TIME-DOMAIN FEATURES ---
    # 1. mean
    means = np.mean(windows, axis=1)
    # 2. std
    stds = np.std(windows, axis=1)
    # 3. variance
    variances = np.var(windows, axis=1)
    # 4. RMS (and compatibility alias rms)
    rms_vals = np.sqrt(np.mean(windows ** 2, axis=1))
    # 5. minimum
    mins = np.min(windows, axis=1)
    # 6. maximum
    maxs = np.max(windows, axis=1)
    # 7. peak_to_peak
    peak_to_peaks = maxs - mins
    # 8. absolute_mean
    abs_means = np.mean(np.abs(windows), axis=1)
    
    # Auxiliary peak
    peaks = np.max(np.abs(windows), axis=1)

    # 9. skewness & 10. kurtosis (Fisher standardized definitions)
    safe_stds = np.where(stds > 1e-12, stds, 1e-12)
    norm_diffs = (windows - means[:, None]) / safe_stds[:, None]
    skewnesses = np.where(stds > 1e-12, np.mean(norm_diffs ** 3, axis=1), 0.0)
    kurtoses = np.where(stds > 1e-12, np.mean(norm_diffs ** 4, axis=1) - 3.0, 0.0)

    # 11. crest_factor: peak / RMS
    safe_rms = np.where(rms_vals > 1e-12, rms_vals, 1e-12)
    crest_factors = np.where(rms_vals > 1e-12, peaks / safe_rms, 0.0)

    # 12. shape_factor: RMS / absolute_mean
    safe_abs_means = np.where(abs_means > 1e-12, abs_means, 1e-12)
    shape_factors = np.where(abs_means > 1e-12, rms_vals / safe_abs_means, 0.0)

    # 13. impulse_factor: peak / absolute_mean
    impulse_factors = np.where(abs_means > 1e-12, peaks / safe_abs_means, 0.0)

    # 14. clearance_factor: peak / (mean(sqrt(|x|)))^2
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(windows)), axis=1)
    sq_mean_sqrt_abs = mean_sqrt_abs ** 2
    safe_clearance_denom = np.where(sq_mean_sqrt_abs > 1e-12, sq_mean_sqrt_abs, 1e-12)
    clearance_factors = np.where(sq_mean_sqrt_abs > 1e-12, peaks / safe_clearance_denom, 0.0)

    # Auxiliary energy
    energies = np.sum(windows ** 2, axis=1)

    # --- FREQUENCY-DOMAIN FEATURES ---
    # Sampling frequency and Nyquist limit
    fs = float(sampling_rate) if sampling_rate > 0 else DEFAULT_FS
    nyquist = fs / 2.0
    rfft = np.fft.rfft(windows, axis=1)
    freqs = np.fft.rfftfreq(window_size, 1.0 / fs)
    psd = np.abs(rfft) ** 2

    # Dominant frequency
    dom_idx = np.argmax(psd, axis=1)
    dominant_frequencies = freqs[dom_idx]

    # Spectral energy & centroid
    spectral_energies = np.sum(psd, axis=1)
    safe_spectral_energies = np.where(spectral_energies > 1e-12, spectral_energies, 1e-12)
    spectral_centroids = np.sum(psd * freqs[None, :], axis=1) / safe_spectral_energies

    # 4 Equal frequency sub-bands (up to Nyquist, e.g. 0-1.5kHz, 1.5-3kHz, 3-4.5kHz, 4.5-6kHz for 12kHz fs)
    b1_edge = nyquist * 0.25
    b2_edge = nyquist * 0.50
    b3_edge = nyquist * 0.75

    m_b1 = (freqs >= 0.0) & (freqs < b1_edge)
    m_b2 = (freqs >= b1_edge) & (freqs < b2_edge)
    m_b3 = (freqs >= b2_edge) & (freqs < b3_edge)
    m_b4 = (freqs >= b3_edge) & (freqs <= nyquist)

    band_energy_0_1500hz = np.sum(psd[:, m_b1], axis=1)
    band_energy_1500_3000hz = np.sum(psd[:, m_b2], axis=1)
    band_energy_3000_4500hz = np.sum(psd[:, m_b3], axis=1)
    band_energy_4500_6000hz = np.sum(psd[:, m_b4], axis=1)

    # frequency_band_energy records the dominant sub-band energy
    freq_band_energies = np.maximum.reduce([
        band_energy_0_1500hz,
        band_energy_1500_3000hz,
        band_energy_3000_4500hz,
        band_energy_4500_6000hz
    ])

    # Window indexing
    window_starts = np.arange(n_windows, dtype=np.int64) * step
    window_ends = window_starts + window_size
    window_ids = np.arange(n_windows, dtype=np.int64)

    ctx = metadata_context or {}
    source_file = ctx.get("source_file", "unknown.mat")
    ch_name = ctx.get("source_channel", "unknown_channel")
    bearing_id = ctx.get("bearing_id", source_file.replace(".mat", ""))
    sensor_loc = ctx.get("sensor_location", "unknown")
    rpm = ctx.get("rpm", None)
    load_cond = ctx.get("load_condition", "0HP")
    fault_label = ctx.get("fault_label", "unknown")
    fault_size = ctx.get("fault_size", None)
    fault_size_mils = ctx.get("fault_size_mils", 0)
    motor_load_hp = ctx.get("motor_load_hp", 0)

    # Deterministic source_id and record_id
    source_id = f"cwru/{source_file}/{ch_name}"
    record_ids = [f"{source_id}_win_{w:04d}" for w in window_ids]

    df_dict = {
        # Provenance and Traceability
        "dataset_name": "cwru",
        "source_file": source_file,
        "source_id": source_id,
        "record_id": record_ids,
        "bearing_id": str(bearing_id),
        "window_id": window_ids,
        "sensor_location": sensor_loc,
        "sampling_frequency": float(fs),
        "RPM": float(rpm) if rpm is not None else np.nan,
        "load_condition": str(load_cond),
        "fault_label": str(fault_label),
        "fault_size": float(fault_size) if fault_size is not None else np.nan,
        # Window parameters
        "window_start": window_starts,
        "window_end": window_ends,
        "window_length": int(window_size),
        # 14 distinct mathematical time-domain metrics
        "mean": means,
        "std": stds,
        "variance": variances,
        "RMS": rms_vals,
        "minimum": mins,
        "maximum": maxs,
        "peak_to_peak": peak_to_peaks,
        "absolute_mean": abs_means,
        "skewness": skewnesses,
        "kurtosis": kurtoses,
        "crest_factor": crest_factors,
        "shape_factor": shape_factors,
        "impulse_factor": impulse_factors,
        "clearance_factor": clearance_factors,
        # Auxiliary time-domain metrics & compatibility aliases
        "rms": rms_vals,  # Documented alias for RMS
        "peak": peaks,
        "energy": energies,
        # Frequency-domain metrics
        "dominant_frequency": dominant_frequencies,
        "spectral_energy": spectral_energies,
        "spectral_centroid": spectral_centroids,
        "frequency_band_energy": freq_band_energies,
        "band_energy_0_1500hz": band_energy_0_1500hz,
        "band_energy_1500_3000hz": band_energy_1500_3000hz,
        "band_energy_3000_4500hz": band_energy_3000_4500hz,
        "band_energy_4500_6000hz": band_energy_4500_6000hz,
        # Backward compatibility metadata fields
        "source_channel": ch_name,
        "fault_size_mils": fault_size_mils,
        "motor_load_hp": motor_load_hp,
        "original_sampling_rate": float(fs),
    }

    df = pd.DataFrame(df_dict)

    # Ensure all numeric values are strictly finite
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if c in ["RPM", "fault_size"]:
            # Nullable metadata columns can have NaN if genuinely unavailable
            continue
        col_vals = df[c].values
        if not np.all(np.isfinite(col_vals)):
            df[c] = np.nan_to_num(col_vals, nan=0.0, posinf=0.0, neginf=0.0)

    return df


def extract_features_for_file(
    channels: Dict[str, np.ndarray],
    file_metadata: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    source_file: str = "unknown.mat",
    sensor_locations: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract features from all vibration channels of a single CWRU file.

    Returns concatenated DataFrame across all valid channels in the file.
    """
    sampling_rate = float(file_metadata.get("sampling_rate_hz", DEFAULT_FS))
    all_features = []
    locations = sensor_locations or {}

    bearing_id = file_metadata.get("bearing_id", source_file.replace(".mat", ""))
    fault_type = file_metadata.get("fault_type", "unknown")
    fault_size_mils = file_metadata.get("fault_size_mils", 0)
    fault_size = float(fault_size_mils) * 0.001 if fault_size_mils is not None else 0.0
    motor_load_hp = file_metadata.get("motor_load_hp", 0)
    rpm = file_metadata.get("rpm", None)
    load_cond = f"{motor_load_hp}HP"

    for ch_name, signal in channels.items():
        if signal is None or not isinstance(signal, np.ndarray) or len(signal) < window_size:
            continue
        if not np.issubdtype(signal.dtype, np.number):
            continue

        loc = locations.get(ch_name, "unknown")
        if loc == "unknown":
            k_upper = ch_name.upper()
            if "_DE_" in k_upper or "DE_TIME" in k_upper:
                loc = "drive_end"
            elif "_FE_" in k_upper or "FE_TIME" in k_upper:
                loc = "fan_end"
            elif "_BA_" in k_upper or "BA_TIME" in k_upper:
                loc = "base"

        ctx = {
            "source_file": source_file,
            "source_channel": ch_name,
            "bearing_id": bearing_id,
            "sensor_location": loc,
            "rpm": rpm,
            "load_condition": load_cond,
            "fault_label": fault_type,
            "fault_size": fault_size,
            "fault_size_mils": fault_size_mils,
            "motor_load_hp": motor_load_hp,
        }

        df = extract_window_features(
            signal=signal,
            window_size=window_size,
            overlap=overlap,
            sampling_rate=sampling_rate,
            metadata_context=ctx,
        )
        if not df.empty:
            all_features.append(df)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
