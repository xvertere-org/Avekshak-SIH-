"""
NASA Battery feature extraction — cycle-level degradation records.

Extracts capacity fade, impedance, and SOH metrics per cycle.
RESTRICTION: Battery features stay in methodology/benchmark namespace.
Never map to engine telemetry channels (CHT, EGT, oil pressure, etc.).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


NOMINAL_REFERENCE_CAPACITY_AH = 2.0  # Standard 18650 rated nominal capacity


def _safe_float(val: Any) -> float:
    """Safely extract float from numpy array or scalar."""
    if val is None:
        return np.nan
    try:
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return np.nan
            val = val.flat[-1]
        f = float(val)
        if np.isinf(f) or np.isnan(f):
            return np.nan
        return f
    except (ValueError, TypeError):
        return np.nan


def _array_stats(arr: Any) -> Dict[str, float]:
    """Compute mean, min, max from array-like data."""
    res = {"mean": np.nan, "min": np.nan, "max": np.nan}
    if arr is None:
        return res
    try:
        if isinstance(arr, (int, float)):
            f = float(arr)
            if not np.isinf(f) and not np.isnan(f):
                return {"mean": f, "min": f, "max": f}
            return res
        nparr = np.asarray(arr, dtype=np.float64)
        valid = nparr[np.isfinite(nparr)]
        if len(valid) > 0:
            res["mean"] = float(np.mean(valid))
            res["min"] = float(np.min(valid))
            res["max"] = float(np.max(valid))
    except Exception:
        pass
    return res


def extract_battery_cycle_records(
    mat_data: Dict,
    battery_id: str,
    source_file: str,
    source_id: str,
    nominal_ref_capacity: float = NOMINAL_REFERENCE_CAPACITY_AH,
) -> pd.DataFrame:
    """
    Extract cycle-level records from a loaded battery .mat structure.

    Each record represents one cycle (charge, discharge, or impedance)
    with physical measurements, SOH (when capacity reference is valid),
    and traceability headers.
    """
    # Identify the top-level battery object
    battery_obj = None
    if battery_id in mat_data:
        battery_obj = mat_data[battery_id]
    else:
        for k in mat_data:
            if not k.startswith("__"):
                battery_obj = mat_data[k]
                break

    if battery_obj is None or not hasattr(battery_obj, "cycle"):
        return pd.DataFrame()

    cycles = battery_obj.cycle
    if not hasattr(cycles, "__len__"):
        cycles = [cycles]

    records = []
    
    for cycle_idx, cycle in enumerate(cycles, start=1):
        cycle_type = str(getattr(cycle, "type", "unknown")).strip().lower()
        amb_temp = _safe_float(getattr(cycle, "ambient_temperature", np.nan))
        
        # Initialize default values
        duration_s = np.nan
        v_stats = {"mean": np.nan, "min": np.nan, "max": np.nan}
        i_stats = {"mean": np.nan, "min": np.nan, "max": np.nan}
        t_stats = {"mean": np.nan, "min": np.nan, "max": np.nan}
        discharge_cap = np.nan
        re_val = np.nan
        rct_val = np.nan

        data = getattr(cycle, "data", None)
        if data is not None:
            # Duration from Time array if present
            time_arr = getattr(data, "Time", None)
            if time_arr is not None:
                try:
                    t_np = np.asarray(time_arr, dtype=np.float64).flatten()
                    valid_t = t_np[np.isfinite(t_np)]
                    if len(valid_t) > 1:
                        duration_s = float(valid_t[-1] - valid_t[0])
                except Exception:
                    pass

            # Voltage, current, and temperature measurements
            v_meas = getattr(data, "Voltage_measured", None)
            if v_meas is not None:
                v_stats = _array_stats(v_meas)

            i_meas = getattr(data, "Current_measured", None)
            if i_meas is not None:
                i_stats = _array_stats(i_meas)

            t_meas = getattr(data, "Temperature_measured", None)
            if t_meas is not None:
                t_stats = _array_stats(t_meas)

            # Discharge capacity
            if cycle_type == "discharge" and hasattr(data, "Capacity"):
                raw_cap = _safe_float(data.Capacity)
                # Validation: no impossible negative capacity
                if raw_cap is not None and not np.isnan(raw_cap) and raw_cap > 0.0:
                    discharge_cap = raw_cap

            # Impedance parameters
            if cycle_type == "impedance":
                if hasattr(data, "Re"):
                    re_val = _safe_float(data.Re)
                if hasattr(data, "Rct"):
                    rct_val = _safe_float(data.Rct)

        # Calculate SOH only when valid discharge capacity and reference are available
        soh = np.nan
        soh_clipped = np.nan
        capacity_loss = np.nan
        if not np.isnan(discharge_cap) and nominal_ref_capacity > 0:
            soh = discharge_cap / nominal_ref_capacity
            soh_clipped = float(np.clip(soh, 0.0, 1.0))
            capacity_loss = nominal_ref_capacity - discharge_cap

        record = {
            "dataset_name": "nasa_battery",
            "source_file": source_file,
            "source_id": source_id,
            "record_id": f"{battery_id}_c{cycle_idx:04d}",
            "battery_id": battery_id,
            "cycle": int(cycle_idx),
            "cycle_type": cycle_type,
            "ambient_temperature": float(amb_temp) if not np.isnan(amb_temp) else np.nan,
            "duration_s": float(duration_s) if not np.isnan(duration_s) else np.nan,
            "voltage_mean": float(v_stats["mean"]),
            "voltage_min": float(v_stats["min"]),
            "voltage_max": float(v_stats["max"]),
            "current_mean": float(i_stats["mean"]),
            "current_min": float(i_stats["min"]),
            "current_max": float(i_stats["max"]),
            "temperature_mean": float(t_stats["mean"]),
            "temperature_min": float(t_stats["min"]),
            "temperature_max": float(t_stats["max"]),
            "discharge_capacity": float(discharge_cap) if not np.isnan(discharge_cap) else np.nan,
            "reference_capacity": float(nominal_ref_capacity),
            "soh": float(soh) if not np.isnan(soh) else np.nan,
            "soh_clipped": float(soh_clipped) if not np.isnan(soh_clipped) else np.nan,
            "capacity_loss": float(capacity_loss) if not np.isnan(capacity_loss) else np.nan,
            "re_electrolyte_resistance": float(re_val) if not np.isnan(re_val) else np.nan,
            "rct_charge_transfer_resistance": float(rct_val) if not np.isnan(rct_val) else np.nan,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df
