"""
Safe formatters and status mappers for SIH26054 Dashboard UI.

Guarantees non-destructive display:
- None or NaN values are never fabricated or converted to zero.
- Renders explicit 'Unavailable' or 'Not Available' text.
- Formats engineering units, time conversions, and status badge mappings.
"""

import math
from typing import Optional, Union, Any
from dashboard.schemas.view_model import StatusLevel


def safe_is_nan(val: Any) -> bool:
    """Check if a value is None, NaN, or positive/negative infinity."""
    if val is None:
        return True
    try:
        f_val = float(val)
        return math.isnan(f_val) or math.isinf(f_val)
    except (ValueError, TypeError):
        return False


def format_value(
    val: Optional[Union[float, int, str]],
    decimals: int = 2,
    unit: str = "",
    default: str = "Unavailable",
) -> str:
    """Safely format a numeric value with decimals and optional unit."""
    if val is None:
        return default
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        formatted = f"{f_val:.{decimals}f}"
        return f"{formatted} {unit}".strip() if unit else formatted
    except (ValueError, TypeError):
        return str(val) if str(val).strip() else default


def format_percent(
    val: Optional[Union[float, int]],
    decimals: int = 1,
    default: str = "Unavailable",
) -> str:
    """Safely format a ratio [0.0, 1.0] or percentage [0.0, 100.0]."""
    if safe_is_nan(val):
        return default
    f_val = float(val)
    # If in 0.0-1.0 range, convert to percent
    pct = f_val * 100.0 if 0.0 <= f_val <= 1.0 else f_val
    return f"{pct:.{decimals}f}%"


def format_health_index(val: Optional[Union[float, int]]) -> str:
    """Safely format Health Index on standard [0.00, 1.00] scale."""
    if safe_is_nan(val):
        return "Unavailable"
    f_val = float(val)
    # Clamp to [0, 1] for display if valid numeric
    clamped = max(0.0, min(1.0, f_val))
    return f"{clamped:.2f}"


def format_rul(
    seconds: Optional[Union[float, int]],
    p05: Optional[Union[float, int]] = None,
    p95: Optional[Union[float, int]] = None,
    default: str = "Unavailable",
) -> str:
    """
    Safely format Remaining Useful Life from seconds to hours with uncertainty bounds.
    Does not invent values if input is NaN or None.
    """
    if safe_is_nan(seconds):
        return default

    f_sec = float(seconds)
    if f_sec <= 0.0:
        return "0.0 hrs (EOL Reached)"

    hrs = f_sec / 3600.0

    if p05 is not None and p95 is not None and not safe_is_nan(p05) and not safe_is_nan(p95):
        p05_hrs = max(0.0, float(p05) / 3600.0)
        p95_hrs = max(0.0, float(p95) / 3600.0)
        return f"{hrs:.1f} hrs [{p05_hrs:.1f} - {p95_hrs:.1f} hrs]"

    return f"{hrs:.1f} hrs ({f_sec / 60.0:.0f} min)"


def format_timestamp(ts: Optional[Union[float, int]]) -> str:
    """Format mission timestamp into standard T+ seconds format."""
    if safe_is_nan(ts):
        return "T+ -- s"
    return f"T+{float(ts):.1f} s"


def format_fault_name(fault: Optional[str]) -> str:
    """Format snake_case fault identifiers into clear operational titles."""
    if not fault or fault.strip() == "" or fault.lower() in ("none", "normal"):
        return "Nominal / None"
    clean = fault.replace("_", " ").title()
    return clean


def map_health_to_status(state: Optional[str]) -> StatusLevel:
    """Map Phase 9 HealthState string to UI StatusLevel."""
    if not state:
        return StatusLevel.UNAVAILABLE
    normalized = str(state).upper()
    if normalized == "HEALTHY":
        return StatusLevel.HEALTHY
    elif normalized == "DEGRADED":
        return StatusLevel.WARNING
    elif normalized in ("SEVERELY_DEGRADED", "CRITICAL"):
        return StatusLevel.CRITICAL
    elif normalized == "INSUFFICIENT_DATA":
        return StatusLevel.UNAVAILABLE
    return StatusLevel.UNKNOWN


def map_anomaly_to_status(status: Optional[str]) -> StatusLevel:
    """Map Phase 7 AnomalyStatus string to UI StatusLevel."""
    if not status:
        return StatusLevel.UNAVAILABLE
    normalized = str(status).upper()
    if normalized == "NORMAL":
        return StatusLevel.HEALTHY
    elif normalized == "WARNING":
        return StatusLevel.WARNING
    elif normalized == "ANOMALY":
        return StatusLevel.CRITICAL
    elif normalized == "INSUFFICIENT_DATA":
        return StatusLevel.UNAVAILABLE
    return StatusLevel.UNKNOWN


def map_rul_status_to_status(status: Optional[str]) -> StatusLevel:
    """Map Phase 11 RULStatus string to UI StatusLevel."""
    if not status:
        return StatusLevel.UNAVAILABLE
    normalized = str(status).upper()
    if normalized == "NOT_DEGRADING":
        return StatusLevel.HEALTHY
    elif normalized in ("ACTIVE_DEGRADATION", "RECOVERING"):
        return StatusLevel.WARNING
    elif normalized == "CRITICAL_EOL_REACHED":
        return StatusLevel.CRITICAL
    elif normalized == "DEGRADED_PROGNOSTIC":
        return StatusLevel.DEGRADED
    elif normalized in ("INSUFFICIENT_DATA", "INSUFFICIENT_HISTORY", "INDETERMINATE_TREND", "EXCEEDS_HORIZON"):
        return StatusLevel.UNAVAILABLE
    return StatusLevel.UNKNOWN
