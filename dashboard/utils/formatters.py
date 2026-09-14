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


# =============================================================================
# Avekshak Centralized User-Facing Terminology Mappings
# =============================================================================

ADVISORY_ACTION_MAP = {
    "NORMAL_MONITORING": "Routine Monitoring",
    "ADVISORY_CAUTION": "Operational Caution",
    "MAINTENANCE_INSPECTION": "Maintenance Inspection Recommended",
    "CRITICAL_ABORT_ACTION": "Immediate Operational Intervention",
    "INSUFFICIENT_DATA": "Insufficient Telemetry Data",
}

HEALTH_STATE_MAP = {
    "HEALTHY": "Healthy Condition",
    "NORMAL": "Healthy Condition",
    "NOT_DEGRADING": "Stable Condition",
    "ACTIVE_DEGRADATION": "Active Degradation",
    "DEGRADED": "Degraded Condition",
    "SEVERELY_DEGRADED": "Severe Degradation",
    "RAPIDLY_DEGRADING": "Rapid Health Decline",
    "CRITICAL": "Critical Condition",
    "CRITICAL_EOL_REACHED": "Critical Limit Reached",
    "INSUFFICIENT_HISTORY": "Insufficient Flight History",
    "INSUFFICIENT_DATA": "Insufficient Data",
}

HEALTH_TREND_MAP = {
    "STABLE": "Stable Trend",
    "NOT_DEGRADING": "Stable Condition",
    "ACTIVE_DEGRADATION": "Active Degradation",
    "RAPIDLY_DEGRADING": "Rapid Health Decline",
    "RECOVERING": "Recovering Trend",
    "INDETERMINATE": "Indeterminate Trend",
    "INDETERMINATE_TREND": "Indeterminate Trend",
    "INSUFFICIENT_HISTORY": "Insufficient Flight History",
}

RUL_STATE_MAP = {
    "NOT_DEGRADING": "Stable Condition",
    "ACTIVE_DEGRADATION": "Active Degradation",
    "SEVERELY_DEGRADED": "Severe Degradation",
    "RAPIDLY_DEGRADING": "Rapid Health Decline",
    "CRITICAL_EOL_REACHED": "Critical Limit Reached",
    "INSUFFICIENT_HISTORY": "Insufficient Flight History",
    "INSUFFICIENT_DATA": "Insufficient Telemetry Data",
    "EXCEEDS_HORIZON": "Exceeds Prediction Horizon",
    "DEGRADED_PROGNOSTIC": "Degraded Prognostic State",
    "RECOVERING": "Recovering",
}

LIMITING_FACTOR_MAP = {
    "REDLINE_CHT": "CHT Thermal Limit",
    "REDLINE_OIL_PRESSURE": "Oil Pressure Limit",
    "REDLINE_OIL_TEMP": "Oil Temperature Limit",
    "REDLINE_RPM": "Engine Speed Limit",
    "GLOBAL_HEALTH_INDEX": "Overall Health Limit",
    "NONE": "None (Nominal)",
    "NULL": "None (Nominal)",
}

FORECAST_STATUS_MAP = {
    "LOADED_PRETRAINED": "Forecast Model Active",
    "BLOCKED_UNAUTHENTICATED_GATED": "Advanced Forecast Unavailable",
    "LOCAL_UNCHECKPOINTED_GRAPH": "Local Model Active (Uncheckpointed)",
    "BUFFERING": "Preparing Forecast",
    "BASELINE": "Baseline Model Active",
    "UNAVAILABLE": "Forecast Unavailable",
}

FORECAST_QUALITY_MAP = {
    "VALID": "Valid Quality",
    "DEGRADED_INPUT": "Degraded Input Data",
    "INSUFFICIENT_DATA": "Insufficient History",
    "LOW_CONFIDENCE": "Low Confidence",
}

DATA_QUALITY_MAP = {
    "NOMINAL": "Nominal Stream",
    "VALID": "Nominal Stream",
    "DEGRADED": "Degraded Stream",
    "OUT_OF_ORDER_REJECTED": "Out-of-Order Timestamp",
    "MISSING": "Signal Missing",
    "DROPOUT": "Signal Dropout",
    "PHYSICALLY_INVALID": "Outside Physical Bounds",
    "WARNING_ENVELOPE": "Caution Range",
    "ISOLATED_BY_PHM": "Sensor Channel Isolated",
    "ISOLATED": "Sensor Channel Isolated",
    "INVALID": "Invalid Signal",
}

CHANNEL_MAP = {
    "rpm": "Engine Speed (RPM)",
    "cht": "Cylinder Head Temperature (CHT)",
    "egt": "Exhaust Gas Temperature (EGT)",
    "oil_pressure": "Oil Pressure",
    "oil_temp": "Oil Temperature",
    "fuel_flow": "Fuel Flow",
    "vibration": "Vibration",
    "throttle": "Throttle",
    "load": "Engine Load",
    "engine_load": "Engine Load",
    "altitude": "Altitude",
    "ambient_temp": "Ambient Temperature",
    "oat": "Ambient Temperature",
}

FAULT_MAP = {
    "none": "Nominal / None",
    "normal": "Nominal / None",
    "cooling_degradation": "Cooling Degradation",
    "lubrication_degradation": "Lubrication Degradation",
    "fuel_injection_abnormality": "Fuel Injection Abnormality",
    "mechanical_degradation": "Mechanical Degradation",
    "sensor_fault": "Sensor Signal Dropout",
}

EVIDENCE_STATUS_MAP = {
    "SUPPORTED": "Physically Consistent",
    "CONSISTENT": "Physically Consistent",
    "PARTIALLY_SUPPORTED": "Partially Supported",
    "CONFLICTING": "Physics Conflict",
    "INSUFFICIENT_DATA": "Insufficient Data",
    "UNAVAILABLE": "Unavailable",
}


def format_action(action_code: Optional[str]) -> str:
    """Format an advisory action code to clear operational English."""
    if not action_code:
        return "Routine Monitoring"
    norm = str(action_code).strip().upper()
    return ADVISORY_ACTION_MAP.get(norm, norm.replace("_", " ").title())


def format_health_state(state: Optional[str]) -> str:
    """Format health state string to clear operational English."""
    if not state:
        return "Unavailable"
    norm = str(state).strip().upper()
    return HEALTH_STATE_MAP.get(norm, norm.replace("_", " ").title())


def format_health_trend(trend: Optional[str]) -> str:
    """Format degradation trend string to clear operational English."""
    if not trend:
        return "Stable Trend"
    norm = str(trend).strip().upper()
    return HEALTH_TREND_MAP.get(norm, norm.replace("_", " ").title())


def format_rul_state(state: Optional[str]) -> str:
    """Format Remaining Useful Life status state."""
    if not state:
        return "Unavailable"
    norm = str(state).strip().upper()
    return RUL_STATE_MAP.get(norm, norm.replace("_", " ").title())


def format_limiting_factor(factor: Optional[str]) -> str:
    """Format prognostic limit / threshold identifier."""
    if not factor or str(factor).strip().upper() in ("NONE", "NULL", ""):
        return "None (Nominal)"
    norm = str(factor).strip().upper()
    return LIMITING_FACTOR_MAP.get(norm, norm.replace("_", " ").title())


def format_forecast_status(status: Optional[str]) -> str:
    """Format telemetry forecasting runtime status."""
    if not status:
        return "Unavailable"
    norm = str(status).strip().upper()
    return FORECAST_STATUS_MAP.get(norm, norm.replace("_", " ").title())


def format_forecast_quality(quality: Optional[str]) -> str:
    """Format telemetry forecasting confidence/quality."""
    if not quality:
        return "Unavailable"
    norm = str(quality).strip().upper()
    return FORECAST_QUALITY_MAP.get(norm, norm.replace("_", " ").title())


def format_data_quality_status(status: Optional[str]) -> str:
    """Format telemetry data quality condition."""
    if not status:
        return "Unavailable"
    norm = str(status).strip().upper()
    return DATA_QUALITY_MAP.get(norm, norm.replace("_", " ").title())


def format_channel(channel: Optional[str]) -> str:
    """Format sensor channel identifier to standard operational label."""
    if not channel:
        return ""
    norm = str(channel).strip().lower()
    return CHANNEL_MAP.get(norm, channel.replace("_", " ").title())


def format_fault(fault: Optional[str]) -> str:
    """Format fault classification into clear operational failure title."""
    if not fault or fault.strip() == "" or fault.lower() in ("none", "normal"):
        return "Nominal / None"
    norm = str(fault).strip().lower()
    return FAULT_MAP.get(norm, fault.replace("_", " ").title())


def format_fault_name(fault: Optional[str]) -> str:
    """Backward-compatible alias for format_fault."""
    return format_fault(fault)


def format_evidence_status(status: Optional[str]) -> str:
    """Format physics consistency or temporal evidence status."""
    if not status:
        return "Unavailable"
    norm = str(status).strip().upper()
    return EVIDENCE_STATUS_MAP.get(norm, norm.replace("_", " ").title())


def format_status(status: Optional[str]) -> str:
    """Format generic status string."""
    if not status:
        return "Unavailable"
    norm = str(status).strip().upper()
    if norm in HEALTH_STATE_MAP:
        return HEALTH_STATE_MAP[norm]
    if norm in RUL_STATE_MAP:
        return RUL_STATE_MAP[norm]
    if norm in DATA_QUALITY_MAP:
        return DATA_QUALITY_MAP[norm]
    return norm.replace("_", " ").title()


def format_error(error: Union[Exception, str], context: str = "") -> str:
    """Sanitize technical exceptions into clear operational descriptions."""
    msg = str(error)
    if "KeyError" in msg or "IndexError" in msg:
        detail = "Data payload is missing required telemetry fields."
    elif "Connection" in msg or "Timeout" in msg:
        detail = "Connection to telemetry stream timed out or was interrupted."
    elif "Memory" in msg or "OOM" in msg:
        detail = "Buffer capacity exceeded during calculation."
    else:
        detail = "An unexpected processing error occurred."

    ctx_prefix = f"**{context} failed.** " if context else ""
    return f"{ctx_prefix}{detail} Please try refreshing or restarting the simulation."


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

