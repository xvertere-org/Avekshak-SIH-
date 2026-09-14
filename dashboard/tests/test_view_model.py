"""
Unit tests for View Model and Formatters in SIH26054 Dashboard.
"""

import math
import pytest
from dashboard.schemas.view_model import StatusLevel, AvailabilityStatus
from dashboard.utils.formatters import (
    format_value,
    format_rul,
    format_percent,
    format_health_index,
    format_timestamp,
    format_fault_name,
    format_fault,
    format_action,
    format_channel,
    format_health_state,
    format_health_trend,
    format_rul_state,
    format_limiting_factor,
    format_forecast_status,
    format_data_quality_status,
    format_error,
    map_health_to_status,
    map_anomaly_to_status,
    map_rul_status_to_status,
    safe_is_nan,
)


def test_format_value():
    assert format_value(12.3456, decimals=2, unit="°C") == "12.35 °C"
    assert format_value(None) == "Unavailable"
    assert format_value(float("nan")) == "Unavailable"
    assert format_value(float("inf")) == "Unavailable"
    assert format_value("custom_str") == "custom_str"


def test_format_rul():
    # 7200 seconds = 2.0 hrs
    assert format_rul(7200.0) == "2.0 hrs (120 min)"
    assert format_rul(7200.0, p05=6000.0, p95=8000.0) == "2.0 hrs [1.7 - 2.2 hrs]"
    assert format_rul(0.0) == "0.0 hrs (EOL Reached)"
    assert format_rul(-10.0) == "0.0 hrs (EOL Reached)"
    assert format_rul(None) == "Unavailable"
    assert format_rul(float("nan")) == "Unavailable"


def test_format_percent():
    assert format_percent(0.854) == "85.4%"
    assert format_percent(85.4) == "85.4%"
    assert format_percent(None) == "Unavailable"
    assert format_percent(float("nan")) == "Unavailable"


def test_format_health_index():
    assert format_health_index(0.923) == "0.92"
    assert format_health_index(1.5) == "1.00"
    assert format_health_index(-0.2) == "0.00"
    assert format_health_index(None) == "Unavailable"
    assert format_health_index(float("nan")) == "Unavailable"


def test_format_timestamp():
    assert format_timestamp(42.5) == "T+42.5 s"
    assert format_timestamp(None) == "T+ -- s"
    assert format_timestamp(float("nan")) == "T+ -- s"


def test_format_fault_name():
    assert format_fault_name("cooling_degradation") == "Cooling Degradation"
    assert format_fault_name("none") == "Nominal / None"
    assert format_fault_name("") == "Nominal / None"
    assert format_fault_name(None) == "Nominal / None"


def test_status_mappers():
    assert map_health_to_status("HEALTHY") == StatusLevel.HEALTHY
    assert map_health_to_status("DEGRADED") == StatusLevel.WARNING
    assert map_health_to_status("CRITICAL") == StatusLevel.CRITICAL
    assert map_health_to_status(None) == StatusLevel.UNAVAILABLE

    assert map_anomaly_to_status("NORMAL") == StatusLevel.HEALTHY
    assert map_anomaly_to_status("WARNING") == StatusLevel.WARNING
    assert map_anomaly_to_status("ANOMALY") == StatusLevel.CRITICAL
    assert map_anomaly_to_status(None) == StatusLevel.UNAVAILABLE

    assert map_rul_status_to_status("NOT_DEGRADING") == StatusLevel.HEALTHY
    assert map_rul_status_to_status("ACTIVE_DEGRADATION") == StatusLevel.WARNING
    assert map_rul_status_to_status("CRITICAL_EOL_REACHED") == StatusLevel.CRITICAL
    assert map_rul_status_to_status(None) == StatusLevel.UNAVAILABLE


def test_avekshak_terminology_layer():
    # Advisory Actions
    assert format_action("NORMAL_MONITORING") == "Routine Monitoring"
    assert format_action("ADVISORY_CAUTION") == "Operational Caution"
    assert format_action("MAINTENANCE_INSPECTION") == "Maintenance Inspection Recommended"
    assert format_action("CRITICAL_ABORT_ACTION") == "Immediate Operational Intervention"
    assert format_action(None) == "Routine Monitoring"

    # Health States
    assert format_health_state("NOT_DEGRADING") == "Stable Condition"
    assert format_health_state("ACTIVE_DEGRADATION") == "Active Degradation"
    assert format_health_state("SEVERELY_DEGRADED") == "Severe Degradation"
    assert format_health_state("CRITICAL_EOL_REACHED") == "Critical Limit Reached"
    assert format_health_state(None) == "Unavailable"

    # Health Trends
    assert format_health_trend("RAPIDLY_DEGRADING") == "Rapid Health Decline"
    assert format_health_trend("STABLE") == "Stable Trend"

    # RUL States
    assert format_rul_state("INSUFFICIENT_HISTORY") == "Insufficient Flight History"
    assert format_rul_state("ACTIVE_DEGRADATION") == "Active Degradation"

    # Limiting Factors
    assert format_limiting_factor("REDLINE_CHT") == "CHT Thermal Limit"
    assert format_limiting_factor("REDLINE_OIL_PRESSURE") == "Oil Pressure Limit"
    assert format_limiting_factor("GLOBAL_HEALTH_INDEX") == "Overall Health Limit"
    assert format_limiting_factor("NONE") == "None (Nominal)"
    assert format_limiting_factor(None) == "None (Nominal)"

    # Forecast States
    assert format_forecast_status("LOADED_PRETRAINED") == "Forecast Model Active"
    assert format_forecast_status("BLOCKED_UNAUTHENTICATED_GATED") == "Advanced Forecast Unavailable"
    assert format_forecast_status("BUFFERING") == "Preparing Forecast"

    # Data Quality States
    assert format_data_quality_status("OUT_OF_ORDER_REJECTED") == "Out-of-Order Timestamp"
    assert format_data_quality_status("MISSING") == "Signal Missing"
    assert format_data_quality_status("DROPOUT") == "Signal Dropout"
    assert format_data_quality_status("PHYSICALLY_INVALID") == "Outside Physical Bounds"
    assert format_data_quality_status("WARNING_ENVELOPE") == "Caution Range"
    assert format_data_quality_status("ISOLATED_BY_PHM") == "Sensor Channel Isolated"

    # Channels
    assert format_channel("rpm") == "Engine Speed (RPM)"
    assert format_channel("cht") == "Cylinder Head Temperature (CHT)"
    assert format_channel("egt") == "Exhaust Gas Temperature (EGT)"
    assert format_channel("oil_pressure") == "Oil Pressure"
    assert format_channel("oil_temp") == "Oil Temperature"
    assert format_channel("fuel_flow") == "Fuel Flow"
    assert format_channel("vibration") == "Vibration"
    assert format_channel("throttle") == "Throttle"
    assert format_channel("engine_load") == "Engine Load"
    assert format_channel("load") == "Engine Load"
    assert format_channel("altitude") == "Altitude"
    assert format_channel("ambient_temp") == "Ambient Temperature"

    # Faults
    assert format_fault("cooling_degradation") == "Cooling Degradation"
    assert format_fault("sensor_fault") == "Sensor Signal Dropout"
    assert format_fault("none") == "Nominal / None"

    # Error Sanitization
    err_str = format_error(KeyError("missing_col"), context="Mission report")
    assert "Mission report failed" in err_str
    assert "KeyError" not in err_str

