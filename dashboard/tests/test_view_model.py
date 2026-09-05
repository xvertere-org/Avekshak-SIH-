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
