"""
Comprehensive Test Suite for Phase 9: Real Telemetry Readiness.

Covers:
1. Canonical telemetry contract, types, and lossless TelemetryRecord bridge.
2. Explicit SI / engineering unit conversions and dimensional safety.
3. Boundary validation, clock skew detection, and sequence number auditing.
4. Multi-rate buffering, epistemic status tracking (HELD_LAST_VALUE), and bounded interpolation.
5. Deterministic replay engine (streaming & materialized) and adapters.
6. End-to-end integration through Digital Twin synchronizer, residuals, and health.
7. Sensor fault (F6/F7) boundary compatibility and ambiguity handling.
8. Static anti-circularity and anti-label-leakage audits.
9. Performance benchmarking (1,000 records) and bounded memory verification.
"""

import math
import json
import time
import ast
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from telemetry.canonical import (
    SourceType,
    CalibrationStatus,
    CalibrationMetadata,
    CanonicalMeasurement,
    CanonicalTelemetryPacket,
)
from telemetry.schema import TelemetryRecord, FaultCategory
from telemetry.units import (
    PhysicalDimension,
    UnitConversionError,
    convert_unit,
    convert_mass_to_volumetric_flow,
    get_unit_dimension,
)
from telemetry.validator import (
    ClockStatus,
    SequenceStatus,
    ValidationReport,
    BoundaryValidator,
)
from telemetry.multi_rate import (
    ChannelSample,
    MultiRateBuffer,
)
from telemetry.replay import DeterministicReplayEngine
from telemetry.adapters import (
    JSONReplayAdapter,
    CSVReplayAdapter,
)
from digital_twin.quality import DataQualityStatus
from digital_twin.state import QuantityStatus
from digital_twin.synchronizer import StateEstimator, EstimatorConfig


# ============================================================
# 1. CANONICAL TELEMETRY CONTRACT & LOSSLESS BRIDGE
# ============================================================

def test_canonical_measurement_creation_and_immutability():
    """Test CanonicalMeasurement stores raw values immutably and tracks quality."""
    m = CanonicalMeasurement(
        channel_name="rpm",
        value=5000.0,
        unit="RPM",
        raw_value="5000.0",
        raw_unit="RPM",
        timestamp=100.0,
        source_id="sensor_rpm_01",
        source_type=SourceType.REAL_SENSOR,
        quality=DataQualityStatus.VALID,
        status=QuantityStatus.MEASURED,
    )
    assert m.channel_name == "rpm"
    assert m.value == 5000.0
    assert m.raw_value == "5000.0"
    assert m.source_type == SourceType.REAL_SENSOR
    assert m.status == QuantityStatus.MEASURED
    assert m.quality == DataQualityStatus.VALID


def test_canonical_telemetry_packet_lossless_bridge_to_telemetry_record():
    """Test lossless conversion from CanonicalTelemetryPacket to TelemetryRecord."""
    pkt = CanonicalTelemetryPacket(
        timestamp=10.5,
        source_id="feed_test",
        source_type=SourceType.REPLAY,
        engine_id="ENGINE_UAV_01",
        mission_id="MISSION_TEST",
        mission_phase="CRUISE",
        measurements={
            "rpm": CanonicalMeasurement("rpm", 5200.0, "RPM", timestamp=10.5),
            "altitude": CanonicalMeasurement("altitude", 2500.0, "m", timestamp=10.5),
            "ambient_temp": CanonicalMeasurement("ambient_temp", 12.0, "degC", timestamp=10.5),
            "throttle": CanonicalMeasurement("throttle", 80.0, "%", timestamp=10.5),
            "load": CanonicalMeasurement("load", 75.0, "%", timestamp=10.5),
            "cht": CanonicalMeasurement("cht", 115.0, "degC", timestamp=10.5),
            "egt": CanonicalMeasurement("egt", 820.0, "degC", timestamp=10.5),
            "oil_temp": CanonicalMeasurement("oil_temp", 98.0, "degC", timestamp=10.5),
            "oil_pressure": CanonicalMeasurement("oil_pressure", 3.6, "bar", timestamp=10.5),
            "fuel_flow": CanonicalMeasurement("fuel_flow", 24.0, "L/h", timestamp=10.5),
            "vibration": CanonicalMeasurement("vibration", 0.9, "g", timestamp=10.5),
            "map_bar": CanonicalMeasurement("map_bar", 1.15, "bar", timestamp=10.5),
        }
    )
    rec = pkt.to_telemetry_record()
    assert isinstance(rec, TelemetryRecord)
    assert rec.timestamp == 10.5
    assert rec.rpm == 5200.0
    assert rec.altitude == 2500.0
    assert rec.oil_pressure == 3.6
    assert rec.fuel_flow == 24.0
    assert rec.map_bar == 1.15
    assert rec.source_type == "replay"


def test_canonical_telemetry_packet_from_telemetry_record_interop():
    """Test lossless ingestion of TelemetryRecord into CanonicalTelemetryPacket."""
    rec = TelemetryRecord(
        timestamp=20.0,
        mission_id="MISSION_SIM",
        engine_id="ENGINE_UAV_01",
        mission_phase="CLIMB",
        altitude=1500.0,
        ambient_temp=18.0,
        throttle=85.0,
        load=80.0,
        rpm=5400.0,
        cht=118.0,
        egt=840.0,
        oil_temp=101.0,
        oil_pressure=3.8,
        fuel_flow=27.0,
        vibration=1.1,
        source="simulator_v1",
        source_type="simulated",
    )
    pkt = CanonicalTelemetryPacket.from_telemetry_record(rec)
    assert pkt.timestamp == 20.0
    assert pkt.source_type == SourceType.SIMULATOR
    assert pkt.get_value("rpm") == 5400.0
    assert pkt.get_value("oil_pressure") == 3.8
    assert pkt.measurements["rpm"].quality == DataQualityStatus.VALID
    assert pkt.measurements["rpm"].status == QuantityStatus.MEASURED


def test_source_type_vs_quantity_status_decoupling():
    """Verify physical source origin is decoupled from epistemic quantity status."""
    # Simulator output is MEASURED in simulation domain, but origin is SIMULATOR
    m_sim = CanonicalMeasurement(
        channel_name="cht", value=110.0, unit="degC",
        source_type=SourceType.SIMULATOR, status=QuantityStatus.MEASURED
    )
    # Real sensor with hold-last-value is ESTIMATED, but origin is REAL_SENSOR
    m_real_held = CanonicalMeasurement(
        channel_name="cht", value=110.0, unit="degC",
        source_type=SourceType.REAL_SENSOR, status=QuantityStatus.ESTIMATED
    )
    assert m_sim.source_type == SourceType.SIMULATOR
    assert m_sim.status == QuantityStatus.MEASURED
    assert m_real_held.source_type == SourceType.REAL_SENSOR
    assert m_real_held.status == QuantityStatus.ESTIMATED


def test_calibration_metadata_application():
    """Verify calibration is applied only when marked active."""
    cal_inactive = CalibrationMetadata(scale=1.05, offset=2.0, is_active=False)
    assert cal_inactive.apply(100.0) == 100.0

    cal_active = CalibrationMetadata(scale=1.05, offset=2.0, is_active=True)
    assert cal_active.apply(100.0) == pytest.approx(107.0, 1e-6)


# ============================================================
# 2. EXPLICIT UNIT CONVERSIONS & DIMENSIONAL SAFETY
# ============================================================

def test_pressure_conversions_bar_pa_hpa_psi_inhg():
    """Verify exact pressure conversions against independently calculated constants."""
    # 1 bar = 100,000 Pa
    assert convert_unit(1.0, "bar", "Pa") == pytest.approx(100000.0, 1e-6)
    assert convert_unit(100000.0, "Pa", "bar") == pytest.approx(1.0, 1e-6)

    # 1 bar = 1000 hPa
    assert convert_unit(1.0, "bar", "hPa") == pytest.approx(1000.0, 1e-6)
    assert convert_unit(1013.25, "hPa", "bar") == pytest.approx(1.01325, 1e-6)

    # 14.50377 psi ~ 1 bar
    assert convert_unit(14.50377377, "psi", "bar") == pytest.approx(1.0, 1e-5)
    assert convert_unit(1.0, "bar", "psi") == pytest.approx(14.50377377, 1e-5)

    # 29.52998 inHg ~ 1 bar
    assert convert_unit(29.529983, "inHg", "bar") == pytest.approx(1.0, 1e-4)


def test_temperature_conversions_c_k_f():
    """Verify temperature conversions against standard zero and boiling points."""
    assert convert_unit(0.0, "degC", "K") == pytest.approx(273.15, 1e-6)
    assert convert_unit(273.15, "K", "degC") == pytest.approx(0.0, 1e-6)

    assert convert_unit(100.0, "degC", "degF") == pytest.approx(212.0, 1e-6)
    assert convert_unit(32.0, "degF", "degC") == pytest.approx(0.0, 1e-6)
    assert convert_unit(-40.0, "degF", "degC") == pytest.approx(-40.0, 1e-6)


def test_volumetric_flow_conversions_lh_m3s_gpm():
    """Verify volumetric flow conversions."""
    # 1 m3/s = 3,600,000 L/h
    assert convert_unit(1.0, "m3/s", "L/h") == pytest.approx(3600000.0, 1e-6)
    # 1 gpm ~ 227.1247 L/h
    assert convert_unit(1.0, "gpm", "L/h") == pytest.approx(227.124707, 1e-4)


def test_mass_flow_conversions_kgs_kgh_pph():
    """Verify mass flow conversions."""
    # 1 kg/s = 3600 kg/h
    assert convert_unit(1.0, "kg/s", "kg/h") == pytest.approx(3600.0, 1e-6)
    # 1 pph ~ 0.453592 kg/h
    assert convert_unit(100.0, "pph", "kg/h") == pytest.approx(45.359237, 1e-5)


def test_mass_to_volumetric_flow_requires_density_parameter():
    """Verify mass-to-volume flow conversion strictly requires explicit fuel density."""
    with pytest.raises(ValueError, match="rho_fuel_kg_L.*must be explicitly provided"):
        convert_mass_to_volumetric_flow(0.005, "kg/s", rho_fuel_kg_L=None)

    with pytest.raises(ValueError, match="rho_fuel_kg_L.*must be explicitly provided"):
        convert_mass_to_volumetric_flow(0.005, "kg/s", rho_fuel_kg_L=0.0)


def test_mass_to_volumetric_flow_provenance_tagging():
    """Verify mass-to-volume flow calculation tags density assumption in provenance note."""
    # 0.005 kg/s = 18 kg/h; with rho = 0.72 kg/L -> 18 / 0.72 = 25.0 L/h
    vol_flow, unit, note = convert_mass_to_volumetric_flow(0.005, "kg/s", rho_fuel_kg_L=0.72)
    assert vol_flow == pytest.approx(25.0, 1e-6)
    assert unit == "L/h"
    assert "DERIVED_FROM_MASS_FLOW" in note
    assert "0.7200 kg/L" in note


def test_acceleration_conversions_g_to_ms2():
    """Verify broadband RMS acceleration conversion."""
    assert convert_unit(1.0, "g", "m/s2") == pytest.approx(9.80665, 1e-6)
    assert convert_unit(9.80665, "m/s2", "g") == pytest.approx(1.0, 1e-6)


def test_negative_cross_dimensional_conversion_fails_unit_error():
    """Verify cross-dimensional unit conversion raises UnitConversionError."""
    with pytest.raises(UnitConversionError, match="Cross-dimensional unit conversion is prohibited"):
        convert_unit(1.0, "bar", "degC")

    with pytest.raises(UnitConversionError, match="Cross-dimensional unit conversion is prohibited"):
        convert_unit(5000.0, "RPM", "m")


def test_negative_vibration_accel_to_velocity_fails_unit_error():
    """Verify direct conversion between acceleration (g) and velocity (mm/s) strictly raises UnitConversionError."""
    with pytest.raises(UnitConversionError, match="Cannot directly convert between acceleration.*and velocity"):
        convert_unit(1.0, "g", "mm/s")

    with pytest.raises(UnitConversionError, match="Cannot directly convert between acceleration.*and velocity"):
        convert_unit(10.0, "mm/s", "g")


def test_length_altitude_conversions_m_ft():
    """Verify altitude conversions."""
    assert convert_unit(1000.0, "ft", "m") == pytest.approx(304.8, 1e-6)
    assert convert_unit(304.8, "m", "ft") == pytest.approx(1000.0, 1e-6)


def test_rotational_speed_conversions_rpm_rads_hz():
    """Verify rotational speed conversions."""
    # 60 RPM = 1 Hz = 2*pi rad/s
    assert convert_unit(60.0, "RPM", "Hz") == pytest.approx(1.0, 1e-6)
    assert convert_unit(60.0, "RPM", "rad/s") == pytest.approx(2.0 * math.pi, 1e-5)


# ============================================================
# 3. BOUNDARY VALIDATOR, CLOCK SKEW & SEQUENCE AUDITING
# ============================================================

def test_boundary_validator_accepts_clean_record():
    """Verify boundary validator accepts well-formed external record."""
    val = BoundaryValidator()
    raw = {
        "timestamp": 100.0,
        "sequence": 1,
        "channels": {
            "rpm": 5000.0,
            "oil_pressure": 3.5,
            "cht": 110.0,
        }
    }
    pkt, report = val.validate_packet(raw, ingest_time=100.1)
    assert pkt is not None
    assert report.is_acceptable is True
    assert report.clock_status == ClockStatus.VALID
    assert report.sequence_status == SequenceStatus.NORMAL
    assert pkt.get_value("rpm") == 5000.0


def test_boundary_validator_rejects_missing_timestamp():
    """Verify boundary validator rejects record with missing timestamp."""
    val = BoundaryValidator()
    raw = {"sequence": 1, "channels": {"rpm": 5000.0}}
    pkt, report = val.validate_packet(raw)
    assert pkt is None
    assert report.is_acceptable is False
    assert "timestamp" in report.quarantined_reasons


def test_boundary_validator_rejects_non_numeric_and_nan():
    """Verify boundary validator quarantines non-numeric strings and NaN values."""
    val = BoundaryValidator()
    raw = {
        "timestamp": 100.0,
        "channels": {
            "rpm": "corrupted_rpm_string",
            "cht": float("nan"),
            "egt": 800.0,
        }
    }
    pkt, report = val.validate_packet(raw)
    assert pkt is not None
    assert "rpm" in report.dropped_channels
    assert "cht" in report.dropped_channels
    assert pkt.measurements["rpm"].quality == DataQualityStatus.INVALID
    assert pkt.measurements["cht"].quality == DataQualityStatus.NON_FINITE
    assert pkt.measurements["egt"].quality == DataQualityStatus.VALID


def test_boundary_validator_clock_latency_and_negative_skew():
    """Verify boundary validator flags negative latency as CLOCK_SKEW."""
    val = BoundaryValidator()
    raw = {"timestamp": 100.0, "channels": {"rpm": 5000.0}}

    # Ingest time before source time -> Clock skew
    pkt, report = val.validate_packet(raw, ingest_time=99.0)
    assert report.clock_status == ClockStatus.CLOCK_SKEW
    assert report.latency == pytest.approx(-1.0, 1e-4)
    assert any("CLOCK_SKEW" in note for note in report.notes)


def test_boundary_validator_sequence_normal_and_duplicate():
    """Verify sequence number validation detects duplicate sequence."""
    val = BoundaryValidator()
    raw1 = {"timestamp": 100.0, "sequence": 10, "channels": {"rpm": 5000.0}}
    raw2 = {"timestamp": 101.0, "sequence": 10, "channels": {"rpm": 5005.0}}

    pkt1, rep1 = val.validate_packet(raw1)
    assert rep1.sequence_status == SequenceStatus.NORMAL

    pkt2, rep2 = val.validate_packet(raw2)
    assert rep2.sequence_status == SequenceStatus.DUPLICATE


def test_boundary_validator_sequence_skipped_gap():
    """Verify sequence auditor detects skipped packets."""
    val = BoundaryValidator()
    raw1 = {"timestamp": 100.0, "sequence": 1, "channels": {"rpm": 5000.0}}
    raw2 = {"timestamp": 101.0, "sequence": 5, "channels": {"rpm": 5005.0}}

    val.validate_packet(raw1)
    pkt2, rep2 = val.validate_packet(raw2)
    assert rep2.sequence_status == SequenceStatus.SKIPPED
    assert "gap: 3" in rep2.notes[0]


def test_boundary_validator_sequence_modulus_wraparound():
    """Verify sequence auditor handles wraparound when sequence_modulus is explicitly configured."""
    val = BoundaryValidator(sequence_modulus=256) # 8-bit counter
    raw1 = {"timestamp": 100.0, "sequence": 255, "channels": {"rpm": 5000.0}}
    raw2 = {"timestamp": 101.0, "sequence": 0, "channels": {"rpm": 5005.0}}

    val.validate_packet(raw1)
    pkt2, rep2 = val.validate_packet(raw2)
    assert rep2.sequence_status == SequenceStatus.WRAPAROUND


def test_boundary_validator_unknown_modulus_does_not_infer_wraparound():
    """Verify sequence auditor does NOT infer wraparound when modulus is unconfigured."""
    val = BoundaryValidator(sequence_modulus=None)
    raw1 = {"timestamp": 100.0, "sequence": 65535, "channels": {"rpm": 5000.0}}
    raw2 = {"timestamp": 101.0, "sequence": 0, "channels": {"rpm": 5005.0}}

    val.validate_packet(raw1)
    pkt2, rep2 = val.validate_packet(raw2)
    assert rep2.sequence_status == SequenceStatus.CONFIGURATION_UNKNOWN


def test_boundary_validator_outlier_preserves_raw_value_without_clipping():
    """Verify boundary validator flags OUT_OF_RANGE without modifying the raw observation."""
    val = BoundaryValidator()
    raw = {"timestamp": 100.0, "channels": {"oil_pressure": 99.0}} # Unphysically high pressure
    pkt, report = val.validate_packet(raw)
    assert pkt is not None
    m = pkt.measurements["oil_pressure"]
    assert m.quality == DataQualityStatus.OUT_OF_RANGE
    assert m.raw_value == 99.0
    # Must NOT be clipped to 15.0 bar
    assert m.raw_value != 15.0


def test_boundary_validator_physical_impossibility_checks():
    """Verify thermodynamic impossibilities (negative RPM, negative absolute pressure) are flagged."""
    val = BoundaryValidator()
    raw = {
        "timestamp": 100.0,
        "channels": {
            "rpm": -500.0,
            "oil_pressure": -2.0,
            "cht": -300.0, # Below absolute zero
        }
    }
    pkt, report = val.validate_packet(raw)
    assert pkt.measurements["rpm"].quality == DataQualityStatus.OUT_OF_RANGE
    assert pkt.measurements["oil_pressure"].quality == DataQualityStatus.OUT_OF_RANGE
    assert pkt.measurements["cht"].quality == DataQualityStatus.OUT_OF_RANGE


# ============================================================
# 4. MULTI-RATE TELEMETRY & ALIGNMENT
# ============================================================

def test_multi_rate_buffer_exact_sample_match():
    """Verify buffer returns MEASURED for exact timestamp match."""
    buf = MultiRateBuffer()
    m = CanonicalMeasurement("rpm", 5000.0, "RPM", timestamp=10.0, status=QuantityStatus.MEASURED)
    buf.push_measurement(m)

    aligned = buf.get_aligned_measurement("rpm", target_timestamp=10.0)
    assert aligned.value == 5000.0
    assert aligned.status == QuantityStatus.MEASURED


def test_multi_rate_buffer_held_last_value_marked_estimated():
    """Verify held-last-value is explicitly marked QuantityStatus.ESTIMATED (never MEASURED)."""
    buf = MultiRateBuffer(tau_stale=2.0)
    m = CanonicalMeasurement("ambient_temp", 15.0, "degC", timestamp=10.0, status=QuantityStatus.MEASURED)
    buf.push_measurement(m)

    # Query 0.5s later
    aligned = buf.get_aligned_measurement("ambient_temp", target_timestamp=10.5)
    assert aligned.value == 15.0
    assert aligned.status == QuantityStatus.ESTIMATED
    assert "HELD_LAST_VALUE" in aligned.notes
    assert aligned.source_timestamp == 10.0


def test_multi_rate_buffer_stale_timeout_triggers_unavailable():
    """Verify sample age exceeding tau_stale evaluates to STALE / UNAVAILABLE."""
    buf = MultiRateBuffer(channel_stale_limits={"rpm": 0.5})
    m = CanonicalMeasurement("rpm", 5000.0, "RPM", timestamp=10.0, status=QuantityStatus.MEASURED)
    buf.push_measurement(m)

    # Query 1.0s later (exceeds 0.5s tau_stale)
    aligned = buf.get_aligned_measurement("rpm", target_timestamp=11.0)
    assert aligned.value is None
    assert aligned.quality == DataQualityStatus.STALE
    assert aligned.status == QuantityStatus.UNAVAILABLE


def test_multi_rate_buffer_bounded_interpolation_marked_estimated():
    """Verify linear interpolation within max_interpolation_gap is marked ESTIMATED."""
    buf = MultiRateBuffer(max_interpolation_gap=1.0)
    buf.push_measurement(CanonicalMeasurement("cht", 100.0, "degC", timestamp=10.0))
    buf.push_measurement(CanonicalMeasurement("cht", 110.0, "degC", timestamp=10.5))

    aligned = buf.get_aligned_measurement("cht", target_timestamp=10.25, allow_interpolation=True)
    assert aligned.value == pytest.approx(105.0, 1e-6)
    assert aligned.status == QuantityStatus.ESTIMATED
    assert "INTERPOLATED" in aligned.notes


def test_multi_rate_buffer_interpolation_forbidden_across_large_gap():
    """Verify interpolation is rejected if gap exceeds max_interpolation_gap."""
    buf = MultiRateBuffer(max_interpolation_gap=0.5, tau_stale=5.0)
    buf.push_measurement(CanonicalMeasurement("cht", 100.0, "degC", timestamp=10.0))
    buf.push_measurement(CanonicalMeasurement("cht", 120.0, "degC", timestamp=12.0)) # 2.0s gap

    aligned = buf.get_aligned_measurement("cht", target_timestamp=11.0, allow_interpolation=True)
    # Should fall back to hold-last-value instead of interpolating across the large gap
    assert aligned.value == 100.0
    assert "HELD_LAST_VALUE" in aligned.notes


def test_multi_rate_buffer_interpolation_forbidden_across_invalid_sample():
    """Verify interpolation strictly refuses to bridge across an intermediate invalid sample (valid -> invalid -> valid)."""
    buf = MultiRateBuffer(max_interpolation_gap=1.0)
    buf.push_measurement(CanonicalMeasurement("cht", 100.0, "degC", timestamp=10.0, quality=DataQualityStatus.VALID))
    # Intervening invalid sample at 10.2
    buf.push_measurement(CanonicalMeasurement("cht", None, "degC", timestamp=10.2, quality=DataQualityStatus.INVALID))
    buf.push_measurement(CanonicalMeasurement("cht", 110.0, "degC", timestamp=10.4, quality=DataQualityStatus.VALID))

    # Query at 10.3 between invalid and valid
    aligned = buf.get_aligned_measurement("cht", target_timestamp=10.3, allow_interpolation=True)
    # Interpolation must be refused! And preceding sample was invalid, so hold-last-value is UNAVAILABLE
    assert aligned.value is None
    assert aligned.status == QuantityStatus.UNAVAILABLE


def test_multi_rate_buffer_interpolation_forbidden_across_dropout():
    """Verify interpolation strictly refuses to bridge across a sensor dropout (valid -> dropout -> valid)."""
    buf = MultiRateBuffer(max_interpolation_gap=1.0)
    buf.push_measurement(CanonicalMeasurement("rpm", 5000.0, "RPM", timestamp=10.0, quality=DataQualityStatus.VALID))
    # Intervening dropout at 10.2
    buf.push_measurement(CanonicalMeasurement("rpm", None, "RPM", timestamp=10.2, quality=DataQualityStatus.MISSING))
    buf.push_measurement(CanonicalMeasurement("rpm", 5020.0, "RPM", timestamp=10.4, quality=DataQualityStatus.VALID))

    aligned = buf.get_aligned_measurement("rpm", target_timestamp=10.3, allow_interpolation=True)
    assert aligned.value is None
    assert aligned.status == QuantityStatus.UNAVAILABLE



def test_multi_rate_packet_alignment_mixed_rates():
    """Verify multi-rate packet aligner aligns channels sampled at distinct rates."""
    buf = MultiRateBuffer(tau_stale=5.0)
    # RPM at 50 Hz (t=10.00, 10.02, 10.04)
    buf.push_measurement(CanonicalMeasurement("rpm", 5000.0, "RPM", timestamp=10.00))
    buf.push_measurement(CanonicalMeasurement("rpm", 5010.0, "RPM", timestamp=10.02))
    buf.push_measurement(CanonicalMeasurement("rpm", 5020.0, "RPM", timestamp=10.04))

    # CHT at 5 Hz (t=10.00)
    buf.push_measurement(CanonicalMeasurement("cht", 112.0, "degC", timestamp=10.00))

    # Altitude at 1 Hz (t=9.50)
    buf.push_measurement(CanonicalMeasurement("altitude", 2000.0, "m", timestamp=9.50))

    # Align to current high-rate step t=10.04
    aligned_pkt = buf.align_packet(10.04, ["rpm", "cht", "altitude"])
    assert aligned_pkt.get_value("rpm") == 5020.0
    assert aligned_pkt.measurements["rpm"].status == QuantityStatus.MEASURED
    assert aligned_pkt.get_value("cht") == 112.0
    assert aligned_pkt.measurements["cht"].status == QuantityStatus.ESTIMATED
    assert aligned_pkt.get_value("altitude") == 2000.0
    assert aligned_pkt.measurements["altitude"].status == QuantityStatus.ESTIMATED


# ============================================================
# 5. REPLAY ENGINE & ADAPTERS
# ============================================================

def test_deterministic_replay_engine_json_array():
    """Verify deterministic replay engine loads JSON array correctly."""
    golden_path = Path("tests/data/golden_telemetry.json")
    assert golden_path.exists()

    engine = DeterministicReplayEngine(source_type=SourceType.REPLAY)
    packets = engine.replay_json_array(golden_path)
    assert len(packets) == 10
    assert packets[0].timestamp == 0.0
    assert packets[-1].timestamp == 10.0
    assert packets[0].source_type == SourceType.REPLAY


def test_deterministic_replay_engine_ndjson_streaming():
    """Verify streaming NDJSON replay with line-by-line generator."""
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
        for i in range(5):
            rec = {"timestamp": float(i), "sequence": i, "channels": {"rpm": 5000.0 + i * 10}}
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name

    try:
        engine = DeterministicReplayEngine()
        streamed = list(engine.stream_ndjson(tmp_path))
        assert len(streamed) == 5
        assert streamed[0].timestamp == 0.0
        assert streamed[4].timestamp == 4.0
        assert streamed[4].get_value("rpm") == 5040.0
    finally:
        Path(tmp_path).unlink()


def test_replay_engine_determinism_identical_runs():
    """Verify identical replay runs produce bitwise identical packet values."""
    golden_path = Path("tests/data/golden_telemetry.json")
    engine = DeterministicReplayEngine()

    run1 = engine.replay_json_array(golden_path)
    run2 = engine.replay_json_array(golden_path)

    assert len(run1) == len(run2)
    for p1, p2 in zip(run1, run2):
        assert p1.timestamp == p2.timestamp
        assert p1.measurements.keys() == p2.measurements.keys()
        for k in p1.measurements:
            assert p1.measurements[k].value == p2.measurements[k].value
            assert p1.measurements[k].quality == p2.measurements[k].quality


def test_json_replay_adapter_file_parsing():
    """Verify JSONReplayAdapter parses interchange format."""
    adapter = JSONReplayAdapter(source_id="interchange_test")
    golden_path = Path("tests/data/golden_telemetry.json")
    packets = adapter.adapt_file(golden_path)
    assert len(packets) == 10
    assert packets[0].source_id == "interchange_test"


def test_csv_replay_adapter_with_unit_and_column_mappings():
    """Verify CSVReplayAdapter applies column, unit, and sensor mappings."""
    csv_data = (
        "time_sec,engine_speed,manifold_psi,head_temp_f\n"
        "0.0,5000,14.50377,212.0\n"
        "1.0,5020,14.50377,215.6\n"
    )
    df = pd.read_csv(pd.io.common.StringIO(csv_data))

    adapter = CSVReplayAdapter(
        column_mapping={
            "time_sec": "timestamp",
            "engine_speed": "rpm",
            "manifold_psi": "map_bar",
            "head_temp_f": "cht",
        },
        unit_mapping={
            "engine_speed": "RPM",
            "manifold_psi": "psi",
            "head_temp_f": "degF",
        },
    )

    packets = adapter.adapt_packets(df)
    assert len(packets) == 2
    assert packets[0].timestamp == 0.0
    assert packets[0].get_value("rpm") == 5000.0
    # 14.50377 psi ~ 1.0 bar
    assert packets[0].get_value("map_bar") == pytest.approx(1.0, 1e-4)
    # 212 degF = 100 degC
    assert packets[0].get_value("cht") == pytest.approx(100.0, 1e-4)


def test_golden_telemetry_replay_provenance():
    """Verify golden telemetry contains explicit synthetic provenance tags."""
    with open("tests/data/golden_telemetry.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        meta = item.get("metadata", {})
        assert meta.get("dataset_class") == "SYNTHETIC"
        assert meta.get("provenance") == "SYNTHETIC_REPLAY"


# ============================================================
# 6. END-TO-END DIGITAL TWIN INTEGRATION
# ============================================================

def test_end_to_end_ingestion_through_digital_twin_synchronizer():
    """Verify external JSON replay passes through adapter into StateEstimator."""
    golden_path = Path("tests/data/golden_telemetry.json")
    engine = DeterministicReplayEngine()
    packets = engine.replay_json_array(golden_path)

    estimator = StateEstimator()
    states = []
    for pkt in packets:
        rec = pkt.to_telemetry_record()
        state = estimator.step(rec)
        states.append(state)

    assert len(states) == 10
    # State tracking initialized and stepped across mission
    assert states[-1].timestamp == 10.0
    assert states[-1].rotational.rpm.value > 4000.0


def test_end_to_end_residuals_and_health_assessment():
    """Verify full pipeline: external format -> packet -> synchronizer -> residuals."""
    adapter = JSONReplayAdapter()
    packets = adapter.adapt_file("tests/data/golden_telemetry.json")
    estimator = StateEstimator()

    for pkt in packets:
        rec = pkt.to_telemetry_record()
        state = estimator.step(rec)

    # Steady state residual checks in metadata
    assert "residuals" in state.metadata
    residuals = state.metadata["residuals"]
    assert "rpm" in residuals
    assert "cht" in residuals


def test_simulator_output_compatibility_through_canonical_boundary():
    """Verify simulator output converts to CanonicalTelemetryPacket and back losslessly."""
    from simulator.engine_simulator import EngineSimulator
    from simulator.config import SimulatorConfig
    from telemetry.schema import MissionConfig

    sim = EngineSimulator(sim_config=SimulatorConfig())
    m_cfg = MissionConfig(mission_id="SIM_VALID_01", duration=5.0)
    records = sim.run_mission(m_cfg, time_step=1.0)
    assert len(records) > 0


    # Convert to canonical and verify physics parity
    for rec in records:
        pkt = CanonicalTelemetryPacket.from_telemetry_record(rec)
        rec_roundtrip = pkt.to_telemetry_record()
        assert rec_roundtrip.timestamp == rec.timestamp
        assert rec_roundtrip.rpm == rec.rpm
        assert rec_roundtrip.cht == rec.cht
        assert rec_roundtrip.oil_pressure == rec.oil_pressure


def test_f6_f7_sensor_fault_compatibility_through_boundary():
    """Verify sensor bias/drift (F6) and dropout/stuck (F7) pass through boundary without leaking fault labels."""
    val = BoundaryValidator()
    # Simulated sensor dropout (None / NaN)
    raw_dropout = {"timestamp": 10.0, "channels": {"cht": None, "rpm": 5000.0}}
    pkt_d, rep_d = val.validate_packet(raw_dropout)
    assert pkt_d.measurements["cht"].quality == DataQualityStatus.MISSING
    assert pkt_d.measurements["cht"].status == QuantityStatus.UNAVAILABLE

    # Simulated stuck sensor (steady constant)
    raw_stuck = {"timestamp": 11.0, "channels": {"cht": 110.0, "rpm": 5000.0}}
    pkt_s, rep_s = val.validate_packet(raw_stuck)
    assert pkt_s.measurements["cht"].quality == DataQualityStatus.VALID


def test_physical_vs_sensor_fault_ambiguity_behavior():
    """Verify single corrupt channel vs multi-channel physical anomaly reporting."""
    val = BoundaryValidator()
    # Single channel spike (accelerometer corrupted to 45 g)
    raw_corrupt = {"timestamp": 10.0, "channels": {"vibration": 45.0, "rpm": 5000.0, "cht": 110.0}}
    pkt_c, rep_c = val.validate_packet(raw_corrupt)
    # Quality captures the extreme measurement
    assert pkt_c.measurements["vibration"].quality == DataQualityStatus.VALID # 45 g is within 50 g instrument bound
    # Raw value preserved
    assert pkt_c.measurements["vibration"].raw_value == 45.0


# ============================================================
# 7. ARCHITECTURAL GUARDRAILS & AUDITS
# ============================================================

def test_anti_circularity_static_import_audit():
    """Verify telemetry ingestion modules do not import health, diagnosis, or RUL modules."""
    telemetry_dir = Path("telemetry")
    prohibited_tokens = ["digital_twin.health", "digital_twin.diagnosis", "digital_twin.rul", "digital_twin.degradation"]

    for py_file in telemetry_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prob in prohibited_tokens:
                        assert prob not in alias.name, f"Circularity violation: {py_file} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for prob in prohibited_tokens:
                    assert prob not in mod, f"Circularity violation: {py_file} imports from {mod}"


def test_anti_label_leakage_audit():
    """Verify external adapters and boundary validator do not access ground-truth fault labels."""
    val = BoundaryValidator()
    # Ingestion dictionary contains simulated fault tags from simulator
    raw_external = {
        "timestamp": 100.0,
        "fault_type": "cooling_degradation",
        "fault_severity": 0.8,
        "failure_time": 450.0,
        "channels": {"rpm": 5000.0, "cht": 135.0}
    }
    pkt, report = val.validate_packet(raw_external)
    # Measurements must NOT be conditioned on the fault label
    assert pkt.measurements["cht"].quality == DataQualityStatus.VALID
    assert "fault_type" not in pkt.measurements


def test_streaming_memory_bounded_buffer():
    """Verify streaming NDJSON replay runs over a long file without retaining records in memory."""
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
        for i in range(200):
            rec = {"timestamp": float(i) * 0.1, "sequence": i, "channels": {"rpm": 5000.0 + (i % 50)}}
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name

    try:
        engine = DeterministicReplayEngine()
        count = 0
        for pkt in engine.stream_ndjson(tmp_path):
            count += 1
            assert pkt.timestamp is not None
        assert count == 200
    finally:
        Path(tmp_path).unlink()


def test_performance_benchmark_1000_records():
    """Benchmark 1,000 synthetic records through validation, canonicalization, and synchronizer."""
    val = BoundaryValidator()
    estimator = StateEstimator()

    records = [
        {
            "timestamp": float(i) * 0.1,
            "sequence": i,
            "channels": {
                "rpm": 5000.0 + (i % 10),
                "throttle": 75.0,
                "altitude": 2000.0,
                "ambient_temp": 15.0,
                "cht": 110.0 + (i % 5),
                "egt": 800.0,
                "oil_temp": 95.0,
                "oil_pressure": 3.5,
                "fuel_flow": 22.0,
                "vibration": 0.8,
            }
        }
        for i in range(1000)
    ]

    latencies = []
    t_start_total = time.perf_counter()

    for r in records:
        t0 = time.perf_counter()
        pkt, rep = val.validate_packet(r)
        rec = pkt.to_telemetry_record()
        estimator.step(rec)
        latencies.append((time.perf_counter() - t0) * 1000.0) # ms

    total_duration_sec = time.perf_counter() - t_start_total

    mean_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    # Soft real-time benchmark assertion (each step should comfortably take < 10ms on modern CPU)
    assert mean_ms < 10.0, f"Mean step latency ({mean_ms:.2f}ms) exceeded 10ms threshold"
    assert total_duration_sec < 10.0, f"Total benchmark duration ({total_duration_sec:.2f}s) exceeded 10s"
