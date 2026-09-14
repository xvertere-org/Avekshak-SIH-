"""
Tests for authoritative sensor-fault identification across all 7 telemetry channels.
Verifies:
1. Dynamic identification of the actual faulty channel across all 7 canonical sensors
   (rpm, cht, oil_pressure, egt, vibration, oil_temp, fuel_flow) - no hardcoded 'cht'.
2. Multi-sensor simultaneous fault handling (2 and 3 simultaneous sensor failures).
3. Ambiguity & uncertainty handling (low confidence, coupled physical deviations -> 'unknown').
4. Safety constraint enforcement: minimum 4 active channels preserved, preventing systemic isolation.
5. Telemetry dropout (NaN) detection and isolation.
6. Integration across FaultDiagnosisPipeline, Orchestrator, and Dashboard adapter.
"""

import math
import pytest
from typing import Dict, Any, List

from fault_diagnosis.sensor_identification import (
    SensorFaultIdentifier,
    SensorIdentificationResult,
    CANONICAL_CHANNELS,
)
from fault_diagnosis.schema import FaultDiagnosisResult, CANONICAL_FAULT_LABELS
from fault_diagnosis.pipeline import FaultDiagnosisPipeline
from fault_diagnosis.features import FeatureExtractor
from fault_diagnosis.classifier import XGBoostFaultClassifier
from orchestrator.adapter import PipelineHandoffAdapter
from orchestrator.schema import DashboardStatePayload
from dashboard.services.adapter import DashboardAdapter


# ---------------------------------------------------------------------------
# Unit Tests: SensorFaultIdentifier
# ---------------------------------------------------------------------------

class TestSensorFaultIdentifier:
    """Direct tests for multi-evidence SensorFaultIdentifier."""

    def test_canonical_channels_completeness(self):
        expected_channels = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
        assert set(CANONICAL_CHANNELS) == set(expected_channels)

    @pytest.mark.parametrize("target_channel", CANONICAL_CHANNELS)
    def test_single_sensor_fault_identification_all_channels(self, target_channel):
        """Verify that every single canonical channel is correctly identified when isolated."""
        identifier = SensorFaultIdentifier(outlier_threshold_sigma=2.5, nominal_bound_sigma=1.5)
        
        # Build residual dictionary: target channel has 4.5 sigma, all others are near 0.2 sigma
        residuals = {f"{ch}_norm_residual": 0.2 for ch in CANONICAL_CHANNELS}
        residuals[f"{target_channel}_norm_residual"] = 4.5
        
        # Also provide clean telemetry
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=[target_channel],
            diagnostic_confidence=0.88,
        )

        assert result.suspect_sensor == target_channel
        assert result.suspect_sensors == [target_channel]
        assert result.sensor_isolation_status == "CONFIRMED"
        # Tuple unpacking test
        primary, suspects, status = result
        assert primary == target_channel
        assert suspects == [target_channel]
        assert status == "CONFIRMED"

        # Explicit check that non-CHT channels do NOT identify as 'cht'
        if target_channel != "cht":
            assert result.suspect_sensor != "cht"

    def test_simultaneous_two_sensor_faults(self):
        """Verify detection of 2 independent simultaneous sensor failures."""
        identifier = SensorFaultIdentifier()
        residuals = {f"{ch}_norm_residual": 0.1 for ch in CANONICAL_CHANNELS}
        residuals["rpm_norm_residual"] = 4.2
        residuals["oil_pressure_norm_residual"] = 3.9
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["rpm", "oil_pressure"],
            diagnostic_confidence=0.90,
        )

        assert set(result.suspect_sensors) == {"rpm", "oil_pressure"}
        assert result.sensor_isolation_status == "CONFIRMED"
        assert result.suspect_sensor in ("rpm", "oil_pressure")

    def test_simultaneous_three_sensor_faults(self):
        """Verify detection of 3 independent simultaneous sensor failures while maintaining >= 4 active."""
        identifier = SensorFaultIdentifier(min_valid_channels_required=4)
        residuals = {f"{ch}_norm_residual": 0.1 for ch in CANONICAL_CHANNELS}
        residuals["rpm_norm_residual"] = 4.0
        residuals["egt_norm_residual"] = 3.8
        residuals["vibration_norm_residual"] = 3.5
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["rpm", "egt", "vibration"],
            diagnostic_confidence=0.85,
        )

        assert set(result.suspect_sensors) == {"rpm", "egt", "vibration"}
        assert result.sensor_isolation_status == "CONFIRMED"
        assert len(result.suspect_sensors) == 3

    def test_safety_guard_rejects_more_than_three_sensor_isolations(self):
        """If 4 or more channels are corrupted, safety guard must report 'unknown' to protect min 4 channels."""
        identifier = SensorFaultIdentifier(min_valid_channels_required=4)
        residuals = {f"{ch}_norm_residual": 0.1 for ch in CANONICAL_CHANNELS}
        residuals["rpm_norm_residual"] = 4.0
        residuals["egt_norm_residual"] = 3.8
        residuals["vibration_norm_residual"] = 3.5
        residuals["oil_temp_norm_residual"] = 3.2
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["rpm", "egt", "vibration", "oil_temp"],
            diagnostic_confidence=0.85,
        )

        # 4 corrupted sensors would leave only 3 active channels (< 4), which violates the digital twin requirement
        assert result.suspect_sensor == "unknown"
        assert result.suspect_sensors == []
        assert result.sensor_isolation_status == "UNCERTAIN"

    def test_low_confidence_reports_unknown(self):
        """When diagnostic confidence is below threshold (< 0.50), reports 'unknown'."""
        identifier = SensorFaultIdentifier(confidence_threshold=0.50)
        residuals = {f"{ch}_norm_residual": 0.1 for ch in CANONICAL_CHANNELS}
        residuals["oil_pressure_norm_residual"] = 5.0
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["oil_pressure"],
            diagnostic_confidence=0.35,  # Low confidence
        )

        assert result.suspect_sensor == "unknown"
        assert result.suspect_sensors == []
        assert result.sensor_isolation_status == "UNCERTAIN"

    def test_coupled_physical_fault_reports_unknown(self):
        """When multiple non-candidate physical channels also deviate, reports 'unknown' (physical fault)."""
        identifier = SensorFaultIdentifier(outlier_threshold_sigma=2.5, nominal_bound_sigma=1.5)
        residuals = {f"{ch}_norm_residual": 0.2 for ch in CANONICAL_CHANNELS}
        # One major outlier
        residuals["cht_norm_residual"] = 4.5
        # Correlated physical channels also deviate (e.g. cooling failure coupling)
        residuals["oil_temp_norm_residual"] = 1.9  # > 1.5 sigma
        residuals["egt_norm_residual"] = 1.8       # > 1.5 sigma
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["cht"],
            diagnostic_confidence=0.85,
        )

        assert result.suspect_sensor == "unknown"
        assert result.suspect_sensors == []
        assert result.sensor_isolation_status == "UNCERTAIN"

    def test_nan_dropout_candidate_identification(self):
        """Verify that sensor dropout (NaN in telemetry) is detected as suspect sensor."""
        identifier = SensorFaultIdentifier()
        residuals = {f"{ch}_norm_residual": 0.1 for ch in CANONICAL_CHANNELS}
        telemetry = {ch: 100.0 for ch in CANONICAL_CHANNELS}
        telemetry["vibration"] = float("nan")

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            diagnostic_confidence=0.80,
        )

        assert result.suspect_sensor == "vibration"
        assert result.suspect_sensors == ["vibration"]
        assert result.sensor_isolation_status == "CONFIRMED"


# ---------------------------------------------------------------------------
# Integration Tests: FaultDiagnosisPipeline & PipelineHandoffAdapter
# ---------------------------------------------------------------------------

class TestPipelineSensorFaultIdentification:
    """Integration tests verifying suspect sensor fields across pipeline stages."""

    def test_pipeline_diagnose_sample_with_sensor_fault(self):
        import pandas as pd
        classifier = XGBoostFaultClassifier()
        extractor = FeatureExtractor()
        extractor.fit(pd.DataFrame([{"mission_phase": "CRUISE"}]))
        pipeline = FaultDiagnosisPipeline(classifier=classifier, feature_extractor=extractor)

        # Mock sample where fuel_flow is the faulty sensor
        sample = {
            "timestamp": 100.0,
            "engine_id": "ENG_001",
            "mission_id": "MSN_TEST",
            "mission_phase": "CRUISE",
            "rpm_norm_residual": 0.1,
            "cht_norm_residual": 0.2,
            "egt_norm_residual": 0.1,
            "oil_temp_norm_residual": 0.2,
            "oil_pressure_norm_residual": 0.1,
            "fuel_flow_norm_residual": 4.8,
            "vibration_norm_residual": 0.1,
        }
        # Add core features needed by FeatureExtractor
        for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
            sample[f"{col}_residual"] = sample.get(f"{col}_norm_residual", 0.0)

        # Fake the classifier prediction to sensor_fault
        classifier._is_trained = True
        classifier.predict = lambda X: ["sensor_fault"]
        import numpy as np
        classifier.predict_proba = lambda X: np.array([[0.0, 0.0, 0.0, 0.0, 0.95, 0.05]])

        result = pipeline.diagnose_sample(
            sample=sample,
            anomaly_status="ANOMALY",
            anomaly_score=0.9,
            observed_telemetry={ch: 100.0 for ch in CANONICAL_CHANNELS},
            contributing_channels=["fuel_flow"],
        )

        assert result.predicted_fault_type == "sensor_fault"
        assert result.suspect_sensor == "fuel_flow"
        assert result.suspect_channel == "fuel_flow"
        assert result.suspect_sensors == ["fuel_flow"]
        assert result.sensor_isolation_status == "CONFIRMED"

        # Verify serialization
        d = result.to_dict()
        assert d["suspect_sensor"] == "fuel_flow"
        assert d["suspect_channel"] == "fuel_flow"
        assert d["suspect_sensors"] == ["fuel_flow"]
        assert d["sensor_isolation_status"] == "CONFIRMED"

    def test_handoff_adapter_passes_correct_suspect_to_health_index(self):
        """Verify PipelineHandoffAdapter.step_health_index receives fuel_flow and not 'cht'."""
        class MockHealthPipeline:
            def __init__(self):
                self.received_context = None

            def process_sample(self, row_dict, optional_context=None):
                self.received_context = optional_context
                return None

        health_pipe = MockHealthPipeline()

        # Create a mock diagnosis result for oil_temp fault
        diag_res = FaultDiagnosisResult(
            timestamp=10.0,
            engine_id="ENG_001",
            mission_id="MSN_001",
            mission_phase="CRUISE",
            predicted_fault_type="sensor_fault",
            class_probabilities={"sensor_fault": 0.9},
            diagnostic_confidence=0.9,
            suspect_sensor="oil_temp",
            suspect_sensors=["oil_temp"],
            sensor_isolation_status="CONFIRMED",
        )

        class MockResidualFrame:
            def to_dataframe(self):
                import pandas as pd
                return pd.DataFrame([{"timestamp": 10.0}])

        PipelineHandoffAdapter.step_health_index(
            pipeline=health_pipe,
            residual_frame=MockResidualFrame(),
            diagnosis_result=diag_res,
        )

        assert health_pipe.received_context is not None
        assert health_pipe.received_context["suspect_channel"] == "oil_temp"
        assert health_pipe.received_context["suspect_channels"] == ["oil_temp"]
        # Explicit check that it is NOT 'cht'
        assert health_pipe.received_context["suspect_channel"] != "cht"

    def test_dashboard_adapter_view_model_mapping(self):
        """Verify that DashboardTelemetryAdapter populates DiagnosticsViewModel with correct suspect sensor."""
        payload = DashboardStatePayload(
            engine_id="ENG_001",
            mission_id="MSN_001",
            timestamp=100.0,
            mission_phase="CRUISE",
            predicted_fault_class="sensor_fault",
            diagnostic_confidence=0.92,
            suspect_sensor="oil_pressure",
            suspect_sensors=["oil_pressure"],
            sensor_isolation_status="CONFIRMED",
            observed_telemetry={ch: 100.0 for ch in CANONICAL_CHANNELS},
        )

        adapter = DashboardAdapter()
        vm = adapter.adapt(payload)

        assert vm.diagnostics.predicted_fault == "sensor_fault"
        assert vm.diagnostics.suspect_sensor == "oil_pressure"
        assert vm.diagnostics.suspect_sensors == ["oil_pressure"]
        assert vm.diagnostics.sensor_isolation_status == "CONFIRMED"
        assert "oil_pressure" in vm.diagnostics.isolated_channels
        assert "cht" not in vm.diagnostics.isolated_channels

    def test_health_calculator_multi_sensor_exclusion(self):
        """Verify that HealthCalculator excludes multiple confirmed suspect sensors."""
        from health_index.calculator import HealthCalculator
        calc = HealthCalculator()

        residuals = {ch: 0.1 for ch in CANONICAL_CHANNELS}
        residuals["rpm"] = 5.0
        residuals["egt"] = 4.5

        (
            raw_hi,
            raw_deg,
            contrib,
            evidence,
            dominant,
            valid,
            missing,
            excluded,
            weights,
            quality,
        ) = calc.compute(
            timestamp=10.0,
            residuals=residuals,
            upstream_fault_type="sensor_fault",
            upstream_confidence=0.85,
            upstream_suspect_channels=["rpm", "egt"],
        )

        assert set(excluded) == {"rpm", "egt"}
        assert "rpm" not in weights
        assert "egt" not in weights
        assert len(weights) == 5
        assert math.isfinite(raw_hi)

    def test_health_calculator_safeguards_min_valid_channels(self):
        """Verify that HealthCalculator never isolates more than len(valid) - min_valid_channels."""
        from health_index.calculator import HealthCalculator
        calc = HealthCalculator()

        residuals = {ch: 0.1 for ch in CANONICAL_CHANNELS}

        (
            raw_hi,
            raw_deg,
            contrib,
            evidence,
            dominant,
            valid,
            missing,
            excluded,
            weights,
            quality,
        ) = calc.compute(
            timestamp=10.0,
            residuals=residuals,
            upstream_fault_type="sensor_fault",
            upstream_confidence=0.85,
            upstream_suspect_channels=["rpm", "egt", "cht", "oil_temp"],  # 4 channels
        )

        # With 7 valid channels and min_valid_channels=4, max isolatable is 3
        assert len(excluded) <= 3
        assert len(weights) >= 4

    def test_dashboard_state_payload_serialization(self):
        """Verify that DashboardStatePayload serializes suspect sensor fields cleanly."""
        payload = DashboardStatePayload(
            engine_id="ENG_001",
            mission_id="MSN_001",
            timestamp=100.0,
            mission_phase="CRUISE",
            predicted_fault_class="sensor_fault",
            suspect_sensor="vibration",
            suspect_sensors=["vibration"],
            sensor_isolation_status="CONFIRMED",
        )
        d = payload.to_dict()
        assert d["diagnosis"]["suspect_sensor"] == "vibration"
        assert d["diagnosis"]["suspect_channel"] == "vibration"
        assert d["diagnosis"]["suspect_sensors"] == ["vibration"]
        assert d["diagnosis"]["sensor_isolation_status"] == "CONFIRMED"

