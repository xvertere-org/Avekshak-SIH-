from fault_diagnosis.sensor_identification import SensorFaultIdentifier

class TestSC004SensorAttribution:
    """
    Regression tests for SC-004: Sensor Fault Misattribution to CHT (Safety Violation).
    Validates that physical overheating (correlated thermal deviations) is not
    suppressed or misclassified as a sensor fault, while single sensor faults and
    simultaneous independent sensor faults continue to be properly isolated.
    """

    def test_genuine_overheating_not_isolated(self):
        """
        Genuine overheating causes multiple thermal channels to deviate (CHT, EGT).
        This must return UNCERTAIN, preventing suppression of the physical emergency.
        """
        identifier = SensorFaultIdentifier()
        residuals = {
            "rpm_norm_residual": 0.5,
            "cht_norm_residual": 4.5,
            "egt_norm_residual": 3.0,
            "oil_temp_norm_residual": 1.0,
            "oil_pressure_norm_residual": 0.5,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        }
        
        telemetry = {
            "rpm": 5500.0,
            "cht": 150.0,
            "egt": 750.0,
            "oil_temp": 110.0,
            "oil_pressure": 4.5,
            "fuel_flow": 25.0,
            "vibration": 0.5,
        }

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["cht", "egt"],
            diagnostic_confidence=0.9
        )
        
        # It must refuse to isolate a single sensor
        assert result.sensor_isolation_status == "UNCERTAIN"
        assert result.suspect_sensor == "unknown"
        assert result.suspect_sensors == []

    def test_single_cht_sensor_fault_isolated(self):
        """
        If ONLY CHT deviates significantly, it is a genuine CHT sensor fault.
        """
        identifier = SensorFaultIdentifier()
        residuals = {
            "rpm_norm_residual": 0.5,
            "cht_norm_residual": 4.5,
            "egt_norm_residual": 0.5,
            "oil_temp_norm_residual": 0.5,
            "oil_pressure_norm_residual": 0.5,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        }
        
        telemetry = {ch: 100.0 for ch in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["cht"],
            diagnostic_confidence=0.9
        )
        
        # It must successfully isolate CHT
        assert result.sensor_isolation_status == "CONFIRMED"
        assert result.suspect_sensor == "cht"
        assert "cht" in result.suspect_sensors

    def test_simultaneous_independent_sensor_faults_isolated(self):
        """
        If two completely uncorrelated sensors (e.g., RPM and Oil Pressure) fail
        simultaneously, they should still be successfully isolated as sensor faults.
        """
        identifier = SensorFaultIdentifier()
        residuals = {
            "rpm_norm_residual": 4.0,
            "cht_norm_residual": 0.5,
            "egt_norm_residual": 0.5,
            "oil_temp_norm_residual": 0.5,
            "oil_pressure_norm_residual": 3.9,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        }
        
        telemetry = {ch: 100.0 for ch in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]}

        result = identifier.identify_suspect_sensors(
            sample=residuals,
            observed_telemetry=telemetry,
            contributing_channels=["rpm", "oil_pressure"],
            diagnostic_confidence=0.9
        )
        
        # It must successfully isolate both without flagging physical coupling
        assert result.sensor_isolation_status == "CONFIRMED"
        assert set(result.suspect_sensors) == {"rpm", "oil_pressure"}
        # primary channel should be the one with highest magnitude
        assert result.suspect_sensor == "rpm"
