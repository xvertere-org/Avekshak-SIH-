"""
Demo Data Provider for SIH26054 Dashboard.

Provides pre-packaged synthetic test bench scenarios for evaluation and demonstration.
STRICT NOTICE: All outputs generated here are explicitly flagged with `is_synthetic_demo = True`
and must be unmistakably labeled as DEMO / SYNTHETIC.
"""

from typing import Dict, Any, List, Tuple
from dashboard.schemas.contracts import Phase13OutputContract


class DemoScenarioProvider:
    """Generates synthetic test scenarios adhering strictly to Phase 13 contracts."""

    @staticmethod
    def get_available_scenarios() -> List[str]:
        return [
            "1. Nominal Healthy Cruise",
            "2. Thermal Degradation (Cooling Conductance Loss)",
            "3. Lubrication Pressure Loss (Hydraulic Failure)",
            "4. CHT Sensor Dropout & Isolation (Instrumentation Fault)",
        ]


    @staticmethod
    def generate_scenario_payload(scenario_name: str, step: int = 15) -> Tuple[Phase13OutputContract, Dict[str, Dict[str, List[float]]]]:
        """
        Generate a single-step contract along with a 30-step historical rolling buffer
        for multi-point charting.
        """
        t_base = float(step) * 1.0  # 1-second steps

        # Build 30-sample historical series
        history: Dict[str, Dict[str, List[float]]] = {
            "timestamps": [float(i) for i in range(max(0, step - 30), step + 1)],
            "rpm": {"timestamps": [], "observed": [], "expected": []},
            "cht": {"timestamps": [], "observed": [], "expected": []},
            "egt": {"timestamps": [], "observed": [], "expected": []},
            "oil_temp": {"timestamps": [], "observed": [], "expected": []},
            "oil_pressure": {"timestamps": [], "observed": [], "expected": []},
            "fuel_flow": {"timestamps": [], "observed": [], "expected": []},
            "vibration": {"timestamps": [], "observed": [], "expected": []},
            "health_index": {"timestamps": [], "hi": []},
        }

        ts_list = history["timestamps"]

        if "1. Nominal Healthy Cruise" in scenario_name:
            # Nominal Healthy
            telemetry = {
                "timestamp": t_base,
                "engine_id": "MALE_UAV_ROT912_01",
                "mission_id": "MIS_ISR_PATROL_01",
                "mission_phase": "CRUISE",
                "altitude": 3200.0,
                "ambient_temp": 14.5,
                "throttle": 75.0,
                "load": 70.0,
                "rpm": 5420.0,
                "cht": 98.4,
                "egt": 685.0,
                "oil_temp": 86.2,
                "oil_pressure": 4.2,
                "fuel_flow": 15.8,
                "vibration": 0.52,
                "source": "synthetic_bench_simulator",
                "source_type": "simulated",
                "simulation_version": "v0.9-sih-demo",
            }
            digital_twin = {
                "nominal_estimates": {
                    "expected_rpm": 5400.0,
                    "expected_cht": 98.0,
                    "expected_egt": 680.0,
                    "expected_oil_temp": 85.0,
                    "expected_oil_pressure": 4.2,
                    "expected_fuel_flow": 15.6,
                    "expected_vibration": 0.50,
                },
                "residuals": {
                    "rpm_residual": 20.0,
                    "cht_residual": 0.4,
                    "egt_residual": 5.0,
                    "oil_temp_residual": 1.2,
                    "oil_pressure_residual": 0.0,
                    "fuel_flow_residual": 0.2,
                    "vibration_residual": 0.02,
                },
            }
            health_index = {
                "smoothed_health_index": 0.98,
                "raw_health_index": 0.99,
                "health_state": "HEALTHY",
                "degradation_rate": 0.0000,
                "degradation_trend": "STABLE",
                "excluded_channels": [],
            }
            anomaly = {
                "anomaly_status": "NORMAL",
                "anomaly_score": 0.04,
                "threshold_score": 0.02,
                "ewma_score": 0.03,
                "contributing_channels": [],
            }
            fault_diagnosis = {
                "predicted_fault_type": "none",
                "diagnostic_confidence": 0.98,
                "class_probabilities": {"none": 0.98, "cooling_degradation": 0.01, "lubrication_degradation": 0.01},
            }
            prognostics = {
                "rul_seconds_median": 72000.0,
                "rul_seconds_p05": 68000.0,
                "rul_seconds_p95": 76000.0,
                "status": "NOT_DEGRADING",
                "limiting_factor": "GLOBAL_HEALTH_INDEX",
                "confidence_score": 0.99,
            }
            forecast = {
                "forecast_quality": "VALID",
                "model_name": "TimesFM-3.0",
                "model_status": "BASELINE",
                "forecast_timestamps": [t_base + float(i) for i in range(1, 17)],
                "predicted_telemetry": {
                    "rpm": [5420.0 + (i * 0.5) for i in range(16)],
                    "cht": [98.4 + (i * 0.1) for i in range(16)],
                    "egt": [685.0 for _ in range(16)],
                    "oil_temp": [86.2 for _ in range(16)],
                    "oil_pressure": [4.2 for _ in range(16)],
                    "fuel_flow": [15.8 for _ in range(16)],
                    "vibration": [0.52 for _ in range(16)],
                },
                "lower_bounds": {
                    "cht": [98.4 - 2.0 for _ in range(16)],
                },
                "upper_bounds": {
                    "cht": [98.4 + 2.0 for _ in range(16)],
                },
            }
            explainability = {
                "summary_explanation": "All engine channels are operating within healthy nominal margins. Residuals are strictly within ±1.5σ deadbands.",
                "physics_evidence": {
                    "status": "SUPPORTED",
                    "diagnosed_fault": "none",
                    "consistency_reason": "All thermal and mechanical channels adhere to nominal baseline dynamics.",
                },
                "shap_evidence": {
                    "top_features": [
                        {"feature_name": "oil_pressure", "feature_value": 4.2, "shap_value": 0.02, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.35},
                        {"feature_name": "cht", "feature_value": 98.4, "shap_value": 0.01, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.30},
                    ],
                    "disclaimer": "SHAP feature attribution does not establish causality.",
                },
            }

            # Populate history
            for t in ts_list:
                history["rpm"]["timestamps"].append(t)
                history["rpm"]["observed"].append(5400.0 + (t % 10))
                history["rpm"]["expected"].append(5400.0)

                history["cht"]["timestamps"].append(t)
                history["cht"]["observed"].append(98.0 + (t % 3) * 0.3)
                history["cht"]["expected"].append(98.0)

                history["egt"]["timestamps"].append(t)
                history["egt"]["observed"].append(680.0 + (t % 5))
                history["egt"]["expected"].append(680.0)

                history["oil_temp"]["timestamps"].append(t)
                history["oil_temp"]["observed"].append(85.5)
                history["oil_temp"]["expected"].append(85.0)

                history["oil_pressure"]["timestamps"].append(t)
                history["oil_pressure"]["observed"].append(4.2)
                history["oil_pressure"]["expected"].append(4.2)

                history["fuel_flow"]["timestamps"].append(t)
                history["fuel_flow"]["observed"].append(15.8)
                history["fuel_flow"]["expected"].append(15.6)

                history["vibration"]["timestamps"].append(t)
                history["vibration"]["observed"].append(0.52)
                history["vibration"]["expected"].append(0.50)

                history["health_index"]["timestamps"].append(t)
                history["health_index"]["hi"].append(0.98)

        elif "2. Thermal Degradation" in scenario_name:
            # Severe CHT rise
            telemetry = {
                "timestamp": t_base,
                "engine_id": "MALE_UAV_ROT912_01",
                "mission_id": "MIS_ISR_PATROL_01",
                "mission_phase": "CRUISE",
                "altitude": 3200.0,
                "ambient_temp": 14.5,
                "throttle": 75.0,
                "load": 70.0,
                "rpm": 5380.0,
                "cht": 144.2,  # Approaching 150°C redline!
                "egt": 710.0,
                "oil_temp": 118.5,  # Secondary thermal coupling
                "oil_pressure": 3.9,
                "fuel_flow": 16.1,
                "vibration": 0.60,
                "source": "synthetic_bench_simulator",
                "source_type": "simulated",
                "simulation_version": "v0.9-sih-demo",
            }
            digital_twin = {
                "nominal_estimates": {
                    "expected_rpm": 5400.0,
                    "expected_cht": 98.0,
                    "expected_egt": 680.0,
                    "expected_oil_temp": 85.0,
                    "expected_oil_pressure": 4.2,
                    "expected_fuel_flow": 15.6,
                    "expected_vibration": 0.50,
                },
                "residuals": {
                    "rpm_residual": -20.0,
                    "cht_residual": 46.2,
                    "egt_residual": 30.0,
                    "oil_temp_residual": 33.5,
                    "oil_pressure_residual": -0.3,
                    "fuel_flow_residual": 0.5,
                    "vibration_residual": 0.10,
                },
            }
            health_index = {
                "smoothed_health_index": 0.48,
                "raw_health_index": 0.45,
                "health_state": "DEGRADED",
                "degradation_rate": -0.0035,
                "degradation_trend": "RAPIDLY_DEGRADING",
                "excluded_channels": [],
                "dominant_degraded_channels": ["cht", "oil_temp"],
            }
            anomaly = {
                "anomaly_status": "ANOMALY",
                "anomaly_score": 0.88,
                "threshold_score": 0.92,
                "ewma_score": 0.85,
                "contributing_channels": ["cht", "oil_temp"],
            }
            fault_diagnosis = {
                "predicted_fault_type": "cooling_degradation",
                "diagnostic_confidence": 0.94,
                "class_probabilities": {"cooling_degradation": 0.94, "lubrication_degradation": 0.04, "none": 0.02},
            }
            prognostics = {
                "rul_seconds_median": 1620.0,  # 27 minutes
                "rul_seconds_p05": 1200.0,     # 20 minutes
                "rul_seconds_p95": 2100.0,     # 35 minutes
                "status": "ACTIVE_DEGRADATION",
                "limiting_factor": "REDLINE_CHT",
                "confidence_score": 0.86,
            }
            forecast = {
                "forecast_quality": "VALID",
                "model_name": "TimesFM-3.0",
                "model_status": "BASELINE",
                "forecast_timestamps": [t_base + float(i) for i in range(1, 17)],
                "predicted_telemetry": {
                    "rpm": [5380.0 for _ in range(16)],
                    "cht": [144.2 + (i * 0.8) for i in range(16)],  # Will cross 150 C
                    "egt": [710.0 for _ in range(16)],
                    "oil_temp": [118.5 + (i * 0.4) for i in range(16)],
                    "oil_pressure": [3.9 for _ in range(16)],
                    "fuel_flow": [16.1 for _ in range(16)],
                    "vibration": [0.60 for _ in range(16)],
                },
                "lower_bounds": {
                    "cht": [144.2 + (i * 0.5) for i in range(16)],
                },
                "upper_bounds": {
                    "cht": [144.2 + (i * 1.1) for i in range(16)],
                },
            }
            explainability = {
                "summary_explanation": "Severe thermal escalation detected on Cylinder Head Temperature (+46.2°C residual) with coupled secondary oil temperature rise (+33.5°C residual). Strong physical support for cooling system conductance failure.",
                "physics_evidence": {
                    "status": "SUPPORTED",
                    "diagnosed_fault": "cooling_degradation",
                    "consistency_reason": "Primary thermal elevation on CHT followed by coupled conductive heating of lubrication oil.",
                },
                "shap_evidence": {
                    "top_features": [
                        {"feature_name": "cht", "feature_value": 144.2, "shap_value": 0.62, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.55},
                        {"feature_name": "oil_temp", "feature_value": 118.5, "shap_value": 0.38, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.35},
                    ],
                    "disclaimer": "SHAP feature attribution does not establish causality.",
                },
            }

            for idx, t in enumerate(ts_list):
                prog_factor = float(idx) / max(1.0, float(len(ts_list)))
                history["cht"]["timestamps"].append(t)
                history["cht"]["observed"].append(98.0 + (46.0 * prog_factor))
                history["cht"]["expected"].append(98.0)

                history["rpm"]["timestamps"].append(t)
                history["rpm"]["observed"].append(5400.0 - (20.0 * prog_factor))
                history["rpm"]["expected"].append(5400.0)

                history["egt"]["timestamps"].append(t)
                history["egt"]["observed"].append(680.0 + (30.0 * prog_factor))
                history["egt"]["expected"].append(680.0)

                history["oil_temp"]["timestamps"].append(t)
                history["oil_temp"]["observed"].append(85.0 + (33.0 * prog_factor))
                history["oil_temp"]["expected"].append(85.0)

                history["oil_pressure"]["timestamps"].append(t)
                history["oil_pressure"]["observed"].append(4.2 - (0.3 * prog_factor))
                history["oil_pressure"]["expected"].append(4.2)

                history["fuel_flow"]["timestamps"].append(t)
                history["fuel_flow"]["observed"].append(15.8 + (0.3 * prog_factor))
                history["fuel_flow"]["expected"].append(15.6)

                history["vibration"]["timestamps"].append(t)
                history["vibration"]["observed"].append(0.52 + (0.08 * prog_factor))
                history["vibration"]["expected"].append(0.50)

                history["health_index"]["timestamps"].append(t)
                history["health_index"]["hi"].append(0.98 - (0.50 * prog_factor))

        elif "3. Lubrication Pressure Loss" in scenario_name:
            # Low oil pressure
            telemetry = {
                "timestamp": t_base,
                "engine_id": "MALE_UAV_ROT912_01",
                "mission_id": "MIS_ISR_PATROL_01",
                "mission_phase": "CRUISE",
                "altitude": 3200.0,
                "ambient_temp": 14.5,
                "throttle": 75.0,
                "load": 70.0,
                "rpm": 5350.0,
                "cht": 108.0,
                "egt": 690.0,
                "oil_temp": 126.0,  # Friction heating
                "oil_pressure": 1.35,  # Close to 1.2 bar redline!
                "fuel_flow": 16.0,
                "vibration": 0.85,  # Bearing vibration
                "source": "synthetic_bench_simulator",
                "source_type": "simulated",
                "simulation_version": "v0.9-sih-demo",
            }
            digital_twin = {
                "nominal_estimates": {
                    "expected_rpm": 5400.0,
                    "expected_cht": 98.0,
                    "expected_egt": 680.0,
                    "expected_oil_temp": 85.0,
                    "expected_oil_pressure": 4.2,
                    "expected_fuel_flow": 15.6,
                    "expected_vibration": 0.50,
                },
                "residuals": {
                    "oil_pressure_residual": -2.85,
                    "oil_temp_residual": 41.0,
                    "vibration_residual": 0.35,
                },
            }
            health_index = {
                "smoothed_health_index": 0.38,
                "raw_health_index": 0.36,
                "health_state": "SEVERELY_DEGRADED",
                "degradation_rate": -0.0050,
                "degradation_trend": "RAPIDLY_DEGRADING",
                "excluded_channels": [],
                "dominant_degraded_channels": ["oil_pressure", "oil_temp", "vibration"],
            }
            anomaly = {
                "anomaly_status": "ANOMALY",
                "anomaly_score": 0.95,
                "threshold_score": 0.98,
                "ewma_score": 0.92,
                "contributing_channels": ["oil_pressure", "oil_temp"],
            }
            fault_diagnosis = {
                "predicted_fault_type": "lubrication_degradation",
                "diagnostic_confidence": 0.96,
                "class_probabilities": {"lubrication_degradation": 0.96, "mechanical_degradation": 0.03, "none": 0.01},
            }
            prognostics = {
                "rul_seconds_median": 720.0,  # 12 minutes
                "rul_seconds_p05": 480.0,    # 8 minutes
                "rul_seconds_p95": 960.0,    # 16 minutes
                "status": "ACTIVE_DEGRADATION",
                "limiting_factor": "REDLINE_OIL_PRESSURE",
                "confidence_score": 0.90,
            }
            forecast = {
                "forecast_quality": "VALID",
                "model_name": "TimesFM-3.0",
                "model_status": "BASELINE",
                "forecast_timestamps": [t_base + float(i) for i in range(1, 17)],
                "predicted_telemetry": {
                    "rpm": [5350.0 for _ in range(16)],
                    "cht": [108.0 for _ in range(16)],
                    "egt": [690.0 for _ in range(16)],
                    "oil_temp": [126.0 + (i * 0.5) for i in range(16)],
                    "oil_pressure": [1.35 - (i * 0.03) for i in range(16)],  # Will breach 1.2 bar
                    "fuel_flow": [16.0 for _ in range(16)],
                    "vibration": [0.85 for _ in range(16)],
                },
            }
            explainability = {
                "summary_explanation": "Critical oil pressure drop (-2.85 bar residual) combined with friction heating (+41.0°C oil temp residual). Rapid boundary lubrication failure predicted.",
                "physics_evidence": {
                    "status": "SUPPORTED",
                    "diagnosed_fault": "lubrication_degradation",
                    "consistency_reason": "Direct hydrodynamic delivery collapse with coupled boundary friction heat generation.",
                },
                "shap_evidence": {
                    "top_features": [
                        {"feature_name": "oil_pressure", "feature_value": 1.35, "shap_value": 0.72, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.60},
                        {"feature_name": "oil_temp", "feature_value": 126.0, "shap_value": 0.40, "direction": "TOWARD_PREDICTED_CLASS", "relative_weight": 0.35},
                    ],
                    "disclaimer": "SHAP feature attribution does not establish causality.",
                },
            }

            for idx, t in enumerate(ts_list):
                prog_factor = float(idx) / max(1.0, float(len(ts_list)))
                for ch in ["rpm", "cht", "egt", "fuel_flow"]:
                    history[ch]["timestamps"].append(t)
                    history[ch]["observed"].append(telemetry[ch])
                    history[ch]["expected"].append(digital_twin["nominal_estimates"][f"expected_{ch}"])

                history["oil_pressure"]["timestamps"].append(t)
                history["oil_pressure"]["observed"].append(4.2 - (2.85 * prog_factor))
                history["oil_pressure"]["expected"].append(4.2)

                history["oil_temp"]["timestamps"].append(t)
                history["oil_temp"]["observed"].append(85.0 + (41.0 * prog_factor))
                history["oil_temp"]["expected"].append(85.0)

                history["vibration"]["timestamps"].append(t)
                history["vibration"]["observed"].append(0.50 + (0.35 * prog_factor))
                history["vibration"]["expected"].append(0.50)

                history["health_index"]["timestamps"].append(t)
                history["health_index"]["hi"].append(0.98 - (0.60 * prog_factor))

        else:
            # 4. Sensor Dropout & Isolation
            telemetry = {
                "timestamp": t_base,
                "engine_id": "MALE_UAV_ROT912_01",
                "mission_id": "MIS_ISR_PATROL_01",
                "mission_phase": "CRUISE",
                "altitude": 3200.0,
                "ambient_temp": 14.5,
                "throttle": 75.0,
                "load": 70.0,
                "rpm": 5410.0,
                "cht": float("nan"),  # Sensor Dropout!
                "egt": 682.0,
                "oil_temp": 85.5,
                "oil_pressure": 4.15,
                "fuel_flow": 15.7,
                "vibration": 0.51,
                "source": "synthetic_bench_simulator",
                "source_type": "simulated",
                "simulation_version": "v0.9-sih-demo",
            }
            digital_twin = {
                "nominal_estimates": {
                    "expected_rpm": 5400.0,
                    "expected_cht": 98.0,
                    "expected_egt": 680.0,
                    "expected_oil_temp": 85.0,
                    "expected_oil_pressure": 4.2,
                    "expected_fuel_flow": 15.6,
                    "expected_vibration": 0.50,
                },
                "residuals": {
                    "cht_residual": float("nan"),
                    "oil_temp_residual": 0.5,
                },
            }
            health_index = {
                "smoothed_health_index": 0.91,
                "raw_health_index": 0.91,
                "health_state": "HEALTHY",
                "degradation_rate": 0.0000,
                "degradation_trend": "STABLE",
                "excluded_channels": ["cht"],  # Isolated!
            }
            anomaly = {
                "anomaly_status": "WARNING",
                "anomaly_score": 0.35,
                "threshold_score": 0.40,
                "contributing_channels": ["cht"],
            }
            fault_diagnosis = {
                "predicted_fault_type": "sensor_fault",
                "diagnostic_confidence": 0.89,
                "class_probabilities": {"sensor_fault": 0.89, "cooling_degradation": 0.08, "none": 0.03},
            }
            prognostics = {
                "rul_seconds_median": 72000.0,
                "status": "DEGRADED_PROGNOSTIC",
                "limiting_factor": "GLOBAL_HEALTH_INDEX",
                "confidence_score": 0.75,  # Penalized by sensor isolation
            }
            forecast = {
                "forecast_quality": "DEGRADED_INPUT",
                "model_name": "TimesFM-3.0",
                "model_status": "BASELINE",
                "forecast_timestamps": [t_base + float(i) for i in range(1, 17)],
                "predicted_telemetry": {
                    "rpm": [5410.0 for _ in range(16)],
                    "cht": [],  # Unavailable due to sensor isolation
                    "egt": [682.0 for _ in range(16)],
                    "oil_temp": [85.5 for _ in range(16)],
                    "oil_pressure": [4.15 for _ in range(16)],
                    "fuel_flow": [15.7 for _ in range(16)],
                    "vibration": [0.51 for _ in range(16)],
                },
            }
            explainability = {
                "summary_explanation": "Cylinder Head Temperature channel has dropped out (NaN) and been isolated by the Health Monitor. Thermally coupled oil temperature remains nominal, confirming sensor instrumentation failure without physical engine damage.",
                "physics_evidence": {

                    "status": "SUPPORTED",
                    "diagnosed_fault": "sensor_fault",
                    "consistency_reason": "Isolated thermal anomaly with zero cross-channel propagation to oil temperature.",
                },
                "shap_evidence": {
                    "top_features": [],
                    "disclaimer": "SHAP feature attribution does not establish causality.",
                },
            }

            for t in ts_list:
                for ch in ["rpm", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
                    history[ch]["timestamps"].append(t)
                    history[ch]["observed"].append(telemetry[ch])
                    history[ch]["expected"].append(digital_twin["nominal_estimates"][f"expected_{ch}"])

                # CHT is NaN
                history["cht"]["timestamps"].append(t)
                history["cht"]["observed"].append(float("nan"))
                history["cht"]["expected"].append(98.0)

                history["health_index"]["timestamps"].append(t)
                history["health_index"]["hi"].append(0.91)

        # Data quality report
        data_quality = {
            "quality_score": 0.88 if "Sensor Dropout" in scenario_name else 0.99,
            "is_regular_sampling": True,
            "sampling_interval_mean": 0.1,
            "issues_detected": ["Channel CHT sensor dropout detected (NaN)"] if "Sensor Dropout" in scenario_name else [],
            "channel_summaries": {
                "rpm": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
                "cht": {"missing_count": 1 if "Sensor Dropout" in scenario_name else 0, "invalid_count": 0, "warning_count": 0},
                "egt": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
                "oil_temp": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
                "oil_pressure": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
                "fuel_flow": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
                "vibration": {"missing_count": 0, "invalid_count": 0, "warning_count": 0},
            },
        }

        contract = Phase13OutputContract(
            timestamp=t_base,
            engine_id="MALE_UAV_ROT912_01",
            mission_id="MIS_ISR_PATROL_01",
            execution_status="COMPLETED",
            is_synthetic_demo=True,
            telemetry=telemetry,
            data_quality=data_quality,
            digital_twin=digital_twin,
            anomaly=anomaly,
            fault_diagnosis=fault_diagnosis,
            health_index=health_index,
            forecast=forecast,
            prognostics=prognostics,
            explainability=explainability,
        )

        return contract, history
