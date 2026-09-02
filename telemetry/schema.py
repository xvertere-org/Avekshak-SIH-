"""
Core data schemas and interfaces for SIH26054 Aero Piston Engine Digital Twin.

All engineering units and provenance properties are strictly typed and documented.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional


class MissionPhase(str, Enum):
    """Standard flight mission phases for MALE UAV operations."""
    PREFLIGHT = "PREFLIGHT"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    LOITER = "LOITER"
    DESCENT = "DESCENT"
    LANDING = "LANDING"
    POSTFLIGHT = "POSTFLIGHT"


class FaultCategory(str, Enum):
    """
    Standard fault categories for aero piston engine simulation and diagnostics.
    Values can be extended without breaking schema contracts.
    """
    NONE = "none"
    COOLING_DEGRADATION = "cooling_degradation"
    INJECTOR_FUEL_ABNORMALITY = "injector_fuel_abnormality"
    LUBRICATION_ISSUE = "lubrication_issue"
    MECHANICAL_VIBRATION_FAULT = "mechanical_vibration_fault"
    SENSOR_DRIFT_FAILURE = "sensor_drift_failure"


@dataclass
class EngineConfig:
    """
    Configuration structure for engine simulation parameters.

    NOTE: Field values in templates are non-authoritative grey-box parameters
    and do not represent certified OEM specifications.
    """
    engine_id: str = "ENGINE_UAV_01"
    model_template_name: str = "GENERIC_MALE_UAV_PISTON_4CYL"
    displacement_cc: Optional[float] = None
    compression_ratio: Optional[float] = None
    max_rpm: Optional[float] = None
    rated_power_hp: Optional[float] = None
    cooling_type: Optional[str] = "air_liquid_hybrid"
    num_cylinders: Optional[int] = 4
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        return cls(**data)


@dataclass
class MissionConfig:
    """
    Flight mission configuration schema for engine simulation profiles.
    """
    mission_id: str
    engine_id: str = "ENGINE_UAV_01"
    altitude: float = 3000.0                # Altitude in meters (m)
    ambient_temperature: float = 15.0       # Ambient temperature in Celsius (°C)
    duration: float = 3600.0                # Total mission duration in seconds (s)
    throttle: float = 75.0                  # Throttle position in percent (0.0 to 100.0 %)
    engine_load: float = 70.0               # Engine load in percent (0.0 to 100.0 %)
    mission_phase: str = MissionPhase.CRUISE.value
    fault_type: str = FaultCategory.NONE.value
    fault_severity: float = 0.0             # Fault severity scale (0.0: healthy, 1.0: maximum degradation)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionConfig":
        return cls(**data)


@dataclass
class TelemetryRecord:
    """
    Standard Telemetry Record for Aero Piston Engines in MALE UAVs.

    Documented Units:
    - timestamp: Seconds since epoch or relative mission elapsed time (s)
    - altitude: Pressure/GPS altitude in meters (m)
    - ambient_temp: Outside Air Temperature (OAT) in Celsius (°C)
    - throttle: Commanded throttle position (0 - 100 %)
    - load: Engine mechanical load factor (0 - 100 %)
    - rpm: Engine crankshaft rotation speed (RPM)
    - cht: Cylinder Head Temperature in Celsius (°C)
    - egt: Exhaust Gas Temperature in Celsius (°C)
    - oil_temp: Engine lubrication oil temperature in Celsius (°C)
    - oil_pressure: Engine lubrication oil pressure in bar (bar)
    - fuel_flow: Fuel consumption flow rate in liters per hour (L/h)
    - vibration: Engine mount vibration amplitude in acceleration units (g)
    - fault_type: Active fault category or 'none'
    - fault_severity: Severity scaling factor (0.0 to 1.0)
    - source: Data generator / feed identifier (e.g. 'simulator_v1', 'bench_dataset')
    - source_type: Data origin category ('simulated', 'synthetic', 'test_bench', 'flight_test')
    - simulation_version: Codebase / model version tag for provenance tracking
    """
    timestamp: float
    mission_id: str
    engine_id: str
    mission_phase: str
    altitude: float                         # meters (m)
    ambient_temp: float                     # Celsius (°C)
    throttle: float                         # percent (%)
    load: float                             # percent (%)
    rpm: float                              # Revolutions Per Minute (RPM)
    cht: float                              # Cylinder Head Temperature (°C)
    egt: float                              # Exhaust Gas Temperature (°C)
    oil_temp: float                         # Oil Temperature (°C)
    oil_pressure: float                     # Oil Pressure (bar)
    fuel_flow: float                        # Fuel Flow (L/h)
    vibration: float                        # Vibration (g)
    fault_type: str = FaultCategory.NONE.value
    fault_severity: float = 0.0
    source: str = "simulator_v1"
    source_type: str = "simulated"
    simulation_version: str = "0.1.0-phase1-stub"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryRecord":
        return cls(**data)


@dataclass
class DigitalTwinState:
    """
    State estimation and residual tracking emitted by the Digital Twin core.
    """
    timestamp: float
    engine_id: str
    observed_telemetry: TelemetryRecord
    nominal_estimates: Dict[str, float] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)
    state_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAssessment:
    """
    Health monitoring and fault diagnosis output from the PHM module.
    """
    timestamp: float
    engine_id: str
    health_index: float                     # 0.0 (failed) to 1.0 (healthy)
    anomaly_detected: bool = False
    anomaly_score: float = 0.0              # 0.0 to 1.0
    fault_category: str = FaultCategory.NONE.value
    fault_confidence: float = 0.0           # 0.0 to 1.0
    active_warnings: list = field(default_factory=list)


@dataclass
class RULPrediction:
    """
    Prognostic Remaining Useful Life (RUL) and trajectory forecast.
    """
    timestamp: float
    engine_id: str
    estimated_rul_hours: float
    confidence_lower_hours: float
    confidence_upper_hours: float
    forecast_trajectory: Dict[str, list] = field(default_factory=dict)
    prediction_confidence: float = 1.0


@dataclass
class ExplanationReport:
    """
    Explainable AI (XAI) feature importance and operational summary.
    """
    timestamp: float
    engine_id: str
    top_contributing_features: Dict[str, float] = field(default_factory=dict)
    shap_summary: Dict[str, Any] = field(default_factory=dict)
    explanation_text: str = ""
