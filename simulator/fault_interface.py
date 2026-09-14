"""
Fault and Degradation State Interface for SIH26054 Aero Piston Engine Simulator.

Defines typed schemas and scheduling contracts for Phase 4 fault injection:
- FaultType enum
- FaultSubsystem enum
- FaultState data model
- FaultSchedule timeline manager

DISCLAIMER:
Phase 4A defines the fault and degradation contract only.
No physical fault dynamics are implemented in this stage; healthy physical
trajectories remain strictly unchanged when no fault is active.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum, EnumType
from typing import Dict, Any, List, Optional, Union


class FaultTypeMeta(EnumType):
    """Metaclass ensuring exactly 6 canonical classes are returned by iter/len for Phase 8-13 compatibility."""
    def __len__(cls):
        return 6

    def __iter__(cls):
        canonical = [
            cls.NONE,
            cls.COOLING_DEGRADATION,
            cls.LUBRICATION_DEGRADATION,
            cls.FUEL_INJECTION_ABNORMALITY,
            cls.MECHANICAL_DEGRADATION,
            cls.SENSOR_FAULT,
        ]
        return iter(canonical)

    def __contains__(cls, member):
        return super().__contains__(member)


class FaultType(str, Enum, metaclass=FaultTypeMeta):
    """
    Standard fault and degradation categories for aero piston engine simulation.
    Supports canonical Phase 4 names and backward-compatible aliases.
    """
    NONE = "none"
    # F1: Injector / Fuel-Delivery Abnormality
    INJECTOR_DELIVERY_ABNORMALITY = "injector_delivery_abnormality"
    FUEL_INJECTION_ABNORMALITY = "fuel_injection_abnormality"  # Compatibility alias
    # F2: Lubrication Degradation
    LUBRICATION_DEGRADATION = "lubrication_degradation"
    # F3: Cooling Degradation
    COOLING_DEGRADATION = "cooling_degradation"
    # F4: Combustion / Misfire Instability
    COMBUSTION_MISFIRE = "combustion_misfire"
    COMBUSTION_INSTABILITY = "combustion_instability"          # Compatibility alias
    # F5: Mechanical / Vibration Degradation
    MECHANICAL_DEGRADATION = "mechanical_degradation"
    # F6 & F7: Sensor Faults (Observation Layer Only)
    SENSOR_FAULT = "sensor_fault"                              # Compatibility general
    SENSOR_BIAS = "sensor_bias"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_STUCK = "sensor_stuck"


class FaultSubsystem(str, Enum):
    """
    Engine physical or data subsystems targeted by fault injection.
    """
    NONE = "none"
    THERMAL = "thermal"
    COOLING = "cooling"
    LUBRICATION = "lubrication"
    FUEL = "fuel"
    DYNAMICS = "dynamics"
    VIBRATION = "vibration"
    SENSOR = "sensor"
    COMBUSTION = "combustion"


class FuelMixtureMode(str, Enum):
    """
    Mixture abnormality modes for FUEL_INJECTION_ABNORMALITY faults.
    """
    LEAN = "lean"
    RICH = "rich"


@dataclass
class FaultState:
    """
    Structured representation of an active or scheduled fault/degradation state.

    Attributes:
        fault_type: Category of the fault (FaultType enum or string).
        severity: Normalized fault intensity in [0.0, 1.0]
                  (0.0: nominal/healthy, 1.0: maximum modeled degradation level).
        active: Boolean flag indicating if the fault is enabled.
        start_time: Simulation timestamp (seconds) at which the fault activates.
        end_time: Simulation timestamp (seconds) at which the fault deactivates (None = indefinite).
        affected_subsystem: Target subsystem (FaultSubsystem enum or string).
        affected_cylinder: Optional target cylinder index (1, 2, 3, 4, or None for all cylinders).
        fault_id: Unique string identifier for tracking.
        mechanism_description: Physical or observation mechanism description.
        parameter_changes: Dictionary of calibrated parameter modifications.
        expected_signatures: List of expected physical/residual signatures.
        parameters: Additional fault-specific parameters (e.g. ramp_duration, mode, target_sensor).
    """
    fault_type: Union[FaultType, str] = FaultType.NONE
    severity: float = 0.0
    active: bool = True
    start_time: float = 0.0
    end_time: Optional[float] = None
    affected_subsystem: Union[FaultSubsystem, str] = FaultSubsystem.NONE
    affected_cylinder: Optional[int] = None
    fault_id: str = ""
    mechanism_description: str = ""
    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    expected_signatures: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate and convert fault_type (handling aliases gracefully)
        if isinstance(self.fault_type, FaultType):
            pass
        elif isinstance(self.fault_type, str):
            ft_str = self.fault_type.lower().strip()
            # Map canonical / aliases
            matched = False
            for name, ft in FaultType.__members__.items():
                if name.lower() == ft_str or ft.value == ft_str:
                    self.fault_type = ft
                    matched = True
                    break
            if not matched:
                valid_types = [t.value for t in FaultType]
                raise ValueError(
                    f"Invalid fault_type '{self.fault_type}'. Supported types: {valid_types}"
                )
        else:
            raise ValueError(f"Invalid fault_type '{self.fault_type}'. Expected FaultType or str.")

        # Validate and convert affected_subsystem
        if isinstance(self.affected_subsystem, str):
            sub_str = self.affected_subsystem.lower().strip()
            matched_sub = False
            for s in FaultSubsystem:
                if s.value == sub_str:
                    self.affected_subsystem = s
                    matched_sub = True
                    break
            if not matched_sub:
                valid_subsystems = [s.value for s in FaultSubsystem]
                raise ValueError(
                    f"Invalid affected_subsystem '{self.affected_subsystem}'. Supported subsystems: {valid_subsystems}"
                )

        # Validate severity bounds [0.0, 1.0]
        if not (0.0 <= float(self.severity) <= 1.0):
            raise ValueError(
                f"Fault severity must be normalized in range [0.0, 1.0], got {self.severity}"
            )
        self.severity = float(self.severity)

        # Validate affected_cylinder in {None, 1, 2, 3, 4}
        if self.affected_cylinder is not None:
            if not isinstance(self.affected_cylinder, int) or self.affected_cylinder not in (1, 2, 3, 4):
                raise ValueError(
                    f"affected_cylinder must be None or an integer in {{1, 2, 3, 4}}, got {self.affected_cylinder}"
                )

        # Validate timing
        self.start_time = float(self.start_time)
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")

        if self.end_time is not None:
            self.end_time = float(self.end_time)
            if self.end_time < self.start_time:
                raise ValueError(
                    f"end_time ({self.end_time}) cannot be earlier than start_time ({self.start_time})"
                )

        # Auto-generate fault_id if empty
        if not self.fault_id:
            ft_val = self.fault_type.value if hasattr(self.fault_type, "value") else str(self.fault_type)
            cyl_tag = f"_cyl{self.affected_cylinder}" if self.affected_cylinder is not None else ""
            self.fault_id = f"FAULT_{ft_val.upper()}{cyl_tag}_{int(self.start_time)}"

    def is_active_at(self, timestamp: float) -> bool:
        """
        Check whether this fault is active at the given simulation timestamp.
        """
        if not self.active or self.fault_type == FaultType.NONE or self.severity <= 0.0:
            return False

        t = float(timestamp)
        if t < self.start_time:
            return False
        if self.end_time is not None and t > self.end_time:
            return False
        return True

    def get_effective_severity(self, timestamp: float) -> float:
        """
        Compute the effective severity at timestamp t, taking into account
        optional ramp-up durations and deterministic periodic/intermittent modulations.
        """
        if not self.is_active_at(timestamp):
            return 0.0

        t = float(timestamp)
        ramp_duration = float(self.parameters.get("ramp_duration", 0.0))

        if ramp_duration > 0.0:
            elapsed = t - self.start_time
            fraction = min(1.0, max(0.0, elapsed / ramp_duration))
            base_sev = self.severity * fraction
        else:
            base_sev = self.severity

        # Deterministic intermittent / periodic modulation (zero random draws)
        if self.parameters.get("intermittent", False) or "period" in self.parameters:
            period = float(self.parameters.get("period", 2.0))
            duty_cycle = float(self.parameters.get("duty_cycle", 0.5))
            if period > 0.0:
                elapsed = t - self.start_time
                if (elapsed % period) >= (duty_cycle * period):
                    return 0.0

        return base_sev

    def to_dict(self) -> Dict[str, Any]:
        """Convert FaultState to dictionary representation."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value if hasattr(self.fault_type, "value") else str(self.fault_type),
            "severity": self.severity,
            "active": self.active,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "affected_subsystem": self.affected_subsystem.value if hasattr(self.affected_subsystem, "value") else str(self.affected_subsystem),
            "affected_cylinder": self.affected_cylinder,
            "mechanism_description": self.mechanism_description,
            "parameter_changes": dict(self.parameter_changes),
            "expected_signatures": list(self.expected_signatures),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultState":
        """Reconstruct FaultState from dictionary."""
        return cls(**data)


# ────────────────────────────────────────────────────────────────────────────
# Authoritative Fault Signature Catalog (Contract for Detection & Diagnosis)
# ────────────────────────────────────────────────────────────────────────────

FAULT_SIGNATURE_CATALOG: Dict[str, Dict[str, Any]] = {
    "F1_INJECTOR_DELIVERY_ABNORMALITY": {
        "fault_type": FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        "class_id": "F1",
        "name": "Injector / Fuel-Delivery Abnormality",
        "affected_subsystem": FaultSubsystem.FUEL,
        "layer": "physical",
        "primary_mechanism": "Effective fuel delivery reduction m_fuel_eff,i = m_fuel_nominal,i * (1 - s * d_fuel)",
        "calibration_parameter": "d_fuel = 0.35",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Reduced chemical power generation",
            "Localized cylinder combustion heat generation drop",
            "Localized EGT lean/starvation shift",
        ],
        "expected_telemetry_signatures": [
            "fuel_flow drop",
            "indicated power drop",
            "local cht_cyl drop/shift",
            "egt shift",
        ],
        "expected_residuals": [
            "negative fuel_flow residual",
            "negative cht residual",
            "egt residual",
        ],
        "temporal_character": "step, ramp, or deterministic intermittent",
        "cylinder_localization_supported": True,
    },
    "F2_LUBRICATION_DEGRADATION": {
        "fault_type": FaultType.LUBRICATION_DEGRADATION,
        "class_id": "F2",
        "name": "Lubrication Degradation",
        "affected_subsystem": FaultSubsystem.LUBRICATION,
        "layer": "physical",
        "primary_mechanism": (
            "Hydraulic oil pressure loss P_oil = P_nominal * (1 - s * d_p_loss), "
            "friction torque increase T_fric = T_fric,nom * (1 + s * d_fric), "
            "oil frictional heat generation increase, cooler heat rejection loss"
        ),
        "calibration_parameter": "d_p_loss = 0.55, d_fric = 0.08, d_heat = 0.20, d_cool = 0.20",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Oil temperature elevation",
            "Subtle engine speed deceleration under fixed throttle",
            "Secondary thermal conduction to CHT",
        ],
        "expected_telemetry_signatures": [
            "oil_pressure drop",
            "oil_temp elevation",
            "subtle rpm drop",
        ],
        "expected_residuals": [
            "negative oil_pressure residual",
            "positive oil_temp residual",
        ],
        "temporal_character": "step or ramp",
        "cylinder_localization_supported": False,
    },
    "F3_COOLING_DEGRADATION": {
        "fault_type": FaultType.COOLING_DEGRADATION,
        "class_id": "F3",
        "name": "Cooling Degradation",
        "affected_subsystem": FaultSubsystem.COOLING,
        "layer": "physical",
        "primary_mechanism": (
            "Radiator convection loss h_rad = h_rad_nom * (1 - s * d_rad), "
            "cylinder air convection loss h_cool = h_cool_nom * (1 - s * d_cool)"
        ),
        "calibration_parameter": "d_rad = 0.55, d_cool = 0.55",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Coolant temperature rise",
            "Cylinder head temperature elevation across heads",
            "Secondary thermal coupling to oil temp",
        ],
        "expected_telemetry_signatures": [
            "coolant_temp rise",
            "cht rise",
            "cht_cyl rise",
            "oil_temp mild rise",
        ],
        "expected_residuals": [
            "positive cht residual",
            "positive coolant_temp residual",
        ],
        "temporal_character": "step or ramp",
        "cylinder_localization_supported": True,
    },
    "F4_COMBUSTION_MISFIRE": {
        "fault_type": FaultType.COMBUSTION_MISFIRE,
        "class_id": "F4",
        "name": "Combustion / Misfire Instability",
        "affected_subsystem": FaultSubsystem.COMBUSTION,
        "layer": "physical",
        "primary_mechanism": (
            "Combustion efficiency degradation eta_comb = eta_comb_nom * (1 - s * d_comb), "
            "indicated power loss, torque pulsation ripple"
        ),
        "calibration_parameter": "d_comb = 0.70",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Indicated power loss",
            "Torque ripple causing 1X vibration surge",
            "Cylinder head heat generation drop",
            "EGT drop from unburned charge",
        ],
        "expected_telemetry_signatures": [
            "indicated power drop",
            "local cht_cyl drop",
            "local egt_cyl drop",
            "vibration RMS increase",
        ],
        "expected_residuals": [
            "negative cht residual",
            "negative egt residual",
            "positive vibration residual",
        ],
        "temporal_character": "step, ramp, or deterministic intermittent pulse train",
        "cylinder_localization_supported": True,
    },
    "F5_MECHANICAL_DEGRADATION": {
        "fault_type": FaultType.MECHANICAL_DEGRADATION,
        "class_id": "F5",
        "name": "Mechanical / Vibration Degradation",
        "affected_subsystem": FaultSubsystem.VIBRATION,
        "layer": "physical",
        "primary_mechanism": (
            "Rotational harmonic order amplification A_1X = A_1X_nom * (1 + s * 1.8), "
            "A_2X = A_2X_nom * (1 + s * 1.8), broadband noise scaling, friction torque increase"
        ),
        "calibration_parameter": "d_vib_1x = 1.8, d_vib_2x = 1.8, d_noise = 2.5, d_fric = 0.08",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Broadband vibration noise increase",
            "Slight friction torque increase",
        ],
        "expected_telemetry_signatures": [
            "vibration RMS elevation",
            "order_1x amplitude elevation",
            "order_2x amplitude elevation",
        ],
        "expected_residuals": [
            "positive vibration residual",
        ],
        "temporal_character": "step or ramp",
        "cylinder_localization_supported": False,
    },
    "F6_SENSOR_BIAS_DRIFT": {
        "fault_type": FaultType.SENSOR_BIAS,
        "class_id": "F6",
        "name": "Sensor Drift / Bias",
        "affected_subsystem": FaultSubsystem.SENSOR,
        "layer": "observation",
        "primary_mechanism": (
            "Observation-layer corruption: constant offset y_obs = y_true + s * b_max "
            "or linear drift y_obs(t) = y_true(t) + s * r_drift * dt. ZERO physical state change."
        ),
        "calibration_parameter": "Per-channel bias/drift limits in config",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Zero physical effect on engine dynamics, thermodynamics, or true states",
        ],
        "expected_telemetry_signatures": [
            "Target channel offset or ramping away from nominal",
        ],
        "expected_residuals": [
            "Unilateral persistent residual on target channel only",
        ],
        "temporal_character": "step (bias) or linear ramp (drift)",
        "cylinder_localization_supported": False,
    },
    "F7_SENSOR_DROPOUT_STUCK": {
        "fault_type": FaultType.SENSOR_DROPOUT,
        "class_id": "F7",
        "name": "Sensor Failure / Dropout / Stuck",
        "affected_subsystem": FaultSubsystem.SENSOR,
        "layer": "observation",
        "primary_mechanism": (
            "Observation-layer failure: complete channel dropout (y_obs = NaN) "
            "or latch at fault onset value (y_obs(t) = y_obs(t_start)). ZERO physical state change."
        ),
        "calibration_parameter": "Observation layer mapping",
        "provenance": "MODEL_CALIBRATION",
        "secondary_effects": [
            "Zero physical effect on engine dynamics, thermodynamics, or true states",
        ],
        "expected_telemetry_signatures": [
            "Target channel NaN / missing or flatlined unchanged across flight transients",
        ],
        "expected_residuals": [
            "NaN residual or discrepancy during flight maneuvers",
        ],
        "temporal_character": "step latch or dropout",
        "cylinder_localization_supported": False,
    },
}

# Provide direct lowercase canonical lookups
for _cat_key in list(FAULT_SIGNATURE_CATALOG.keys()):
    _entry = FAULT_SIGNATURE_CATALOG[_cat_key]
    _ft = _entry.get("fault_type")
    if _ft is not None:
        _ft_val = _ft.value if hasattr(_ft, "value") else str(_ft)
        FAULT_SIGNATURE_CATALOG[_ft_val] = _entry
FAULT_SIGNATURE_CATALOG["sensor_bias"] = FAULT_SIGNATURE_CATALOG.get("F6_SENSOR_BIAS_DRIFT", {})
FAULT_SIGNATURE_CATALOG["sensor_drift"] = FAULT_SIGNATURE_CATALOG.get("F6_SENSOR_BIAS_DRIFT", {})
FAULT_SIGNATURE_CATALOG["sensor_dropout"] = FAULT_SIGNATURE_CATALOG.get("F7_SENSOR_DROPOUT_STUCK", {})
FAULT_SIGNATURE_CATALOG["sensor_stuck"] = FAULT_SIGNATURE_CATALOG.get("F7_SENSOR_DROPOUT_STUCK", {})


class FaultSchedule:
    """
    Timeline manager for scheduling and querying multiple engine faults.
    """

    def __init__(self, faults: Optional[List[FaultState]] = None):
        self._faults: List[FaultState] = []
        if faults:
            for f in faults:
                self.add_fault(f)

    def add_fault(self, fault: FaultState) -> None:
        """Add a fault state to the schedule."""
        if not isinstance(fault, FaultState):
            raise TypeError(f"Expected FaultState instance, got {type(fault)}")
        self._faults.append(fault)

    def get_active_faults(self, timestamp: float) -> List[FaultState]:
        """Return all faults that are currently active at the given timestamp."""
        return [f for f in self._faults if f.is_active_at(timestamp)]

    def get_primary_fault(self, timestamp: float) -> Optional[FaultState]:
        """
        Return the primary active fault with highest effective severity at timestamp.
        Returns None if no fault is active.
        """
        active = self.get_active_faults(timestamp)
        if not active:
            return None
        return max(active, key=lambda f: f.get_effective_severity(timestamp))

    def clear(self) -> None:
        """Clear all scheduled faults."""
        self._faults.clear()

    def __len__(self) -> int:
        return len(self._faults)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize schedule to a list of dictionaries."""
        return [f.to_dict() for f in self._faults]
