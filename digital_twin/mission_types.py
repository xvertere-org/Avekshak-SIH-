"""
Phase 10 Mission Reliability & What-If Simulation: Types and Specifications.

Defines typed, immutable schemas for mission specifications, environmental profiles,
control profiles, envelope monitoring thresholds with strict provenance, heuristic
mission risk indicators, mission results, and comparative scenario metrics.

DISCLAIMER & NON-CLAIMS:
This module is for grey-box engineering simulation and what-if decision support only.
It does NOT provide certified reliability prediction, fleet statistics, MTBF,
survival probabilities, or airworthiness certification.
All scenarios are MODEL SCENARIO RESULTS / SYNTHETIC.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np

from simulator.subsystems.atmosphere import Atmosphere
from simulator.config import SimulatorConfig
from simulator.fault_interface import FaultSchedule, FaultState


class MissionPhase(str, Enum):
    """Standard flight/mission phases for aero propulsion simulation."""
    START = "START"
    TAXI = "TAXI"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    LANDING = "LANDING"


class ThresholdClassification(str, Enum):
    """
    Authoritative classification of operating thresholds.
    Distinguishes certified OEM operating limits from grey-box model envelopes
    and data-quality bounds.
    """
    OEM_REFERENCE_LIMIT = "OEM_REFERENCE_LIMIT"
    MODEL_ENVELOPE = "MODEL_ENVELOPE"
    DATA_QUALITY_BOUND = "DATA_QUALITY_BOUND"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"


class ThresholdDirection(str, Enum):
    """Direction of threshold excursion."""
    MAX = "MAX"
    MIN = "MIN"


@dataclass(frozen=True)
class EnvelopeThreshold:
    """
    Strictly provenanced operating envelope threshold definition.
    """
    channel: str
    numeric_value: float
    unit: str
    direction: ThresholdDirection
    classification: ThresholdClassification
    source_doc: str
    source_section: str
    engine_variant: str = "Rotax 914 UL/F"
    notes: str = ""


# Authoritative Registry of Envelope Limits based on Phase 1/2 verified provenance
# Sources:
# - EASA TCDS E.122 Issue 06, Section IV (Operating Limitations)
# - BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1
AUTHORITATIVE_ENVELOPE_THRESHOLDS: List[EnvelopeThreshold] = [
    EnvelopeThreshold(
        channel="rpm",
        numeric_value=5800.0,
        unit="RPM",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Takeoff speed limit (max 5 minutes)",
    ),
    EnvelopeThreshold(
        channel="rpm",
        numeric_value=5500.0,
        unit="RPM",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum continuous engine speed",
    ),
    EnvelopeThreshold(
        channel="rpm",
        numeric_value=1400.0,
        unit="RPM",
        direction=ThresholdDirection.MIN,
        classification=ThresholdClassification.MODEL_ENVELOPE,
        source_doc="Rotax 914 Operators Manual Ed. 4 / Rev. 0",
        source_section="Section 2.1 Operating Limits",
        notes="Nominal idle speed floor",
    ),
    EnvelopeThreshold(
        channel="map",
        numeric_value=1.350,
        unit="bar",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum takeoff manifold pressure (39.9 inHg / 1350 hPa)",
    ),
    EnvelopeThreshold(
        channel="map",
        numeric_value=1.200,
        unit="bar",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum continuous manifold pressure (35.4 inHg / 1200 hPa)",
    ),
    EnvelopeThreshold(
        channel="cht",
        numeric_value=135.0,
        unit="deg_C",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum cylinder head temperature with conventional 50/50 water-glycol coolant",
    ),
    EnvelopeThreshold(
        channel="coolant_temp",
        numeric_value=120.0,
        unit="deg_C",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum coolant exit temperature",
    ),
    EnvelopeThreshold(
        channel="oil_temp",
        numeric_value=130.0,
        unit="deg_C",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum oil inlet temperature",
    ),
    EnvelopeThreshold(
        channel="oil_temp",
        numeric_value=50.0,
        unit="deg_C",
        direction=ThresholdDirection.MIN,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="Rotax 914 Operators Manual Ed. 4 / Rev. 0",
        source_section="Section 2.1 Operating Limits",
        notes="Minimum operational oil temperature before takeoff power",
    ),
    EnvelopeThreshold(
        channel="oil_pressure",
        numeric_value=0.8,
        unit="bar",
        direction=ThresholdDirection.MIN,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Minimum permissible oil pressure at idle",
    ),
    EnvelopeThreshold(
        channel="oil_pressure",
        numeric_value=2.0,
        unit="bar",
        direction=ThresholdDirection.MIN,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Minimum normal operational oil pressure",
    ),
    EnvelopeThreshold(
        channel="oil_pressure",
        numeric_value=5.0,
        unit="bar",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="EASA TCDS E.122 Issue 06",
        source_section="Section IV Operating Limitations",
        notes="Maximum normal operational oil pressure",
    ),
    EnvelopeThreshold(
        channel="egt",
        numeric_value=950.0,
        unit="deg_C",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="Rotax 914 Operators Manual Ed. 4 / Rev. 0",
        source_section="Section 2.1 Operating Limits",
        notes="Maximum exhaust gas temperature at takeoff rating (5 min)",
    ),
    EnvelopeThreshold(
        channel="egt",
        numeric_value=910.0,
        unit="deg_C",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.OEM_REFERENCE_LIMIT,
        source_doc="Rotax 914 Operators Manual Ed. 4 / Rev. 0",
        source_section="Section 2.1 Operating Limits",
        notes="Maximum continuous exhaust gas temperature",
    ),
    # Vibration threshold: EXPLICITLY NOT AN OEM LIMIT.
    # Rotax does not publish an OEM g-RMS limit in TCDS E.122 or OM Section 2.1.
    # This is strictly a grey-box model envelope / engineering heuristic limit.
    EnvelopeThreshold(
        channel="vibration",
        numeric_value=1.20,
        unit="g_rms",
        direction=ThresholdDirection.MAX,
        classification=ThresholdClassification.MODEL_ENVELOPE,
        source_doc="SIH26054 Grey-Box Propulsion Contract",
        source_section="Section 3.5 Dynamic Limits",
        notes="Simulated overall engine vibration envelope warning; NOT a certified OEM limit.",
    ),
]


@dataclass
class EnvelopeViolationEvent:
    """Record of an observed parameter violating an envelope threshold."""
    timestamp: float
    channel: str
    observed_value: float
    threshold_value: float
    unit: str
    direction: ThresholdDirection
    classification: ThresholdClassification
    source_doc: str
    duration_s: float
    scenario_id: str


@dataclass(frozen=True)
class EnvironmentProfile:
    """
    Environmental profile over the mission timeline.

    Reuses existing ISA Atmosphere model directly without double counting.
    Hot-day offset (temp_offset_k) is added exactly once to T_ISA(altitude).
    """
    initial_altitude_m: float = 1000.0
    altitude_schedule: Optional[Callable[[float], float]] = None  # f(t) -> altitude_m
    temp_offset_k: float = 0.0  # Hot-day delta: T_amb = T_ISA(alt) + temp_offset_k
    is_synthetic_high_altitude: bool = False  # Explicit flag for synthetic 4500m scenarios

    def get_conditions(self, t: float, atmosphere: Atmosphere) -> Tuple[float, float, float, float]:
        """
        Evaluate atmosphere at time t.
        Returns (altitude_m, t_amb_c, p_amb_pa, density_factor).
        """
        alt = self.altitude_schedule(t) if self.altitude_schedule is not None else self.initial_altitude_m
        # Clamp altitude to standard tropospheric bounds
        alt_safe = max(0.0, min(11000.0, float(alt)))

        # Evaluate existing ISA Atmosphere
        atmo = atmosphere.compute(altitude_m=alt_safe)

        # Apply hot-day temperature offset exactly once to ISA temperature
        t_amb_c = atmo.temperature_c + self.temp_offset_k
        p_amb_pa = atmo.pressure_pa
        density_factor = atmo.density_factor

        return alt_safe, t_amb_c, p_amb_pa, density_factor


@dataclass(frozen=True)
class ControlProfile:
    """
    Control/actuation profile over the mission timeline.
    """
    initial_throttle_pct: float = 75.0
    throttle_schedule: Optional[Callable[[float], float]] = None  # f(t) -> throttle_pct in [0, 100]
    initial_load_pct: float = 75.0
    load_schedule: Optional[Callable[[float], float]] = None      # f(t) -> load_pct

    def get_throttle(self, t: float) -> float:
        if self.throttle_schedule is not None:
            thr = self.throttle_schedule(t)
        else:
            thr = self.initial_throttle_pct
        return max(0.0, min(100.0, float(thr)))

    def get_load(self, t: float) -> float:
        if self.load_schedule is not None:
            load = self.load_schedule(t)
        else:
            load = self.initial_load_pct
        return max(0.0, min(100.0, float(load)))


@dataclass(frozen=True)
class MissionSpec:
    """
    Typed, immutable mission specification.
    """
    mission_id: str
    duration_s: float
    dt_s: float
    environment: EnvironmentProfile
    controls: ControlProfile
    fault_schedule: Optional[FaultSchedule] = None
    random_seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.duration_s <= 0.0:
            raise ValueError(f"Mission duration must be strictly positive, got {self.duration_s}")
        if self.dt_s <= 0.0:
            raise ValueError(f"Mission timestep dt must be strictly positive, got {self.dt_s}")
        if self.dt_s > self.duration_s:
            raise ValueError(f"Mission dt ({self.dt_s}) cannot exceed mission duration ({self.duration_s})")


@dataclass
class MissionRiskIndex:
    """
    Deterministic engineering risk heuristic for scenario comparison.

    MANDATORY DISCLAIMER:
    This index is an engineering heuristic combining health degradation,
    envelope excursions, degraded state durations, and RUL depletion.
    It is STRICTLY NOT a probability of failure, mission survival probability,
    or certified reliability indicator.
    """
    score: float  # [0.0, 1.0] where 0.0 is nominal healthy and 1.0 is extreme risk
    health_component: float       # Health degradation contribution
    envelope_component: float     # Envelope violation contribution
    duration_component: float     # Time spent in degraded/critical state contribution
    rul_component: float          # RUL depletion contribution
    risk_index_type: str = "ENGINEERING_HEURISTIC"
    claim_class: str = "MODEL_SCENARIO_RESULT"
    disclaimer: str = (
        "Engineering heuristic index for comparative scenario analysis only. "
        "Strictly NOT a statistical failure probability or certified airworthiness metric."
    )


@dataclass
class MissionMetrics:
    """
    Descriptive, deterministic summary metrics for a simulated mission scenario.
    """
    mission_duration_s: float
    step_count: int
    valid_telemetry_fraction: float
    min_hi: float
    mean_hi: float
    final_hi: float
    min_subsystem_health: Dict[str, float]
    max_cht_c: float
    max_egt_c: float
    min_oil_pressure_bar: float
    max_oil_temp_c: float
    max_vibration_g: float
    mean_fuel_flow_l_h: float
    max_fuel_flow_l_h: float
    time_below_watch_s: float       # HI < 0.85
    time_below_degraded_s: float    # HI < 0.70
    time_below_critical_s: float    # HI < 0.50
    time_in_diagnostic_state_s: float
    min_estimated_rul_h: Optional[float]
    final_estimated_rul_h: Optional[float]
    envelope_event_count: int
    total_envelope_excursion_duration_s: float
    risk_index: MissionRiskIndex


@dataclass
class MissionResult:
    """
    Typed, structured output of a completed mission simulation.
    """
    mission_id: str
    scenario_id: str
    timestamps_s: List[float]
    telemetry_history: Optional[List[Any]]  # Populated if full_trajectory=True
    health_trajectory: List[float]          # Overall HI(t)
    subsystem_health_trajectories: Dict[str, List[float]]
    degradation_trajectory: List[float]     # D(t)
    rul_trajectory: List[Optional[float]]   # RUL(t)
    envelope_events: List[EnvelopeViolationEvent]
    metrics: MissionMetrics
    seed: int
    provenance: Dict[str, Any]
    streaming_mode: bool = False


@dataclass
class ScenarioComparisonResult:
    """
    Comparative delta analysis between a Baseline mission and an alternative What-If scenario.
    """
    baseline_id: str
    scenario_id: str
    delta_min_hi: float
    delta_mean_hi: float
    delta_final_hi: float
    delta_time_below_watch_s: float
    delta_time_below_degraded_s: float
    delta_time_below_critical_s: float
    delta_min_rul_h: Optional[float]
    delta_final_rul_h: Optional[float]
    delta_max_cht_c: float
    delta_max_egt_c: float
    delta_min_oil_pressure_bar: float
    delta_max_vibration_g: float
    delta_envelope_events: int
    delta_risk_score: float
    qualitative_interpretation: str
    claim_class: str = "MODEL_SCENARIO_RESULT"
