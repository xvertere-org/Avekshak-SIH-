"""
Canonical State Representation for SIH26054 Digital Twin.

Establishes a typed, unit-aware, inspectable canonical state vector across all
engine subsystems for the BRP-Rotax 914 UL/F aero-piston digital twin.

Key architectural rules:
1. Strict separation of input, latent state, observed telemetry, and predicted telemetry.
2. QuantityStatus tracks state variable provenance (MEASURED, ESTIMATED, PREDICTED, DERIVED, UNAVAILABLE).
3. ThermalState houses the single canonical oil temperature state; LubricationState provides a read-only property view.
4. Electrical state is explicitly cataloged as UNAVAILABLE (honest disclosure, no fabricated physics).
5. Heuristic synchronization confidence is strictly deterministic and non-statistical.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import math


class QuantityStatus(str, Enum):
    """Provenance and epistemological status of a physical quantity."""
    MEASURED = "MEASURED"                    # Directly from calibrated sensor observation
    ESTIMATED = "ESTIMATED"                  # Synchronized via observer / estimator correction
    PREDICTED = "PREDICTED"                  # Forward physics prediction without sensor correction
    DERIVED = "DERIVED"                      # Kinematically or algebraically computed from other states
    DEFAULT_ASSUMED = "DEFAULT_ASSUMED"      # Static reference parameter or baseline initial condition
    UNAVAILABLE = "UNAVAILABLE"              # Uninstrumented or unmodeled channel


class EngineOperatingRegime(str, Enum):
    """Engine operating regime classification based on thermal and rotational state."""
    COLD_START = "COLD_START"                # Sub-operating temperature, cranking / starting
    WARMING = "WARMING"                      # Transitional warming (CHT/oil below nominal threshold)
    STEADY_OPERATION = "STEADY_OPERATION"    # Normal operating thermal and RPM regime


class SynchronizationStatus(str, Enum):
    """Overall Digital Twin state synchronization status."""
    INITIALIZING = "INITIALIZING"                      # Initial steps before filter settle
    SYNCHRONIZED = "SYNCHRONIZED"                      # Full valid telemetry, residuals bounded
    PARTIALLY_SYNCHRONIZED = "PARTIALLY_SYNCHRONIZED"  # Subset of channels valid / missing telemetry
    DEGRADED_OBSERVABILITY = "DEGRADED_OBSERVABILITY"  # Telemetry degraded or significant sensor dropouts
    STALE = "STALE"                                    # Observation age exceeds timeout, pure physics fallback
    UNSYNCHRONIZED = "UNSYNCHRONIZED"                  # Critical divergence or complete observation failure


@dataclass
class PhysicalQuantity:
    """A typed physical quantity with value, unit, provenance status, and metadata."""
    value: float
    unit: str
    status: QuantityStatus
    timestamp: float
    source: str = "digital_twin"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": round(self.value, 4) if not math.isnan(self.value) else None,
            "unit": self.unit,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass
class RotationalState:
    """Rotational dynamics and power transmission state."""
    rpm: PhysicalQuantity                        # Engine crankshaft speed [RPM]
    omega_rad_s: PhysicalQuantity                # Engine crankshaft angular speed [rad/s]
    propeller_rpm: PhysicalQuantity              # Propeller speed via 51/21 gearbox [RPM]
    engine_torque_nm: PhysicalQuantity          # Net indicated engine torque [N*m]
    propeller_torque_nm: PhysicalQuantity       # Absorbed propeller torque [N*m]
    indicated_power_w: PhysicalQuantity         # Indicated engine power [W]
    power_target_w: PhysicalQuantity            # Target mechanical power demand [W]
    engine_load_pct: PhysicalQuantity           # Dimensionless engine load [%]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rpm": self.rpm.to_dict(),
            "omega_rad_s": self.omega_rad_s.to_dict(),
            "propeller_rpm": self.propeller_rpm.to_dict(),
            "engine_torque_nm": self.engine_torque_nm.to_dict(),
            "propeller_torque_nm": self.propeller_torque_nm.to_dict(),
            "indicated_power_w": self.indicated_power_w.to_dict(),
            "power_target_w": self.power_target_w.to_dict(),
            "engine_load_pct": self.engine_load_pct.to_dict(),
        }


@dataclass
class AirBoostState:
    """Turbocharger, intake manifold, and atmospheric air-path state."""
    map_bar: PhysicalQuantity                           # Manifold absolute pressure [bar]
    manifold_pressure_pa: PhysicalQuantity              # MAP in SI units [Pa]
    charge_air_temp_c: PhysicalQuantity                 # Charge air temperature after compressor [°C]
    ambient_pressure_bar: PhysicalQuantity              # Ambient static pressure [bar]
    ambient_temp_c: PhysicalQuantity                    # Ambient static air temperature [°C]
    air_mass_flow_kg_s: PhysicalQuantity                # Intake air mass flow rate [kg/s]
    pressure_ratio: PhysicalQuantity                    # Compressor pressure ratio P2/P1 [-]
    wastegate_position: PhysicalQuantity                # Wastegate effective opening fraction [0.0 - 1.0]
    turbo_shaft_speed: PhysicalQuantity                 # Turbocharger shaft speed [RPM] (unobserved surrogate)
    compressor_aerodynamic_efficiency: PhysicalQuantity  # Compressor isentropic efficiency [-] (surrogate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_bar": self.map_bar.to_dict(),
            "manifold_pressure_pa": self.manifold_pressure_pa.to_dict(),
            "charge_air_temp_c": self.charge_air_temp_c.to_dict(),
            "ambient_pressure_bar": self.ambient_pressure_bar.to_dict(),
            "ambient_temp_c": self.ambient_temp_c.to_dict(),
            "air_mass_flow_kg_s": self.air_mass_flow_kg_s.to_dict(),
            "pressure_ratio": self.pressure_ratio.to_dict(),
            "wastegate_position": self.wastegate_position.to_dict(),
            "turbo_shaft_speed": self.turbo_shaft_speed.to_dict(),
            "compressor_aerodynamic_efficiency": self.compressor_aerodynamic_efficiency.to_dict(),
        }


@dataclass
class CombustionState:
    """Fuel injection, combustion efficiency, and mixture state."""
    fuel_mass_flow_kg_s: PhysicalQuantity      # Fuel mass consumption rate [kg/s]
    fuel_flow_l_h: PhysicalQuantity            # Volumetric fuel flow rate [L/h]
    air_fuel_ratio: PhysicalQuantity           # Trapped air-to-fuel ratio [-]
    combustion_efficiency: PhysicalQuantity    # Thermal combustion efficiency factor [-]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fuel_mass_flow_kg_s": self.fuel_mass_flow_kg_s.to_dict(),
            "fuel_flow_l_h": self.fuel_flow_l_h.to_dict(),
            "air_fuel_ratio": self.air_fuel_ratio.to_dict(),
            "combustion_efficiency": self.combustion_efficiency.to_dict(),
        }


@dataclass
class ThermalState:
    """
    Multi-cylinder head temperatures, exhaust gas temperatures, coolant, and oil temperature.
    NOTE: oil_temp_c is canonically stored here, reflecting the engine thermal ODE network.
    """
    cht_cyl1_c: PhysicalQuantity               # Cylinder 1 cylinder head temperature [°C]
    cht_cyl2_c: PhysicalQuantity               # Cylinder 2 cylinder head temperature [°C]
    cht_cyl3_c: PhysicalQuantity               # Cylinder 3 cylinder head temperature [°C]
    cht_cyl4_c: PhysicalQuantity               # Cylinder 4 cylinder head temperature [°C]
    egt_cyl1_c: PhysicalQuantity               # Cylinder 1 exhaust gas temperature [°C]
    egt_cyl2_c: PhysicalQuantity               # Cylinder 2 exhaust gas temperature [°C]
    egt_cyl3_c: PhysicalQuantity               # Cylinder 3 exhaust gas temperature [°C]
    egt_cyl4_c: PhysicalQuantity               # Cylinder 4 exhaust gas temperature [°C]
    coolant_temp_c: PhysicalQuantity           # Effective cooling circuit temperature [°C]
    oil_temp_c: PhysicalQuantity               # Canonical engine oil temperature [°C]
    q_heads_w: PhysicalQuantity                # Total heat rejected to cylinder heads [W]
    q_radiator_w: PhysicalQuantity             # Heat rejected by coolant radiator [W]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cht_cyl1_c": self.cht_cyl1_c.to_dict(),
            "cht_cyl2_c": self.cht_cyl2_c.to_dict(),
            "cht_cyl3_c": self.cht_cyl3_c.to_dict(),
            "cht_cyl4_c": self.cht_cyl4_c.to_dict(),
            "egt_cyl1_c": self.egt_cyl1_c.to_dict(),
            "egt_cyl2_c": self.egt_cyl2_c.to_dict(),
            "egt_cyl3_c": self.egt_cyl3_c.to_dict(),
            "egt_cyl4_c": self.egt_cyl4_c.to_dict(),
            "coolant_temp_c": self.coolant_temp_c.to_dict(),
            "oil_temp_c": self.oil_temp_c.to_dict(),
            "q_heads_w": self.q_heads_w.to_dict(),
            "q_radiator_w": self.q_radiator_w.to_dict(),
        }


class LubricationState:
    """
    Engine lubrication and oil circuit state.
    Guarantees single source of truth: oil_temp_c is an explicit property view of ThermalState.oil_temp_c.
    """
    def __init__(
        self,
        oil_pressure_bar: PhysicalQuantity,
        thermal_state: ThermalState,
    ):
        self.oil_pressure_bar = oil_pressure_bar
        self._thermal_state = thermal_state

    @property
    def oil_temp_c(self) -> PhysicalQuantity:
        """Read-only view of canonical oil temperature stored in ThermalState."""
        return self._thermal_state.oil_temp_c

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oil_pressure_bar": self.oil_pressure_bar.to_dict(),
            "oil_temp_c": self.oil_temp_c.to_dict(),
        }


@dataclass
class VibrationState:
    """Engine structural vibration harmonics and overall RMS level."""
    vibration_rms_g: PhysicalQuantity          # Overall vibration RMS level [g]
    order_1x_freq_hz: PhysicalQuantity         # 1st engine order frequency [Hz]
    order_2x_freq_hz: PhysicalQuantity         # 2nd engine order frequency [Hz]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vibration_rms_g": self.vibration_rms_g.to_dict(),
            "order_1x_freq_hz": self.order_1x_freq_hz.to_dict(),
            "order_2x_freq_hz": self.order_2x_freq_hz.to_dict(),
        }


@dataclass
class ElectricalState:
    """
    Electrical subsystem state.
    Explicit honest disclosure: electrical physics are unmodeled in Phase 3.
    Values are NaN with status UNAVAILABLE (no fabricated equations).
    """
    battery_voltage_v: PhysicalQuantity = field(
        default_factory=lambda: PhysicalQuantity(
            value=float("nan"),
            unit="V",
            status=QuantityStatus.UNAVAILABLE,
            timestamp=0.0,
            notes="Electrical subsystem unmodeled in Phase 3",
        )
    )
    battery_current_a: PhysicalQuantity = field(
        default_factory=lambda: PhysicalQuantity(
            value=float("nan"),
            unit="A",
            status=QuantityStatus.UNAVAILABLE,
            timestamp=0.0,
            notes="Electrical subsystem unmodeled in Phase 3",
        )
    )
    bus_voltage_v: PhysicalQuantity = field(
        default_factory=lambda: PhysicalQuantity(
            value=float("nan"),
            unit="V",
            status=QuantityStatus.UNAVAILABLE,
            timestamp=0.0,
            notes="Electrical subsystem unmodeled in Phase 3",
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_voltage_v": self.battery_voltage_v.to_dict(),
            "battery_current_a": self.battery_current_a.to_dict(),
            "bus_voltage_v": self.bus_voltage_v.to_dict(),
        }


@dataclass
class CanonicalTwinState:
    """
    Complete canonical state vector for the Rotax 914 UL/F Digital Twin.
    Encapsulates all subsystem states, operating regime, synchronization status,
    and heuristic confidence indicator.
    """
    timestamp: float
    regime: EngineOperatingRegime
    sync_status: SynchronizationStatus
    heuristic_confidence: float
    rotational: RotationalState
    air_boost: AirBoostState
    combustion: CombustionState
    thermal: ThermalState
    lubrication: LubricationState
    vibration: VibrationState
    electrical: ElectricalState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime.value,
            "sync_status": self.sync_status.value,
            "heuristic_confidence": round(self.heuristic_confidence, 4),
            "rotational": self.rotational.to_dict(),
            "air_boost": self.air_boost.to_dict(),
            "combustion": self.combustion.to_dict(),
            "thermal": self.thermal.to_dict(),
            "lubrication": self.lubrication.to_dict(),
            "vibration": self.vibration.to_dict(),
            "electrical": self.electrical.to_dict(),
            "metadata": self.metadata,
        }
