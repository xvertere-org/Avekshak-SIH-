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
from enum import Enum
from typing import Dict, Any, List, Optional, Union


class FaultType(str, Enum):
    """
    Standard fault and degradation categories for aero piston engine simulation.
    """
    NONE = "none"
    COOLING_DEGRADATION = "cooling_degradation"
    LUBRICATION_DEGRADATION = "lubrication_degradation"
    FUEL_INJECTION_ABNORMALITY = "fuel_injection_abnormality"
    MECHANICAL_DEGRADATION = "mechanical_degradation"
    SENSOR_FAULT = "sensor_fault"


class FaultSubsystem(str, Enum):
    """
    Engine physical or data subsystems targeted by fault injection.
    """
    NONE = "none"
    THERMAL = "thermal"
    LUBRICATION = "lubrication"
    FUEL = "fuel"
    DYNAMICS = "dynamics"
    VIBRATION = "vibration"
    SENSOR = "sensor"


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
        parameters: Additional fault-specific parameters (e.g. ramp_duration, target_sensor, leak_rate).
    """
    fault_type: Union[FaultType, str] = FaultType.NONE
    severity: float = 0.0
    active: bool = True
    start_time: float = 0.0
    end_time: Optional[float] = None
    affected_subsystem: Union[FaultSubsystem, str] = FaultSubsystem.NONE
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate and convert fault_type
        if isinstance(self.fault_type, str):
            try:
                self.fault_type = FaultType(self.fault_type.lower())
            except ValueError:
                valid_types = [t.value for t in FaultType]
                raise ValueError(
                    f"Invalid fault_type '{self.fault_type}'. Supported types: {valid_types}"
                )

        # Validate and convert affected_subsystem
        if isinstance(self.affected_subsystem, str):
            try:
                self.affected_subsystem = FaultSubsystem(self.affected_subsystem.lower())
            except ValueError:
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
        optional ramp-up durations specified in parameters.
        """
        if not self.is_active_at(timestamp):
            return 0.0

        t = float(timestamp)
        ramp_duration = float(self.parameters.get("ramp_duration", 0.0))

        if ramp_duration > 0.0:
            elapsed = t - self.start_time
            fraction = min(1.0, max(0.0, elapsed / ramp_duration))
            return self.severity * fraction
        return self.severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert FaultState to dictionary representation."""
        return {
            "fault_type": self.fault_type.value if hasattr(self.fault_type, "value") else str(self.fault_type),
            "severity": self.severity,
            "active": self.active,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "affected_subsystem": self.affected_subsystem.value if hasattr(self.affected_subsystem, "value") else str(self.affected_subsystem),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultState":
        """Reconstruct FaultState from dictionary."""
        return cls(**data)


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
