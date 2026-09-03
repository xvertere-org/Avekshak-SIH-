"""
Simulator module for SIH26054 Aero Piston Engine Digital Twin.
"""

from simulator.base import BaseEngineSimulator
from simulator.config import (
    SimulatorConfig,
    TierAParameters,
    TierCParameters,
    TierDParameters,
)
from simulator.fault_interface import (
    FaultType,
    FaultSubsystem,
    FaultState,
    FaultSchedule,
)
from simulator.sensor_faults import (
    SensorFaultMode,
    SensorChannel,
    SensorFaultProcessor,
)
from simulator.engine_simulator import EngineSimulator
from simulator.telemetry_generator import TelemetryGenerator
from simulator.subsystems.atmosphere import Atmosphere, ISAState
from simulator.subsystems.mission import MissionProfile, FlightPhase, MissionStep, PhaseSegment
from simulator.subsystems.dynamics import RotationalDynamics, OperatingPoint
from simulator.subsystems.fuel import FuelSystem, FuelState
from simulator.subsystems.thermal import ThermalSystem, ThermalState
from simulator.subsystems.lubrication import LubricationSystem, LubricationState
from simulator.subsystems.vibration import VibrationSystem, VibrationState

__all__ = [
    "BaseEngineSimulator",
    "SimulatorConfig",
    "TierAParameters",
    "TierCParameters",
    "TierDParameters",
    "FaultType",
    "FaultSubsystem",
    "FaultState",
    "FaultSchedule",
    "SensorFaultMode",
    "SensorChannel",
    "SensorFaultProcessor",
    "EngineSimulator",
    "TelemetryGenerator",
    "Atmosphere",
    "ISAState",
    "MissionProfile",
    "FlightPhase",
    "MissionStep",
    "PhaseSegment",
    "RotationalDynamics",
    "OperatingPoint",
    "FuelSystem",
    "FuelState",
    "ThermalSystem",
    "ThermalState",
    "LubricationSystem",
    "LubricationState",
    "VibrationSystem",
    "VibrationState",
]

