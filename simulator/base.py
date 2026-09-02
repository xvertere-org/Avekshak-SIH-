"""
Base abstract interface for Physics-Informed Engine Simulators.
"""

from abc import ABC, abstractmethod
from typing import List
from telemetry.schema import MissionConfig, EngineConfig, TelemetryRecord


class BaseEngineSimulator(ABC):
    """
    Abstract interface for aero piston engine simulation.
    Future phases will implement reduced-order physics-informed grey-box dynamics.
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config

    @abstractmethod
    def step(self, mission_config: MissionConfig, time_step: float = 1.0) -> TelemetryRecord:
        """
        Advance the simulation by one discrete time step.
        """
        pass

    @abstractmethod
    def run_mission(self, mission_config: MissionConfig, time_step: float = 1.0) -> List[TelemetryRecord]:
        """
        Execute a complete simulated mission profile.
        """
        pass
