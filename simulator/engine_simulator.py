"""
Physics-Informed Aero Piston Engine Simulator interface and Phase 1 stub.

DISCLAIMER:
This simulator is designed as a future reduced-order physics-informed/grey-box model.
It is NOT a computational fluid dynamics (CFD) solver or an authoritative certified OEM engine model.
"""

from typing import List, Optional
from simulator.base import BaseEngineSimulator
from telemetry.schema import MissionConfig, EngineConfig, TelemetryRecord, MissionPhase, FaultCategory


class EngineSimulator(BaseEngineSimulator):
    """
    Modular engine simulator interface.
    Phase 1: Stub returning schema-compliant telemetry without actual physics computation.
    """

    def __init__(self, engine_config: Optional[EngineConfig] = None):
        if engine_config is None:
            engine_config = EngineConfig()
        super().__init__(engine_config)
        self.current_time = 0.0

    def step(self, mission_config: MissionConfig, time_step: float = 1.0) -> TelemetryRecord:
        """
        Produce a single step of telemetry based on mission inputs.
        Phase 1 provides a synthetic nominal/fault-flagged stub record.
        """
        self.current_time += time_step

        # Default placeholder values compliant with schema
        record = TelemetryRecord(
            timestamp=self.current_time,
            mission_id=mission_config.mission_id,
            engine_id=mission_config.engine_id or self.engine_config.engine_id,
            mission_phase=mission_config.mission_phase,
            altitude=mission_config.altitude,
            ambient_temp=mission_config.ambient_temperature,
            throttle=mission_config.throttle,
            load=mission_config.engine_load,
            rpm=2400.0 * (mission_config.throttle / 100.0),
            cht=180.0,
            egt=720.0,
            oil_temp=85.0,
            oil_pressure=4.2,
            fuel_flow=22.5,
            vibration=0.8,
            fault_type=mission_config.fault_type,
            fault_severity=mission_config.fault_severity,
            source="simulator_v1_stub",
            source_type="simulated",
            simulation_version="0.1.0-phase1-stub",
        )
        return record

    def run_mission(self, mission_config: MissionConfig, time_step: float = 1.0) -> List[TelemetryRecord]:
        """
        Execute mission simulation loop.
        Phase 1 generates stub records across the mission duration.
        """
        records: List[TelemetryRecord] = []
        steps = int(max(1, mission_config.duration / time_step))
        
        # In Phase 1 stub mode, limit max generated records for dry-run efficiency
        max_stub_steps = min(steps, 10)
        for _ in range(max_stub_steps):
            records.append(self.step(mission_config, time_step))
        return records
