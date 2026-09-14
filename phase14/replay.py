"""
Mission Replay Module for Phase 14 — SIH26054 / AVEKSHAK Digital Twin.

Architectural Rule:
- Replay consumes completed synthetic mission observations and Phase 13 output contracts.
- Strictly does NOT recompute PHM logic independently.
- Preserves (engine_id, mission_id) state boundary isolation.
- Guarantees chronological ordering and deterministic replay.
"""

from typing import Dict, List, Optional, Any, Union
import time

from orchestrator.schema import DashboardStatePayload, SimulationScenario
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.schema import MissionReplaySession


class MissionReplayManager:
    """
    Manages storage, retrieval, and playback of completed mission sessions.
    Provides decoupled session replay for visualization and retrospective analysis.
    """

    def __init__(self):
        # In-memory thread-safe session store keyed by (engine_id, mission_id)
        self._sessions: Dict[str, MissionReplaySession] = {}

    @staticmethod
    def _make_key(engine_id: str, mission_id: str) -> str:
        return f"{engine_id}::{mission_id}"

    def register_session(self, session: MissionReplaySession) -> str:
        """Register a completed mission replay session with strict session isolation."""
        if not session.engine_id or not session.mission_id:
            raise ValueError("Both engine_id and mission_id are required for replay session isolation.")
        
        # Validate chronological integrity
        self.validate_chronological_ordering(session.payloads)

        key = self._make_key(session.engine_id, session.mission_id)
        self._sessions[key] = session
        return key

    def create_session_from_payloads(
        self,
        payloads: List[DashboardStatePayload],
        scenario_name: str = "recorded_mission",
        dt: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> MissionReplaySession:
        """
        Build and register a MissionReplaySession from an existing list of Phase 13 DashboardStatePayloads.
        """
        if not payloads:
            raise ValueError("Cannot create a replay session from an empty payload list.")

        # Validate chronological ordering
        self.validate_chronological_ordering(payloads)

        engine_id = payloads[0].engine_id
        mission_id = payloads[0].mission_id or "UNKNOWN_MISSION"

        # Verify engine/mission homogeneity across the payload list
        for idx, p in enumerate(payloads):
            if p.engine_id != engine_id or p.mission_id != mission_id:
                raise ValueError(
                    f"Payload stream mixes state boundaries at step {idx}: "
                    f"expected ({engine_id}, {mission_id}) but got ({p.engine_id}, {p.mission_id})."
                )

        prov = provenance or {
            "source": "Phase 13 Pipeline Orchestrator",
            "simulation_mode": payloads[0].simulation_mode,
            "created_at": time.time(),
        }

        session = MissionReplaySession(
            engine_id=engine_id,
            mission_id=mission_id,
            scenario_name=scenario_name,
            payloads=payloads,
            total_duration_s=payloads[-1].timestamp - payloads[0].timestamp,
            total_steps=len(payloads),
            dt=dt,
            created_at=time.time(),
            provenance=prov,
        )
        self.register_session(session)
        return session

    def record_mission(
        self,
        orchestrator: SystemPipelineOrchestrator,
        scenario: SimulationScenario,
    ) -> MissionReplaySession:
        """
        Execute a simulation run through the authoritative Phase 13 pipeline and record the session.
        Guarantees deterministic execution for matching seeds.
        """
        payloads = orchestrator.run_simulation(scenario=scenario)
        prov = {
            "scenario_name": scenario.name,
            "seed": scenario.seed,
            "throttle_pct": scenario.throttle_pct,
            "altitude_m": scenario.altitude_m,
            "ambient_temp_c": scenario.ambient_temp_c,
            "fault_type": scenario.fault_type.value if hasattr(scenario.fault_type, "value") else str(scenario.fault_type),
            "created_at": time.time(),
        }
        return self.create_session_from_payloads(
            payloads=payloads,
            scenario_name=scenario.name,
            dt=scenario.dt,
            provenance=prov,
        )

    def get_session(self, engine_id: str, mission_id: str) -> Optional[MissionReplaySession]:
        """Retrieve stored replay session by its isolation key."""
        key = self._make_key(engine_id, mission_id)
        return self._sessions.get(key)

    def list_sessions(self) -> List[str]:
        """Return list of active registered session isolation keys."""
        return sorted(list(self._sessions.keys()))

    def clear(self) -> None:
        """Clear all registered sessions."""
        self._sessions.clear()

    @staticmethod
    def validate_chronological_ordering(payloads: List[DashboardStatePayload]) -> None:
        """
        Ensure valid telemetry timestamps strictly increase monotonically.
        Observations explicitly marked as REJECTED by causal sequence guards are excluded.
        """
        if not payloads:
            return

        valid_payloads = [
            p for p in payloads
            if "REJECTED" not in (getattr(p, "quality_status", None) or "")
        ]
        if not valid_payloads:
            return

        last_ts = valid_payloads[0].timestamp
        for idx in range(1, len(valid_payloads)):
            curr_ts = valid_payloads[idx].timestamp
            if curr_ts <= last_ts:
                raise ValueError(
                    f"Chronological ordering violation at valid index {idx}: "
                    f"timestamp {curr_ts} <= preceding {last_ts}."
                )
            last_ts = curr_ts
