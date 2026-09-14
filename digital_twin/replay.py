"""
Deterministic Telemetry Replay Engine for SIH26054 Digital Twin.

Enables causal, step-by-step historical replay of recorded telemetry datasets
through the state estimator and synchronizer.

Guarantees:
- Deterministic replay for identical runtime, configuration, and input conditions.
- Accurate temporal sequence playback with gap detection and sub-stepping.
- Direct output of canonical state trajectories and synchronization diagnostics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from digital_twin.state import CanonicalTwinState, SynchronizationStatus
from digital_twin.synchronizer import StateEstimator, EstimatorConfig
from telemetry.schema import TelemetryRecord
from telemetry.ingestion import CanonicalTelemetryFrame
from simulator.config import SimulatorConfig


@dataclass
class ReplayStepRecord:
    """Individual diagnostic step record from a replay run."""
    step_index: int
    timestamp: float
    actual_dt: float
    substeps_executed: int
    telemetry_valid: bool
    sync_status: SynchronizationStatus
    heuristic_confidence: float
    residuals: Dict[str, float]
    canonical_state: CanonicalTwinState

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "actual_dt": round(self.actual_dt, 4),
            "substeps_executed": self.substeps_executed,
            "telemetry_valid": self.telemetry_valid,
            "sync_status": self.sync_status.value,
            "heuristic_confidence": round(self.heuristic_confidence, 4),
            "residuals": {k: round(v, 4) for k, v in self.residuals.items()},
        }


class DigitalTwinReplay:
    """
    Deterministic replay executor for the Digital Twin.
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        sim_config: Optional[SimulatorConfig] = None,
    ):
        self.config = config or EstimatorConfig()
        self.sim_config = sim_config or SimulatorConfig()
        self.estimator = StateEstimator(config=self.config, sim_config=self.sim_config)

    def reset(self) -> None:
        """Reset internal estimator."""
        self.estimator.reset()

    def replay(
        self,
        telemetry_series: Union[List[TelemetryRecord], pd.DataFrame, CanonicalTelemetryFrame],
    ) -> List[ReplayStepRecord]:
        """
        Execute deterministic replay over a telemetry sequence.

        Args:
            telemetry_series: Sequence of TelemetryRecords, a pandas DataFrame,
                              or a CanonicalTelemetryFrame.

        Returns:
            List of ReplayStepRecord diagnostic objects.
        """
        self.reset()
        records: List[ReplayStepRecord] = []

        # Normalize input to an iterable of records / row dicts
        if isinstance(telemetry_series, CanonicalTelemetryFrame):
            df = telemetry_series.to_dataframe()
            rows = df.to_dict(orient="records")
        elif isinstance(telemetry_series, pd.DataFrame):
            rows = telemetry_series.to_dict(orient="records")
        else:
            rows = telemetry_series

        last_t: Optional[float] = None

        for idx, row in enumerate(rows):
            # Extract timestamp
            if isinstance(row, TelemetryRecord):
                t_curr = float(row.timestamp)
            elif isinstance(row, dict):
                t_curr = float(row.get("timestamp", idx * self.sim_config.default_dt))
            else:
                t_curr = float(getattr(row, "timestamp", idx * self.sim_config.default_dt))

            if last_t is not None:
                actual_dt = max(1e-4, t_curr - last_t)
            else:
                actual_dt = self.sim_config.default_dt
            last_t = t_curr

            # Calculate expected sub-steps
            if actual_dt <= self.config.dt_max:
                substeps = 1
            else:
                num_full = int(actual_dt // self.config.dt_max)
                rem = actual_dt - (num_full * self.config.dt_max)
                substeps = num_full + (1 if rem > 1e-6 else 0)

            # Step estimator
            state = self.estimator.step(row, dt=actual_dt)

            report_meta = state.metadata.get("quality_report", {})
            is_valid = report_meta.get("overall_valid", True)
            res_dict = state.metadata.get("residuals", {})

            step_record = ReplayStepRecord(
                step_index=idx,
                timestamp=t_curr,
                actual_dt=actual_dt,
                substeps_executed=substeps,
                telemetry_valid=is_valid,
                sync_status=state.sync_status,
                heuristic_confidence=state.heuristic_confidence,
                residuals=res_dict,
                canonical_state=state,
            )
            records.append(step_record)

        return records
