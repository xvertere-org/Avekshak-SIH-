"""
Phase 10 Mission Reliability & What-If Simulation: Mission Simulator.

Implements the deterministic mission simulation layer coordinating:
- Environment Profile (ISA Atmosphere + single hot-day offset)
- Control Profile (throttle / load profiles)
- EngineSimulator (grey-box Rotax 914 physics)
- Authoritative Envelope Monitoring (EASA TCDS / OM limits, Model Envelope for vibration)
- Digital Twin causal assessment (residual, health, diagnosis, degradation, RUL)
- Deterministic metric aggregation and heuristic risk scoring

DISCLAIMER:
All outputs are MODEL SCENARIO RESULTS / SYNTHETIC.
This module is strictly for engineering analysis and decision support.
It does NOT provide certified flight safety, fleet reliability, or airworthiness certification.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from telemetry.schema import TelemetryRecord, DigitalTwinState
from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from simulator.subsystems.atmosphere import Atmosphere
from digital_twin.twin_model import DigitalTwin
from digital_twin.health import HealthState
from digital_twin.mission_types import (
    MissionSpec,
    MissionResult,
    MissionMetrics,
    MissionRiskIndex,
    EnvelopeThreshold,
    EnvelopeViolationEvent,
    ThresholdDirection,
    ThresholdClassification,
    AUTHORITATIVE_ENVELOPE_THRESHOLDS,
)


class MissionSimulator:
    """
    Deterministic mission simulator and scenario execution engine.
    Ensures complete state isolation and reproducibility between runs.
    """

    def __init__(
        self,
        sim_config: Optional[SimulatorConfig] = None,
        custom_thresholds: Optional[List[EnvelopeThreshold]] = None,
    ):
        self.sim_config = sim_config or SimulatorConfig()
        self.thresholds = custom_thresholds if custom_thresholds is not None else AUTHORITATIVE_ENVELOPE_THRESHOLDS
        self.atmosphere = Atmosphere(tier_a=self.sim_config.tier_a)

    def run_mission(
        self,
        spec: MissionSpec,
        full_trajectory: bool = True,
        scenario_id: Optional[str] = None,
    ) -> MissionResult:
        """
        Execute a mission deterministically with strict state isolation.

        Args:
            spec: Immutable MissionSpec defining environmental, control, and fault schedules.
            full_trajectory: If True, retains full TelemetryRecord history (O(N) memory).
                             If False, retains streaming summary metrics only.
            scenario_id: Optional scenario label. Defaults to spec.mission_id.

        Returns:
            Fully populated and typed MissionResult.
        """
        effective_scenario_id = scenario_id or spec.mission_id

        # STRICT ISOLATION: instantiate clean, independent simulator and digital twin instances
        # using the exact random seed configured in the mission specification.
        engine_sim = EngineSimulator(sim_config=self.sim_config, seed=spec.random_seed)
        engine_sim.reset(seed=spec.random_seed)

        twin = DigitalTwin(sim_config=self.sim_config)

        # Simulation timeline variables
        dt = spec.dt_s
        duration = spec.duration_s
        t = 0.0

        # Trajectory storage
        timestamps: List[float] = []
        telemetry_history: Optional[List[TelemetryRecord]] = [] if full_trajectory else None
        health_trajectory: List[float] = []
        subsystem_health_trajectories: Dict[str, List[float]] = {
            "thermal": [],
            "lubrication": [],
            "fuel": [],
            "mechanical": [],
            "combustion": [],
        }
        degradation_trajectory: List[float] = []
        rul_trajectory: List[Optional[float]] = []

        # Envelope tracking state
        envelope_events: List[EnvelopeViolationEvent] = []
        active_excursions: Dict[Tuple[str, ThresholdDirection], float] = {}  # (channel, dir) -> start_time
        total_excursion_duration = 0.0

        # Metrics accumulators
        valid_telemetry_count = 0
        step_count = 0
        time_below_watch = 0.0       # HI < 0.85
        time_below_degraded = 0.0    # HI < 0.70
        time_below_critical = 0.0    # HI < 0.50
        time_in_diagnostic_state = 0.0

        max_cht = -float("inf")
        max_egt = -float("inf")
        min_oil_p = float("inf")
        max_oil_t = -float("inf")
        max_vib = -float("inf")
        fuel_flow_sum = 0.0
        max_fuel_flow = -float("inf")

        min_subsystem_health = {
            "thermal": 1.0,
            "lubrication": 1.0,
            "fuel": 1.0,
            "mechanical": 1.0,
            "combustion": 1.0,
        }

        # Step loop
        while t < duration + 1e-9:
            step_count += 1
            timestamps.append(t)

            # 1. Environment evaluation (single-source ISA Atmosphere + hot-day offset)
            alt_m, t_amb_c, p_amb_pa, density_factor = spec.environment.get_conditions(t, self.atmosphere)

            # 2. Control evaluation
            throttle_pct = spec.controls.get_throttle(t)

            # 3. Physics step in EngineSimulator
            # Passes fault schedule directly to simulator for physics modulation.
            record = engine_sim.step(
                throttle_pct=throttle_pct,
                altitude_m=alt_m,
                temp_offset_k=spec.environment.temp_offset_k,
                dt=dt,
                fault_state=spec.fault_schedule,
            )

            # Record telemetry validity
            if record is not None and not math.isnan(record.rpm):
                valid_telemetry_count += 1

            if full_trajectory and telemetry_history is not None:
                telemetry_history.append(record)

            # 4. Envelope monitoring against authoritative limits
            self._check_envelope(
                record=record,
                t=t,
                dt=dt,
                scenario_id=effective_scenario_id,
                active_excursions=active_excursions,
                envelope_events=envelope_events,
            )

            # 5. Causal Digital Twin update
            # ANTI-LEAKAGE: twin.update receives ONLY physical telemetry record.
            # No ground truth fault labels or scenario configurations are passed.
            twin_state: DigitalTwinState = twin.update(record)

            # 6. Extract Health, Degradation, RUL assessments
            hi_val = 1.0
            if twin_state.health_assessment is not None:
                ha = twin_state.health_assessment
                if not math.isnan(ha.HI_smooth):
                    hi_val = float(ha.HI_smooth)
                elif not math.isnan(ha.HI_raw):
                    hi_val = float(ha.HI_raw)

                # Subsystem health
                for sub_k, sub_ass in ha.subsystems.items():
                    sub_lower = sub_k.lower()
                    if sub_lower in subsystem_health_trajectories:
                        s_score = sub_ass.score if not math.isnan(sub_ass.score) else 1.0
                        subsystem_health_trajectories[sub_lower].append(float(s_score))
                        min_subsystem_health[sub_lower] = min(min_subsystem_health[sub_lower], float(s_score))
            health_trajectory.append(hi_val)

            # Degradation index D(t)
            d_val = 0.0
            if twin_state.degradation_assessment is not None:
                d_val = float(twin_state.degradation_assessment.degradation_index)
            degradation_trajectory.append(d_val)

            # RUL assessment
            rul_val = None
            if twin_state.rul_assessment is not None and twin_state.rul_assessment.rul_median is not None:
                rul_val = float(twin_state.rul_assessment.rul_median)
            rul_trajectory.append(rul_val)

            # 7. Accumulate threshold times (Phase 5/6 standards: 0.85, 0.70, 0.50)
            if hi_val < 0.85:
                time_below_watch += dt
            if hi_val < 0.70:
                time_below_degraded += dt
            if hi_val < 0.50:
                time_below_critical += dt

            # Diagnostic state time
            if twin_state.diagnosis_result is not None:
                diag = twin_state.diagnosis_result
                if diag.status in ("CONFIRMED", "SUSPECTED") and diag.primary_fault not in ("none", "unknown", ""):
                    time_in_diagnostic_state += dt

            # Extrema
            if not math.isnan(record.cht):
                max_cht = max(max_cht, record.cht)
            if not math.isnan(record.egt):
                max_egt = max(max_egt, record.egt)
            if not math.isnan(record.oil_pressure):
                min_oil_p = min(min_oil_p, record.oil_pressure)
            if not math.isnan(record.oil_temp):
                max_oil_t = max(max_oil_t, record.oil_temp)
            if not math.isnan(record.vibration):
                max_vib = max(max_vib, record.vibration)
            if not math.isnan(record.fuel_flow):
                fuel_flow_sum += record.fuel_flow
                max_fuel_flow = max(max_fuel_flow, record.fuel_flow)

            # Advance clock
            t += dt

        # Close any open envelope excursions at mission end
        for (ch, direction), start_t in list(active_excursions.items()):
            excursion_dur = t - start_t
            total_excursion_duration += excursion_dur

        # Calculate final summary metrics
        min_hi = float(np.min(health_trajectory)) if health_trajectory else 1.0
        mean_hi = float(np.mean(health_trajectory)) if health_trajectory else 1.0
        final_hi = health_trajectory[-1] if health_trajectory else 1.0

        valid_ruls = [r for r in rul_trajectory if r is not None]
        min_rul = min(valid_ruls) if valid_ruls else None
        final_rul = valid_ruls[-1] if valid_ruls else None

        valid_fraction = valid_telemetry_count / max(1, step_count)
        mean_fuel_flow = fuel_flow_sum / max(1, step_count)

        # Compute deterministic heuristic Mission Risk Index
        risk_index = self._calculate_risk_index(
            min_hi=min_hi,
            envelope_event_count=len(envelope_events),
            time_below_degraded=time_below_degraded,
            time_below_critical=time_below_critical,
            final_rul=final_rul,
            duration=duration,
        )

        metrics = MissionMetrics(
            mission_duration_s=round(duration, 3),
            step_count=step_count,
            valid_telemetry_fraction=round(valid_fraction, 4),
            min_hi=round(min_hi, 4),
            mean_hi=round(mean_hi, 4),
            final_hi=round(final_hi, 4),
            min_subsystem_health={k: round(v, 4) for k, v in min_subsystem_health.items()},
            max_cht_c=round(max_cht, 2) if max_cht != -float("inf") else 0.0,
            max_egt_c=round(max_egt, 2) if max_egt != -float("inf") else 0.0,
            min_oil_pressure_bar=round(min_oil_p, 3) if min_oil_p != float("inf") else 0.0,
            max_oil_temp_c=round(max_oil_t, 2) if max_oil_t != -float("inf") else 0.0,
            max_vibration_g=round(max_vib, 3) if max_vib != -float("inf") else 0.0,
            mean_fuel_flow_l_h=round(mean_fuel_flow, 2),
            max_fuel_flow_l_h=round(max_fuel_flow, 2) if max_fuel_flow != -float("inf") else 0.0,
            time_below_watch_s=round(time_below_watch, 2),
            time_below_degraded_s=round(time_below_degraded, 2),
            time_below_critical_s=round(time_below_critical, 2),
            time_in_diagnostic_state_s=round(time_in_diagnostic_state, 2),
            min_estimated_rul_h=round(min_rul, 2) if min_rul is not None else None,
            final_estimated_rul_h=round(final_rul, 2) if final_rul is not None else None,
            envelope_event_count=len(envelope_events),
            total_envelope_excursion_duration_s=round(total_excursion_duration, 2),
            risk_index=risk_index,
        )

        return MissionResult(
            mission_id=spec.mission_id,
            scenario_id=effective_scenario_id,
            timestamps_s=timestamps,
            telemetry_history=telemetry_history,
            health_trajectory=health_trajectory,
            subsystem_health_trajectories=subsystem_health_trajectories,
            degradation_trajectory=degradation_trajectory,
            rul_trajectory=rul_trajectory,
            envelope_events=envelope_events,
            metrics=metrics,
            seed=spec.random_seed,
            provenance={
                "engine_variant": "Rotax 914 UL/F",
                "atmosphere_source": "ISA Troposphere Model",
                "envelope_source": "EASA TCDS E.122 / Rotax OM 914",
                "claim_class": "MODEL_SCENARIO_RESULT",
                "full_trajectory": full_trajectory,
            },
            streaming_mode=not full_trajectory,
        )

    def _check_envelope(
        self,
        record: TelemetryRecord,
        t: float,
        dt: float,
        scenario_id: str,
        active_excursions: Dict[Tuple[str, ThresholdDirection], float],
        envelope_events: List[EnvelopeViolationEvent],
    ) -> None:
        """
        Validate observed telemetry against the authoritative envelope threshold registry.
        """
        val_map = {
            "rpm": record.rpm,
            "map": record.map_bar,
            "cht": record.cht,
            "coolant_temp": record.coolant_temp,
            "oil_temp": record.oil_temp,
            "oil_pressure": record.oil_pressure,
            "egt": record.egt,
            "vibration": record.vibration,
        }

        for thresh in self.thresholds:
            val = val_map.get(thresh.channel)
            if val is None or math.isnan(val):
                continue

            is_violation = False
            if thresh.direction == ThresholdDirection.MAX and val > thresh.numeric_value:
                is_violation = True
            elif thresh.direction == ThresholdDirection.MIN and val < thresh.numeric_value:
                is_violation = True

            key = (thresh.channel, thresh.direction)
            if is_violation:
                if key not in active_excursions:
                    active_excursions[key] = t
                    # Record newly triggered event
                    envelope_events.append(
                        EnvelopeViolationEvent(
                            timestamp=round(t, 3),
                            channel=thresh.channel,
                            observed_value=round(val, 3),
                            threshold_value=thresh.numeric_value,
                            unit=thresh.unit,
                            direction=thresh.direction,
                            classification=thresh.classification,
                            source_doc=thresh.source_doc,
                            duration_s=round(dt, 3),
                            scenario_id=scenario_id,
                        )
                    )
                else:
                    # Update duration of last event for this key
                    if envelope_events and envelope_events[-1].channel == thresh.channel:
                        envelope_events[-1].duration_s = round(t - active_excursions[key], 3)
            else:
                if key in active_excursions:
                    del active_excursions[key]

    def _calculate_risk_index(
        self,
        min_hi: float,
        envelope_event_count: int,
        time_below_degraded: float,
        time_below_critical: float,
        final_rul: Optional[float],
        duration: float,
    ) -> MissionRiskIndex:
        """
        Calculate deterministic engineering risk heuristic index.

        DISCLAIMER:
        This is an engineering heuristic combining health loss, envelope excursions,
        and degraded state duration. It is NOT a statistical failure probability.
        """
        # Component 1: Health loss (0 = healthy HI 1.0; 1 = HI 0.0)
        c_health = max(0.0, min(1.0, 1.0 - min_hi))

        # Component 2: Envelope violation count (saturates at 10 events)
        c_envelope = min(1.0, envelope_event_count / 10.0)

        # Component 3: Degraded and Critical duration fraction
        c_duration = min(1.0, (time_below_degraded * 1.0 + time_below_critical * 2.0) / max(1.0, duration))

        # Component 4: RUL depletion (relative to 2000h nominal TBO)
        if final_rul is not None:
            c_rul = max(0.0, min(1.0, 1.0 - (final_rul / 2000.0)))
        else:
            c_rul = 0.0

        # Weighted composite score: health (40%), duration (30%), envelope (20%), RUL (10%)
        composite_score = 0.40 * c_health + 0.30 * c_duration + 0.20 * c_envelope + 0.10 * c_rul
        composite_score = round(max(0.0, min(1.0, composite_score)), 4)

        return MissionRiskIndex(
            score=composite_score,
            health_component=round(c_health, 4),
            envelope_component=round(c_envelope, 4),
            duration_component=round(c_duration, 4),
            rul_component=round(c_rul, 4),
        )
