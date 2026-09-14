"""
Reusable Mission Profiles and Environmental Generation for Rotax 914 Population.

Defines 8 canonical mission profiles covering the complete operational envelope:
1. GROUND_IDLE
2. TAKEOFF_CLIMB
3. CRUISE
4. HIGH_ALTITUDE_CRUISE (near critical altitude 4572 m)
5. DESCENT
6. RAPID_THROTTLE_TRANSITION
7. ENDURANCE
8. HOT_DAY_OPERATION (ISA + 23 K)

Atmospheric pressure and temperature are strictly derived via ISA barometric equations
to ensure physical consistency without unphysical pressure-altitude jumps.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator
import math
import numpy as np

from simulator.subsystems.mission import FlightPhase, MissionStep, PhaseSegment
from simulator.subsystems.atmosphere import Atmosphere, ISAState
from simulator.config import TierAParameters


class CanonicalMission(str, Enum):
    """Authoritative 8 canonical mission profiles for Phase 7 population validation."""
    GROUND_IDLE = "GROUND_IDLE"
    TAKEOFF_CLIMB = "TAKEOFF_CLIMB"
    CRUISE = "CRUISE"
    HIGH_ALTITUDE_CRUISE = "HIGH_ALTITUDE_CRUISE"
    DESCENT = "DESCENT"
    RAPID_THROTTLE_TRANSITION = "RAPID_THROTTLE_TRANSITION"
    ENDURANCE = "ENDURANCE"
    HOT_DAY_OPERATION = "HOT_DAY_OPERATION"


@dataclass
class MissionTrajectory:
    """
    Complete mission trajectory description supporting continuous interpolation
    and discrete step generation.
    """
    mission_name: CanonicalMission
    total_duration_s: float
    dt: float
    segments: List[PhaseSegment]
    seed: Optional[int] = None
    temp_offset_base_k: float = 0.0

    def generate_steps(self) -> List[MissionStep]:
        """
        Generate discrete MissionStep instances interpolated at time step self.dt.
        Applies deterministic seed-controlled smooth perturbation if seed is provided.
        """
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        steps: List[MissionStep] = []
        current_time = 0.0

        for seg in self.segments:
            seg_duration = seg.duration_s
            n_steps = max(1, int(round(seg_duration / self.dt)))

            for i in range(n_steps):
                progress = float(i) / float(n_steps)
                # Linear interpolations
                throt = seg.throttle_start_pct + progress * (seg.throttle_end_pct - seg.throttle_start_pct)
                alt = seg.altitude_start_m + progress * (seg.altitude_end_m - seg.altitude_start_m)
                spd = seg.airspeed_start_ms + progress * (seg.airspeed_end_ms - seg.airspeed_start_ms)
                t_off = seg.temp_offset_k + self.temp_offset_base_k

                # Deterministic slight seed jitter if rng is active
                if rng is not None:
                    throt_jitter = float(rng.normal(0.0, 0.4))
                    alt_jitter = float(rng.normal(0.0, 1.5))
                    t_jitter = float(rng.normal(0.0, 0.15))
                    throt = np.clip(throt + throt_jitter, 0.0, 100.0)
                    alt = max(0.0, alt + alt_jitter)
                    t_off = t_off + t_jitter

                steps.append(
                    MissionStep(
                        timestamp_s=round(current_time, 3),
                        phase=seg.phase,
                        throttle_pct=float(np.clip(throt, 0.0, 100.0)),
                        altitude_m=float(max(0.0, alt)),
                        airspeed_ms=float(max(0.0, spd)),
                        temp_offset_k=float(t_off),
                        progress_pct=round(progress, 4),
                    )
                )
                current_time += self.dt

        return steps


def build_canonical_trajectory(
    mission: CanonicalMission,
    seed: Optional[int] = None,
    dt: float = 0.5,
    duration_scale: float = 1.0,
) -> MissionTrajectory:
    """
    Construct an authoritative MissionTrajectory for any of the 8 canonical mission profiles.
    """
    scale = max(0.2, float(duration_scale))

    if mission == CanonicalMission.GROUND_IDLE:
        segments = [
            PhaseSegment(
                phase=FlightPhase.TAKEOFF,
                duration_s=60.0 * scale,
                throttle_start_pct=0.0,
                throttle_end_pct=5.0,
                altitude_start_m=0.0,
                altitude_end_m=5.0,
                airspeed_start_ms=0.0,
                airspeed_end_ms=5.0,
                temp_offset_k=0.0,
            ),
            PhaseSegment(
                phase=FlightPhase.TAKEOFF,
                duration_s=60.0 * scale,
                throttle_start_pct=5.0,
                throttle_end_pct=10.0,
                altitude_start_m=5.0,
                altitude_end_m=10.0,
                airspeed_start_ms=5.0,
                airspeed_end_ms=10.0,
                temp_offset_k=0.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=120.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.TAKEOFF_CLIMB:
        segments = [
            PhaseSegment(
                phase=FlightPhase.TAKEOFF,
                duration_s=60.0 * scale,
                throttle_start_pct=100.0,
                throttle_end_pct=100.0,
                altitude_start_m=0.0,
                altitude_end_m=200.0,
                airspeed_start_ms=15.0,
                airspeed_end_ms=32.0,
                temp_offset_k=0.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CLIMB,
                duration_s=180.0 * scale,
                throttle_start_pct=90.0,
                throttle_end_pct=90.0,
                altitude_start_m=200.0,
                altitude_end_m=2500.0,
                airspeed_start_ms=32.0,
                airspeed_end_ms=42.0,
                temp_offset_k=0.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=240.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.CRUISE:
        segments = [
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=240.0 * scale,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=2500.0,
                altitude_end_m=2500.0,
                airspeed_start_ms=45.0,
                airspeed_end_ms=46.0,
                temp_offset_k=0.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=240.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.HIGH_ALTITUDE_CRUISE:
        segments = [
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=240.0 * scale,
                throttle_start_pct=82.0,
                throttle_end_pct=82.0,
                altitude_start_m=4500.0,
                altitude_end_m=4500.0,
                airspeed_start_ms=48.0,
                airspeed_end_ms=48.0,
                temp_offset_k=-5.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=240.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.DESCENT:
        segments = [
            PhaseSegment(
                phase=FlightPhase.DESCENT,
                duration_s=200.0 * scale,
                throttle_start_pct=42.0,
                throttle_end_pct=38.0,
                altitude_start_m=3000.0,
                altitude_end_m=300.0,
                airspeed_start_ms=44.0,
                airspeed_end_ms=34.0,
                temp_offset_k=0.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=200.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.RAPID_THROTTLE_TRANSITION:
        # Rapid throttle steps: 25% -> 90% -> 40% -> 95% -> 50%
        dur = 30.0 * scale
        segments = [
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=dur,
                throttle_start_pct=25.0,
                throttle_end_pct=25.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=38.0,
                airspeed_end_ms=38.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=dur,
                throttle_start_pct=90.0,
                throttle_end_pct=90.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=42.0,
                airspeed_end_ms=42.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=dur,
                throttle_start_pct=40.0,
                throttle_end_pct=40.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=39.0,
                airspeed_end_ms=39.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=dur,
                throttle_start_pct=95.0,
                throttle_end_pct=95.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=43.0,
                airspeed_end_ms=43.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=dur,
                throttle_start_pct=50.0,
                throttle_end_pct=50.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=40.0,
                airspeed_end_ms=40.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=5 * dur,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.ENDURANCE:
        segments = [
            PhaseSegment(
                phase=FlightPhase.LOITER,
                duration_s=400.0 * scale,
                throttle_start_pct=65.0,
                throttle_end_pct=65.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
                airspeed_start_ms=38.0,
                airspeed_end_ms=38.0,
                temp_offset_k=2.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=400.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    elif mission == CanonicalMission.HOT_DAY_OPERATION:
        # High ambient temp: ISA + 23 K (~38°C at sea level)
        segments = [
            PhaseSegment(
                phase=FlightPhase.CLIMB,
                duration_s=100.0 * scale,
                throttle_start_pct=90.0,
                throttle_end_pct=90.0,
                altitude_start_m=0.0,
                altitude_end_m=1200.0,
                airspeed_start_ms=25.0,
                airspeed_end_ms=40.0,
                temp_offset_k=23.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=150.0 * scale,
                throttle_start_pct=80.0,
                throttle_end_pct=80.0,
                altitude_start_m=1200.0,
                altitude_end_m=1200.0,
                airspeed_start_ms=42.0,
                airspeed_end_ms=42.0,
                temp_offset_k=23.0,
            ),
        ]
        return MissionTrajectory(
            mission_name=mission,
            total_duration_s=250.0 * scale,
            dt=dt,
            segments=segments,
            seed=seed,
            temp_offset_base_k=0.0,
        )

    raise ValueError(f"Unknown canonical mission: {mission}")
