"""
Mission Profile generator for SIH26054.

Provides flight phase definitions, profile construction, and discrete time-step interpolation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Iterator, Optional, Dict, Any


class FlightPhase(str, Enum):
    """MALE UAV mission flight phases."""
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    LOITER = "LOITER"
    DESCENT = "DESCENT"
    LANDING = "LANDING"


@dataclass
class PhaseSegment:
    """
    Configuration for a single flight mission phase segment.
    """
    phase: FlightPhase
    duration_s: float
    throttle_start_pct: float
    throttle_end_pct: float
    altitude_start_m: float
    altitude_end_m: float
    airspeed_start_ms: float = 30.0
    airspeed_end_ms: float = 45.0
    temp_offset_k: float = 0.0


@dataclass
class MissionStep:
    """
    Interpolated flight mission conditions at a discrete simulation time step.
    """
    timestamp_s: float
    phase: FlightPhase
    throttle_pct: float                     # 0.0 to 100.0 %
    altitude_m: float                       # meters
    airspeed_ms: float                      # meters per second (airflow cooling proxy)
    temp_offset_k: float                    # Temperature offset from ISA standard
    progress_pct: float                     # Segment completion fraction (0.0 to 1.0)


class MissionProfile:
    """
    Orchestrates flight mission profiles and yields discrete time-step steps.
    """

    def __init__(self, mission_id: str = "MISSION_MALE_UAV_001", segments: Optional[List[PhaseSegment]] = None):
        self.mission_id = mission_id
        self.segments = segments or self._default_male_uav_profile()

    @staticmethod
    def _default_male_uav_profile() -> List[PhaseSegment]:
        """
        Default standard 6-phase MALE UAV profile.
        NOTE: Durations and profiles are Tier D engineering assumptions for demonstration and testing.
        """
        return [
            PhaseSegment(
                phase=FlightPhase.TAKEOFF,
                duration_s=60.0,
                throttle_start_pct=100.0,
                throttle_end_pct=100.0,
                altitude_start_m=0.0,
                altitude_end_m=150.0,
                airspeed_start_ms=15.0,
                airspeed_end_ms=30.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CLIMB,
                duration_s=300.0,
                throttle_start_pct=90.0,
                throttle_end_pct=90.0,
                altitude_start_m=150.0,
                altitude_end_m=3000.0,
                airspeed_start_ms=30.0,
                airspeed_end_ms=40.0,
            ),
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=600.0,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=3000.0,
                altitude_end_m=3000.0,
                airspeed_start_ms=45.0,
                airspeed_end_ms=48.0,
            ),
            PhaseSegment(
                phase=FlightPhase.LOITER,
                duration_s=400.0,
                throttle_start_pct=60.0,
                throttle_end_pct=60.0,
                altitude_start_m=3000.0,
                altitude_end_m=3000.0,
                airspeed_start_ms=38.0,
                airspeed_end_ms=38.0,
            ),
            PhaseSegment(
                phase=FlightPhase.DESCENT,
                duration_s=300.0,
                throttle_start_pct=40.0,
                throttle_end_pct=40.0,
                altitude_start_m=3000.0,
                altitude_end_m=200.0,
                airspeed_start_ms=45.0,
                airspeed_end_ms=35.0,
            ),
            PhaseSegment(
                phase=FlightPhase.LANDING,
                duration_s=60.0,
                throttle_start_pct=25.0,
                throttle_end_pct=10.0,
                altitude_start_m=200.0,
                altitude_end_m=0.0,
                airspeed_start_ms=30.0,
                airspeed_end_ms=15.0,
            ),
        ]

    @property
    def total_duration_s(self) -> float:
        """Total duration of all mission segments in seconds."""
        return sum(seg.duration_s for seg in self.segments)

    def generate_steps(self, dt: float = 0.1) -> Iterator[MissionStep]:
        """
        Yield interpolated mission steps across the complete mission duration.
        """
        current_time = 0.0
        for seg in self.segments:
            seg_duration = max(dt, seg.duration_s)
            steps_in_segment = int(round(seg_duration / dt))
            
            for step_idx in range(steps_in_segment):
                # Fraction through the segment [0.0, 1.0]
                frac = step_idx / float(steps_in_segment) if steps_in_segment > 1 else 0.0

                throttle = seg.throttle_start_pct + frac * (seg.throttle_end_pct - seg.throttle_start_pct)
                altitude = seg.altitude_start_m + frac * (seg.altitude_end_m - seg.altitude_start_m)
                airspeed = seg.airspeed_start_ms + frac * (seg.airspeed_end_ms - seg.airspeed_start_ms)

                yield MissionStep(
                    timestamp_s=current_time,
                    phase=seg.phase,
                    throttle_pct=throttle,
                    altitude_m=altitude,
                    airspeed_ms=airspeed,
                    temp_offset_k=seg.temp_offset_k,
                    progress_pct=frac,
                )
                current_time += dt
