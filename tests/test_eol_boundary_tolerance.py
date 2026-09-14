"""
Regression test suite for Health Index End-of-Life (EOL) boundary evaluation.

Validates numerical precision handling, floating-point representations,
EWMA smoothing convergence, RUL gating, and dashboard state adaptation
around the critical HI = 0.35 boundary.
"""

import math
import numpy as np
import pytest

from prognostics.pipeline import RULPipeline
from prognostics.schema import RULStatus, RULResult, EOLCriteriaConfig, DEFAULT_HI_EOL_TOLERANCE
from prognostics.threshold import WeakestLinkEOLEvaluator
from health_index.schema import (
    HealthIndexResult,
    HealthState,
    DegradationTrend,
    HealthDataQuality,
    HealthIndexConfig,
)
from health_index.smoothing import CausalEWMASmoother
from health_index.degradation import DegradationTracker
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import StatusLevel


def make_health_result(
    timestamp: float,
    hi_smooth: float,
    engine_id: str = "ENG_01",
    mission_id: str = "MSN_01",
    health_state: str = "DEGRADED",
    degradation_rate: float = -0.001,
) -> HealthIndexResult:
    """Construct deterministic HealthIndexResult fixture for prognostics testing."""
    all_ch = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
    return HealthIndexResult(
        timestamp=timestamp,
        engine_id=engine_id,
        mission_id=mission_id,
        mission_phase="CRUISE",
        raw_health_index=hi_smooth,
        smoothed_health_index=hi_smooth,
        raw_degradation_score=1.0 - hi_smooth,
        degradation_rate=degradation_rate,
        degradation_trend=DegradationTrend.DEGRADING.value,
        health_state=health_state,
        data_quality=HealthDataQuality.VALID.value,
        valid_channels=all_ch,
        missing_channels=[],
        excluded_channels=[],
        dominant_degraded_channels=["cht"],
        channel_contributions={"cht": 0.3},
        channel_degradation_evidence={"cht": 0.5},
        effective_channel_weights={},
    )


def prime_pipeline(pipe: RULPipeline, engine_id: str, mission_id: str, start_hi: float = 0.50, end_hi: float = 0.36, n_points: int = 40):
    """Accumulate sufficient history (>= 32s) for degradation slope estimation."""
    ts = np.linspace(1.0, 40.0, n_points)
    his = np.linspace(start_hi, end_hi, n_points)
    for t, h in zip(ts, his):
        hr = make_health_result(timestamp=float(t), hi_smooth=float(h), engine_id=engine_id, mission_id=mission_id)
        pipe.process_assessment(hr)


# =========================================================================
# 1. EXACT BOUNDARY
# =========================================================================

def test_exact_035_declares_eol():
    """Exact HI = 0.35 must declare CRITICAL_EOL_REACHED with RUL = 0.0s."""
    pipe = RULPipeline()
    prime_pipeline(pipe, "ENG_01", "MSN_EXACT", start_hi=0.50, end_hi=0.36, n_points=40)

    hr = make_health_result(timestamp=41.0, hi_smooth=0.35, engine_id="ENG_01", mission_id="MSN_EXACT")
    res = pipe.process_assessment(hr)

    assert res.status == RULStatus.CRITICAL_EOL_REACHED
    assert res.rul_seconds_median == 0.0
    assert res.rul_seconds_p05 == 0.0
    assert res.rul_seconds_p95 == 0.0
    assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"


# =========================================================================
# 2. FLOATING-POINT REPRESENTATION AROUND 0.35
# =========================================================================

@pytest.mark.parametrize("hi_val", [
    0.35000000000000003,  # Typical IEEE 754 precision artifact of 0.15*0.35 + 0.85*0.35
    0.35000000001,        # +1e-11 delta
    0.350000001,          # +1e-9 delta
    0.350001,            # +1e-6 delta (within 1e-5 tolerance)
    0.3500099,           # Near upper edge of tolerance
])
def test_floating_point_representation_around_035(hi_val):
    """Values at or effectively equal to 0.35 within tolerance must declare EOL."""
    pipe = RULPipeline()
    prime_pipeline(pipe, "ENG_01", f"MSN_FP_{hi_val}", start_hi=0.50, end_hi=0.36, n_points=40)

    hr = make_health_result(timestamp=41.0, hi_smooth=hi_val, engine_id="ENG_01", mission_id=f"MSN_FP_{hi_val}")
    res = pipe.process_assessment(hr)

    assert res.status == RULStatus.CRITICAL_EOL_REACHED
    assert res.rul_seconds_median == 0.0
    assert res.rul_seconds_p05 == 0.0
    assert res.rul_seconds_p95 == 0.0
    assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"


# =========================================================================
# 3. JUST BELOW BOUNDARY
# =========================================================================

@pytest.mark.parametrize("hi_val", [
    0.349999999,  # -1e-9 delta
    0.349999,     # -1e-6 delta
    0.349,        # -1e-3 delta
    0.30,         # Substantially below EOL
    0.0,          # Zero health
])
def test_just_below_boundary(hi_val):
    """Values below 0.35 must strictly declare CRITICAL_EOL_REACHED."""
    pipe = RULPipeline()
    prime_pipeline(pipe, "ENG_01", f"MSN_BELOW_{hi_val}", start_hi=0.50, end_hi=0.36, n_points=40)

    hr = make_health_result(timestamp=41.0, hi_smooth=hi_val, engine_id="ENG_01", mission_id=f"MSN_BELOW_{hi_val}")
    res = pipe.process_assessment(hr)

    assert res.status == RULStatus.CRITICAL_EOL_REACHED
    assert res.rul_seconds_median == 0.0
    assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"


# =========================================================================
# 4. JUST ABOVE BOUNDARY (ACTIVE RANGE)
# =========================================================================

@pytest.mark.parametrize("hi_val", [
    0.35002,  # Safely above 1e-5 tolerance boundary
    0.3501,   # +1e-4 delta
    0.351,    # +1e-3 delta
    0.36,     # +0.01 delta
    0.40,     # Degraded operational range
])
def test_just_above_boundary(hi_val):
    """Values legitimately above EOL tolerance must NOT declare EOL and must evaluate active degradation."""
    pipe = RULPipeline()
    prime_pipeline(pipe, "ENG_01", f"MSN_ABOVE_{hi_val}", start_hi=0.50, end_hi=0.36, n_points=40)

    hr = make_health_result(timestamp=41.0, hi_smooth=hi_val, engine_id="ENG_01", mission_id=f"MSN_ABOVE_{hi_val}")
    res = pipe.process_assessment(hr)

    assert res.status != RULStatus.CRITICAL_EOL_REACHED
    assert res.status in (RULStatus.ACTIVE_DEGRADATION, RULStatus.INDETERMINATE_TREND)
    if res.status == RULStatus.ACTIVE_DEGRADATION:
        assert res.rul_seconds_median is not None
        assert res.rul_seconds_median > 0.0


# =========================================================================
# 5. EWMA-GENERATED BOUNDARY VALUES
# =========================================================================

def test_ewma_generated_boundary_convergence():
    """Verify that when raw HI reaches 0.35, EWMA convergence declares EOL and CRITICAL health state."""
    smoother = CausalEWMASmoother(alpha=0.15)
    tracker = DegradationTracker(HealthIndexConfig())
    pipe = RULPipeline()

    # Prime degradation history from 0.50 down to 0.36 over 40 seconds
    for t in range(1, 41):
        r = 0.50 - (0.50 - 0.36) * (t / 40.0)
        sm = smoother.update(r, timestamp=float(t))
        tracker.update_and_evaluate(float(t), sm)
        hr = make_health_result(float(t), sm)
        pipe.process_assessment(hr)

    # Physical raw health reaches 0.35 at t = 41
    # Trace steps until smoothed HI reaches threshold tolerance
    eol_declared = False
    for step in range(1, 80):
        t = 40.0 + step
        sm = smoother.update(0.35, timestamp=t)
        _, _, state = tracker.update_and_evaluate(t, sm)
        hr = make_health_result(t, sm, health_state=state)
        res = pipe.process_assessment(hr)

        if sm <= (0.35 + DEFAULT_HI_EOL_TOLERANCE):
            assert res.status == RULStatus.CRITICAL_EOL_REACHED
            assert res.rul_seconds_median == 0.0
            assert state == HealthState.CRITICAL.value
            eol_declared = True

    assert eol_declared, "EWMA pipeline must reach and declare EOL"


# =========================================================================
# 6. RUL BLOCKING AT EOL
# =========================================================================

def test_rul_blocking_at_eol():
    """Verify that no positive RUL estimate or trajectory forecast leaks through when EOL is reached."""
    pipe = RULPipeline()
    prime_pipeline(pipe, "ENG_01", "MSN_BLOCK", start_hi=0.50, end_hi=0.36, n_points=40)

    # Test with boundary floating point value
    hr = make_health_result(timestamp=41.0, hi_smooth=0.35000000001, engine_id="ENG_01", mission_id="MSN_BLOCK")
    res = pipe.process_assessment(hr)

    assert res.status == RULStatus.CRITICAL_EOL_REACHED
    assert res.rul_seconds_median == 0.0
    assert res.rul_seconds_p05 == 0.0
    assert res.rul_seconds_p95 == 0.0
    assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"
    assert res.trajectory_type == "IMMEDIATE_BREACH"
    assert res.confidence_score == 1.0


# =========================================================================
# 7. DASHBOARD BEHAVIOR WHEN RUL IS BLOCKED AT EOL
# =========================================================================

def test_dashboard_behavior_when_rul_is_blocked():
    """Verify dashboard adapter properly maps EOL-blocked prognostic payload to CRITICAL status."""
    adapter = DashboardAdapter()

    # Payload reflecting system at EOL boundary
    payload = {
        "timestamp": 45.0,
        "engine_id": "ENG_01",
        "mission_id": "MSN_DASH_EOL",
        "mission_phase": "CRUISE",
        "health": {
            "smoothed_health_index": 0.35000000001,
            "raw_health_index": 0.35,
            "health_state": "CRITICAL",
            "degradation_rate": -0.001,
            "degradation_trend": "DEGRADING",
            "dominant_channels": ["cht"],
            "channel_contributions": {"cht": 0.5},
        },
        "prognostics": {
            "rul_seconds_median": 0.0,
            "rul_seconds_p05": 0.0,
            "rul_seconds_p95": 0.0,
            "status": "CRITICAL_EOL_REACHED",
            "limiting_factor": "GLOBAL_HEALTH_INDEX",
        },
        "advisory": {
            "action_code": "CRITICAL_ABORT_ACTION",
            "headline": "CRITICAL LIMIT REACHED / IMMEDIATE INTERVENTION ADVISED",
            "recommended_action": "Terminate operation immediately.",
            "urgency": "CRITICAL",
        },
    }

    vm = adapter.adapt(payload)

    assert vm.overview.overall_status == StatusLevel.CRITICAL
    assert vm.overview.health_card.status == StatusLevel.CRITICAL
    assert vm.overview.rul_card.status == StatusLevel.CRITICAL
    assert vm.prognostics.rul_state == "CRITICAL_EOL_REACHED"
    assert vm.prognostics.point_rul_seconds == 0.0
    assert vm.prognostics.limiting_factor == "GLOBAL_HEALTH_INDEX"


# =========================================================================
# 8. REPEATED EXECUTIONS & FRESH-PROCESS DETERMINISM
# =========================================================================

def test_repeated_executions_determinism():
    """Verify repeated executions across multiple fresh pipeline instances yield identical EOL decisions."""
    for trial in range(10):
        pipe = RULPipeline()
        prime_pipeline(pipe, "ENG_01", f"MSN_TRIAL_{trial}", start_hi=0.50, end_hi=0.36, n_points=40)
        hr = make_health_result(timestamp=41.0, hi_smooth=0.350000001, engine_id="ENG_01", mission_id=f"MSN_TRIAL_{trial}")
        res = pipe.process_assessment(hr)

        assert res.status == RULStatus.CRITICAL_EOL_REACHED
        assert res.rul_seconds_median == 0.0
        assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"
