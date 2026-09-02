"""
Pytest Test Suite for Phase 3 Validation Framework.

Automates regression testing for all Phase 3 validation gates:
- Physical envelope and bounds
- Directional monotonicity
- Dynamic lag transient hierarchy
- Empirical timestep stability sweep
- Channel-specific steady-state convergence
- Cross-channel coherence and correlations
- Multi-RPM vibration order harmonic tracking
- Representative 6-phase flight mission
- Golden baseline summary validity
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from validation.validation_runner import ValidationRunner
from validation.calibration import (
    get_parameter_inventory,
    run_inertia_sensitivity_sweep,
    run_thermal_capacitance_sweep,
)


@pytest.fixture(scope="module")
def validation_results():
    """Run all validation suites once for the test module."""
    runner = ValidationRunner(seed=42)
    return runner.run_all()


def test_suite_physical_bounds(validation_results):
    """Verify physical bounds pass across all operating conditions."""
    s1 = validation_results["detailed_results"]["physical_bounds"]
    assert s1["passed"] is True
    assert s1["conditions_tested"] >= 5


def test_suite_monotonicity(validation_results):
    """Verify monotonic relationships across throttle, altitude, RPM, and load."""
    s2 = validation_results["detailed_results"]["monotonicity"]
    assert s2["passed"] is True
    assert s2["tests"]["throttle_vs_power"]["pass_monotonic"] is True
    assert s2["tests"]["altitude_vs_density"]["pass_monotonic"] is True
    assert s2["tests"]["rpm_vs_oil_pressure"]["pass_monotonic"] is True


def test_suite_transient_hierarchy(validation_results):
    """Verify dynamic lag hierarchy: tau_RPM < tau_EGT < tau_CHT <= tau_Oil_Temp."""
    s3 = validation_results["detailed_results"]["transient_hierarchy"]
    assert s3["passed"] is True
    times = s3["settling_times_s"]
    assert times["rpm"] < times["egt"]
    assert times["egt"] < times["cht"]
    assert times["cht"] <= times["oil_temp"]


def test_suite_timestep_stability_sweep(validation_results):
    """Verify numerical stability across timesteps [0.05, 0.1, 0.2, 0.5, 1.0] s."""
    s4 = validation_results["detailed_results"]["timestep_stability_sweep"]
    assert s4["passed"] is True
    assert s4["recommended_dt_s"] == 0.1
    for dt_key, dt_res in s4["results"].items():
        assert dt_res["stable"] is True
        assert dt_res["has_nan"] is False
        assert dt_res["has_inf"] is False


def test_suite_steady_state_convergence(validation_results):
    """Verify channel-specific steady-state convergence criteria."""
    s5 = validation_results["detailed_results"]["steady_state_convergence"]
    assert s5["passed"] is True
    channels = s5["channels"]
    assert channels["rpm"]["passed"] is True
    assert channels["cht"]["passed"] is True
    assert channels["egt"]["passed"] is True
    assert channels["oil_temp"]["passed"] is True
    assert channels["oil_pressure"]["passed"] is True
    assert channels["fuel_flow"]["passed"] is True
    assert channels["vibration"]["passed"] is True


def test_suite_cross_channel_coherence(validation_results):
    """Verify physical cross-channel correlations."""
    s6 = validation_results["detailed_results"]["cross_channel_coherence"]
    assert s6["passed"] is True
    corrs = s6["correlations"]
    assert corrs["throttle_vs_rpm"] > 0.85
    assert corrs["throttle_vs_fuel_flow"] > 0.85
    assert corrs["rpm_vs_oil_pressure"] > 0.75
    assert corrs["fuel_flow_vs_cht"] > 0.50


def test_suite_vibration_orders(validation_results):
    """Verify 1x and 2x crankshaft order harmonic peak tracking via FFT across RPM points."""
    s7 = validation_results["detailed_results"]["vibration_orders"]
    assert s7["passed"] is True
    for rpm_key, rpm_res in s7["operating_points"].items():
        assert rpm_res["orders"]["order_1.0x"]["pass"] is True
        assert rpm_res["orders"]["order_2.0x"]["pass"] is True
        assert rpm_res["orders"]["order_1.0x"]["error_pct"] < 4.0
        assert rpm_res["orders"]["order_2.0x"]["error_pct"] < 4.0


def test_suite_representative_mission(validation_results):
    """Verify representative 6-phase flight mission execution and continuity."""
    s8 = validation_results["detailed_results"]["representative_mission"]
    assert s8["passed"] is True
    assert s8["time_continuous"] is True
    assert len(s8["phases_covered"]) == 6


def test_golden_baseline_file():
    """Verify golden baseline summary file exists and contains valid quantiles."""
    path = Path("data") / "golden_baseline_summary.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["golden_baseline_version"] == "1.0.0-phase3"
    assert data["provenance"]["seed"] == 42
    assert data["provenance"]["dt_s"] == 0.1
    assert "channel_quantiles" in data
    assert "rpm" in data["channel_quantiles"]
    assert data["channel_quantiles"]["rpm"]["max"] > 5000.0


def test_calibration_parameter_inventory_and_sweeps():
    """Verify parameter inventory definitions and sensitivity sweeps."""
    inventory = get_parameter_inventory()
    assert len(inventory) >= 15
    tiers = {p.tier for p in inventory}
    assert "Tier A" in tiers
    assert "Tier C" in tiers
    assert "Tier D" in tiers

    # Inertia sensitivity sweep
    inertia_res = run_inertia_sensitivity_sweep([0.20, 0.35])
    assert len(inertia_res) == 2

    # Thermal capacitance sweep
    thermal_res = run_thermal_capacitance_sweep([0.8, 1.2])
    assert len(thermal_res) == 2
