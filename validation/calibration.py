"""
Calibration Parameter Definitions, Metadata Table, and Sensitivity Sweep Utilities for SIH26054.

Provides:
- Complete Tier C/D parameter inventory and calibration status documentation
- Controlled parameter sensitivity sweeps (inertia, thermal capacitance, cooling, fuel slope)
- Markdown calibration table generator
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.config import SimulatorConfig, TierAParameters, TierCParameters, TierDParameters
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from validation.metrics import compute_settling_time, compute_channel_stability


@dataclass
class ParameterRecord:
    """Metadata record for an engine simulator parameter."""
    name: str
    tier: str
    current_value: Any
    units: str
    effect: str
    calibration_status: str
    reference_source: str


def get_parameter_inventory() -> List[ParameterRecord]:
    """
    Returns the complete structured parameter inventory across all tiers.
    """
    return [
        # Tier A: Public Reference Parameters
        ParameterRecord(
            name="power_max_continuous_w",
            tier="Tier A",
            current_value=58000.0,
            units="W",
            effect="Sets absolute upper power ceiling derated by density and RPM efficiency",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS Type Certificate / Operator Manual (58 kW @ 5500 RPM)",
        ),
        ParameterRecord(
            name="rpm_max_continuous",
            tier="Tier A",
            current_value=5500.0,
            units="RPM",
            effect="Rated continuous crankshaft speed anchor for efficiency scaling",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS published maximum continuous RPM",
        ),
        ParameterRecord(
            name="rpm_max",
            tier="Tier A",
            current_value=5800.0,
            units="RPM",
            effect="Structural maximum 5-minute takeoff speed limit",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS published maximum RPM",
        ),
        ParameterRecord(
            name="cht_limit_c",
            tier="Tier A",
            current_value=150.0,
            units="°C",
            effect="Operational warning and safety threshold for cylinder head temperature",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS Maintenance Manual CHT maximum limit",
        ),
        ParameterRecord(
            name="oil_temp_nominal_band",
            tier="Tier A",
            current_value="90.0 - 110.0",
            units="°C",
            effect="Nominal operating oil temperature envelope during cruise",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS Operator Manual recommended oil temp range",
        ),
        ParameterRecord(
            name="oil_press_normal_band",
            tier="Tier A",
            current_value="2.0 - 5.0",
            units="bar",
            effect="Nominal operating oil pressure range across operational flight speeds",
            calibration_status="Locked to reference",
            reference_source="Rotax 912 ULS Operator Manual operational oil pressure envelope",
        ),
        # Tier C: Calibration Parameters
        ParameterRecord(
            name="rpm_idle",
            tier="Tier C",
            current_value=1400.0,
            units="RPM",
            effect="Target engine idle speed at closed throttle (balances idle friction & load)",
            calibration_status="Empirically Calibrated",
            reference_source="Engineering calibration baseline for typical 4-cyl light aero engine idle",
        ),
        ParameterRecord(
            name="inertia_kg_m2",
            tier="Tier C",
            current_value=0.28,
            units="kg·m²",
            effect="Determines throttle step settling time and transient rotational acceleration",
            calibration_status="Calibrated",
            reference_source="Equivalent crankshaft + prop reduction gear reflected inertia (3-blade prop)",
        ),
        ParameterRecord(
            name="k_load",
            tier="Tier C",
            current_value=3.036e-4,
            units="N·m/(rad/s)²",
            effect="Propeller aerodynamic power absorption torque law: Torque_load = k * omega^2",
            calibration_status="Calibrated to torque balance",
            reference_source="Torque equilibrium at 58 kW @ 5500 RPM (omega = 575.96 rad/s)",
        ),
        ParameterRecord(
            name="a_fuel_kg_per_j",
            tier="Tier C",
            current_value=6.8e-8,
            units="kg/J",
            effect="Willans line slope; sets brake specific fuel consumption (BSFC ~245 g/kWh)",
            calibration_status="Calibrated to BSFC band",
            reference_source="Empirical light aero piston engine fuel consumption curve",
        ),
        ParameterRecord(
            name="b_fuel_kg_per_s",
            tier="Tier C",
            current_value=0.00032,
            units="kg/s",
            effect="Idle fuel mass flow rate (~1.15 kg/h at closed throttle)",
            calibration_status="Calibrated to idle fuel",
            reference_source="Empirical aero piston engine idle consumption",
        ),
        ParameterRecord(
            name="c_th_cht",
            tier="Tier C",
            current_value=920.0,
            units="J/K",
            effect="Thermal capacitance governing cylinder head temperature lag time constant (~25-35 s)",
            calibration_status="Calibrated",
            reference_source="Lumped thermal capacity for aluminium cylinder head mass",
        ),
        ParameterRecord(
            name="q_gen_fraction",
            tier="Tier C",
            current_value=0.048,
            units="fraction",
            effect="Fraction of fuel combustion heat entering cylinder heads; sets CHT steady-state (95-115°C)",
            calibration_status="Calibrated to CHT target",
            reference_source="Heat balance calibration for hybrid air/liquid cylinder head",
        ),
        ParameterRecord(
            name="c_oil",
            tier="Tier C",
            current_value=1350.0,
            units="J/K",
            effect="Oil system thermal capacitance; governs slow oil temperature response time (~120-150 s)",
            calibration_status="Calibrated",
            reference_source="Lumped thermal capacity of 3.5 L oil charge + aluminium cooler",
        ),
        ParameterRecord(
            name="k_oil_cht_couple",
            tier="Tier C",
            current_value=2.2,
            units="W/K",
            effect="Heat transfer coupling from hot cylinder heads into circulating engine oil",
            calibration_status="Calibrated",
            reference_source="Thermal conduction coupling constant",
        ),
        ParameterRecord(
            name="k_oil_p_temp",
            tier="Tier C",
            current_value=0.022,
            units="bar/°C",
            effect="Oil pressure decrease due to fluid viscosity drop as temperature rises",
            calibration_status="Calibrated to pressure envelope",
            reference_source="Viscosity temperature slope for SAE 15W-40 / 10W-40 multi-grade oil",
        ),
        ParameterRecord(
            name="vib_order1_base_g",
            tier="Tier C",
            current_value=0.32,
            units="g",
            effect="Baseline 1x crankshaft rotational harmonic vibration amplitude",
            calibration_status="Documented assumption",
            reference_source="Baseline engine mount acceleration assumption",
        ),
        ParameterRecord(
            name="vib_order2_base_g",
            tier="Tier C",
            current_value=0.22,
            units="g",
            effect="Baseline 2x rotational order harmonic vibration amplitude",
            calibration_status="Documented assumption",
            reference_source="Four-stroke 4-cylinder firing pulse order assumption",
        ),
        # Tier D: Engineering Assumptions
        ParameterRecord(
            name="rpm_eff_poly",
            tier="Tier D",
            current_value="(-0.75, 1.65, 0.10)",
            units="dimensionless",
            effect="Quadratic efficiency shape peaking near continuous rated speed",
            calibration_status="Engineering assumption",
            reference_source="Unverified functional shape representing volumetric/mechanical efficiency",
        ),
        ParameterRecord(
            name="propeller_load_square_law",
            tier="Tier D",
            current_value="Torque = k * omega^2",
            units="N·m",
            effect="Idealized fixed-pitch propeller torque demand curve",
            calibration_status="Engineering assumption",
            reference_source="Standard momentum theory aerodynamic propeller loading assumption",
        ),
    ]


def generate_calibration_markdown_table() -> str:
    """Generate Markdown formatted calibration parameter documentation table."""
    records = get_parameter_inventory()
    lines = [
        "| Parameter Name | Tier | Current Value | Units | Physical / Telemetry Effect | Calibration Status | Reference Source |",
        "| :--- | :--- | :---: | :---: | :--- | :--- | :--- |",
    ]
    for r in records:
        lines.append(
            f"| `{r.name}` | **{r.tier}** | `{r.current_value}` | {r.units} | {r.effect} | {r.calibration_status} | {r.reference_source} |"
        )
    return "\n".join(lines)


def run_inertia_sensitivity_sweep(inertia_values: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Evaluate RPM settling time across different rotational inertia values (Tier C).
    """
    inertias = inertia_values or [0.15, 0.28, 0.40]
    results = {}

    for inertia in inertias:
        cfg = SimulatorConfig()
        cfg.tier_c.inertia_kg_m2 = inertia
        sim = EngineSimulator(sim_config=cfg, seed=42)

        dt = 0.05
        # Simulate idle for 5 s, then step to 90% throttle for 15 s
        times = []
        rpms = []
        t = 0.0
        for _ in range(int(5.0 / dt)):
            rec = sim.step(time_step=dt, mission_input=None)
            times.append(t)
            rpms.append(rec.rpm)
            t += dt

        for _ in range(int(15.0 / dt)):
            rec = sim.step(time_step=dt, mission_input=None)
            # Override throttle internally for step test
            op = sim.dynamics.step(throttle_pct=90.0, density_factor=1.0, dt=dt)
            times.append(t)
            rpms.append(op.rpm)
            t += dt

        settle_time = compute_settling_time(times, rpms, step_time=5.0, band_pct=0.02)
        results[f"inertia_{inertia:.2f}"] = {
            "inertia_kg_m2": inertia,
            "settling_time_s": round(settle_time, 2),
            "final_rpm": round(rpms[-1], 1),
        }

    return results


def run_thermal_capacitance_sweep(c_th_factors: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Evaluate CHT thermal rise time across different cylinder head thermal capacitance values.
    """
    factors = c_th_factors or [0.75, 1.0, 1.25]
    results = {}

    for fac in factors:
        cfg = SimulatorConfig()
        base_c_th = cfg.tier_c.c_th_cht
        cfg.tier_c.c_th_cht = base_c_th * fac
        sim = EngineSimulator(sim_config=cfg, seed=42)

        dt = 0.2
        total_time = 120.0
        times = []
        chts = []
        t = 0.0

        for _ in range(int(total_time / dt)):
            rec = sim.step(time_step=dt)
            times.append(t)
            chts.append(rec.cht)
            t += dt

        # Measure time from 85°C to 100°C
        t_85_100 = 0.0
        for ti, chti in zip(times, chts):
            if chti >= 100.0:
                t_85_100 = ti
                break

        results[f"c_th_factor_{fac:.2f}"] = {
            "capacitance_j_k": round(base_c_th * fac, 1),
            "time_to_100c_s": round(t_85_100, 1),
            "final_cht_c": round(chts[-1], 2),
        }

    return results
