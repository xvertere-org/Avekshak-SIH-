"""
Phase 4: Health Monitoring, Fault Injection, Uncertainty Quantification, and Prognostics Script.

Executes and evaluates the 9 controlled aero-piston benchmark scenarios:
1. Nominal healthy cruise
2. Gradual cooling degradation
3. Abrupt lubrication degradation
4. Thermal load increase
5. CHT sensor drift
6. RPM sensor bias
7. Intermittent oil-pressure sensor dropout
8. Transient throttle disturbance
9. High-altitude/high-load cruise

Generates:
- reports/phase4_health_prognostics_report.json
- reports/phase4_health_prognostics_report.md
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import FaultState, FaultType, FaultSubsystem
from telemetry.ingestion import TelemetryIngestor
from ml.tasks.residual_correction import (
    GreyBoxResidualPipeline,
    ResidualCorrectionConfig,
    CANONICAL_TARGET_CHANNELS,
    TARGET_TO_PHYSICS_MAP,
)
from ml.tasks.health_prognostics import (
    EngineHealthState,
    AlertClassification,
    DegradationTrendState,
    HealthMonitoringOutput,
    HealthPrognosticsPipeline,
    get_standard_fault_catalog,
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def _generate_telemetry(
    phase: FlightPhase,
    duration_s: float,
    throttle_start: float,
    throttle_end: float,
    alt_start: float,
    alt_end: float,
    fault_schedule: Optional[List[FaultState]] = None,
    seed: int = 42,
    dt: float = 0.2,
) -> pd.DataFrame:
    """Generate raw simulation telemetry."""
    sim = EngineSimulator(seed=seed)
    seg = PhaseSegment(
        phase=phase,
        duration_s=duration_s,
        throttle_start_pct=throttle_start,
        throttle_end_pct=throttle_end,
        altitude_start_m=alt_start,
        altitude_end_m=alt_end,
    )
    profile = MissionProfile(segments=[seg])
    records = sim.run(profile, dt=dt, fault_schedule=fault_schedule)
    frame = TelemetryIngestor.ingest(records)
    df = frame.to_dataframe()
    df["timestamp"] = np.arange(len(df)) * dt
    return df


def build_and_evaluate_scenarios():
    print("=================================================================")
    print("SIH26054 Phase 4: Health Monitoring & Prognostics Benchmarking")
    print("=================================================================")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dt = 0.2

    # -------------------------------------------------------------
    # 1. Fit Grey-Box Digital Twin Residual Pipeline (Training data only)
    # -------------------------------------------------------------
    print("\n[Step 1] Fitting Grey-Box Pipeline on nominal training flights...")
    df_train_cruise = _generate_telemetry(
        phase=FlightPhase.CRUISE,
        duration_s=50.0,
        throttle_start=75.0,
        throttle_end=75.0,
        alt_start=2000.0,
        alt_end=2000.0,
        seed=101,
        dt=dt,
    )
    df_train_climb = _generate_telemetry(
        phase=FlightPhase.CLIMB,
        duration_s=50.0,
        throttle_start=65.0,
        throttle_end=85.0,
        alt_start=1500.0,
        alt_end=3000.0,
        seed=102,
        dt=dt,
    )
    train_df = pd.concat([df_train_cruise, df_train_climb], ignore_index=True)
    train_df["timestamp"] = np.arange(len(train_df)) * dt

    gb_config = ResidualCorrectionConfig(
        targets=CANONICAL_TARGET_CHANNELS,
        model_type="ridge",
        ridge_alpha=1.0,
        clipping_safety=True,
        default_dt=dt,
    )
    gb_pipeline = GreyBoxResidualPipeline(config=gb_config)
    gb_pipeline.fit(train_df, dt=dt)

    # Calibrate baseline scales for HealthPrognosticsPipeline strictly from training residuals
    train_clean = gb_pipeline.validate_telemetry_dataframe(train_df)
    train_physics = gb_pipeline.generate_physics_predictions(train_clean, dt=dt)
    train_residuals = pd.DataFrame(index=train_clean.index)
    for ch in CANONICAL_TARGET_CHANNELS:
        phys_col = TARGET_TO_PHYSICS_MAP.get(ch)
        if phys_col and phys_col in train_physics.columns:
            train_residuals[ch] = train_clean[ch] - train_physics[phys_col]

    base_health_pipeline = HealthPrognosticsPipeline(
        persistence_steps=5,
        hysteresis_recovery_steps=4,
        z_score_threshold=2.5,
        recovery_z_threshold=1.2,
        ewma_alpha=0.20,
        eol_threshold=0.35,
    )
    base_health_pipeline.calibrate_baseline_scales(train_residuals)
    print("Baseline residual dispersion scales calibrated strictly on training partition.")

    # -------------------------------------------------------------
    # 2. Define the 9 Controlled Evaluation Scenarios
    # -------------------------------------------------------------
    print("\n[Step 2] Executing 9 controlled evaluation scenarios...")
    catalog = get_standard_fault_catalog()

    scenario_definitions = [
        {
            "id": "scenario_1",
            "name": "Nominal healthy cruise",
            "type": "nominal",
            "description": "Unperturbed cruise condition at 75% throttle and 2000m altitude. Demonstrates false-alarm suppression and RUL withholding.",
            "duration": 50.0,
            "fault_onset": None,
            "fault_obj": None,
            "sensor_mutation": None,
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.HEALTHY,
            "expected_alert": AlertClassification.NOMINAL,
            "is_sensor_fault": False,
        },
        {
            "id": "scenario_2",
            "name": "Gradual cooling degradation",
            "type": "physical_degradation",
            "description": "Progressive coolant heat exchanger fouling (40% conductance loss) starting at t=20s. Evaluates thermal lag detection and time-to-threshold.",
            "duration": 60.0,
            "fault_onset": 20.0,
            "fault_obj": FaultState(
                fault_type=FaultType.COOLING_DEGRADATION,
                affected_subsystem=FaultSubsystem.COOLING,
                start_time=20.0,
                end_time=60.0,
                severity=0.40,
            ),
            "sensor_mutation": None,
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.COOLING_DEGRADATION,
            "expected_alert": AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION,
            "is_sensor_fault": False,
        },
        {
            "id": "scenario_3",
            "name": "Abrupt lubrication degradation",
            "type": "physical_degradation",
            "description": "Sudden oil pump pressure relief valve sticking (35% pressure loss) at t=15s. Evaluates step-fault detection delay and rapid decline.",
            "duration": 50.0,
            "fault_onset": 15.0,
            "fault_obj": FaultState(
                fault_type=FaultType.LUBRICATION_DEGRADATION,
                affected_subsystem=FaultSubsystem.LUBRICATION,
                start_time=15.0,
                end_time=50.0,
                severity=0.35,
            ),
            "sensor_mutation": None,
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.LUBRICATION_DEGRADATION,
            "expected_alert": AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION,
            "is_sensor_fault": False,
        },
        {
            "id": "scenario_4",
            "name": "Thermal load increase",
            "type": "physical_degradation",
            "description": "Multi-cylinder thermal stress from high sustained climb load at high ambient temperature. Evaluates thermal subsystem scoring.",
            "duration": 50.0,
            "fault_onset": 10.0,
            "fault_obj": FaultState(
                fault_type=FaultType.COOLING_DEGRADATION,
                affected_subsystem=FaultSubsystem.COOLING,
                start_time=10.0,
                end_time=50.0,
                severity=0.30,
            ),
            "sensor_mutation": None,
            "throttle_start": 82.0,
            "throttle_end": 85.0,
            "alt_start": 2200.0,
            "alt_end": 2200.0,
            "expected_state": EngineHealthState.COOLING_DEGRADATION,
            "expected_alert": AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION,
            "is_sensor_fault": False,
        },
        {
            "id": "scenario_5",
            "name": "CHT sensor drift",
            "type": "sensor_fault",
            "description": "Unilateral +0.8 °C/s thermocouple drift starting at t=10s. Other thermal channels (coolant, oil) remain nominal. Evaluates cross-channel discriminator.",
            "duration": 50.0,
            "fault_onset": 10.0,
            "fault_obj": None,
            "sensor_mutation": ("drift", "cht", 10.0, 0.8),
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.SENSOR_ANOMALY,
            "expected_alert": AlertClassification.SENSOR_ANOMALY,
            "is_sensor_fault": True,
        },
        {
            "id": "scenario_6",
            "name": "RPM sensor bias",
            "type": "sensor_fault",
            "description": "Abrupt +150 RPM pickup bias offset at t=15s. MAP and fuel flow remain dynamic and uncorrupted. Discriminator isolates single-channel sensor bias.",
            "duration": 50.0,
            "fault_onset": 15.0,
            "fault_obj": None,
            "sensor_mutation": ("bias", "rpm", 15.0, 150.0),
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.SENSOR_ANOMALY,
            "expected_alert": AlertClassification.SENSOR_ANOMALY,
            "is_sensor_fault": True,
        },
        {
            "id": "scenario_7",
            "name": "Intermittent oil-pressure sensor dropout",
            "type": "sensor_fault",
            "description": "Sensor signal dropout (NaN injection) between t=20s and t=35s. Physical lubrication loop remains healthy. Evaluates missing-data handling.",
            "duration": 50.0,
            "fault_onset": 20.0,
            "fault_obj": None,
            "sensor_mutation": ("dropout", "oil_pressure", 20.0, 35.0),
            "throttle_start": 75.0,
            "throttle_end": 75.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.SENSOR_ANOMALY,
            "expected_alert": AlertClassification.SENSOR_ANOMALY,
            "is_sensor_fault": True,
        },
        {
            "id": "scenario_8",
            "name": "Transient throttle disturbance",
            "type": "transient_disturbance",
            "description": "5-second throttle pulse (+15% throttle step from t=15s to t=20s), returning to cruise. Evaluates persistence counter and hysteresis recovery.",
            "duration": 50.0,
            "fault_onset": 15.0,
            "fault_obj": None,
            "sensor_mutation": None,
            "throttle_start": 70.0,
            "throttle_end": 70.0,
            "alt_start": 2000.0,
            "alt_end": 2000.0,
            "expected_state": EngineHealthState.HEALTHY,
            "expected_alert": AlertClassification.NOMINAL,
            "is_sensor_fault": False,
            "transient_pulse": (15.0, 20.0, 85.0),
        },
        {
            "id": "scenario_9",
            "name": "High-altitude/high-load cruise",
            "type": "nominal_stress",
            "description": "Sustained high cruise load (85% throttle) at 3500m altitude. Confirms physics baseline accommodates legitimate operational workload without false alarms.",
            "duration": 50.0,
            "fault_onset": None,
            "fault_obj": None,
            "sensor_mutation": None,
            "throttle_start": 85.0,
            "throttle_end": 85.0,
            "alt_start": 3500.0,
            "alt_end": 3500.0,
            "expected_state": EngineHealthState.HEALTHY,
            "expected_alert": AlertClassification.NOMINAL,
            "is_sensor_fault": False,
        },
    ]

    scenario_results: List[Dict[str, Any]] = []

    for sc in scenario_definitions:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"\n  Evaluating {sc_id}: {sc_name}...")

        # 1. Generate base telemetry
        fault_sched = [sc["fault_obj"]] if sc["fault_obj"] is not None else None
        sim_df = _generate_telemetry(
            phase=FlightPhase.CRUISE,
            duration_s=sc["duration"],
            throttle_start=sc["throttle_start"],
            throttle_end=sc["throttle_end"],
            alt_start=sc["alt_start"],
            alt_end=sc["alt_end"],
            fault_schedule=fault_sched,
            seed=300 + int(sc_id.split("_")[1]),
            dt=dt,
        )

        # Handle transient throttle pulse if configured
        if "transient_pulse" in sc and sc["transient_pulse"] is not None:
            p_start, p_end, p_thr = sc["transient_pulse"]
            pulse_mask = (sim_df["timestamp"] >= p_start) & (sim_df["timestamp"] <= p_end)
            sim_df.loc[pulse_mask, "throttle"] = p_thr
            # Slight transient RPM / MAP dynamic response
            sim_df.loc[pulse_mask, "rpm"] += 200.0
            sim_df.loc[pulse_mask, "map_bar"] += 0.12

        # 2. Apply observation-layer sensor mutations
        obs_df = sim_df.copy()
        if sc["sensor_mutation"] is not None:
            m_type, m_channel, m_start, m_param = sc["sensor_mutation"]
            if m_type == "drift":
                drift_mask = obs_df["timestamp"] >= m_start
                dt_drift = obs_df.loc[drift_mask, "timestamp"] - m_start
                obs_df.loc[drift_mask, m_channel] += dt_drift * m_param
            elif m_type == "bias":
                bias_mask = obs_df["timestamp"] >= m_start
                obs_df.loc[bias_mask, m_channel] += m_param
            elif m_type == "dropout":
                m_end = m_param
                drop_mask = (obs_df["timestamp"] >= m_start) & (obs_df["timestamp"] <= m_end)
                obs_df.loc[drop_mask, m_channel] = np.nan

        # 3. Generate physics and grey-box predictions
        preds = gb_pipeline.predict(obs_df, dt=dt)

        # 4. Run step-by-step causal processing through fresh pipeline instance
        pipeline = HealthPrognosticsPipeline(
            persistence_steps=5,
            hysteresis_recovery_steps=4,
            z_score_threshold=2.5,
            recovery_z_threshold=1.2,
            ewma_alpha=0.20,
            eol_threshold=0.35,
        )
        pipeline.calibrate_baseline_scales(train_residuals)

        step_outputs: List[HealthMonitoringOutput] = []
        detection_time: Optional[float] = None
        false_alarms = 0
        missed_detections = 0
        fault_onset = sc["fault_onset"]

        for idx in range(len(obs_df)):
            t = float(obs_df.loc[idx, "timestamp"])
            obs_dict = {ch: float(obs_df.loc[idx, ch]) if pd.notna(obs_df.loc[idx, ch]) else None for ch in CANONICAL_TARGET_CHANNELS}
            phys_dict = {f"{ch}_expected": float(preds["physics"][ch][idx]) for ch in CANONICAL_TARGET_CHANNELS}
            corr_dict = {ch: float(preds["corrected"][ch][idx]) for ch in CANONICAL_TARGET_CHANNELS}

            out = pipeline.process_step(
                timestamp=t,
                observed=obs_dict,
                physics_expected=phys_dict,
                corrected_prediction=corr_dict,
            )
            step_outputs.append(out)

            is_alarm = (out.alert_classification != AlertClassification.NOMINAL)

            # False alarm tracking (alarms before fault onset or on nominal flights)
            if fault_onset is None:
                if is_alarm:
                    false_alarms += 1
            else:
                if t < fault_onset and is_alarm:
                    false_alarms += 1
                elif t >= fault_onset and is_alarm and detection_time is None:
                    detection_time = t

        # Detection metrics
        if fault_onset is not None:
            if detection_time is not None:
                detection_delay_s = round(detection_time - fault_onset, 2)
                detected = True
            else:
                detection_delay_s = None
                detected = False
                missed_detections = 1
        else:
            detection_delay_s = None
            detected = False

        final_out = step_outputs[-1]
        mid_out = step_outputs[len(step_outputs) // 2]
        unc_widths = [
            u.uncertainty_width for u in final_out.channel_uncertainties.values()
        ]
        mean_unc_width = float(np.mean(unc_widths)) if unc_widths else 0.0

        res_entry = {
            "scenario_id": sc_id,
            "name": sc_name,
            "scenario_type": sc["type"],
            "description": sc["description"],
            "is_sensor_fault": sc["is_sensor_fault"],
            "fault_onset_s": fault_onset,
            "detected": detected,
            "detection_delay_s": detection_delay_s,
            "false_alarms": false_alarms,
            "missed_detections": missed_detections,
            "initial_health_score": round(step_outputs[0].health_score, 4),
            "mid_health_score": round(mid_out.health_score, 4),
            "final_health_score": round(final_out.health_score, 4),
            "final_health_state": final_out.health_state.value,
            "final_alert_classification": final_out.alert_classification.value,
            "final_degradation_severity": round(final_out.degradation_severity, 4),
            "affected_channels": final_out.affected_channels,
            "affected_subsystems": final_out.affected_subsystems,
            "final_trend_state": final_out.trend_state.value,
            "trend_slope_per_s": final_out.trend_slope_per_s,
            "rul_state": final_out.rul_state,
            "time_to_threshold_s": final_out.time_to_threshold_s,
            "prognostics_reason": final_out.prognostics_reason,
            "mean_uncertainty_width": round(mean_unc_width, 3),
            "data_quality_status": final_out.data_quality_status,
        }
        scenario_results.append(res_entry)

        print(f"    Detection: {detected} (Delay: {detection_delay_s}s) | False alarms: {false_alarms}")
        print(f"    Alert: {final_out.alert_classification.value} | State: {final_out.health_state.value}")
        print(f"    HI: {final_out.health_score:.3f} | Trend: {final_out.trend_state.value} | RUL: {final_out.rul_state} ({final_out.time_to_threshold_s}s)")

    # -------------------------------------------------------------
    # 3. Compile Reports
    # -------------------------------------------------------------
    report_json_path = REPORTS_DIR / "phase4_health_prognostics_report.json"
    report_md_path = REPORTS_DIR / "phase4_health_prognostics_report.md"

    summary_stats = {
        "total_scenarios_evaluated": len(scenario_results),
        "fault_scenarios_count": sum(1 for s in scenario_results if s["fault_onset_s"] is not None),
        "nominal_scenarios_count": sum(1 for s in scenario_results if s["fault_onset_s"] is None),
        "faults_detected": sum(1 for s in scenario_results if s["detected"]),
        "false_alarm_events": sum(s["false_alarms"] for s in scenario_results),
        "missed_detection_events": sum(s["missed_detections"] for s in scenario_results),
        "rul_withheld_scenarios": sum(1 for s in scenario_results if s["rul_state"] == "UNAVAILABLE"),
        "rul_computed_scenarios": sum(1 for s in scenario_results if s["rul_state"] == "AVAILABLE"),
    }

    full_report_data = {
        "phase": 4,
        "title": "SIH26054 Phase 4: Health Monitoring, Fault Injection, Uncertainty, and Prognostics Report",
        "domain_qualification": {
            "telemetry_source": "Controlled physics-based aero-piston digital twin simulation",
            "certification_status": "Technical demonstration benchmark only. Zero claims of complete Rotax 914 OEM validation, FAA certification, or certified flight-test validation.",
            "health_score_meaning": "Bounded engineering health indicator [0.0, 1.0] where 1.0 represents nominal operational baseline and 0.0 represents functional critical threshold (HI <= 0.35).",
            "uncertainty_label": "Engineering uncertainty estimates derived from rolling residual dispersion. Not statistically calibrated Bayesian or frequentist coverage intervals.",
        },
        "summary": summary_stats,
        "scenarios": scenario_results,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"\nSaved structured JSON report to: {report_json_path}")

    # Generate Markdown Report
    lines = [
        "# SIH26054 Phase 4: Health Monitoring, Fault Injection, Uncertainty Quantification, and Prognostics Report",
        "",
        "> [!IMPORTANT]",
        "> **Operational Scope & Domain Disclaimer**: All telemetry evaluated herein is generated from controlled physics-based simulation benchmarks. External prognostic datasets (Paderborn, CWRU, FEMTO, C-MAPSS) serve as auxiliary component benchmarks and do not validate the complete Rotax 914 aero-piston engine. No claims of OEM validation, FAA airworthiness certification, or real-world maintenance directive generation are made.",
        "",
        "## Executive Summary",
        "",
        f"- **Scenarios Evaluated**: {summary_stats['total_scenarios_evaluated']} controlled benchmark scenarios (6 fault/degradation scenarios, 3 nominal/transient stress scenarios).",
        f"- **Fault Detection Rate**: {summary_stats['faults_detected']} / {summary_stats['fault_scenarios_count']} detected ({summary_stats['faults_detected'] / max(1, summary_stats['fault_scenarios_count']) * 100:.1f}%).",
        f"- **False Alarms**: {summary_stats['false_alarm_events']} across all nominal and pre-fault intervals (persistence counters prevent single-point false alarms).",
        f"- **RUL Withheld Count**: {summary_stats['rul_withheld_scenarios']} / {summary_stats['total_scenarios_evaluated']} scenarios (RUL returned as `UNAVAILABLE` when degradation trend is absent or unsupported).",
        "",
        "## Benchmark Scenario Results",
        "",
        "| Scenario | Type | Detection | Delay (s) | False Alarms | Alert Class | Health State | Final HI | Trend | RUL State | RUL (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :--- | :---: | :---: |",
    ]

    for s in scenario_results:
        delay_str = f"{s['detection_delay_s']:.1f}" if s['detection_delay_s'] is not None else "N/A"
        rul_val = f"{s['time_to_threshold_s']:.1f}" if s['time_to_threshold_s'] is not None else s['prognostics_reason'][:25] + "..." if s['prognostics_reason'] else "N/A"
        lines.append(
            f"| **{s['name']}** | `{s['scenario_type']}` | {'✅ Yes' if s['detected'] else '—'} | {delay_str} | {s['false_alarms']} | `{s['final_alert_classification']}` | `{s['final_health_state']}` | {s['final_health_score']:.3f} | `{s['final_trend_state']}` | `{s['rul_state']}` | {rul_val} |"
        )

    lines.extend([
        "",
        "## Scenario Details and Discriminator Logic",
        "",
    ])

    for s in scenario_results:
        lines.extend([
            f"### {s['scenario_id'].upper()}: {s['name']}",
            f"- **Scenario Description**: {s['description']}",
            f"- **Classification**: `{s['final_alert_classification']}` (Health State: `{s['final_health_state']}`)",
            f"- **Affected Subsystems**: `{', '.join(s['affected_subsystems']) if s['affected_subsystems'] else 'None'}`",
            f"- **Affected Channels**: `{', '.join(s['affected_channels']) if s['affected_channels'] else 'None'}`",
            f"- **Health Indicator**: Initial: `{s['initial_health_score']}` -> Mid: `{s['mid_health_score']}` -> Final: `{s['final_health_score']}`",
            f"- **Degradation Trend**: `{s['final_trend_state']}` (Slope: `{s['trend_slope_per_s']} s⁻¹`)",
            f"- **RUL Determination**: State = `{s['rul_state']}` | Value = `{s['time_to_threshold_s']} s` | Reason: *{s['prognostics_reason']}*",
            f"- **Engineering Uncertainty**: Mean prediction interval width = `±{s['mean_uncertainty_width'] / 2.0:.2f}` units",
            f"- **Data Quality Status**: `{s['data_quality_status']}`",
            "",
        ])

    lines.extend([
        "## Methodological Foundation & Threshold Origins",
        "",
        "1. **Health Index Bounds**: Engineered to $[0.0, 1.0]$ using a sigmoidal transfer function mapping multi-channel z-score excursions to degradation severity. Critical threshold defined at $HI \\le 0.35$.",
        "2. **Sensor vs. Physical Discrimination**: Single-channel excursions with uncorrelated companion channels are categorized as `SENSOR_ANOMALY` or `MODEL_DISAGREEMENT`. Correlated deviations across thermodynamically linked channels (e.g., CHT + Coolant Temp) are qualified as `POSSIBLE_PHYSICAL_DEGRADATION`.",
        "3. **Persistence and Hysteresis**: Detection requires $N_{\\text{persist}} = 5$ consecutive timesteps exceeding $z = 2.5$. Recovery requires $N_{\\text{recov}} = 4$ consecutive timesteps below $z = 1.2$, preventing alarm chatter in noise deadbands.",
        "4. **Engineering Uncertainty Estimates**: Computed from rolling residual standard deviations and nominal model dispersion strictly calibrated on training partitions. Labeled transparently as engineering uncertainty estimates.",
        "5. **Causal Prognostics Policy**: Slope is estimated strictly over causal history ($W = 50$ steps) via Theil-Sen robust regression. If slope $\\ge -1\\times 10^{-4}\\text{ s}^{-1}$ or degradation is absent, RUL is withheld as `UNAVAILABLE` with an explicit reason string.",
        "",
        "## Reproduction Commands",
        "```bash",
        "python scripts/generate_phase4_health_prognostics_results.py",
        "pytest -v tests/test_phase4_health_prognostics.py",
        "```",
    ])

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved human-readable Markdown report to: {report_md_path}")
    print("\nPhase 4 evaluation script completed successfully.")


if __name__ == "__main__":
    build_and_evaluate_scenarios()
