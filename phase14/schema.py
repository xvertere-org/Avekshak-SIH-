"""
Schema contracts for Phase 14 — Mission Replay, Reporting & What-If Analysis.
SIH26054 / AVEKSHAK Digital Twin.

Strict architectural rules:
- Consumes authoritative Phase 13 DashboardStatePayload without duplicating PHM logic.
- Maintains (engine_id, mission_id) state boundary isolation.
- Enforces non-fabrication and clear disclaimers for synthetic simulation decision-support.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from orchestrator.schema import DashboardStatePayload, SimulationScenario


@dataclass
class MissionReplaySession:
    """
    Encapsulates a completed synthetic mission payload sequence for chronological replay.
    Replay consumes existing Phase 13 outputs and does NOT recompute PHM logic.
    """
    engine_id: str
    mission_id: str
    scenario_name: str
    payloads: List[DashboardStatePayload]
    total_duration_s: float
    total_steps: int
    dt: float = 1.0
    created_at: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.payloads:
            self.total_steps = 0
            self.total_duration_s = 0.0
        else:
            self.total_steps = len(self.payloads)
            self.total_duration_s = max(0.0, self.payloads[-1].timestamp - self.payloads[0].timestamp)

    def get_step(self, step_idx: int) -> DashboardStatePayload:
        """Retrieve payload at a specific chronological index."""
        if not self.payloads:
            raise ValueError(f"Replay session for engine '{self.engine_id}' mission '{self.mission_id}' is empty.")
        clamped_idx = max(0, min(step_idx, len(self.payloads) - 1))
        return self.payloads[clamped_idx]

    def get_history(self, up_to_step: int) -> Dict[str, Any]:
        """Extract multi-channel history up to the designated replay step for synchronized charts."""
        if not self.payloads:
            return {"timestamps": [], "health_index": {"timestamps": [], "hi": []}}

        sub = self.payloads[: max(1, min(up_to_step + 1, len(self.payloads)))]
        channels = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
        history: Dict[str, Any] = {
            "timestamps": [p.timestamp for p in sub],
            "health_index": {
                "timestamps": [p.timestamp for p in sub],
                "hi": [p.smoothed_health_index if p.smoothed_health_index is not None and not np.isnan(p.smoothed_health_index) else float("nan") for p in sub],
            },
        }
        for ch in channels:
            history[ch] = [p.observed_telemetry.get(ch, float("nan")) for p in sub]
        return history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "scenario_name": self.scenario_name,
            "total_steps": self.total_steps,
            "total_duration_s": self.total_duration_s,
            "dt": self.dt,
            "provenance": self.provenance,
        }


@dataclass
class MissionReportSummary:
    """
    Concise engineering-style mission report generated from an executed mission.
    Adheres to strict safety boundaries and decision-support phrasing.
    """
    # MISSION
    engine_id: str
    mission_id: str
    duration_s: float
    mission_phase: str
    simulation_mode: str
    scenario_name: str
    scenario_metadata: Dict[str, Any]

    # TELEMETRY EXTREMES & STATS
    rpm_min: float
    rpm_max: float
    rpm_mean: float
    cht_peak: float
    cht_mean: float
    egt_peak: float
    egt_mean: float
    oil_temp_peak: float
    oil_temp_mean: float
    oil_pressure_min: float
    oil_pressure_mean: float
    vibration_max: float
    vibration_mean: float
    fuel_flow_total: float
    fuel_flow_mean: float

    # PHM SUMMARY
    anomaly_events_count: int
    dominant_anomaly_channels: List[str]
    final_diagnosis: str
    final_diagnosis_probability: float
    evidence_quality: str  # e.g. HIGH, MEDIUM, LOW, VALID
    final_health_index: float
    min_health_index: float
    degradation_trend: str
    final_rul_seconds: Optional[float]
    rul_p05_seconds: Optional[float]
    rul_p95_seconds: Optional[float]
    limiting_factor: str
    forecast_status: str
    forecast_source: str

    # EXPLAINABILITY
    dominant_shap_evidence: List[Dict[str, Any]]
    physics_evidence_status: str
    physics_consistency_reason: str
    temporal_evidence_status: str
    fused_evidence_headline: str
    recommended_operator_action: str

    # DECISION SUPPORT
    advisory_action_code: str
    advisory_headline: str
    advisory_urgency: str
    advisory_assessment: str  # GO, CAUTION, or MAINTENANCE

    # PROVENANCE & LIMITATIONS
    provenance: Dict[str, Any]
    disclaimer: str = (
        "Advisory decision-support summary based on synthetic simulation telemetry and project-defined "
        "functional-failure/EOL assumptions. Not certified airworthiness limits, flight validation, or OEM directive."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": {
                "engine_id": self.engine_id,
                "mission_id": self.mission_id,
                "duration_s": round(self.duration_s, 2),
                "mission_phase": self.mission_phase,
                "simulation_mode": self.simulation_mode,
                "scenario_name": self.scenario_name,
                "scenario_metadata": self.scenario_metadata,
            },
            "telemetry_statistics": {
                "rpm": {"min": round(self.rpm_min, 1), "max": round(self.rpm_max, 1), "mean": round(self.rpm_mean, 1)},
                "cht_deg_c": {"peak": round(self.cht_peak, 2), "mean": round(self.cht_mean, 2)},
                "egt_deg_c": {"peak": round(self.egt_peak, 2), "mean": round(self.egt_mean, 2)},
                "oil_temp_deg_c": {"peak": round(self.oil_temp_peak, 2), "mean": round(self.oil_temp_mean, 2)},
                "oil_pressure_bar": {"min": round(self.oil_pressure_min, 2), "mean": round(self.oil_pressure_mean, 2)},
                "vibration_g": {"max": round(self.vibration_max, 3), "mean": round(self.vibration_mean, 3)},
                "fuel_flow_l_h": {"total_integrated": round(self.fuel_flow_total, 2), "mean": round(self.fuel_flow_mean, 2)},
            },
            "phm_summary": {
                "anomaly_events_count": self.anomaly_events_count,
                "dominant_anomaly_channels": self.dominant_anomaly_channels,
                "final_diagnosis": self.final_diagnosis,
                "final_diagnosis_probability": round(self.final_diagnosis_probability, 4),
                "evidence_quality": self.evidence_quality,
                "final_health_index": round(self.final_health_index, 4) if not np.isnan(self.final_health_index) else None,
                "min_health_index": round(self.min_health_index, 4) if not np.isnan(self.min_health_index) else None,
                "degradation_trend": self.degradation_trend,
                "final_rul_seconds": round(self.final_rul_seconds, 1) if self.final_rul_seconds is not None else None,
                "rul_p05_seconds": round(self.rul_p05_seconds, 1) if self.rul_p05_seconds is not None else None,
                "rul_p95_seconds": round(self.rul_p95_seconds, 1) if self.rul_p95_seconds is not None else None,
                "limiting_factor": self.limiting_factor,
                "forecast_status": self.forecast_status,
                "forecast_source": self.forecast_source,
            },
            "explainability": {
                "dominant_shap_evidence": self.dominant_shap_evidence,
                "physics_evidence_status": self.physics_evidence_status,
                "physics_consistency_reason": self.physics_consistency_reason,
                "temporal_evidence_status": self.temporal_evidence_status,
                "fused_evidence_headline": self.fused_evidence_headline,
                "recommended_operator_action": self.recommended_operator_action,
            },
            "decision_support": {
                "advisory_assessment": self.advisory_assessment,
                "action_code": self.advisory_action_code,
                "headline": self.advisory_headline,
                "urgency": self.advisory_urgency,
            },
            "provenance_and_limitations": {
                "provenance": self.provenance,
                "disclaimer": self.disclaimer,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize engineering report to standard JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render publication-ready Markdown engineering mission report."""
        hi_str = f"{self.final_health_index:.3f}" if not np.isnan(self.final_health_index) else "N/A"
        min_hi_str = f"{self.min_health_index:.3f}" if not np.isnan(self.min_health_index) else "N/A"
        rul_str = f"{self.final_rul_seconds:.1f} s" if self.final_rul_seconds is not None else "INSUFFICIENT_HISTORY"
        rul_bnds = f"[{self.rul_p05_seconds:.1f} s - {self.rul_p95_seconds:.1f} s]" if (self.rul_p05_seconds is not None and self.rul_p95_seconds is not None) else "N/A"

        md = f"""# MISSION ENGINEERING REPORT -- {self.mission_id}
**Engine ID**: `{self.engine_id}` | **Scenario**: `{self.scenario_name}` | **Duration**: `{self.duration_s:.1f} s`  
**Simulation Mode**: `{self.simulation_mode}` | **Final Advisory**: **{self.advisory_assessment}**

---

## 1. Executive Summary & Advisory Decision Support
- **Advisory Assessment**: `{self.advisory_assessment}` ({self.advisory_action_code})
- **Urgency**: `{self.advisory_urgency}`
- **Headline**: {self.advisory_headline}
- **Recommended Operator Action**: {self.recommended_operator_action}

> **Decision Support Boundary Notice**:  
> {self.disclaimer}

---

## 2. Telemetry Statistics & Operational Peaks
| Channel | Observed Peak / Min | Mission Mean | Operational Reference / Unit |
| :--- | :--- | :--- | :--- |
| **Engine RPM** | Min: `{self.rpm_min:.0f}` / Max: `{self.rpm_max:.0f}` | `{self.rpm_mean:.0f}` | RPM |
| **Cylinder Head Temp (CHT)** | Peak: `{self.cht_peak:.1f}` deg C | `{self.cht_mean:.1f}` deg C | Max Warn: 135 deg C, Crit: 150 deg C |
| **Exhaust Gas Temp (EGT)** | Peak: `{self.egt_peak:.1f}` deg C | `{self.egt_mean:.1f}` deg C | Max Warn: 850 deg C, Crit: 880 deg C |
| **Oil Temperature** | Peak: `{self.oil_temp_peak:.1f}` deg C | `{self.oil_temp_mean:.1f}` deg C | Max Warn: 110 deg C, Crit: 130 deg C |
| **Oil Pressure** | Min: `{self.oil_pressure_min:.2f}` bar | `{self.oil_pressure_mean:.2f}` bar | Min Warn: 2.0 bar, Crit: 1.5 bar |
| **Engine Vibration** | Max: `{self.vibration_max:.3f}` g | `{self.vibration_mean:.3f}` g | Max Warn: 1.2 g, Crit: 1.8 g |
| **Fuel Flow** | Integrated Total: `{self.fuel_flow_total:.2f}` L | `{self.fuel_flow_mean:.2f}` L/h | Nominal: 15-25 L/h |

---

## 3. PHM Health, Diagnosis & Prognostics
- **Anomaly Detection**: `{self.anomaly_events_count}` persistent anomaly events detected.
- **Dominant Anomaly Channels**: `{', '.join(self.dominant_anomaly_channels) if self.dominant_anomaly_channels else 'None'}`
- **Diagnosis Result**: `{self.final_diagnosis}`
- **Diagnosis Probability**: `{self.final_diagnosis_probability:.4f}`
- **Evidence Quality**: `{self.evidence_quality}`
- **Health Index (HI)**: Final `{hi_str}` | Minimum Observed `{min_hi_str}`
- **Degradation Trend**: `{self.degradation_trend}`
- **Projected RUL**: `{rul_str}` (Confidence Interval: `{rul_bnds}`)
- **Prognostic Limiting Factor**: `{self.limiting_factor}`
- **Forecaster Status**: `{self.forecast_status}` (Source: `{self.forecast_source}`)

---

## 4. Multi-Source Explainability Evidence
- **Dominant SHAP Feature Attributions**:
"""
        if self.dominant_shap_evidence:
            for item in self.dominant_shap_evidence[:5]:
                md += f"  - `{item.get('feature', 'unknown')}`: weight `{item.get('weight', 0.0):+.4f}`\n"
        else:
            md += "  - No SHAP attribution deviations recorded during this mission profile.\n"

        md += f"""- **Physics Consistency**: `{self.physics_evidence_status}` -- {self.physics_consistency_reason or 'Nominal thermal and mechanical dynamics.'}
- **Temporal Degradation**: `{self.temporal_evidence_status}`
- **Fused Synthesis**: {self.fused_evidence_headline or 'Nominal baseline operation.'}

---

## 5. Provenance & Limitations
- **Data Provenance**: Synthetic numerical simulation based on Rotax 912 grey-box dynamics.
- **Model Assumptions**: Prognostics based on project-defined simulated functional-failure/EOL assumptions.
- **Non-Certification Clause**: This report is an engineering decision-support tool. It does NOT constitute certified airworthiness inspection or OEM maintenance release.
"""
        return md


@dataclass
class WhatIfComparisonResult:
    """
    Comparative evaluation of Baseline Mission vs What-If Mission.
    Both scenarios execute through the existing Simulation -> Phase 13 pipeline.
    """
    baseline_scenario: SimulationScenario
    whatif_scenario: SimulationScenario
    baseline_payloads: List[DashboardStatePayload]
    whatif_payloads: List[DashboardStatePayload]

    # COMPARATIVE METRICS (Where available)
    comparison_summary_headline: str
    simulated_projection_narrative: str

    # BASELINE VS WHAT-IF SUMMARY PAIRS
    baseline_health_index: float
    whatif_health_index: float
    delta_health_index: float

    baseline_rul_seconds: Optional[float]
    whatif_rul_seconds: Optional[float]
    delta_rul_seconds: Optional[float]

    baseline_peak_cht: float
    whatif_peak_cht: float
    delta_peak_cht: float

    baseline_advisory_assessment: str  # GO / CAUTION / MAINTENANCE
    whatif_advisory_assessment: str    # GO / CAUTION / MAINTENANCE

    baseline_limiting_factor: str
    whatif_limiting_factor: str

    baseline_anomaly_count: int
    whatif_anomaly_count: int

    # DETAILED COMPARISONS
    telemetry_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Simulated projection based on comparative synthetic simulation runs through the Phase 13 pipeline. "
        "Project-defined simulated functional-failure/EOL assumptions; not guaranteed real-world flight outcomes."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "headline": self.comparison_summary_headline,
                "narrative": self.simulated_projection_narrative,
                "baseline_advisory": self.baseline_advisory_assessment,
                "whatif_advisory": self.whatif_advisory_assessment,
            },
            "metrics": {
                "health_index": {
                    "baseline": round(self.baseline_health_index, 4) if not np.isnan(self.baseline_health_index) else None,
                    "whatif": round(self.whatif_health_index, 4) if not np.isnan(self.whatif_health_index) else None,
                    "delta": round(self.delta_health_index, 4) if not np.isnan(self.delta_health_index) else None,
                },
                "projected_rul_seconds": {
                    "baseline": round(self.baseline_rul_seconds, 1) if self.baseline_rul_seconds is not None else None,
                    "whatif": round(self.whatif_rul_seconds, 1) if self.whatif_rul_seconds is not None else None,
                    "delta": round(self.delta_rul_seconds, 1) if self.delta_rul_seconds is not None else None,
                },
                "peak_cht_deg_c": {
                    "baseline": round(self.baseline_peak_cht, 2),
                    "whatif": round(self.whatif_peak_cht, 2),
                    "delta": round(self.delta_peak_cht, 2),
                },
                "limiting_factor": {
                    "baseline": self.baseline_limiting_factor,
                    "whatif": self.whatif_limiting_factor,
                },
                "anomaly_events_count": {
                    "baseline": self.baseline_anomaly_count,
                    "whatif": self.whatif_anomaly_count,
                },
            },
            "telemetry_comparison": self.telemetry_comparison,
            "disclaimer": self.disclaimer,
        }
