# Phase 14 — Mission Replay, Engineering Reporting & What-If Analysis
**SIH26054 / AVEKSHAK Digital Twin**

## 1. Overview & Architectural Scope
Phase 14 represents the operational decision-support and post-mission evaluation layer for the SIH26054 Rotax 912 grey-box aero engine digital twin. It wraps around the authoritative backend (`SystemPipelineOrchestrator`) and consumption contracts (`DashboardStatePayload`) developed and frozen in Phases 1–13.

Phase 14 delivers three capabilities:
1. **Mission Replay**: Chronological playback and scrubbing of executed synthetic missions using existing Phase 13 output contracts without recomputing PHM logic in the presentation layer.
2. **Mission Reporting**: Generation and export of structured engineering mission reports (Markdown and JSON) aggregating operational extremes, PHM events, multi-source explainability, and advisory decisions.
3. **Mission What-If Analysis**: Comparative trajectory evaluation contrasting baseline flight conditions against modified operational profiles or simulated fault-stress scenarios through the authoritative Phase 13 pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 14 ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
┌──────────────┐              ┌──────────────┐              ┌────────────────┐
│  replay.py   │              │ reporting.py │              │   what_if.py   │
│ Chronological│              │ Engineering  │              │ Comparative    │
│ Replay Mgmt  │              │ Report Gen   │              │ Trajectory Eval│
└──────┬───────┘              └──────┬───────┘              └───────┬────────┘
       │                             │                              │
       └─────────────────────────────┼──────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │          schema.py           │
                      │ MissionReplaySession         │
                      │ MissionReportSummary         │
                      │ WhatIfComparisonResult       │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
        Authoritative Frozen Backend (Phases 1–13)
        - SystemPipelineOrchestrator
        - DashboardStatePayload
```

---

## 2. Component Design & Contracts

### 2.1 Mission Replay (`phase14/replay.py`)
- **Isolation Key**: `(engine_id, mission_id)` strictly governs state isolation.
- **Chronological Integrity**: Replay payloads must monotonically increase in timestamp (`t[i] > t[i-1]`). Scrambled or duplicate time frames are rejected.
- **Zero Fabrication**: Replay serves stored or cached `DashboardStatePayload` frames directly, ensuring that presentation layers never re-execute or approximate PHM algorithms.
- **Timestep Navigation**: Provides clamped lookup (`get_step(index)`) and synchronized multi-channel history extraction (`get_history(up_to_step)`).

### 2.2 Engineering Mission Reporting (`phase14/reporting.py`)
- **Operational Extremes Aggregation**:
  - Engine RPM (min, max, mean)
  - Cylinder Head Temperature (CHT peak, mean)
  - Exhaust Gas Temperature (EGT peak, mean)
  - Oil Temperature (peak, mean)
  - Oil Pressure (min, mean)
  - Mechanical Vibration (peak, mean)
  - Integrated Fuel Flow & Burn Total
- **PHM Health Summary**:
  - Persistent anomaly count and contributing channels
  - Final diagnosis classification and diagnosis probability
  - Evidence quality (e.g. HIGH/MEDIUM/LOW/VALID) clearly distinguished from ML probabilities
  - Health Index trajectory (final and minimum) and degradation trend
  - Projected RUL point estimate and confidence interval (`[p05, p95]`)
  - Prognostic limiting factor and forecaster source status
- **Explainability & Decision Support**:
  - Dominant SHAP feature attributions
  - Physics consistency assessment and temporal degradation tracking
  - Advisory assessment (`GO`, `CAUTION`, `MAINTENANCE`)
- **Export Formats**: Standardized JSON (`to_json()`) and publication-ready Markdown (`to_markdown()`).

### 2.3 Mission What-If Analysis (`phase14/what_if.py`)
- **Authoritative RUL Pathway**: Projected RUL is **never** independently computed by Phase 14. Both baseline and what-if missions execute through the identical:
  $$\text{EngineSimulator} \longrightarrow \text{Phase 13 Orchestrator} \longrightarrow \text{Phase 11 RUL}$$
  pathway, honoring all TimesFM context buffering and Theil-Sen degradation models.
- **Separation of Concerns**:
  - *Mission-Condition What-If*: Altitude ($h$), Ambient Temperature ($T_{amb}$), Commanded Throttle ($\delta_{th}$), Mission Duration ($t_{dur}$).
  - *Simulated Fault-Stress Scenarios*: Fault type and injected severity. Ground-truth labels are sanitized prior to inference.
- **Comparative Summaries**:
  - Top headline and simulated projection narrative
  - Side-by-side metric deltas ($\Delta \text{HI}$, $\Delta \text{RUL}$, $\Delta \text{CHT}_{peak}$)
  - Comparative advisory shift (e.g., `GO` $\rightarrow$ `MAINTENANCE`)

---

## 3. Dashboard Integration
The Streamlit dashboard (`dashboard/app.py`) cleanly exposes Phase 14 via dedicated navigation views added without altering the existing 5 views:
- **`6. Mission Replay`** (`dashboard/pages/replay_page.py`): Interactive timestep scrubber, live status ribbons, synchronized KPI cards, and multi-trace timeline charts.
- **`7. Mission Report`** (`dashboard/pages/report_page.py`): Engineering report preview with download buttons for Markdown (`.md`) and JSON (`.json`).
- **`8. What-If Comparison`** (`dashboard/pages/what_if_page.py`): Side-by-side scenario parameter sliders, comparative delta indicators, and dual Plotly trajectory overlay charts (Health Index and CHT).

---

## 4. Safety & Claim Boundaries
All Phase 14 outputs adhere to strict aerospace software engineering safety conventions:
1. **Decision Support**: All assessments are classified as `"advisory GO / CAUTION / MAINTENANCE assessment"`. The system **never** claims autonomous flight clearance or safety-critical override authority.
2. **Failure Thresholds**: Prognostic limits represent `"project-defined simulated functional-failure/EOL assumptions"` based on grey-box thermal and lubrication bounds, not certified OEM/FAA airworthiness limits.
3. **Projections**: Future trajectories are explicitly designated as `"simulated projections"`, never guaranteed mission outcomes.
4. **No Airworthiness Claims**: Outputs do not claim flight qualification, OEM certification, or certified airworthiness release.

---

## 5. Verification & Test Coverage
The Phase 14 implementation is accompanied by a dedicated test suite in `phase14/tests/`:
- `test_phase14_replay.py`: Deterministic replay across runs, chronological ordering enforcement, engine/mission isolation, empty payload rejection, and history extraction.
- `test_phase14_reporting.py`: Operational statistical peaks, PHM summaries, advisory wording, safety claim boundary audits, and JSON/Markdown serialization.
- `test_phase14_what_if.py`: Comparative delta calculation, end-to-end pipeline execution, fault-stress comparisons, zero ML leakage, and strict reuse of the Phase 11 RUL pathway.
