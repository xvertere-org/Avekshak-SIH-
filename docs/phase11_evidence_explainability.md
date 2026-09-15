# Phase 11 — Evidence, Traceability & Engineering Explainability

## 1. Overview & Architectural Philosophy

Phase 11 introduces an auditable, deterministic evidence and engineering explainability layer positioned strictly downstream of the Rotax 914 grey-box Digital Twin pipeline (Phases 1–10).

The design follows a fundamental architectural invariant: **the explainability layer observes and extracts already-computed runtime states; it never recomputes them, alters them, or feeds back into upstream models.**

```
+-----------------------------------------------------------------------------------+
|                        UPSTREAM RUNTIME PIPELINE (PHASES 1 - 10)                  |
|                                                                                   |
|  Physical/Replayed Telemetry                                                      |
|           ↓                                                                       |
|  Telemetry Quality Validation (Phase 3, 9)                                        |
|           ↓                                                                       |
|  State Synchronization & Estimation (Phase 3)                                     |
|           ↓                                                                       |
|  Physics Model Nominal Prediction (Phase 2, 4)                                    |
|           ↓                                                                       |
|  Dynamic Residual Generation (Phase 5)                                            |
|           ↓                                                                       |
|  Subsystem & Engine Health Assessment (Phase 5)                                   |
|           ↓                                                                       |
|  Temporal Anomaly Detection (Phase 6)                                             |
|           ↓                                                                       |
|  Physics-Informed Fault Diagnosis (Phase 6)                                       |
|           ↓                                                                       |
|  Degradation Tracking & RUL Prognostics (Phase 8)                                 |
|           ↓                                                                       |
|  Mission Simulation & What-If Evaluator (Phase 10)                                |
+-----------------------------------------------------------------------------------+
                                         ↓
+-----------------------------------------------------------------------------------+
|               PHASE 11: AUDITABLE EVIDENCE & EXPLAINABILITY LAYER                 |
|                                                                                   |
|  1. Evidence Extraction & Snapshotting (digital_twin/evidence.py)                |
|     - ChannelEvidence & SubsystemEvidence                                        |
|     - AnomalyEvidence & DiagnosisEvidence                                        |
|     - PrognosticEvidence & CompletenessAudit                                     |
|     - Machine-Readable Traceability Chains                                       |
|                                                                                   |
|  2. Explainability & Rendering Engine (digital_twin/explainability.py)            |
|     - Deterministic Plain-Text Renderer                                          |
|     - What-If Scenario Delta Explainer                                           |
|     - Deterministic Byte-Identical JSON Serializer (SHA-256)                      |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Non-Goals & Regulatory Disclaimers

The Phase 11 layer strictly enforces the following epistemological boundaries:
1. **No New ML / Neural Models:** Zero integration of XGBoost, TimesFM, deep neural networks, or black-box explainers (e.g., surrogate model approximations).
2. **No Calculation Reconstruction:** Health indices ($HI$), anomaly scores ($S_{\text{anom}}$), diagnostic compatibility scores, degradation slopes, and RUL projections are extracted directly from upstream outputs.
3. **No Probability Inventions:** Heuristic compatibility scores (e.g., $S = 0.56$) are explicitly documented as engineering compatibility metrics, **never** as calibrated statistical probabilities (e.g., "56% probability of failure").
4. **Traceability $\neq$ Causal Proof:** Dependency graphs represent computational traces and evidentiary consistency ("supports", "is consistent with"), not formal econometric or counterfactual causal proofs.
5. **Non-Certification:** All telemetry, residuals, and scenarios are synthetic grey-box models. They do not constitute certified OEM maintenance directives, flight airworthiness certification, or operational UAV flight validation.

---

## 3. Evidence Object Model (`digital_twin/evidence_types.py`)

### 3.1 Data Classification
Prevents conflating measured sensor data with synthetic or modeled values:
- `MEASURED`: Physical sensor observation from telemetry.
- `PREDICTED`: Expected nominal value produced by grey-box physics estimation.
- `DERIVED`: Computed difference or normalized metric (e.g., raw residual, z-score).
- `ESTIMATED`: State estimate reconstructed through multi-rate synchronizer.
- `DEFAULT_ASSUMED`: Unobserved quantity assigned via standard atmospheric/engine default.
- `UNAVAILABLE`: Missing, invalid, or coverage-gated quantity.

### 3.2 Epistemic Provenance
Every parameter, threshold, and limit carries explicit epistemic tagging:
- `OEM_REFERENCE`: Sourced directly from official documentation (e.g., EASA TCDS E.122, Rotax Operators Manual).
- `MODEL_CALIBRATION`: Empirical grey-box physics tuning parameter (e.g., residual scaling factors).
- `ENGINEERING_HEURISTIC`: Heuristic threshold or weighting rule (e.g., anomaly threshold $\tau = 0.018$, subsystem weights).
- `SYNTHETIC_VALIDATION`: Test bench scenario parameter.
- `DATA_QUALITY_RULE`: Sensor admissibility boundary (e.g., physical plausibility bounds).

### 3.3 Machine-Readable Reason Codes (`ExplanationReasonCode`)
Deterministic, structured enum codes derived directly from upstream indicators:
- **Health & Subsystems:** `HEALTH_CHANNEL_DEGRADED`, `SUBSYSTEM_HEALTH_REDUCED`, `ENGINE_HEALTH_REDUCED`, `NOMINAL_OPERATION`.
- **Temporal Anomalies:** `ANOMALY_THRESHOLD_CROSSED`, `ANOMALY_PERSISTENCE_SATISFIED`, `ANOMALY_RECOVERY`.
- **Quality & Coverage:** `INSUFFICIENT_DATA`, `SENSOR_QUALITY_DEGRADED`, `SENSOR_DROPOUT`, `SENSOR_STUCK`, `SENSOR_BIAS`, `SENSOR_DRIFT`, `SENSOR_LOCALIZATION_FAVORED`.
- **Physical Signatures:** `FAULT_SIGNATURE_MATCH`, `CYLINDER_LOCALIZATION`, `RESIDUAL_DIRECTIONAL_SUPPORT`, `THERMAL_LIMIT_APPROACH`, `LUBRICATION_PRESSURE_DEVIATION`, `COMBUSTION_EGT_DEVIATION`, `MECHANICAL_VIBRATION_DEVIATION`.
- **Prognostics & RUL:** `RUL_TREND_SUPPORTED`, `RUL_UNAVAILABLE`, `RUL_NON_DEGRADING`.
- **Mission & Envelope:** `MISSION_ENVELOPE_EVENT`, `MODEL_ASSUMPTION_ACTIVE`.

---

## 4. Lineage & Traceability Chains

Every step evidence record constructs explicit computational lineages.

### Example: F4 Combustion Misfire Lineage
```
[1_TELEMETRY_INGESTION] telemetry_stream[t=20.0s] -> channel_observations (observed_at_sensor: channels=21)
         ↓
[2_PHYSICS_ESTIMATION] DigitalTwinModel -> nominal_predictions (physics_simulated: predictions=15)
         ↓
[3_RESIDUAL_GENERATION] observed_vs_predicted -> residual_vector (subtraction_and_normalization: residuals=15)
         ↓
[4_SUBSYSTEM_EVALUATION] primary_channels -> subsystem_health_scores (arithmetic_subsystem_aggregation: subsystems=6)
         ↓
[5_ENGINE_HEALTH_INDEX] subsystems -> HI_raw_and_HI_smooth (weighted_subsystem_mean_and_causal_ewma: HI=0.8851)
         ↓
[1_ANOMALY_DETECTION] HI_raw -> temporal_detection_status (threshold_and_persistence: status=ANOMALOUS)
         ↓
[2_FAULT_HYPOTHESIS_EVALUATION] residual_vector_and_cylinder_spread -> ranked_hypotheses (rule_matching: top_fault=COMBUSTION_MISFIRE, score=0.5600)
         ↓
[3_CYLINDER_LOCALIZATION] runner_spread_residuals -> localized_cylinder (divergence_localization: cylinder=2)
```

---

## 5. Human-Readable Plain-Text Renderer

The explainability engine renders deterministic plain text strictly formatted into six audit sections:

```
ENGINE HEALTH: 0.8851 (State: WATCH)

Primary contributors:
  - Combustion health: 0.6562
  - Rotational health: 0.7818
  - Thermal health: 1.0000
  - Lubrication health: 1.0000
  - Fuel health: 1.0000
  - Mechanical health: 1.0000

Supporting evidence:
  - EGT residual: -69.50 °C (normalized z = -2.70)
  - RPM residual: -389.20 RPM (normalized z = -3.89)
  - Cylinder 2 localized via runner spread

Diagnosis:
  Primary hypothesis: COMBUSTION_MISFIRE
  Diagnostic confidence heuristic: 0.5600 (Status: SUSPECTED)

Data quality:
  Status: GOOD (Observability coverage: 100.0%)

Interpretation:
  The modeled telemetry signature is consistent with COMBUSTION_MISFIRE affecting cylinder 2.

Limitations:
  - Synthetic grey-box validation; not operational or certified engine diagnosis.
  - Diagnostic confidence and compatibility scores are engineering heuristics, not calibrated probabilities.
  - RUL estimates are model-defined projections to horizon D_EOL and do not represent OEM maintenance limits.
```

---

## 6. What-If Scenario Delta Explanations

Scenario comparisons (Phase 10 What-If) produce `WhatIfScenarioDeltaEvidence` detailing:
1. **Delta Metrics:** $\Delta \text{Max CHT}$, $\Delta \text{Max EGT}$, $\Delta \text{Min Oil Pressure}$, $\Delta \text{Mean HI}$, $\Delta \text{Risk Index}$.
2. **Modeled Contributions:** Partitions sensitivity into environment (ISA offset), control (throttle rating), and fault physics.
3. **Envelope Events Delta:** Authoritative OEM limit exceedances.

---

## 7. Completeness Audit Metric

Completeness is rigorously defined as an exact ratio of present fields to required fields:
$$\text{Completeness Ratio} = \frac{N_{\text{present}}}{N_{\text{required}}}$$

Where $N_{\text{required}}$ comprises:
- Baseline Core: `timestamp`, `engine_id`, `source_telemetry_timestamp`, `canonical_timestamp`, `channel_evidence`, `subsystem_evidence`, `traceability_chains`, `data_quality_status`, `observability_coverage`, `synchronization_status`, `limitations`.
- Active PHM Context: `hi_raw`, `hi_smooth`, `engine_health_state` (when health active), `anomaly_evidence` (when detector active), `diagnosis_evidence` (when diagnoser active), `prognostic_evidence` (when RUL active).

No arbitrary "99% explainable" marketing metrics are permitted.

---

## 8. Deterministic Serialization & Verification

Every `StepEvidenceRecord` and `WhatIfScenarioDeltaEvidence` supports deterministic JSON serialization:
- Sorted dictionary keys (`sort_keys=True`).
- Fixed float precision (rounded to 4 decimal places).
- Canonical SHA-256 hash computation:
  $$\text{SHA-256}(\text{serialized\_bytes}_A) \equiv \text{SHA-256}(\text{serialized\_bytes}_B)$$
Replaying identical telemetry twice produces identical SHA-256 hashes.
