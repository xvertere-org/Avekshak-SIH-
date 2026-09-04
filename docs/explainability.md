# Phase 12: Explainability & Evidence Fusion

## 1. System Overview

Phase 12 provides the authoritative **Explainability & Evidence Fusion** layer for project **SIH26054** (*AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero-Piston Engines used in MALE UAVs*).

> [!IMPORTANT]
> **Core Architectural Rule**:
> The explainability layer **interprets existing outputs** from upstream phases. It **NEVER changes, overrides, retrains, or replaces Phase 7–11 decisions**.
> 
> **Explicit Engineering Disclaimers**:
> 1. *"SHAP feature attribution does not establish causality."*
> 2. *"Physics consistency is supporting engineering evidence, not proof of fault causation."*
> 3. *"Project-defined EOL criteria are not certified OEM/FAA limits."*
> 4. *"Validation is limited to synthetic/project scenarios; real aero-piston-engine validation is not established."*

```
+----------------------------------------------------------------------------------------------------+
|                                PHASE 12 EVIDENCE FUSION ARCHITECTURE                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Phase 8 XGBoost Diagnosis]    [Phase 6 Normalized Residuals]    [Phase 9 Health Index & Isolation] |
|              |                                 |                                 |                 |
|              v                                 v                                 v                 |
|     +------------------+             +-------------------+             +--------------------+      |
|     |  SHAP Explainer  |             |  Physics Evidence |             |   Health Evidence  |      |
|     | (Local TreeSHAP) |             | (Rule Consistency)|             | (Passthrough Cont.)|      |
|     +------------------+             +-------------------+             +--------------------+      |
|              |                                 |                                 |                 |
|              +---------------------------------+---------------------------------+                 |
|                                                |                                                   |
|  [Phase 10 Future Forecast]                    |                    [Phase 11 Prognostic RUL]       |
|              |                                 |                                 |                 |
|              v                                 v                                 v                 |
|     +--------------------+           +-------------------+             +--------------------+      |
|     | Temporal Evidence  | --------> |  Evidence Fusion  | <---------- |    RUL Evidence    |      |
|     | (Causal History)   |           |      Engine       |             | (EOL Provenance)   |      |
|     +--------------------+           +-------------------+             +--------------------+      |
|                                                |                                                   |
|                                                v                                                   |
|                             +-------------------------------------+                                |
|                             |      ExplainabilityResult           |                                |
|                             |  - Overall Quality: HIGH/MED/LOW    |                                |
|                             |  - Operator Summary Narrative       |                                |
|                             |  - Isolated Provenance Metadata     |                                |
|                             +-------------------------------------+                                |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Five Distinct Evidence Streams

Phase 12 strictly distinguishes between five independent evidence dimensions:

### A. Machine Learning Feature Attribution (SHAP)
- **Source**: Phase 8 `XGBoostFaultClassifier`.
- **Methodology**: Lundberg TreeSHAP computed natively via XGBoost's C++ core (`pred_contribs=True`).
- **Outputs**: Top-$k$ feature contributions with signed values, directions (`TOWARD_PREDICTED_CLASS` vs `AWAY_FROM_PREDICTED_CLASS`), and relative contribution weights.
- **Boundary**: Does not retrain or alter Phase 8 models. If the classifier is unavailable, returns `MODEL_UNAVAILABLE` without fabricating attribution.

### B. Deterministic Physics Consistency
- **Source**: Phase 6 normalized residuals ($\sigma$) and Phase 9 sensor isolation.
- **Methodology**: Evaluates domain physical relationships against the Phase 9 deadband ($\tau_{\text{nominal}} = 1.5\sigma$):
  - `COOLING_DEGRADATION`: Primary CHT residual $\uparrow$, coupled oil temperature residual $\uparrow$ (delayed conduction).
  - `LUBRICATION_DEGRADATION`: Hydrodynamic oil pressure residual $\downarrow$, frictional oil temperature residual $\uparrow$.
  - `FUEL_INJECTION_ABNORMALITY`: Lean branch (EGT $\uparrow$, fuel flow $\downarrow$) vs Rich branch (EGT $\downarrow$, fuel flow $\uparrow$).
  - `MECHANICAL_DEGRADATION`: Structural vibration residual $\uparrow$.
  - `SENSOR_FAULT`: Isolated channel anomaly without coupled physical response.
  - `NONE`: All physical residuals within nominal operating band ($[-1.5, +1.5]\sigma$).
- **Evidence Status**: Categorized as `SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONFLICTING`, or `INSUFFICIENT_DATA`.

### C. Health Index & Subsystem Degradation Evidence
- **Source**: Phase 9 `HealthIndexResult`.
- **Methodology**: Direct passthrough interpretation. Exposes current Health Index ($HI \in [0, 1]$), operational health state, dominant degraded channels, and dynamic effective weights.
- **Sensor Isolation**: If Phase 9 isolated an observation channel (`excluded_channels`), that status is preserved.

### D. Causal Temporal Evolution
- **Source**: Phase 9 degradation rate/trend and Phase 10 trajectory forecasting status.
- **Methodology**: Assesses whether degradation is `WORSENING` ($\dot{HI} < -0.0005\text{ s}^{-1}$), `STABLE`, or `IMPROVING`. Strictly causal (never inspects future timestamps).

### E. Prognostic RUL & EOL Limiting Factors
- **Source**: Phase 11 `RULResult`.
- **Methodology**: Interprets median RUL point estimate, P05/P95 uncertainty bounds, EOL limiting factor (`GLOBAL_HEALTH_INDEX` vs physical redline), and EOL provenance.

---

## 3. Sensor Fault Isolation & False Alarm Protection

Phase 12 strictly inherits the sensor-isolation semantics established in Phase 9. An isolated sensor channel **must never create fake physical evidence**:

- **Example**: If cylinder head temperature (CHT) is isolated as an observation fault (e.g. thermocouple bias):
  - Physical cooling degradation evidence is marked as `CONFLICTING` or `INSUFFICIENT_DATA` because the primary thermal channel is unreliable.
  - Sensor fault evidence is marked as `SUPPORTED` due to the isolated observation divergence.
  - Prevents single-instrumentation errors from triggering false physical subsystem alarms.

---

## 4. Synthetic Scenario Validation Results

The explainability layer was validated on the project's canonical synthetic aero-piston failure scenarios:

| Scenario | Diagnosed Fault | ML Confidence | Physics Status | Overall Quality | Top SHAP Attribution | Consistency Summary |
|---|---|---|---|---|---|---|
| **Healthy Nominal Cruise** | `none` | 98% | `SUPPORTED` | **`HIGH`** | `oil_temp_norm_residual` | All observed physical residuals remain within the healthy nominal envelope. |
| **Cooling Degradation** | `cooling_degradation` | 94% | `SUPPORTED` | **`HIGH`** | `cht_norm_residual` | Elevated CHT and coupled oil temperature residuals match cooling-degradation relationships. |
| **Lubrication Degradation** | `lubrication_degradation` | 96% | `SUPPORTED` | **`HIGH`** | `oil_pressure_norm_residual` | Drop in oil pressure residual coupled with elevated oil temperature matches lubrication degradation. |
| **Fuel Injection Abnormality** | `fuel_injection_abnormality` | 91% | `SUPPORTED` | **`HIGH`** | `egt_norm_residual` | Elevated EGT combined with depressed fuel flow matches lean mixture abnormality. |
| **Mechanical Degradation** | `mechanical_degradation` | 95% | `SUPPORTED` | **`HIGH`** | `vibration_norm_residual` | Elevated structural vibration residual is consistent with mechanical degradation. |
| **Sensor Fault** | `sensor_fault` | 89% | `SUPPORTED` | **`HIGH`** | `cht_norm_residual` | CHT isolated by Phase 9 heuristic as observation anomaly without coupled physical divergence. |
| **Conflicting Case** | `cooling_degradation` | 75% | `CONFLICTING` | **`LOW`** | `cht_norm_residual` | Depressed CHT residual contradicts expected thermal elevation for cooling degradation. |
| **Insufficient Data Guard** | `cooling_degradation` | 80% | `INSUFFICIENT_DATA` | **`INSUFFICIENT_DATA`** | `cht_norm_residual` | Insufficient valid physical residual channels (< 4 valid) to evaluate physical consistency. |

---

## 5. Causality and Leakage Guarantees

Phase 12 enforces strict temporal causality:
1. **No Future Leakage**: At timestamp $t$, explanations only consume telemetry, residuals, and upstream outputs at or before $t$.
2. **Future-Modification Invariance**: An explanation computed at $t$ is bit-for-bit identical regardless of whether future samples exist or are modified.
3. **Engine & Mission Session Isolation**: Internal explanation history is strictly keyed by `(engine_id, mission_id)`.
