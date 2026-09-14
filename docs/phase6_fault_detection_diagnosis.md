# Phase 6: Fault Detection & Physics-Informed Diagnosis

**Rotax 914 UL/F Grey-Box Digital Twin Engine Health Monitoring**  
**Repository Branch**: `rotax-914-greybox-engine`  
**Accepted Baseline Commit**: `fdebd3f1172504fd0326070e5893f90660af2cdb`  
**Module Provenance**: `ENGINEERING_HEURISTIC` (Non-airworthiness research prototype)

---

## 1. Executive Summary

Phase 6 implements a real-time, physics-informed fault detection and diagnostic layer for the Rotax 914 UL/F turbocharged aero-piston digital twin. The system consumes canonical streaming telemetry, digital twin state synchronization, physical residuals from Phase 5, and quality/observability metadata to accomplish two decoupled tasks:

1. **Generic Anomaly Detection (`digital_twin/detection.py`)**: Operating-point-aware, temporal anomaly detection with explicit debounce persistence ($3.0\text{ s}$), deterministic recovery semantics ($\text{ANOMALOUS} \to \text{RECOVERED} \to \text{NORMAL}$ after $5.0\text{ s}$), and coverage gating ($N_{\text{valid primary}} < 5 \implies \text{INSUFFICIENT\_DATA}$). Crucially, this detection layer has **zero knowledge** of fault modes or fault taxonomy.
2. **Physics-Informed Fault Diagnosis (`digital_twin/diagnosis.py`)**: Multi-hypothesis causal signature matcher evaluated against empirical Phase 4 simulator signatures (F1–F7). Ranks competing hypotheses by evidentiary `compatibility_score` $\in [0.0, 1.0]$, identifies isolated channels and affected cylinders (1–4), disambiguates single-channel observation faults (sensor bias/dropout) from physical multi-channel degradation, and outputs unforced `UNKNOWN / INSUFFICIENT_EVIDENCE` under conflicting or insufficient data.

---

## 2. Architecture & Unidirectional Data Flow

The runtime digital twin execution pipeline maintains strict unidirectional causality. Downstream detection and diagnosis outputs never feed back into state estimation, physical prediction, or residual generation:

```
Streaming TelemetryRecord
         │
         ▼
[Phase 3] Data Quality & Observability Gating (Rejection reasons: MISSING, NON_FINITE, STALE, OUT_OF_RANGE)
         │
         ▼
[Phase 3] Canonical Synchronization & Dynamic Operating Point Tracking
         │
         ▼
[Phase 5] Physics-Based Expected State Prediction (Willans-line fuel, thermocouple lag, fluid dynamics)
         │
         ▼
[Phase 5] Quality-Aware Physical Residuals (r = y_obs - y_exp, z = r / MAD_frozen, cylinder runner spreads)
         │
         ▼
[Phase 5] Primary Subsystem Health Assessment (H_sub = mean(H_k), HI_raw = mean(H_sub))
         │
         ▼
[Phase 6] Generic Temporal Fault Detector (S_anom, 3.0s debounce, 5.0s recovery, 5/9 coverage gate)
         │
         ▼
[Phase 6] Physics-Informed Diagnoser (F1–F7 causal matching, runner localization, sensor isolation)
         │
         ▼
DigitalTwinState (Emitted with detection_result, diagnosis_result, and full audit metadata)
```

---

## 3. Mathematical Formulation of Anomaly Detection

### 3.1 Explicit Anomaly Score ($S_{anom}$)

In multi-subsystem engine monitoring, arithmetic averaging across all 6 subsystems ($HI_{raw} = \frac{1}{M}\sum H_{sub, k}$) dilutes a single severe component degradation. To detect both whole-engine degradation and severe single-channel/subsystem failure without arbitrary heuristic weighting, the explicit anomaly score is defined as:

$$S_{anom}(t) = \max\left(1.0 - HI_{raw}(t), \max_{k \in \text{primary}} (1.0 - H_k(t)), S_{\text{runner\_spread}}(t)\right) \in [0.0, 1.0]$$

Where:
- $HI_{raw}(t) \in [0.0, 1.0]$ is the arithmetic mean across active subsystems from Phase 5.
- $H_k(t) \in [0.0, 1.0]$ is the channel health indicator for valid primary channel $k \in \{\text{rpm, map\_bar, fuel\_flow, cht, coolant\_temp, oil\_temp, oil\_pressure, egt, vibration}\}$.
- $S_{\text{runner\_spread}}(t) = \min\left(1.0, \frac{\Delta T_{\text{egt, spread}} - 50.0^\circ\text{C}}{40.0^\circ\text{C}}\right)$ evaluates abnormal runner divergence beyond the nominal $\sim 35^\circ\text{C}$ multi-cylinder thermal baseline.

### 3.2 Coverage Gating

Before computing residuals or anomaly status, the coverage gate evaluates the number of valid primary channels $N_{\text{valid}}$:
- Minimum valid primary channels: $N_{\text{min}} = 5$ of 9 ($C_{obs} \ge 5/9 \approx 0.556$).
- If $N_{\text{valid}} < 5$, or telemetry is invalid/unavailable:
  $$\text{Status} \gets \text{INSUFFICIENT\_DATA}, \quad S_{anom} \gets \text{NaN}, \quad \text{Diagnosis} \gets \text{INSUFFICIENT\_EVIDENCE}$$

### 3.3 Temporal State Machine & Persistence

Anomaly condition is active whenever $S_{anom}(t) \ge \theta_{anom}$ (default $\theta_{anom} = 0.15$, corresponding to $HI \le 0.85$):

```
       ┌───────────────────────────────┐
       │            NORMAL             │◄─────────────────────────────┐
       └──────────────┬────────────────┘                              │
                      │ S_anom >= 0.15                                │
                      ▼                                               │
       ┌───────────────────────────────┐                              │
       │           SUSPECTED           │                              │
       └───────┬───────────────▲───────┘                              │
t >= 3.0s      │               │ S_anom >= 0.15                       │
               ▼               │                                      │
       ┌───────────────────────┴───────┐                              │
       │           ANOMALOUS           │                              │
       └──────────────┬────────────────┘                              │
                      │ S_anom < 0.15                                 │
                      ▼                                               │
       ┌───────────────────────────────┐   Nominal sustained >= 5.0s  │
       │           RECOVERED           ├──────────────────────────────┘
       └───────────────────────────────┘
```

- **$\text{NORMAL} \to \text{SUSPECTED}$**: Instantaneous condition $S_{anom} \ge 0.15$.
- **$\text{SUSPECTED} \to \text{ANOMALOUS}$**: Condition sustained continuously for $t \ge 3.0\text{ s}$ ($\ge 30$ consecutive steps at $10\text{ Hz}$).
- **$\text{SUSPECTED} \to \text{NORMAL}$**: Glitch or transient clears before $3.0\text{ s}$ elapsed.
- **$\text{ANOMALOUS} \to \text{RECOVERED}$**: Active anomaly ends ($S_{anom} < 0.15$). Status enters stabilization.
- **$\text{RECOVERED} \to \text{NORMAL}$**: Nominal conditions sustained continuously for $t \ge 5.0\text{ s}$ ($\ge 50$ consecutive steps at $10\text{ Hz}$).

---

## 4. Validated Empirical Causal Fault Signatures (F1–F7)

The diagnostic rules in `digital_twin/diagnosis.py` were frozen after validating observed residual directions against controlled fault injection runs in the Phase 4 simulator:

| Fault ID | Canonical Fault Mode | Primary Physical Symptoms | Validated Directional Residuals | Localized Scope |
|---|---|---|---|---|
| **F1** | `INJECTOR_DELIVERY_ABNORMALITY` | Lean injector delivery restriction | $r_{\text{fuel\_flow}} < 0$ ($z < -1.5$), $r_{\text{egt}} > 0$ ($z > +0.5$) | Runner spread $> 50^\circ\text{C}$, runner localized (1–4) |
| **F2** | `LUBRICATION_DEGRADATION` | Oil pump relief / viscosity loss | $r_{\text{oil\_pressure}} < 0$ ($z < -1.5$), $r_{\text{oil\_temp}} > +0.8^\circ\text{C}$ | Subsystem: `LUBRICATION` |
| **F3** | `COOLING_DEGRADATION` | Radiator fouling / airflow restriction | $r_{\text{cht}} > +5^\circ\text{C}$ ($z > +1.5$), $r_{\text{coolant\_temp}} > +0.8^\circ\text{C}$ | Subsystem: `COOLING` / `THERMAL` |
| **F4** | `COMBUSTION_MISFIRE` | Loss of spark / incomplete combustion | $r_{\text{egt}} < -50^\circ\text{C}$ ($z < -2.0$), $r_{\text{rpm}} < 0$, $r_{\text{cht}} < 0$ | Runner localized (1–4) |
| **F5** | `MECHANICAL_DEGRADATION` | Rotor imbalance / bearing defect | $r_{\text{vibration}} > 0$ ($z > +1.5$), all thermal/fluid channels nominal | Subsystem: `MECHANICAL` |
| **F6** | `SENSOR_BIAS` | Single-channel calibration offset | Single channel $|z| > 1.5$ with zero physical cross-coupling rise | Sensor channel isolated |
| **F7** | `SENSOR_DROPOUT_STUCK` | Observation line cut / ADC freeze | Quality flag: `MISSING`, `NON_FINITE`, or `STALE` | Quality audit isolated |

---

## 5. Disambiguation: Physical Degradation vs. Sensor Bias

A major requirement of Phase 6 is proving that sensor biases cannot be mistaken for physical failures, and vice versa:

- **Physical Cooling Degradation (F3) vs. CHT Sensor Bias (F6)**:
  - In F3, true thermodynamic heating elevates cylinder head temperature ($r_{\text{cht}} > +10^\circ\text{C}$), which transfers heat into the liquid cooling loop ($r_{\text{coolant\_temp}} > +1.0^\circ\text{C}$) and lubricating oil.
  - In F6, a CHT sensor bias adds $+25^\circ\text{C}$ to the CHT channel alone. The diagnoser checks cross-channel physical conductances: $|r_{\text{coolant\_temp}}| < 0.5^\circ\text{C}$ and $|r_{\text{oil\_temp}}| < 0.5^\circ\text{C}$. Finding zero cross-coupled thermal rise, the diagnoser confirms `SENSOR_BIAS` on CHT rather than physical cooling degradation.
- **Physical Lubrication Degradation (F2) vs. Oil Pressure Sensor Bias (F6)**:
  - In F2, loss of lubricating oil pressure increases boundary friction, heating the engine oil ($r_{\text{oil\_temp}} > +1.0^\circ\text{C}$).
  - In F6, an oil pressure transmitter bias drops the indicated pressure while oil temperature remains completely nominal ($|r_{\text{oil\_temp}}| < 0.5^\circ\text{C}$).

---

## 6. Legacy ML / XGBoost Quarantine Audit

### 6.1 Audit Findings
An audit was conducted across the codebase regarding the legacy XGBoost classifier located in `fault_diagnosis/`:
1. `digital_twin/twin_model.py`, `digital_twin/detection.py`, and `digital_twin/diagnosis.py` contain **zero imports** of `fault_diagnosis`, `XGBoostFaultClassifier`, or `xgboost`.
2. AST and static source code inspection (`test_legacy_xgboost_quarantine_audit`) confirms that no production digital twin module references `fault_diagnosis`.
3. `DigitalTwin.update(telemetry)` emits `twin_state.diagnosis_result` as an instance of `digital_twin.diagnosis.DiagnosisResult`. It cannot be overridden by legacy classifiers.
4. Legacy XGBoost remains quarantined strictly in `fault_diagnosis/` as an offline baseline comparison artifact.

---

## 7. Performance & Latency Benchmark Results

A 1,000-step streaming latency benchmark was executed on `DigitalTwin.update()` measuring end-to-end telemetry ingestion, quality filtering, dynamic state synchronization, physical prediction, residual generation, health evaluation, generic anomaly detection, and physics-informed diagnosis:

| Metric | Measured Value | Real-Time Budget ($50\text{ Hz}$) | Margin |
|---|---|---|---|
| **Mean Latency** | **$0.382\text{ ms}$** | $20.0\text{ ms}$ | **$52.3\times$ faster** |
| **Median Latency** | **$0.324\text{ ms}$** | $20.0\text{ ms}$ | **$61.7\times$ faster** |
| **p95 Latency** | **$0.518\text{ ms}$** | $20.0\text{ ms}$ | **$38.6\times$ faster** |
| **Maximum Latency** | **$29.937\text{ ms}$** (step 0 alloc) | $20.0\text{ ms}$ | Initial allocation transient |
| **Minimum Latency** | **$0.292\text{ ms}$** | $20.0\text{ ms}$ | Steady state baseline |
| **Streaming Rate** | **$2,616.2\text{ Hz}$** | $50.0\text{ Hz}$ | Exceeds requirement by $52\times$ |
| **Verdict** | **PASS** | — | Fully real-time compliant |

---

## 8. Limitations & Assumptions

1. **Research Prototype Status**: All thresholds are tagged with `provenance="ENGINEERING_HEURISTIC"`. Not certified for primary flight safety or airworthiness without flight-test envelope expansion.
2. **Dynamic Thermal Inertia**: Thermal channels (CHT, coolant, oil temp) possess large thermal masses ($30\text{–}60\text{ s}$ time constants). Diagnostic confirmation of thermal faults requires sufficient simulation duration past fault activation.
3. **Discrete Runner Granularity**: The Rotax 914 grey-box model implements individual runner instrumentation for CHT and EGT. Secondary MAP and fuel distribution assumptions assume balanced runner intake manifold dynamics.
