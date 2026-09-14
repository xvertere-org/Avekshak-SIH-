# Phase 6: Fault Detection & Physics-Informed Diagnosis

**Rotax 914 UL/F Grey-Box Digital Twin Engine Health Monitoring**  
**Repository Branch**: `rotax-914-greybox-engine`  
**Accepted Baseline Commit**: `fdebd3f1172504fd0326070e5893f90660af2cdb`  
**Module Provenance**: `ENGINEERING_HEURISTIC` (Non-airworthiness research prototype)

---

## 1. Executive Summary & Blocker Corrections

Phase 6 implements a real-time, physics-informed fault detection and diagnostic layer for the Rotax 914 UL/F turbocharged aero-piston digital twin. The system consumes canonical streaming telemetry, digital twin state synchronization, physical residuals from Phase 5, and quality/observability metadata to accomplish two strictly decoupled tasks:

1. **Generic Anomaly Detection (`digital_twin/detection.py`)**: Operating-point-aware, temporal anomaly detection with explicit debounce persistence ($3.0\text{ s}$), deterministic recovery semantics ($\text{ANOMALOUS} \to \text{RECOVERED} \to \text{NORMAL}$ after $5.0\text{ s}$), and coverage gating ($N_{\text{valid primary}} < 5 \implies \text{INSUFFICIENT\_DATA}$). Crucially, this detection layer has **zero knowledge** of fault modes, fault schedules, or fault taxonomy.
2. **Physics-Informed Fault Diagnosis (`digital_twin/diagnosis.py`)**: Multi-hypothesis causal signature matcher evaluated against empirical Phase 4 simulator signatures (F1–F7). Ranks competing hypotheses by evidentiary `compatibility_score` $\in [0.0, 1.0]$, identifies isolated channels and affected cylinders (1–4), disambiguates single-channel observation faults (sensor bias, drift, dropout, stuck) from physical multi-channel degradation, and outputs unforced `UNKNOWN / INSUFFICIENT_EVIDENCE` under conflicting or insufficient data.

### Authoritative Architecture Corrections Applied

- **BLOCKER 1 Resolution (Cylinder Vote Removal)**: In strict compliance with the authorized architecture, individual cylinder head and exhaust temperatures (`cht_cyl1..4`, `egt_cyl1..4`) and runner-spread metrics are **completely excluded** from the engine-level anomaly score and persistence timer. The authoritative engine-level anomaly equation is:
  $$S_{anom}(t) = 1.0 - HI_{raw}(t)$$
  Cylinder and runner evidence is reserved strictly for `PhysicsInformedDiagnoser` for localized cylinder attribution, runner imbalance tracking, and hypothesis ranking.
- **BLOCKER 2 Resolution (Expanded Verification Coverage)**: Test suite expanded from 16 to **32 focused, meaningful tests** across 6 distinct categories (Detection, Physical Faults, Sensor Faults, Diagnosis, Adversarial, Integrity), with 100% pass rate.
- **Latency Characterization**: Comprehensive reporting separating cold-start step 0 initialization overhead ($0.63\text{ ms}$ post-import, spiking up to $29.94\text{ ms}$ under cold runtime allocation / OS GC) from steady-state streaming percentiles (p50: $0.32\text{ ms}$, p95: $0.56\text{ ms}$, p99: $0.76\text{ ms}$). Transparently documented as **PASS WITH LIMITATIONS** against the strict worst-case single-step budget ($20.0\text{ ms}$).

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
[Phase 6] Generic Temporal Fault Detector (S_anom = 1 - HI_raw, 3.0s debounce, 5.0s recovery, 5/9 coverage gate)
         │
         ▼
[Phase 6] Physics-Informed Diagnoser (F1–F7 causal matching, runner localization, sensor isolation)
         │
         ▼
DigitalTwinState (Emitted with detection_result, diagnosis_result, and full audit metadata)
```

---

## 3. Mathematical Formulation of Anomaly Detection

### 3.1 Authoritative Engine-Level Anomaly Score ($S_{anom}$)

The generic detector evaluates physical degradation strictly at the composite engine level:

$$S_{anom}(t) = \max\left(0.0, \min\left(1.0, 1.0 - HI_{raw}(t)\right)\right) \in [0.0, 1.0]$$

Where:
- $HI_{raw}(t) = \frac{1}{M}\sum_{m=1}^M H_{sub, m}(t)$ is the arithmetic mean across active primary subsystems from Phase 5.
- Subsystem health $H_{sub, m}(t) = \frac{1}{K_m}\sum_{k=1}^{K_m} H_k(t)$ is the arithmetic mean across valid primary channels owned by subsystem $m$.
- Primary channels: `rpm`, `map_bar`, `fuel_flow`, `cht`, `coolant_temp`, `oil_temp`, `oil_pressure`, `egt`, `vibration`.
- Cylinder runner spreads (`cht_cyl1..4`, `egt_cyl1..4`) contribute **zero** votes to $S_{anom}(t)$.

### 3.2 Coverage Gating

Before computing residuals or anomaly status, the coverage gate evaluates the number of valid primary channels $N_{\text{valid}}$:
- Minimum valid primary channels: $N_{\text{min}} = 5$ of 9 ($C_{obs} \ge 5/9 \approx 0.556$).
- If $N_{\text{valid}} < 5$, or telemetry is invalid/unavailable:
  $$\text{Status} \gets \text{INSUFFICIENT\_DATA}, \quad S_{anom} \gets \text{NaN}, \quad \text{Diagnosis} \gets \text{INSUFFICIENT\_EVIDENCE}$$

### 3.3 Temporal State Machine & Persistence

Anomaly condition is active whenever $S_{anom}(t) \ge \theta_{anom}$ (where $\theta_{anom} = 0.018$, corresponding to $HI_{raw} \le 0.982$):

```
       ┌───────────────────────────────┐
       │            NORMAL             │◄─────────────────────────────┐
       └──────────────┬────────────────┘                              │
                      │ S_anom >= 0.018                               │
                      ▼                                               │
       ┌───────────────────────────────┐                              │
       │           SUSPECTED           │                              │
       └───────┬───────────────▲───────┘                              │
t >= 3.0s      │               │ S_anom >= 0.018                      │
               ▼               │                                      │
       ┌───────────────────────┴───────┐                              │
       │           ANOMALOUS           │                              │
       └──────────────┬────────────────┘                              │
                      │ S_anom < 0.018                                │
                      ▼                                               │
       ┌───────────────────────────────┐   Nominal sustained >= 5.0s  │
       │           RECOVERED           ├──────────────────────────────┘
       └───────────────────────────────┘
```

- **$\text{NORMAL} \to \text{SUSPECTED}$**: Instantaneous condition $S_{anom} \ge 0.018$.
- **$\text{SUSPECTED} \to \text{ANOMALOUS}$**: Condition sustained continuously for $t \ge 3.0\text{ s}$ ($\ge 30$ consecutive steps at $10\text{ Hz}$).
- **$\text{SUSPECTED} \to \text{NORMAL}$**: Glitch or transient clears before $3.0\text{ s}$ elapsed.
- **$\text{ANOMALOUS} \to \text{RECOVERED}$**: Instantaneous condition $S_{anom} < 0.018$.
- **$\text{RECOVERED} \to \text{NORMAL}$**: Nominal condition sustained continuously for $t \ge 5.0\text{ s}$ ($\ge 50$ consecutive steps at $10\text{ Hz}$).
- **$\text{RECOVERED} \to \text{SUSPECTED}$**: Anomaly reoccurs before $5.0\text{ s}$ recovery completed.

---

## 4. Physics-Informed Multi-Hypothesis Diagnosis

The `PhysicsInformedDiagnoser` evaluates observed normalized residuals ($z_k = \frac{r_k}{\text{MAD}_k}$), physical cross-channel consistency, data quality rejection flags, and cylinder runner spreads against empirical causal signatures.

### 4.1 Provenance Hierarchy

Every diagnostic heuristic is explicitly tagged:
- `MODEL_CALIBRATION`: Derived from Rotax 914 physics definitions (thermodynamics, heat transfer equations, hydraulic models).
- `ENGINEERING_HEURISTIC`: Empirically validated diagnostic matching rule based on physical domain knowledge.
- `SYNTHETICALLY_VALIDATED`: Validated against synthetic datasets generated by the Phase 4 engine simulator.

### 4.2 Canonical Fault Taxonomy & Signatures

| Fault ID | Canonical Name | Primary Physical Signature | Cross-Channel Consistency Invariant | Localization | Provenance |
|---|---|---|---|---|---|
| **F1** | `INJECTOR_DELIVERY_ABNORMALITY` | $z_{\text{fuel}} < -1.5$, $z_{\text{egt}} > +1.5$ (lean) | Thermocouple response lag observed | Cylinder runner $\arg\max(\Delta T_{egt, k})$ | `ENGINEERING_HEURISTIC` |
| **F2** | `LUBRICATION_DEGRADATION` | $z_{\text{oil\_p}} < -1.5$, $r_{\text{oil\_p}} < -0.2\text{ bar}$ | $r_{\text{oil\_t}} > +0.8^\circ\text{C}$ (frictional heating) | Engine-level (Subsystem: LUBRICATION) | `MODEL_CALIBRATION` |
| **F3** | `COOLING_DEGRADATION` | $z_{\text{cht}} > +1.5$ or $z_{\text{cool\_t}} > +1.5$ | Both CHT & coolant temps rise; oil pressure nominal | Engine-level (Subsystem: THERMAL) | `MODEL_CALIBRATION` |
| **F4** | `COMBUSTION_MISFIRE` | $r_{\text{egt}} < -100^\circ\text{C}$, $z_{\text{egt}} < -3.0$ | Runner collapse $\Delta T_{egt, k} < -100^\circ\text{C}$ | Cylinder runner $\arg\min(\Delta T_{egt, k})$ | `SYNTHETICALLY_VALIDATED` |
| **F5** | `MECHANICAL_DEGRADATION` | $z_{\text{vib}} > +1.5$ (vibration surge) | Combustion & thermal states nominal ($|z| < 1.5$) | Engine-level (Subsystem: MECHANICAL) | `ENGINEERING_HEURISTIC` |
| **F6** | `SENSOR_BIAS` | Single channel $|z_k| > 1.5$, window drift rate $\le \theta_{\text{drift}}$ | Coupled channels show zero rise ($|r_{\text{coupled}}| < 0.5$) | Channel-level isolation | `SYNTHETICALLY_VALIDATED` |
| **F6** | `SENSOR_DRIFT` | Single channel $|z_k| > 1.5$, window drift rate $> \theta_{\text{drift}}$ | Coupled channels show zero rise ($|r_{\text{coupled}}| < 0.5$) | Channel-level isolation | `SYNTHETICALLY_VALIDATED` |
| **F7** | `SENSOR_DROPOUT` | Telemetry flag `MISSING`, `NON_FINITE`, `DROPOUT` | Quality validator channel rejection | Channel-level isolation | `SYNTHETICALLY_VALIDATED` |
| **F7** | `SENSOR_STUCK` | Telemetry flag `STALE`, bitwise identical float | Quality validator channel rejection | Channel-level isolation | `SYNTHETICALLY_VALIDATED` |

---

## 5. Measured F1–F7 Validation Table

The following table records actual generated numerical measurements from simulation and DigitalTwin runs (exported to `evidence/phase6_fault_diagnosis_matrix.json`):

| Fault Mode | Severity | Operating Point | Primary Residuals | Normalized Residuals ($z$) | Cylinder Evidence | Diagnosis Result | Compat. | Conf. | Expected Observed | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| **Nominal Cruise** | 0.0 | Cruise (75%, 2000m) | CHT: -0.07°C, Oil_P: +0.015 bar, Fuel: +0.16 L/h | CHT: -0.01, Oil_P: +0.03, Fuel: +0.08 | EGT spread: 32.5°C, CHT spread: 3.1°C | `HEALTHY` | 1.00 | 1.00 | YES | `SYNTHETICALLY_VALIDATED` |
| **F1 Injector Lean** | 0.75 | Cruise (75%, 2000m) | Fuel: -4.33 L/h, EGT: +46.05°C, CHT: -9.86°C | Fuel: -2.17, EGT: +1.84, CHT: -0.99 | EGT spread: 34.6°C, Runners: [29.7, 43.6, 64.4, 36.7] | `INJECTOR_DELIVERY_ABNORMALITY` | 0.65 | 0.65 | YES | `ENGINEERING_HEURISTIC` |
| **F2 Lubrication** | 0.50 | Cruise (75%, 2000m) | Oil_P: -1.02 bar, Oil_T: +2.11°C, CHT: -0.02°C | Oil_P: -2.04, Oil_T: +0.26, CHT: -0.00 | N/A (Engine-level) | `LUBRICATION_DEGRADATION` | 0.85 | 0.85 | YES | `MODEL_CALIBRATION` |
| **F3 Cooling** | 0.85 | Cruise (75%, 2000m) | CHT: +34.52°C, Coolant: +3.68°C, Oil_T: +0.65°C | CHT: +3.45, Coolant: +0.61, Oil_T: +0.08 | N/A (Engine-level) | `COOLING_DEGRADATION` | 0.65 | 0.65 | YES | `MODEL_CALIBRATION` |
| **F4 Misfire** | 0.60 | Cruise (75%, 2000m) | EGT: -128.4°C, RPM: -382 RPM, CHT: -14.2°C | EGT: -5.14, RPM: -3.82, CHT: -1.42 | Runner 3 EGT: -145.2°C, Cyl localized: 3 | `COMBUSTION_MISFIRE` | 0.85 | 0.85 | YES | `SYNTHETICALLY_VALIDATED` |
| **F5 Mechanical** | 0.65 | Cruise (75%, 2000m) | Vib: +0.84 g, EGT: +1.2°C, Oil_P: +0.01 bar | Vib: +4.20, EGT: +0.05, Oil_P: +0.02 | N/A (Engine-level) | `MECHANICAL_DEGRADATION` | 1.00 | 0.95 | YES | `ENGINEERING_HEURISTIC` |
| **F6 Sensor Bias** | +25°C step | Cruise (75%, 2000m) | CHT: +25.0°C, Coolant: -0.02°C, Oil_T: +0.24°C | CHT: +2.50, Coolant: -0.00, Oil_T: +0.03 | N/A (Channel: cht) | `SENSOR_BIAS` | 0.92 | 0.92 | YES | `SYNTHETICALLY_VALIDATED` |
| **F6 Sensor Drift** | -0.3 bar/s | Cruise (75%, 2000m) | Oil_P: -1.80 bar, Oil_T: +0.25°C, RPM: +12 RPM | Oil_P: -3.60, Oil_T: +0.03, RPM: +0.12 | N/A (Channel: oil_pressure) | `SENSOR_DRIFT` | 0.92 | 0.92 | YES | `SYNTHETICALLY_VALIDATED` |
| **F7 Dropout** | NaN packet | Cruise (75%, 2000m) | Fuel: NaN, CHT: nominal, Oil_P: nominal | Fuel: NaN, others nominal | N/A (Channel: fuel_flow) | `SENSOR_DROPOUT` | 0.95 | 0.84 | YES | `SYNTHETICALLY_VALIDATED` |
| **F7 Stuck** | 82.5°C const | Cruise (75%, 2000m) | Coolant: stuck, others nominal | Coolant: static, others nominal | N/A (Channel: coolant_temp) | `SENSOR_STUCK` | 0.95 | 0.84 | YES | `SYNTHETICALLY_VALIDATED` |

---

## 6. Execution Latency Characterization

A 1,000-step continuous benchmark was executed via `scripts/benchmark_phase6_latency.py` evaluating end-to-end processing (`DigitalTwin.update(rec)`: ingestion $\to$ quality validation $\to$ synchronization $\to$ prediction $\to$ residuals $\to$ health index $\to$ detection $\to$ diagnosis).

### 6.1 Measured Latency Metrics

| Metric | Measured Value | Budget | Compliance Status |
|---|---|---|---|
| **Steady-State Mean** | **0.388 ms** | 20.0 ms (50 Hz) | **PASS** (Effective rate: 2,576 Hz) |
| **Steady-State Median** | **0.328 ms** | 20.0 ms (50 Hz) | **PASS** |
| **Steady-State p95** | **0.559 ms** | 20.0 ms (50 Hz) | **PASS** |
| **Steady-State p99** | **0.763 ms** | 20.0 ms (50 Hz) | **PASS** |
| **Cold-Start Step 0** | **0.631 ms** (post-import) | 20.0 ms | **PASS** (initial call) |
| **Overall Worst-Case Maximum** | **28.994 ms** (spiking to 29.937 ms) | 20.0 ms | **EXCEEDS BUDGET** (Runtime GC / OS scheduling jitter) |

### 6.2 Latency Finding & Honest Verdict

- **Steady-State Streaming**: Over 99% of streaming executions execute in $< 0.77\text{ ms}$, representing a $26\times$ safety margin beneath the $20.0\text{ ms}$ real-time budget.
- **Worst-Case Jitter**: A single isolated step spike of $28.994\text{ ms}$ (and previously $29.937\text{ ms}$) was observed across 1,000 steps due to CPython memory allocation garbage collection and Windows thread context switching.
- **Official Latency Verdict**: **PASS WITH LIMITATIONS** (Steady-state streaming exceeds 50 Hz real-time requirements; single-step worst-case maximum exceeds 20.0 ms budget without real-time kernel priority pinning).

---

## 7. Comprehensive Verification Suite Breakdown

The expanded verification suite in `tests/test_fault_detection_diagnosis_phase6.py` comprises **32 focused tests** (100% passing):

### 7.1 Detection Lifecycle (7 tests)
1. `test_detection_healthy_steady_state`: Nominal cruise produces `NORMAL`, $S_{anom} = 0.0$, `is_anomalous = False`.
2. `test_detection_healthy_transient`: Throttle steps (60% $\to$ 100% $\to$ 40%) do not confirm `ANOMALOUS`.
3. `test_detection_short_anomaly_under_3s`: Fault active for 1.8s transitions to `SUSPECTED` but never `ANOMALOUS`.
4. `test_detection_persistent_anomaly_over_3s`: Fault active for 4.5s confirms `ANOMALOUS` ($t \ge 3.0\text{ s}$).
5. `test_detection_recovery_behavior`: Cleared fault enters `RECOVERED` and returns to `NORMAL` after 5.0s.
6. `test_detection_insufficient_data_coverage_gate`: Valid primary channels $< 5$ forces `INSUFFICIENT_DATA`.
7. `test_cylinder_only_anomaly_does_not_create_engine_anomaly`: **BLOCKER 1 REGRESSION TEST**: Isolated cylinder 1 runner spread (+75°C) preserves $HI_{raw} \ge 0.98$, $S_{anom} < 0.018$, and detector remains strictly `NORMAL`.

### 7.2 Physical Faults (5 tests)
8. `test_f1_injector_abnormality`: Fuel flow drop + EGT rise diagnosed as `INJECTOR_DELIVERY_ABNORMALITY`.
9. `test_f2_lubrication_degradation`: Oil pressure drop + oil temp rise diagnosed as `LUBRICATION_DEGRADATION`.
10. `test_f3_cooling_degradation`: Thermal rise across CHT and coolant diagnosed as `COOLING_DEGRADATION`.
11. `test_f4_combustion_misfire`: Severe EGT drop ($<-100^\circ\text{C}$) diagnosed as `COMBUSTION_MISFIRE`, localized to cyl 3.
12. `test_f5_mechanical_degradation`: Vibration surge without fluid/thermal rise diagnosed as `MECHANICAL_DEGRADATION`.

### 7.3 Sensor Faults (4 separate tests)
13. `test_f6_sensor_bias`: Step offset (+25°C) on CHT without thermal rise diagnosed as `SENSOR_BIAS`.
14. `test_f6_sensor_drift`: Monotonic ramp (-0.3 bar/s) on oil pressure diagnosed as `SENSOR_DRIFT`.
15. `test_f7_sensor_dropout`: Missing / NaN fuel flow packet flagged by quality layer as `SENSOR_DROPOUT`.
16. `test_f7_sensor_stuck`: Bitwise frozen coolant temp reading flagged by quality layer as `SENSOR_STUCK`.

### 7.4 Diagnosis Disambiguation & Localization (6 tests)
17. `test_diagnosis_cylinder_1_localization`: Localized fuel delivery restriction attributed to Cylinder 1.
18. `test_diagnosis_cylinder_3_localization`: Localized misfire collapse attributed to Cylinder 3.
19. `test_diagnosis_ambiguous_evidence_yields_unknown`: Unclassifiable multi-subsystem signature yields `UNKNOWN` / `is_ambiguous`.
20. `test_diagnosis_conflicting_evidence_insufficient_evidence`: Corrupted multi-channel input yields `INSUFFICIENT_EVIDENCE`.
21. `test_diagnosis_sensor_fault_not_promoted_to_physical`: Large oil pressure bias without oil temp rise is NOT promoted to F2.
22. `test_diagnosis_physical_fault_not_reduced_to_sensor`: Genuine physical lubrication degradation is NOT reduced to sensor bias.

### 7.5 Adversarial Stress (6 tests)
23. `test_adversarial_healthy_high_altitude`: High altitude cruise (4500m, 0.57 bar) maintains healthy `NORMAL` status.
24. `test_adversarial_rapid_throttle_transition`: Fast throttle cycling (30% $\to$ 90% $\to$ 20% $\to$ 80%) does not confirm `ANOMALOUS`.
25. `test_adversarial_noisy_vibration`: Zero-mean Gaussian sensor noise ($\sigma = 0.08\text{ g}$) produces no false alarms.
26. `test_adversarial_missing_egt`: Single missing EGT channel preserves 8/9 coverage without coverage failure.
27. `test_adversarial_stale_observation`: Frozen bus observations handled gracefully without numerical overflow.
28. `test_adversarial_single_channel_corruption`: Out-of-range spike ($10^9$ on coolant_temp) rejected by data quality.

### 7.6 Integrity & Quarantine (4 tests)
29. `test_integrity_anti_leakage`: Static signature inspection verifies zero ground-truth parameters (`fault_state`, `severity`, etc.).
30. `test_integrity_non_circular_dependency`: Static AST verification confirms non-circular imports and zero simulator imports.
31. `test_integrity_deterministic_reset_replay`: `twin.reset()` wipes detector/diagnoser history; identical streams replay bitwise deterministically.
32. `test_integrity_legacy_xgboost_quarantine`: Verifies legacy `fault_diagnosis/` and `XGBoostFaultClassifier` are not imported or reachable.

---

## 8. Final Repository Verification & Git Status

- **Phase 6 Test Suite**: **32 passed, 0 failed** in 1.66s.
- **Full Repository Test Suite (`pytest tests/ -q`)**: **468 passed, 2 skipped** in 58.88s.
- **Working Tree**: Clean.
- **Target Branch**: `rotax-914-greybox-engine`.
- **Target Branch State**: Ahead of origin by approved commits; `main` branch **untouched**.
