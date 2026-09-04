# Phase 11: Remaining Useful Life (RUL) Estimation & Prognostics

Phase 11 provides the **Remaining Useful Life (RUL) & Prognostics** layer for project **SIH26054** (*AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero-Piston Engines used in MALE UAVs*).

> [!NOTE]
> **Validation Scope**: The prognostics pipeline is **validated on the project's synthetic degradation scenarios**. Statistical calibration and airworthiness certification for **real aero-piston engines are NOT established**.

Phase 11 answers the critical flight operations question:
> **"Given the current engine health and forecasted degradation trajectory, how much operational flight time remains before the engine reaches an End-of-Life (EOL) functional failure limit?"**

```
+----------------------------------------------------------------------------------------------------+
|                                    PHASE 11 RUL ARCHITECTURE                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Phase 9 Health Index]               [Phase 10 Telemetry Forecast]       [Phase 9 Sensor Isolation]|
|         |                                          |                                     |         |
|         | HI_smooth, dHI/dt                        | Predicted channels                  | Excluded|
|         v                                          v                                     v         |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 1: Pipeline Context Guards (Warmup duration >= 32.0 s, Active valid channels >= 4)       | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v                                                                                          |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 2: Weakest-Link EOL Evaluator (HI <= 0.35 OR unisolated physical redlines breached)      | |
|  |          -> If breached: RULStatus.CRITICAL_EOL_REACHED (RUL = 0.0 s)                           | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v (HI > 0.35, no redlines breached)                                                        |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 3: Robust Theil-Sen Median Pairwise Slope Estimation (dHI/dt, SE(beta))                  | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v                                                                                          |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 4: Mathematically Exhaustive State Machine (Partitions all finite (HI, dHI/dt))          | |
|  |                                                                                               | |
|  |   * HI >= 0.85:                                                                               | |
|  |       - dHI/dt > +0.0005                 --> RECOVERING (RUL = NaN)                            | |
|  |       - -0.001 <= dHI/dt <= +0.0005      --> NOT_DEGRADING (RUL = NaN)                         | |
|  |       - dHI/dt < -0.001                  --> ACTIVE_DEGRADATION (Dispatch Prognostics)         | |
|  |                                                                                               | |
|  |   * 0.35 < HI < 0.85:                                                                         | |
|  |       - dHI/dt > +0.0005                 --> RECOVERING (RUL = NaN)                            | |
|  |       - -0.0005 <= dHI/dt <= +0.0005     --> INDETERMINATE_TREND (RUL = NaN)                   | |
|  |       - dHI/dt < -0.0005                 --> ACTIVE_DEGRADATION (Dispatch Prognostics)         | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v (Active Degradation)                                                                     |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 5: Dual-Horizon Synthesis & Phase 10 Handoff Safety Guard                                 | |
|  |   * LOADED_PRETRAINED + horizon in {16, 32}  --> TIMESFM_FORECAST_ASSISTED (H = 16 or 32 s)    | |
|  |   * LOCAL_UNCHECKPOINTED_GRAPH               --> REJECTED (H = 0 s, Theil-Sen fallback)         | |
|  |   * BLOCKED_UNAUTHENTICATED_GATED            --> REJECTED (H = 0 s, Theil-Sen fallback)         | |
|  |   * BASELINE                                 --> BASELINE_EWMA_ASSISTED (H = 16 or 32 s)        | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v                                                                                          |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 6: Monte Carlo Trajectory Propagation (M = 500 realizations)                             | |
|  |   * Health state perturbation: delta_HI ~ N(0, 0.02^2)                                          | |
|  |   * Degradation slope perturbation: delta_beta ~ N(0, SE(beta)^2)                               | |
|  |   * EOL threshold tolerance: HI_EOL ~ U(0.33, 0.37)                                             | |
|  |   * Outputs: Median (P50), Conservative lower bound (P05), Optimistic upper bound (P95)         | |
|  +-----------------------------------------------------------------------------------------------+ |
|         |                                                                                          |
|         v                                                                                          |
|  +-----------------------------------------------------------------------------------------------+ |
|  | STEP 7: Authoritative RUL Output Packaging                                                      | |
|  |   * If P50 > 86400 s (24 h): RULStatus.EXCEEDS_HORIZON (>24h)                                  | |
|  |   * If sensor isolated in Phase 9: RULStatus.DEGRADED_PROGNOSTIC (20% confidence discount)      | |
|  |   * Otherwise: RULStatus.ACTIVE_DEGRADATION                                                     | |
|  +-----------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Phase 10 Handoff Safety Contract

Prognostics relies on upstream forecasting from Phase 10. To prevent unverified models from corrupting safety-critical RUL calculations, `prognostics.trajectory.DualHorizonSynthesizer` strictly enforces the following handoff policy:

| Phase 10 Model Status | Horizon | Prognostic Usability | Resulting Trajectory Type | Rationale |
|---|---|---|---|---|
| `LOADED_PRETRAINED` | 16 or 32 s | **USABLE** ($H \in \{16, 32\}$) | `TIMESFM_FORECAST_ASSISTED` | Genuine checkpoint verified with causal preconditioning. |
| `LOCAL_UNCHECKPOINTED_GRAPH` | Any | **STRICTLY REJECTED** ($H = 0$) | `ROBUST_LINEAR_PRIMARY` | Architectural/API verification only. Never influences RUL. |
| `BLOCKED_UNAUTHENTICATED_GATED` | Any | **STRICTLY REJECTED** ($H = 0$) | `ROBUST_LINEAR_PRIMARY` | Checkpoint blocked or unauthenticated. Falls back to Theil-Sen. |
| `BASELINE` | 16 or 32 s | **CONDITIONALLY USABLE** ($H \in \{16, 32\}$) | `BASELINE_EWMA_ASSISTED` | Transparent baseline mode. Never misrepresented as TimesFM. |

---

## 3. Mathematically Exhaustive State Machine

The state machine classifies engine operational health into mutually exclusive states.

### Proof of Mutual Exclusivity and Exhaustiveness
For all finite $(HI, \dot{HI}) \in (-\infty, +\infty) \times (-\infty, +\infty)$:
1. If $HI \le 0.35 \implies \text{CRITICAL\_EOL\_REACHED}$.
2. For all $HI > 0.35$:
   - Case $HI \ge 0.85$:
     $$\dot{HI} \in (-\infty, -0.001) \implies \text{ACTIVE\_DEGRADATION}$$
     $$\dot{HI} \in [-0.001, +0.0005] \implies \text{NOT\_DEGRADING}$$
     $$\dot{HI} \in (+0.0005, +\infty) \implies \text{RECOVERING}$$
     Since $(-\infty, -0.001) \cup [-0.001, +0.0005] \cup (+0.0005, +\infty) = \mathbb{R}$, this partition is complete and disjoint.
   - Case $0.35 < HI < 0.85$:
     $$\dot{HI} \in (-\infty, -0.0005) \implies \text{ACTIVE\_DEGRADATION}$$
     $$\dot{HI} \in [-0.0005, +0.0005] \implies \text{INDETERMINATE\_TREND}$$
     $$\dot{HI} \in (+0.0005, +\infty) \implies \text{RECOVERING}$$
     Since $(-\infty, -0.0005) \cup [-0.0005, +0.0005] \cup (+0.0005, +\infty) = \mathbb{R}$, this partition is complete and disjoint.

Every possible combination maps to exactly one and only one state.

---

## 4. Weakest-Link End-of-Life (EOL) Failure Boundaries

All EOL criteria are project-defined simulated engineering boundaries for prototype demonstration:

| Boundary Name | Channel | Value | Comp | Provenance Tag | Engineering Rationale |
|---|---|---|---|---|---|
| **Health Index EOL** | `health_index` | $0.35$ | $\le$ | `phase_9_critical_state_boundary` | Phase 9 threshold for CRITICAL state; multi-subsystem divergence $> 4\sigma$. |
| **CHT Redline** | `cht` | $150.0\ ^\circ\text{C}$ | $\ge$ | `telemetry_warning_bound_repurposed` | Repurposed operational warning bound representing cylinder head thermal ceiling. |
| **Minimum Oil Pressure** | `oil_pressure` | $1.2\text{ bar}$ | $\le$ | `project_defined_failure_assumption` | Simulated hydrodynamic film collapse limit (above 0.8 bar idle minimum). |
| **Maximum Oil Temp** | `oil_temp` | $140.0\ ^\circ\text{C}$ | $\ge$ | `project_defined_failure_assumption` | Simulated lubricant thermal cracking limit (exceeding 130 C warning bound). |
| **Structural Vibration** | `vibration` | $3.5\text{ g}$ | $\ge$ | `telemetry_warning_bound_repurposed` | Repurposed operational warning bound representing severe mechanical unbalance. |

### Sensor Isolation Inheritance
If Phase 9 isolates an observation-layer sensor fault (e.g. CHT thermocouple bias), that channel is excluded from physical redline evaluations. This prevents instrumentation artifacts from causing false EOL alarms.

---

## 5. Engineering Uncertainty Assumptions

> [!IMPORTANT]
> **Prognostic Uncertainty Disclaimer**:
> The Monte Carlo distributions ($M=500$) implemented in Phase 11 represent **physics-informed engineering uncertainty assumptions for synthetic-data prognostics, NOT statistically calibrated real-engine confidence distributions**.

Uncertainty components:
1. **Health State Uncertainty**: $\delta_{HI} \sim \mathcal{N}(0, 0.02^2)$
2. **Slope Estimation Uncertainty**: $\delta_\beta \sim \mathcal{N}(0, \text{SE}(\beta)^2)$
3. **Threshold Tolerance**: $HI_{\text{EOL}} \sim \mathcal{U}(0.33, 0.37)$

The pipeline reports:
- $P_{50}$: Median point estimate (s)
- $P_{05}$: Conservative lower bound (s)
- $P_{95}$: Optimistic upper bound (s)

---

## 6. NASA PHM08 & Evaluation Metrics

Phase 11 implements the NASA PHM08 asymmetric scoring function (Saxena et al., 2008):
$$d_j = \widehat{\text{RUL}}_j - \text{RUL}_{\text{true}, j}$$
$$s_j = \begin{cases} \exp(-d_j / 13) - 1, & d_j < 0 \quad (\text{conservative / early}) \\ \exp(d_j / 10) - 1, & d_j \ge 0 \quad (\text{hazardous / late}) \end{cases}$$
$$S = \sum_{j=1}^N s_j$$

### Mathematical Explanation of Score Magnitude ($\sim 10^{30}$)
In the original Saxena et al. (2008) formulation, the late penalty exponent uses a fixed divisor of 10 ($s = \exp(d / 10) - 1$). In challenge datasets where time is measured in operational flight cycles (e.g. $d \in [5, 50]$ cycles), penalties remain between $\exp(0.5) \approx 1.65$ and $\exp(5.0) \approx 148.4$. 

However, in our telemetry pipeline, time is tracked in seconds ($s$). When a degradation trend is late by $d = +700\text{ s}$ relative to a premature physical redline breach, the penalty exponent evaluates to:
$$s = \exp\left(\frac{700}{10}\right) - 1 = \exp(70) - 1 \approx 2.51 \times 10^{30}$$
This exponential explosion is mathematically exact and expected when unscaled second-units are fed into the standard PHM08 formulation with divisor 10. Late predictions are severely and exponentially penalized as hazardous flight conditions.

### Prediction Interval Coverage Probability (PICP) & Width (MPIW)
- **$\text{PICP} \ge 90\%$** is treated as an **evaluation target**, not a hard acceptance gate.
- **MPIW** (Mean Prediction Interval Width, $P_{95} - P_{05}$) is **always reported alongside PICP** to evaluate interval sharpness.
- **Uncertainty Calibration Status**: The empirical PICP across 853 synthetic evaluations was **52.75%**, meaning the $\ge 90\%$ target was **NOT achieved**. The intervals must **NOT be described as statistically calibrated**. They reflect physics-informed engineering assumptions for synthetic simulation, not real-engine calibrated distributions.

---

## 7. Empirical Synthetic RUL Performance Evaluation

The prognostics pipeline was evaluated in a strict, leakage-safe protocol on the project's synthetic degradation scenarios. At each observation timestamp $t_{\text{obs}}$, all future telemetry was strictly hidden from the pipeline. Predictions were compared against the simulated ground-truth EOL timestamps ($T_{\text{EOL}}$).

### Evaluation Matrix & Benchmark Results

| Scenario | Category | EOL Boundary | True EOL (s) | Active $N$ | MAE (s) | RMSE (s) | PHM08 Score | PICP (%) | MPIW (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Coupled Multi-Fault** (Thermal+Lube+Mech) | Physical Simulator | `REDLINE_CHT` | 68.0 | 36 | 7.16 | 8.15 | 34.47 | 41.7% | 83.24 |
| **Severe Lubrication Degradation** | Physical Simulator | `REDLINE_OIL_TEMP` | 206.0 | 111 | 253.88 | 331.59 | $8.16 \times 10^{30}$ | 27.9% | 4268.99 |
| **Healthy Nominal Cruise** (Negative Control) | Physical Simulator | None | None | 0 | N/A | N/A | N/A | N/A | N/A |
| **Fast Progressive Wear** | Controlled Progressive | `GLOBAL_HEALTH_INDEX` | 131.0 | 84 | 44.87 | 92.20 | $1.16 \times 10^{17}$ | 64.3% | 58.97 |
| **Moderate Progressive Wear** | Controlled Progressive | `GLOBAL_HEALTH_INDEX` | 274.0 | 213 | 56.40 | 78.14 | $2.20 \times 10^{13}$ | 44.1% | 92.58 |
| **Gradual Long Wear** | Controlled Progressive | `GLOBAL_HEALTH_INDEX` | 277.0 | 212 | 43.53 | 70.06 | $2.76 \times 10^{14}$ | 54.7% | 81.61 |
| **Stochastic Brownian Wear** | Controlled Progressive | `GLOBAL_HEALTH_INDEX` | 249.0 | 197 | 37.30 | 88.17 | $2.21 \times 10^{28}$ | 71.1% | 83.35 |
| **GLOBAL COMBINED** | **ALL** | **Multi-Factor** | — | **853** | **71.27** | **140.31** | **$8.18 \times 10^{30}$** | **52.75%** | **627.49** |

### Key Empirical Findings:
1. **Coupled Physical Degradation Accuracy**: On multi-subsystem degradation where CHT redline governs failure, the pipeline achieved an MAE of **7.16 seconds** and RMSE of **8.15 seconds** with a low NASA PHM08 score of **34.47**.
2. **Exponential Penalty in NASA PHM08**: In single-fault lubrication degradation, the engine Health Index degraded mildly while the physical oil temperature rose rapidly to breach 140°C. Because the Theil-Sen HI extrapolation anticipated a longer wear horizon than the sudden redline crossing, late predictions incurred the characteristic exponential penalty of $\exp(d/10)$ under the NASA PHM08 metric, resulting in a large penalty score ($8.16 \times 10^{30}$).
3. **Uncertainty Interval Coverage**: The global Prediction Interval Coverage Probability (**PICP**) achieved **52.75%** with a Mean Prediction Interval Width (**MPIW**) of **627.49 seconds**. As mandated, the $\ge 90\%$ threshold was treated honestly as an evaluation target rather than forcing an artificial pass.
4. **False Alarm Immunity (Negative Control)**: On nominal healthy cruise missions, the pipeline produced **zero false positive RUL estimations**, correctly dispatching 117 observations as `NOT_DEGRADING` and 31 as `RECOVERING`.

---

## 8. Monte Carlo CPU Latency Benchmark

The Monte Carlo trajectory propagation engine ($M = 500$ realizations) was benchmarked across all $N = 853$ prognostic evaluations:

| Metric | Measured Value | Performance Target | Compliance Status |
|---|---|---|---|
| **Realization Count ($M$)** | 500 samples | 500 samples | Nominal specification |
| **Mean Latency** | **2.60 ms** | $< 150.0\text{ ms}$ | **MEASURED COMPLIANT** |
| **Median Latency** | **2.46 ms** | $< 150.0\text{ ms}$ | **MEASURED COMPLIANT** |
| **P95 Latency** | **4.19 ms** | $< 150.0\text{ ms}$ | **MEASURED COMPLIANT** |
| **P99 Latency** | **7.09 ms** | $< 150.0\text{ ms}$ | **MEASURED COMPLIANT** |

The measured P95 latency of **4.19 ms** is well below the operational $< 150.0\text{ ms}$ performance target on the evaluation hardware, ensuring seamless real-time execution in 1 Hz avionics telemetry streams.

---

## 9. Verification & Validation Status Summary

| Area | Status | Verification Scope & Findings |
|---|---|---|
| **Unit & Behavioral Tests** | 🟢 **PASS** | 19/19 dedicated tests pass (`tests/test_prognostics.py`), verifying state machine exhaustiveness, boundary conditions, Phase 10 handoff rejection, and metric formulations. |
| **Phases 1–10 Regression** | 🟢 **PASS** | 273/273 regression tests pass with zero modifications to Phases 1–10 code (292 total repository tests passing). |
| **Point-Estimation RUL** | 🟢 **EMPIRICALLY EVALUATED** | Evaluated on 853 synthetic prognostic observations across physical and progressive wear scenarios; MAE (71.27 s) and RMSE (140.31 s) documented. |
| **Uncertainty Calibration** | 🟡 **NOT YET SATISFACTORY** | Measured PICP was 52.75% against the $\ge 90\%$ evaluation target; MPIW was 627.49 s. Intervals are uncalibrated engineering assumptions. |
| **Monte Carlo Latency** | 🟢 **TARGET MET** | $M=500$ CPU latency measured at 2.60 ms mean / 4.19 ms P95 ($<150\text{ ms}$ performance target). |
| **Real-Engine Airworthiness** | ❌ **NOT ESTABLISHED** | Synthetic simulator and engineering assumptions only; real aero-piston accuracy/airworthiness is NOT established. |


