# Phase 8 — Degradation Modeling & Remaining Useful Life (RUL)

## Scientific Disclaimer & Epistemic Boundary

> [!IMPORTANT]
> **SYNTHETIC RESEARCH PROTOTYPE ONLY**
> The degradation models, robust trend estimation algorithms, and remaining useful life (RUL) extrapolation equations implemented in Phase 8 represent a grey-box digital twin engineering prototype for UAV propulsion research and software verification.
>
> They DO NOT claim:
> - Rotax 914 OEM failure distributions or certified fleet wear statistics.
> - Operational flightworthiness certification or maintenance approval (EASA CS-E / FAA FAR 33).
> - Direct physical end-of-life (EOL) measurement.
> - Accurate real-world engine-life prediction.
>
> NASA C-MAPSS and TimesFM foundation models remain **quarantined / legacy-only**; no turbofan-to-piston equivalence is fabricated.

---

## 1. Architectural Design & Hierarchy

Phase 8 sits strictly downstream in the unidirectional Digital Twin pipeline:

```text
Phase 2 Physics Simulation (Rotax 914 Grey-Box Engine Dynamics)
        ↓
Phase 3 State Synchronizer & Observability Registry
        ↓
Phase 4 Causal Fault Injection Layer
        ↓
Phase 5 Residual Generation & Health Assessment (H_sub, HI_raw, HI_smooth)
        ↓
Phase 6 Anomaly Detection (S_anom = 1 - HI_raw) & Multi-Hypothesis Diagnosis
        ↓
Phase 7 Physics-Constrained Synthetic Engine Population (Train/Val/Test Splits)
        ↓
Phase 8 Degradation State & Uncertainty-Aware RUL Estimation Layer
```

### Architectural Invariants
1. **Unidirectional Dependency**: Telemetry $\to$ Synchronizer $\to$ Residuals $\to$ Health $\to$ Degradation Estimator $\to$ RUL Estimator.
2. **Zero Circular Feedback**: RUL projections and degradation states NEVER feed back into the simulator dynamics, synchronizer, residual generators, health evaluator, anomaly detector, or diagnoser.
3. **Zero Ground-Truth Label Leakage**: The runtime estimators take only observable timestamps, health assessments, residual vectors, and operational scenario context. They never inspect `fault_type`, `fault_id`, `fault_severity`, `failure_time`, or `EOL_time`.

---

## 2. Degradation State Representation

Phase 8 defines an explicit, typed degradation state across 6 core engine subsystem dimensions:

```python
class DegradationSubsystem(str, Enum):
    THERMAL_DEGRADATION = "THERMAL_DEGRADATION"
    LUBRICATION_DEGRADATION = "LUBRICATION_DEGRADATION"
    COMBUSTION_DEGRADATION = "COMBUSTION_DEGRADATION"
    MECHANICAL_DEGRADATION = "MECHANICAL_DEGRADATION"
    FUEL_SYSTEM_DEGRADATION = "FUEL_SYSTEM_DEGRADATION"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"
```

### Primary Degradation Index
The normalized health-derived degradation index $D(t)$ is computed from Phase 5 EWMA-smoothed health:
$$D(t) = 1.0 - HI_{smooth}(t), \quad D(t) \in [0.0, 1.0]$$
with instantaneous raw indicator:
$$D_{raw}(t) = 1.0 - HI_{raw}(t)$$

### Subsystem Degradation Breakdown
Each subsystem maintains an independent degradation indicator $D_{sub}(t) = 1.0 - H_{sub}(t)$, tracking localized trends without corrupting engine-level invariants.

---

## 3. Robust Theil-Sen Trend Estimation

Rather than relying on Ordinary Least Squares (OLS) which is sensitive to single-point telemetry outliers, Phase 8 employs the non-parametric **Theil-Sen robust estimator** ($29.3\%$ breakdown point).

### Slope Formulation
For $N$ historical observations $(t_i, D_i)$ within rolling window $W$:
$$s_{ij} = \frac{D_j - D_i}{t_j - t_i}, \quad \forall 1 \le i < j \le N$$
The robust median degradation slope is:
$$\text{slope}_{median} = \text{median}(\{s_{ij}\})$$

### Uncertainty Quantiles
Rather than assuming Gaussian errors, slope uncertainty bounds are extracted directly from empirical quantiles of historical pairwise slopes:
$$\text{slope}_{low} = Q_{0.15}(\{s_{ij}\})$$
$$\text{slope}_{high} = Q_{0.85}(\{s_{ij}\})$$

### Execution Speed & Real-Time Decimation
To guarantee soft real-time execution ($< 5.0\text{ms}$ latency):
- When $N > 50$, the rolling buffer is deterministically decimated to 50 evenly spaced points spanning the window $[t_0, t_{curr}]$.
- The maximum evaluated pairs is bounded to $M = \frac{50 \times 49}{2} = 1,225$, requiring $< 15\mu\text{s}$ per evaluation.

---

## 4. Operational Degradation Regimes

Degradation trajectories are categorized into 4 mutually exclusive regimes based on robust slope per hour ($dD/dh$):

| Regime | Condition | Physical Meaning |
|---|---|---|
| `INSUFFICIENT_DATA` | $N < 10$ samples or $\Delta t < 15\text{s}$ or $C_{data} < 0.35$ | Gated: insufficient evidence to infer trend |
| `STABLE` | $|dD/dh| < 0.02/\text{hour}$ | Nominal engine baseline or zero-mean noise |
| `DEGRADING` | $0.02 \le dD/dh < 0.20/\text{hour}$ | Gradual persistent progressive wear |
| `RAPID_DEGRADATION` | $dD/dh \ge 0.20/\text{hour}$ | Rapid local degradation trend |

> [!NOTE]
> **Non-Linear Wear Classification**: `RAPID_DEGRADATION` captures steep local degradation trends (e.g. accelerating quadratic wear). The estimator extrapolates linearly using the instantaneous local Theil-Sen slope; it does NOT fit higher-order polynomial curvatures or claim non-linear wear parameter identification. For accelerating trajectories, linear extrapolation provides a conservative instantaneous projection that will overestimate remaining time as wear accelerates further.

---

## 5. End-of-Life (EOL) Horizon

### Provenance: `ENGINEERING_HEURISTIC`
Rotax does not publish proprietary or certified health-index-to-failure curves. Therefore, EOL is defined as a computational model horizon:
$$D_{EOL} = 0.50 \iff HI_{EOL} = 0.50$$

> **Definition**: Model-defined computational horizon, not an OEM maintenance limit or airworthiness limit.

---

## 6. Uncertainty-Aware RUL Equations & Horizon Projections

For linear progressive wear to horizon $D_{EOL}$, remaining headroom is:
$$\Delta D = \max(0.0, D_{EOL} - D(t))$$

Under operational scenario stress multiplier $S_{stress}$:
$$\text{effective\_slope} = \text{slope}_{median} \cdot S_{stress}$$

### Projections
$$RUL_{median} = \frac{\Delta D}{\text{effective\_slope} \cdot 3600.0} \quad [\text{hours}]$$
$$RUL_{low} = \frac{\Delta D}{\text{slope}_{high} \cdot S_{stress} \cdot 3600.0} \quad [\text{hours (pessimistic / fastest wear)}]$$
$$RUL_{high} = \frac{\Delta D}{\text{slope}_{low} \cdot S_{stress} \cdot 3600.0} \quad [\text{hours (optimistic / slowest wear)}]$$

If $\text{slope}_{low} \le 0$, $RUL_{high}$ is set to `None` (unbounded upper horizon under model assumptions).

### Empirical Uncertainty Intervals vs. Confidence Intervals
- **Interval Ordering**: $RUL_{low} \le RUL_{median} \le RUL_{high}$ holds in 100% of cases by mathematical construction ($Q_{0.15} \le \text{median} \le Q_{0.85}$). This proves mathematical ordering, NOT statistical confidence calibration.
- **True Empirical Coverage**: Evaluated by determining whether the true known future horizon falls within $[RUL_{low}, RUL_{high}]$ on unseen synthetic test trajectories ($100\%$ on evaluated linear trajectories).
- **Terminology**: Designated as **empirical uncertainty intervals under the estimator**, not formal Bayesian credible intervals.

### Minimum Data Gating & Non-Degrading Behavior
A successful RUL system must know when it does NOT have enough evidence. RUL returns `None` for hours when:
- Status is `INSUFFICIENT_DATA` ($N < 10$ or window $< 15\text{s}$).
- Status is `STABLE` (no degradation trend detected).
- Status is `NON_DEGRADING` (negative degradation slope / clearing disturbance).
- Status is `DATA_QUALITY_DEGRADED` ($C_{data} < 0.35$ or valid fraction $< 0.70$).
- Status is `ALREADY_BEYOND_MODEL_HORIZON` ($D(t) \ge D_{EOL} \implies RUL = 0.0$).

---

## 7. Operational Scenario Multipliers

Phase 8 models sensitivity to operational flight envelopes via explicit stress multipliers. All scenario multipliers are engineering heuristics rather than physically fitted damage laws:

| Scenario | Multiplier ($S_{stress}$) | Physical Rationale | Provenance |
|---|---|---|---|
| `CURRENT_PROFILE` | $1.00$ | Baseline observed mission profile | `ENGINEERING_HEURISTIC` |
| `NORMAL_MISSION` | $1.00$ | Nominal cruise/climb profile | `ENGINEERING_HEURISTIC` |
| `HIGH_ALTITUDE` | $1.15$ | Lower air density, higher turbocharger pressure ratio, reduced radiator mass flow | `ENGINEERING_HEURISTIC` |
| `HOT_DAY` | $1.30$ | Elevated ambient heat rejection limit, increased thermal accumulation | `ENGINEERING_HEURISTIC` |
| `HIGH_LOAD` | $1.50$ | Sustained maximum continuous rating (5500 RPM, 115 HP) | `ENGINEERING_HEURISTIC` |

---

## 8. Causal Fault vs. Degradation Separation & Limitations

Phase 8 explicitly differentiates transient fault disturbances from progressive degradation:

- **Case A (Transient Fault)**: A temporary F3 cooling disturbance increases residuals and raises $D(t)$ briefly. When cleared, temperatures normalize and $D(t) \to 0$. As slope becomes negative or near zero, status returns to `NON_DEGRADING` / `STABLE`, preventing false RUL countdown latching. The rolling window retains transient elevation until the window duration ($300\text{s}$) has completely elapsed, after which regime returns to `STABLE`.
- **Case B (Persistent Wear)**: Continuous synthetic wear produces monotonic positive Theil-Sen slope, shrinking RUL hours, and high trend confidence.
- **Case C (Telemetry Noise)**: Zero-mean Gaussian noise yields $|dD/dh| < 0.02$, maintaining `STABLE` status with null RUL hours.
- **Case D (Severe Sensor Bias / Dropout)**: Invalid or large sensor bias observations degrade $C_{data}$ and $C_{obs}$, triggering `DATA_QUALITY_DEGRADED` rather than manufacturing false mechanical wear.
- **Case E (Slow Sensor Drift Limitation)**: Slow unflagged sensor drift below data quality rejection thresholds can masquerade as gradual physical degradation. This is a fundamental single-sensor observability limitation.

---

## 9. Parameter Provenance Catalog

| Parameter | Value | Provenance | Scientific Documentation |
|---|---|---|---|
| `window_duration_s` | $300.0\text{s}$ | `ENGINEERING_HEURISTIC` | Rolling history buffer for trend extraction |
| `min_observations` | $10$ | `ENGINEERING_HEURISTIC` | Minimum points before fitting regression |
| `min_window_duration_s` | $15.0\text{s}$ | `ENGINEERING_HEURISTIC` | Minimum temporal span for causal stability |
| `min_valid_data_fraction` | $0.70$ | `ENGINEERING_HEURISTIC` | Reject telemetry streams with $>30\%$ missing data |
| `stable_slope_threshold_per_hour`| $0.02/\text{hr}$ | `ENGINEERING_HEURISTIC` | Threshold separating noise from persistent drift |
| `rapid_slope_threshold_per_hour` | $0.20/\text{hr}$ | `ENGINEERING_HEURISTIC` | Threshold indicating steep local degradation trend |
| `eol_threshold` ($D_{EOL}$) | $0.50$ | `ENGINEERING_HEURISTIC` | Model-defined computational horizon ($HI = 0.50$) |
| `high_altitude_stress` | $1.15$ | `ENGINEERING_HEURISTIC` | Assumed atmospheric density and turbo boost stress factor |
| `hot_day_stress` | $1.30$ | `ENGINEERING_HEURISTIC` | Assumed ISA+20K thermal rejection penalty factor |
| `high_load_stress` | $1.50$ | `ENGINEERING_HEURISTIC` | Assumed maximum continuous power exposure factor |

---

## 10. Performance & Verification Metrics

- **Total Phase 8 Tests**: 50 non-tautological unit and integration tests (`tests/test_phase8_rul.py`).
- **Full Regression**: All 563 repository tests (Phases 1–8) pass without regressions.
- **Latency Benchmarks (1000 consecutive updates)**:
  - **Phase 8 Isolated (Degradation + RUL)**: Mean: $1.40\text{ms}$, Median: $1.25\text{ms}$, P95: $2.14\text{ms}$, Max: $2.68\text{ms}$.
  - **End-to-End DigitalTwin (Physics + Observability + Health + Anomaly + Phase 8)**: Mean: $2.17\text{ms}$, Median: $2.17\text{ms}$, P95: $3.45\text{ms}$, Max: $29.86\text{ms}$.
  - **Real-Time Classification**: Soft real-time compliant ($< 5.0\text{ms}$ p95 budget satisfied). No hard real-time guarantees.
- **Memory Boundedness**: History queue is strictly bounded to $\le 305$ elements over 2,500 streaming updates ($300\text{s}$ rolling window at $1\text{Hz}$). Zero unbounded array growth.
- **Validation Matrix**: 19 base scenarios and 65 synthetic horizon prediction scenarios evaluated in `evidence/phase8_rul_matrix.json`. Zero false degradation or false collapse events.

