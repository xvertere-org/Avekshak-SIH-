# Phase 9 — Health Index + Degradation Tracking

## 1. Objective & Scope

Phase 9 implements the deterministic, causal health assessment layer of the SIH26054 predictive health monitoring (PHM) system.

While Phase 7 answers:
> **"Is the engine behaving abnormally?"** (Binary / Unsupervised Anomaly Detection)

And Phase 8 answers:
> **"What fault is most likely causing the abnormal behavior?"** (Supervised Multiclass Fault Diagnosis)

Phase 9 answers:
> **"How healthy is the engine right now, and is its condition getting worse?"** (Continuous Degradation Tracking)

Phase 9 maps physical deviations across normalized residual channels into a continuous, bounded Health Index (HI) and degradation velocity without introducing another machine learning model.

> [!IMPORTANT]
> **HI $\ne$ RUL Boundary:** The Health Index represents the current physical operational integrity on a bounded $[0.0, 1.0]$ scale. It is **never** remaining useful life (RUL), is **never** remaining flight hours, and must **never** be described as "percentage of engine life remaining."

> [!CAUTION]
> **Engineering Threshold Disclaimer:** Channel weights, threshold parameters, and health bands represent project-defined engineering assumptions for the SIH26054 aero-piston prototype. They are **not** certified aerospace airworthiness or OEM engine operating limits.

---

## 2. Mathematical Formulation

### 2.1 Channel Degradation Evidence $d_i$
Each core normalized residual $r_i \in \{\text{rpm, cht, egt, oil\_temp, oil\_pressure, fuel\_flow, vibration}\}$ is mapped to channel degradation evidence $d_i \in [0.0, 1.0]$:

$$d_i = \begin{cases} 
0.0, & |r_i| \le \tau_{\text{nominal}} \\
\frac{|r_i| - \tau_{\text{nominal}}}{\tau_{\text{critical}} - \tau_{\text{nominal}}}, & \tau_{\text{nominal}} < |r_i| < \tau_{\text{critical}} \\
1.0, & |r_i| \ge \tau_{\text{critical}}
\end{cases}$$

- $\tau_{\text{nominal}} = 1.5\sigma$ (nominal flight noise and minor governor fluctuations deadband)
- $\tau_{\text{critical}} = 5.0\sigma$ (severe physical divergence threshold)
- If $r_i$ is `NaN` (e.g. sensor dropout), $d_i$ is preserved as `NaN`. Missing values are **never** replaced by 0.0.

### 2.2 Baseline Channel Weights

| Channel | Default Weight $w_i$ | Failure Mode Safety Rationale |
|---|---|---|
| `oil_pressure` | 0.20 | Direct barrier against journal bearing seizure and crankshaft lockup |
| `cht` | 0.18 | Piston crown thermal seizure and cylinder head structural warping |
| `egt` | 0.16 | Detonation, exhaust valve burnout, severe combustion imbalance |
| `oil_temp` | 0.16 | Thermal degradation of oil film viscosity, accelerated friction wear |
| `vibration` | 0.14 | Bearing raceway spalling, connecting rod fatigue, propeller imbalance |
| `rpm` | 0.10 | Governor tracking error, severe friction drag |
| `fuel_flow` | 0.06 | Fuel metering discrepancy, injection flow abnormality |

Weights sum to 1.0, are non-negative, and are fully configurable via `HealthIndexConfig`.

---

## 3. Data Quality, Missing Channels & Dynamic Renormalization

Telemetry channels can suffer temporary dropouts or disconnections, producing `NaN`.

1. **Insufficient Evidence Threshold**: If fewer than 4 of the 7 core channels contain valid numeric values, physical engine health cannot be reliably determined. The pipeline returns:
   - `health_state = "INSUFFICIENT_DATA"`
   - `data_quality = "INSUFFICIENT_DATA"`
   - `raw_health_index = NaN`
   - `smoothed_health_index = NaN`
   - `degradation_rate = NaN`
   - `degradation_trend = "INSUFFICIENT_DATA"`

2. **Dynamic Weight Renormalization**: When at least 4 channels are valid, active channel weights are dynamically normalized:
   $$w'_i = \frac{w_i}{\sum_{j \in \mathcal{A}} w_j}$$
   The resulting $w'_i$ are preserved in `effective_channel_weights` for complete transparency and auditability.

---

## 4. Sensor-Fault Decoupling & Isolation

Sensor faults are observation-layer anomalies (e.g., stuck thermocouple, drift, or wiring disconnection). A sensor fault must **not** automatically plunge the physical Health Index to zero.

### 4.1 Isolation Criteria
A channel is isolated from the physical Health Index calculation under two deterministic conditions:
1. **Explicit Upstream Context (Optional)**: Phase 8 diagnosis reports `predicted_fault_type == "sensor_fault"` with $\ge 0.60$ confidence.
2. **Deterministic Disconnect Heuristic (Core Phase 9)**: One channel exhibits $|r_k| \ge 4.5\sigma$ while all other valid physical channels remain within $|r_j| \le 1.2\sigma$ continuously over a timestamped span $\ge 5.0\text{ s}$ without timestamp gaps $> 2.0\text{ s}$.

> [!NOTE]
> This disconnect rule is a project-defined observation-quality heuristic intended to identify likely sensor anomalies. It is **not** a certified physical impossibility criterion.

### 4.2 Effect of Isolation
- Isolated channels are placed into `excluded_channels`.
- The physical Health Index continues computing across the remaining uncorrupted physical channels with dynamically renormalized weights.
- `data_quality` is flagged as `SENSOR_ISOLATED`.

---

## 5. Causal Smoothing & Degradation Tracking

### 5.1 Causal EWMA Filter
Instantaneous combustion flutter is smoothed using an Exponentially Weighted Moving Average:
$$\text{smoothed\_health\_index}(t) = \alpha \cdot \text{raw\_health\_index}(t) + (1 - \alpha) \cdot \text{smoothed\_health\_index}(t - \Delta t)$$
- Strict Causality: $\text{smoothed\_health\_index}(t)$ depends **only** on data at or before time $t$. No centered windows or future samples.
- Default $\alpha = 0.15$.

### 5.2 Timestamp-Based Degradation Rate
Degradation rate uses actual elapsed telemetry timestamps over a configurable horizon $\Delta t_{\text{horizon}} = 10.0\text{ s}$:
$$\text{rate}(t) = \frac{\text{smoothed\_health\_index}(t) - \text{smoothed\_health\_index}(t_{\text{ref}})}{t - t_{\text{ref}}}$$
Where $t_{\text{ref}}$ is the latest sample at or before $t - 10.0\text{ s}$.

**Initial History Rule**: When elapsed history $t - t_{\text{start}} < 10.0\text{ s}$, `degradation_rate` is `NaN` and `degradation_trend` is `INSUFFICIENT_HISTORY`.

### 5.3 Degradation Trends
- `RAPIDLY_DEGRADING`: $\text{rate} \le -0.010\text{ s}^{-1}$ (losing $>1\%$ health per second)
- `DEGRADING`: $-0.010 < \text{rate} \le -0.001\text{ s}^{-1}$
- `IMPROVING`: $\text{rate} \ge +0.001\text{ s}^{-1}$
- `STABLE`: otherwise
- `INSUFFICIENT_HISTORY`: elapsed history $< 10.0\text{ s}$
- `INSUFFICIENT_DATA`: valid physical channels $< 4$

### 5.4 Health States
- `HEALTHY`: $HI_{\text{smooth}} \ge 0.85$
- `DEGRADED`: $0.60 \le HI_{\text{smooth}} < 0.85$
- `SEVERELY_DEGRADED`: $0.35 \le HI_{\text{smooth}} < 0.60$
- `CRITICAL`: $HI_{\text{smooth}} < 0.35$
- `INSUFFICIENT_DATA`: valid channels $< 4$

### 5.5 Mission State Isolation & Timestamp Robustness
- **Mission Isolation**: All smoother, degradation tracker, and sensor isolation states are strictly keyed by `(engine_id, mission_id)`. Subsequent missions executed on the same engine never inherit EWMA state, degradation history, or timestamps from preceding flights.
- **Non-Positive Elapsed Time ($dt \le 0$)**: Duplicate timestamps ($dt = 0$) or out-of-order samples ($dt < 0$) deterministically yield `degradation_rate = NaN` and `degradation_trend = INSUFFICIENT_DATA` without corrupting causal history or encountering division by zero.
- **Timestamp Gaps**: If telemetry gap exceeds `max_timestamp_gap_s` (default $5.0\text{ s}$), degradation history is broken and smoother is reset to avoid calculating synthetic degradation across missing time horizons. `INSUFFICIENT_HISTORY` is reported until a fresh continuous window of $\ge 10.0\text{ s}$ is accumulated.

---

## 6. Phase 7 and Phase 8 Decoupling

Phase 9 is completely independent of upstream phases:
- `HealthIndexPipeline.process_dataframe(df)` runs on `ResidualFrame` alone without requiring Phase 7 or Phase 8.
- When available, upstream outputs (`predicted_fault_type`, `anomaly_status`, etc.) can be passed as optional context for audit and enriched provenance.

---

## 7. Assumptions and Limitations

1. **Synthetic Telemetry Baseline**: Validated exclusively on physics-informed synthetic aero-piston telemetry from the project digital twin.
2. **Generalization Boundary**: Real MALE-UAV engine generalization is **not established**.
3. **Observation-Layer Decoupling**: Sensor isolation is a heuristic; novel physical failure modes manifesting strictly in a single channel initially could be classified as observation anomalies until cross-coupling develops.
