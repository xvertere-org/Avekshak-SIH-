# Phase 5: Physics-Based Residual Generation and Health Assessment

## 1. System Architecture & Dependency Flow

Phase 5 establishes a causal, physics-grounded health monitoring layer for the Rotax 914 UL/F Digital Twin without machine-learning models, neural networks, or airworthiness-certified diagnostic classifications. 

The architecture strictly adheres to a unidirectional dependency flow:

```
Telemetry Record
       │
       ▼
Quality & Observability Analysis (Phase 2 & Phase 3)
       │
       ▼
Canonical Twin State Synchronization (Phase 3 Estimator)
       │
       ▼
Twin Dynamic Prediction (Grey-Box Physics Model)
       │
       ▼
Residual Generation & Normalization (Phase 5)
       │
       ▼
Channel & Subsystem Health Indicators (Phase 5)
       │
       ▼
Engine Health State & Composite Assessment (Phase 5)
```

### Strict Non-Circularity Guarantee
Under no circumstances does health assessment feed back into the Kalman filter, observer, or dynamic prediction models. The estimator tracks state purely from telemetry and physics, ensuring that degraded health assessments never perturb state estimation or dynamic predictions.

---

## 2. Channel Registry & Subsystem Primary Mapping

To eliminate double-counting and artificial vote inflation, the engine defines exactly **9 Primary Channels**, each assigned to a single owning subsystem:

| Primary Channel | Units | Primary Owning Subsystem | Default Normalization Scale ($\sigma_k$) |
|---|---|---|---|
| `rpm` | RPM | `ROTATIONAL` | $50.0$ RPM |
| `map_bar` | bar | `AIR_INDUCTION` | $0.03$ bar |
| `fuel_flow` | L/h | `FUEL` | $1.0$ L/h |
| `cht` | °C | `THERMAL` | $4.0$ °C |
| `coolant_temp` | °C | `THERMAL` | $3.0$ °C |
| `oil_temp` | °C | `THERMAL` | $3.0$ °C |
| `oil_pressure` | bar | `LUBRICATION` | $0.25$ bar |
| `egt` | °C | `COMBUSTION` | $15.0$ °C |
| `vibration` | g | `MECHANICAL` | $0.15$ g |

### Secondary and Contextual Channels (Zero Engine-Level Vote Weight)
1. **`charge_air_temp`**: Retained as secondary model-consistency diagnostic evidence. It has **zero vote weight** in engine-level health calculation to prevent artificial penalty from ambient fluctuations.
2. **Per-Cylinder Channels (`cht_cyl_1..4`, `egt_cyl_1..4`)**: Evaluated for runner imbalance and localization. They do not introduce additional votes into the engine-level physical health calculation.
3. **Cylinder Spread Metrics (`cylinder_cht_spread`, `cylinder_egt_spread`)**: Evaluated for imbalance diagnostics within combustion and thermal subsystems.

---

## 3. Mathematical Formulations

### 3.1 Physical Residual
For each observable channel $k$:
$$r_k(t) = y_k^{obs}(t) - \hat{y}_k^{pred}(t)$$
where $y_k^{obs}$ is the validated telemetry measurement and $\hat{y}_k^{pred}$ is the grey-box twin's expected value. Physical residuals retain their true physical engineering units.

### 3.2 Normalized Residual
Using frozen scale factor $\sigma_k^{cal}$:
$$z_k(t) = \frac{r_k(t)}{\sigma_k^{cal}}$$
In nominal operation, $\mathbb{E}[r_k] = 0$.

### 3.3 Channel Health Indicator
Using heuristic thresholds $\tau_{nom} = 1.5$ and $\tau_{crit} = 5.0$:
$$h_k(t) = \begin{cases} 0.0, & |z_k(t)| \le \tau_{nom} \\ \frac{|z_k(t)| - \tau_{nom}}{\tau_{crit} - \tau_{nom}}, & \tau_{nom} < |z_k(t)| < \tau_{crit} \\ 1.0, & |z_k(t)| \ge \tau_{crit} \end{cases}$$
$$H_k(t) = 1.0 - h_k(t)$$
where $H_k \in [0, 1]$.

### 3.4 Subsystem and Engine Physical Health Aggregation
Each subsystem aggregates its valid primary channels strictly via the **arithmetic mean**:
$$H_{sub}(S_k) = \frac{1}{|\mathcal{K}_{sub}^{valid}|} \sum_{k \in \mathcal{K}_{sub}^{valid}} H_k(t)$$
Engine physical health ($H_{phys}$ / $HI_{raw}$) is the **equal-weighted arithmetic mean** over all active subsystems ($\mathcal{S}_{active}$):
$$H_{phys}(t) = \frac{1}{|\mathcal{S}_{active}|} \sum_{S_k \in \mathcal{S}_{active}} H_{sub}(S_k)$$

#### Separate Explicit Diagnostic Metrics
To support localized inspection without corrupting the physics-consistency index $HI_{raw}$:
- `worst_channel_score = min_{k} H_k`: Tracked per subsystem and at the engine level as a diagnostic alerting indicator.
- `min_subsystem_score = min_{S_k} H_{sub}(S_k)`: Tracked at the engine level to identify the primary degraded subsystem.
These metrics do **not** affect $HI_{raw}$ or $H_{phys}$.

### 3.5 Observability Coverage & Gating
With 9 primary channels:
$$C_{obs} = \frac{N_{valid\_primary}}{9}$$
- **Minimum Valid Channels**: $N_{min} = 5$ ($C_{obs} \ge 5/9 \approx 0.556$).
- **Coverage Gate**: If $N_{valid\_primary} < 5$, the evaluator immediately transitions to:
  $$\text{State} = \text{UNAVAILABLE}, \quad HI_{raw} = \text{NaN}$$
  No artificial physical degradation is fabricated due to missing telemetry.

### 3.6 Data Quality Confidence Independence
$C_{data}$ is sourced directly from Phase 3 `heuristic_confidence` (based on sensor quality, staleness, and rate-of-change).
$C_{data}$ and $C_{obs}$ are tracked strictly independently of $H_{phys}$. Telemetry degradation reduces confidence but **never penalizes $HI_{raw}$**.

An optional composite index is provided for reporting:
$$HI_{cov\_adj} = H_{phys} \cdot C_{obs} \cdot C_{data}$$

### 3.7 Temporal Persistence and EWMA Smoothing
1. **EWMA Smoothing**:
   $$\bar{H}_t = (1 - \alpha) \bar{H}_{t-1} + \alpha H_{phys}(t), \quad \alpha = 0.15$$
2. **Persistence State Machine**:
   - $\tau_{nom} = 1.5$, $\tau_{crit} = 5.0$.
   - Transitions to `WATCH`, `DEGRADED`, or `CRITICAL` require sustained violation for $\Delta t \ge 3.0\text{ s}$.
   - Recovery transitions require sustained nominal conditions for $\Delta t \ge 5.0\text{ s}$.

---

## 4. Frozen Scale Calibration Lifecycle

The baseline calibration scale must be derived once from a dedicated **healthy calibration dataset** and permanently frozen:
1. **Calibration Phase**: Run nominal engine operating profile over a healthy flight profile ($N \ge 100$ steps).
2. **MAD Estimation**: Compute median absolute deviation:
   $$\sigma_k = 1.4826 \cdot \text{median}(|r_k - \text{median}(r_k)|)$$
   Floor at $\sigma_{min} > 0$ to prevent division by zero.
3. **Freeze Calibration**: Store scales in an immutable `FrozenScaleCalibration` instance.
4. **Independent Evaluation**: Run the evaluation or streaming pipeline using the frozen calibration object. Calibration scales are **never updated** during streaming operation.

---

## 5. Causal Fault Verification Matrix (F1–F7)

Verification tests confirm that physical faults propagate through the engine subsystems in accordance with thermodynamic and mechanical laws:

| Fault Code | Injected Fault Mechanism | Directional Causal Assertion | Secondary / Localization Signatures |
|---|---|---|---|
| **F1** | Injector delivery restriction (Cyl 1) | $r_{fuel\_flow} < 0$, $r_{egt} > 0$, $r_{cht} < 0$ | $r_{egt\_cyl\_1} > 0$, runner imbalance $> 0$ |
| **F2** | Lubrication oil pump wear | $r_{oil\_pressure} < 0$, $r_{oil\_temp} > 0$ | Friction heating elevates thermal load |
| **F3** | Radiator heat rejection loss | $r_{coolant\_temp} > 0$, $r_{cht} > 0$ | Water jacket heat accumulation |
| **F4** | Ignition misfire (Cyl 3) | $r_{vibration} > 0$, localized cylinder deviation | Unburnt charge in runner 3 |
| **F5** | Mechanical bearing wear | $r_{vibration} > 0$ | Kinetic vibration escalation |
| **F6** | Sensor CHT bias | CHT channel shifts by bias; cross-channel physics healthy | $r_{coolant\_temp}, r_{oil\_temp}, r_{egt}$ remain nominal |
| **F7** | Sensor CHT dropout (NaN) | Sensor rejected; $C_{obs} = 8/9$; $HI_{raw}$ stays healthy | Channel status flagged `UNAVAILABLE` |

---

## 6. Execution Benchmarking & Performance

A 1,000-step continuous simulation benchmark of `DigitalTwin.update(telemetry)` on the host system demonstrated:
- **Mean Step Latency**: **$0.314\text{ ms}$**
- **95th Percentile (p95)**: **$0.470\text{ ms}$**
- **99th Percentile (p99)**: **$0.751\text{ ms}$**
- **Maximum Step Latency**: **$20.388\text{ ms}$** (initial warm-up)

At a nominal $50\text{ Hz}$ telemetry rate ($20\text{ ms}$ cycle budget), the combined grey-box twin, residual generator, and health evaluator consumes $< 2\%$ of the available cycle time.

---

## 7. Non-Airworthiness Prototype Disclaimer

> [!CAUTION]
> **RESEARCH PROTOTYPE ONLY**: This health assessment architecture is an academic grey-box engineering prototype developed for simulation and algorithmic evaluation under SIH26054. The normalization thresholds ($\tau_{nom}=1.5$, $\tau_{crit}=5.0$), EWMA parameters ($\alpha=0.15$), and persistence limits ($\Delta t=3.0\text{ s}$) are **`ENGINEERING_HEURISTIC`** and **`FROZEN_SYNTHETIC_BASELINE_MAD`** parameters. They do not constitute OEM-certified airworthiness limits, DO-178C/DO-254 compliance, or FAA/EASA certified flight clearances for the Rotax 914 UL/F engine.
