# Digital Twin State Model, Synchronization & Observability Specification

**Project**: SIH26054 — AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs  
**Authoritative Reference Architecture**: BRP-Rotax 914 UL/F (EASA TCDS E.122)  
**Phase**: Phase 3 — Digital Twin State Synchronization & Observability  
**Status**: Authoritative Technical Specification  

---

## 1. Digital Twin Architecture

The Phase 3 Digital Twin State Estimation and Observability architecture decouples incoming mission commands from internal physical states, forward predictions, telemetry observations, residuals, and state corrections.

```
[Pilot / Autopilot Commands] (throttle, altitude, ambient temp, airspeed proxy, mission phase)
            │
            ▼
[Reduced-Order Physics Engine] (Rotational Dynamics, Turbocharger, Thermal ODEs, Lubrication)
            │
            ▼
[Predicted Telemetry (y_pred)]
            │
            ├──────────────────────────┐
            ▼                          ▼
[Residual Generator] ◄─── [Data Quality Validator] ◄─── [Raw Observed Telemetry (y_obs)]
  (r = y_obs - y_pred)      (Bounds & Temporal Checks)
            │                          │
            ▼                          ▼
[Bounded State Estimator] ◄────────────┘
  (Sub-stepping, clamp, lag filter)
            │
            ▼
[Canonical State Vector & Heuristic Confidence]
```

### Core Architecture Principles:
1. **Sensor Fault Isolation**: Observed RPM and sensor measurements are **never** fed into the forward physics model as inputs to calculate expected engine load or thermal generation.
2. **Deterministic & Bausal**: The twin operates with complete numerical reproducibility; identical inputs and configurations produce identical state trajectories.
3. **Transparent Heuristics**: State tracking confidence is formulated from explicit weights and never falsely claimed as statistical probability.

---

## 2. Canonical State Vector Definition

The internal state of the twin is formally captured in `CanonicalTwinState` (`digital_twin/state.py`), partitioned into typed, unit-aware subsystem state containers:

### A. Rotational State (`RotationalState`)
* Crankshaft speed $\omega_{crank}$ (`rpm`, [RPM] and `omega_rad_s`, [rad/s])
* Propeller shaft speed $\omega_{prop}$ (`propeller_rpm`, [RPM], kinematically linked via 51/21 gearbox)
* Indicated and target mechanical power (`indicated_power_w`, `power_target_w`, [W])
* Net engine torque and propeller load torque (`engine_torque_nm`, `propeller_torque_nm`, [N*m])
* Engine load fraction (`engine_load_pct`, [%])

### B. Air & Boost State (`AirBoostState`)
* Manifold Absolute Pressure (`map_bar`, [bar] and `manifold_pressure_pa`, [Pa])
* Compressor charge air temperature (`charge_air_temp_c`, [°C])
* Static ambient pressure & temperature (`ambient_pressure_bar`, [bar], `ambient_temp_c`, [°C])
* Intake air mass flow rate (`air_mass_flow_kg_s`, [kg/s])
* Compressor pressure ratio $\Pi_c$ (`pressure_ratio`, [-])
* TCU wastegate duty fraction (`wastegate_position`, [ratio 0.0–1.0])
* Internal turbo spool speed (`turbo_shaft_speed`, [RPM], surrogate)
* Compressor aerodynamic efficiency (`compressor_aerodynamic_efficiency`, [-], surrogate)

### C. Combustion State (`CombustionState`)
* Fuel mass consumption rate (`fuel_mass_flow_kg_s`, [kg/s])
* Volumetric fuel supply flow rate (`fuel_flow_l_h`, [L/h])
* Air-fuel ratio AFR (`air_fuel_ratio`, [-])
* Thermal combustion efficiency (`combustion_efficiency`, [-])

### D. Thermal State (`ThermalState`)
* Per-cylinder head temperatures (`cht_cyl1_c` .. `cht_cyl4_c`, [°C])
* Per-cylinder exhaust gas temperatures (`egt_cyl1_c` .. `egt_cyl4_c`, [°C])
* Effective cooling circuit temperature (`coolant_temp_c`, [°C])
* **Canonical Oil Temperature** (`oil_temp_c`, [°C])
* Heat rejected to heads and radiator (`q_heads_w`, `q_radiator_w`, [W])

### E. Lubrication State (`LubricationState`)
* Main gallery oil pressure (`oil_pressure_bar`, [bar])
* **Oil Temperature View** (`oil_temp_c`): Exposes a **read-only property view** directly accessing `ThermalState.oil_temp_c`. Guarantees single-source-of-truth integrity.

### F. Vibration State (`VibrationState`)
* Structural crankcase vibration RMS level (`vibration_rms_g`, [g])
* 1st engine order frequency ($f_{1X} = \text{RPM} / 60$, [Hz])
* 2nd engine order frequency ($f_{2X} = 2 \cdot f_{1X}$, [Hz])

### G. Electrical Subsystem (`ElectricalState`)
* Channels: `battery_voltage_v`, `battery_current_a`, `bus_voltage_v`.
* **Honest Disclosure**: Unmodeled in Phase 3. Explicitly initialized to `NaN` with status `QuantityStatus.UNAVAILABLE` and `DataQualityStatus.UNAVAILABLE`. No fake battery equations are fabricated.

---

## 3. Strict State Classification Boundaries

The architecture enforces absolute boundary separation between six operational categories:

| Category | Notation | Definition |
| :--- | :---: | :--- |
| **Command / Input** | $u$ | Pilot/autopilot commands: throttle, altitude, ambient temperature, mission phase. |
| **Latent Physical State** | $x$ | Internal dynamic ODE states: crank speed, turbo spool, head thermal capacitances, oil thermal capacitance. |
| **Observed Telemetry** | $y_{obs}$ | Telemetry measurements received from sensors with data quality audit tags. |
| **Predicted Telemetry** | $y_{pred}$ | Forward nominal physics predictions: $y_{pred} = h(x_{pred}, u)$. |
| **Residuals** | $r, r_{norm}$ | Signed error ($r = y_{obs} - y_{pred}$) and scale-normalized dimensionless residual ($r_{norm} = r / \sigma$). |
| **Confidence Indicator** | $C_{synced}$ | Deterministic heuristic tracking metric bounded in $[0.0, 1.0]$. |

---

## 4. Predicted Telemetry Formulation

Nominal forward predictions $y_{pred}$ are computed causally from command inputs $u$ and current latent states $x$:

1. **Rotational Speed**:
   $$J_{eq} \frac{d\omega}{dt} = Q_{eng}(\text{throttle}, \rho_{amb}, \text{MAP}) - Q_{prop}(\omega) - Q_{fric}(\omega)$$
2. **Manifold Absolute Pressure**:
   $$\text{MAP} = P_{amb} \cdot \Pi_c(\dot{m}_{air}, \text{wastegate})$$
3. **Fuel Flow Rate (Willans-line model)**:
   $$\dot{m}_{fuel} = a_{fuel} \cdot P_{target} + b_{fuel}, \quad \text{Flow}_{L/h} = \frac{\dot{m}_{fuel}}{\rho_{fuel}} \times 3600$$
4. **Exhaust Gas Temperature (EGT)**:
   $$\text{EGT}_{target} = T_{base} + k_{load} \cdot \text{Load} + k_{rpm} \cdot \Delta\text{RPM} - k_{\rho} \cdot \rho_{factor}$$
5. **Cylinder Head Temperature (CHT)**:
   $$C_{th} \frac{dT_{cht}}{dt} = \dot{Q}_{comb} - h_{cool}(v_{air}, \omega) \cdot (T_{cht} - T_{amb})$$
6. **Oil Circuit Temperature & Pressure**:
   $$C_{oil} \frac{dT_{oil}}{dt} = \dot{Q}_{oil,gen} - h_{oil} \cdot (T_{oil} - T_{amb}) - k_{couple} \cdot (T_{oil} - T_{cht})$$
   $$P_{oil} = P_{base} + k_p \cdot \omega - k_T \cdot (T_{oil} - 50.0)$$

---

## 5. State Synchronization Algorithm & Sub-Stepping Policy

### Bounded Estimator Correction Equations
The synchronized state estimate is updated via dimensionally consistent observer innovations:

For dynamic states (RPM, MAP, fuel flow, oil pressure, vibration):
$$\hat{x}_{synced} = x_{pred} + K_i \cdot \Delta t_{eff} \cdot \text{clamp}(y_{obs,i} - y_{pred,i}, -\delta_{max,i}, +\delta_{max,i})$$
where:
* $K_i$ has dimensions $[1/\text{s}]$.
* $\Delta t_{eff} = \min(\Delta t, \Delta t_{max})$.
* $K_i \cdot \Delta t_{eff} \le 1.0$ guarantees non-divergent numerical stability.
* $\delta_{max,i}$ clamps maximum allowable single-step innovation.

For thermal states (CHT, EGT, oil temp, coolant temp):
$$\hat{T}_{synced} = T_{pred} + \left(\frac{\Delta t_{eff}}{\tau_{sync}}\right) \cdot \text{clamp}(T_{obs} - T_{pred}, -\Delta T_{max}, +\Delta T_{max})$$
where $\tau_{sync}$ values are **estimator filter calibration parameters** ($\tau_{sync,cht} = 2.0\text{ s}, \tau_{sync,oil} = 5.0\text{ s}$). They are **not** experimentally measured physical Rotax time constants, but calibration parameters designed to prevent unphysical step discontinuities in reduced-order tracking.

### Deterministic Gap Sub-Stepping Policy (No Time Compression)
When an observation interval $\Delta t_{actual} = t_{curr} - t_{last}$ exceeds $\Delta t_{max}$ ($0.2\text{ s}$):
1. Compute $N = \lfloor \Delta t_{actual} / \Delta t_{max} \rfloor$.
2. Step forward physics $N$ times using $\Delta t_{max}$.
3. Step forward physics 1 remainder step with $\Delta t_{rem} = \Delta t_{actual} - (N \cdot \Delta t_{max})$.
4. Final state timestamp strictly equals $t_{curr}$.
5. Zero physical time compression or state truncation.

---

## 6. Timestamp Semantics & Irregular Intervals

1. **Advancing Timestamps ($t_{curr} > t_{last}$)**: Processed normally with $\Delta t = t_{curr} - t_{last}$.
2. **Duplicate Timestamps ($t_{curr} == t_{last}$)**: Flagged `DataQualityStatus.DUPLICATE_TIMESTAMP`. Packet is rejected from state update without advancing physics.
3. **Non-Monotonic Timestamps ($t_{curr} < t_{last}$)**: Flagged `DataQualityStatus.NON_MONOTONIC_TIMESTAMP`. Packet is rejected without time rewind.
4. **Excessive Future Timestamps ($t_{curr} > t_{sim} + \Delta t_{horizon}$)**: Flagged `DataQualityStatus.FUTURE_TIMESTAMP`. Evaluated when an external reference clock is provided; rejected from state update.
5. **Observation Staleness**: If $\Delta t_{actual} > \tau_{stale}$ ($2.0\text{ s}$), status transitions to `STALE`. Estimator applies zero observer correction and falls back to pure physics prediction.

---

## 7. Telemetry Data Quality Taxonomy

The validator (`digital_twin/quality.py`) audits every telemetry packet:

* `VALID`: Value within instrument limits and timestamp monotonic. Advancing timestamp with constant value is recognized as legitimate steady-state and remains `VALID`.
* `MISSING`: Field absent or `None`.
* `NON_FINITE`: `NaN` or `Inf`.
* `OUT_OF_RANGE`: Measurement value violates instrument physical capability bounds (e.g. RPM $< 0$ or $> 7500$, CHT $< -50\ ^\circ\text{C}$ or $> 300\ ^\circ\text{C}$).
* `STALE`: Packet age exceeds timeout.
* `DUPLICATE_TIMESTAMP`: Duplicate timestamp packet.
* `NON_MONOTONIC_TIMESTAMP`: Time inversion packet.
* `FUTURE_TIMESTAMP`: Packet timestamp exceeds forward horizon.
* `INVALID`: Structural format failure.
* `UNAVAILABLE`: Uninstrumented channel.

> [!IMPORTANT]
> **Physical Abnormality $\neq$ Sensor Invalidity**:
> An elevated CHT of 145 °C or a high residual is a physical anomaly, NOT an invalid sensor reading. It remains `VALID` as long as it is within physical sensor capability bounds.

---

## 8. Residual Definitions & Normalization

* **Raw Signed Residual**:
  $$r_i = y_{obs,i} - y_{pred,i}$$
* **Dimensionless Normalized Residual**:
  $$r_{norm,i} = \frac{y_{obs,i} - y_{pred,i}}{\sigma_i}$$

### Residual Scales ($\sigma_i$):
* RPM: $50.0\text{ RPM}$
* MAP: $0.05\text{ bar}$
* CHT: $5.0\ ^\circ\text{C}$
* EGT: $20.0\ ^\circ\text{C}$
* Oil Pressure: $0.3\text{ bar}$
* Oil Temperature: $3.0\ ^\circ\text{C}$
* Coolant Temperature: $3.0\ ^\circ\text{C}$
* Fuel Flow: $1.5\text{ L/h}$
* Vibration: $0.1\text{ g}$

---

## 9. Authoritative Observability Registry

The singleton `ObservabilityRegistry` (`digital_twin/observability.py`) is the sole authority for variable classifications:

| Variable | Unit | Classification | Source / Derivation Rule |
| :--- | :---: | :--- | :--- |
| `rpm` | RPM | `DIRECTLY_OBSERVED` | Calibrated inductive crank pickup |
| `cht_cyl1..4` | °C | `DIRECTLY_OBSERVED` | Cylinder head thermocouples |
| `egt_cyl1..4` | °C | `DIRECTLY_OBSERVED` | Exhaust manifold thermocouples |
| `oil_pressure` | bar | `DIRECTLY_OBSERVED` | Oil pressure transducer |
| `oil_temp` | °C | `DIRECTLY_OBSERVED` | Oil temperature sensor |
| `fuel_flow` | L/h | `DIRECTLY_OBSERVED` | Volumetric flow meter |
| `coolant_temp` | °C | `DIRECTLY_OBSERVED` | Coolant circuit sensor |
| `vibration_rms` | g | `DIRECTLY_OBSERVED` | Accelerometer RMS |
| `propeller_rpm` | RPM | `INDIRECTLY_OBSERVED` | Kinematically linked via 51/21 gearbox ($2.42857:1$) |
| `power_target_w` | W | `MODEL_DERIVED` | Target brake power lookup |
| `indicated_power_w` | W | `MODEL_DERIVED` | Cylinder indicated expansion power |
| `engine_torque_nm` | N*m | `MODEL_DERIVED` | Indicated drive torque |
| `propeller_torque_nm` | N*m | `MODEL_DERIVED` | Propeller absorption torque ($k_{prop} \cdot \omega_{prop}^2$) |
| `map_bar` | bar | `MODEL_DERIVED` | Compressor boost + intercooler drop |
| `pressure_ratio` | - | `MODEL_DERIVED` | Compressor pressure ratio |
| `wastegate_pos` | ratio | `MODEL_DERIVED` | TCU wastegate opening fraction |
| `q_heads_w` | W | `MODEL_DERIVED` | Heat rejected to cylinder heads |
| `q_radiator_w` | W | `MODEL_DERIVED` | Radiator heat dissipation |
| `turbo_shaft_speed`| RPM | `UNOBSERVED` | Surrogate spool dynamics |
| `compressor_eff` | - | `UNOBSERVED` | Surrogate isentropic efficiency |
| `battery_voltage_v`| V | `UNAVAILABLE` | Unmodeled in Phase 3 |
| `battery_current_a`| A | `UNAVAILABLE` | Unmodeled in Phase 3 |
| `bus_voltage_v` | V | `UNAVAILABLE` | Unmodeled in Phase 3 |

**Observability Coverage Ratio**:
$$O_{obsv} = \frac{N_{observed}}{N_{modeled}} = \frac{9}{18} = 0.5000 \quad (50.0\%)$$

> [!NOTE]
> **Observability Coverage Meaning**:
> This metric represents the **fraction of registered canonical state variables classified as directly or indirectly observed** within the model catalog. It does **NOT** represent a formal nonlinear observability-rank analysis (e.g. Lie derivatives or observability Gramian), nor does it claim the physical engine is "50% observable".

---

## 10. Heuristic Synchronization Confidence

The tracking confidence $C_{synced}$ is computed deterministically from explicit, documented weights summing to 1.0:

### Zero-Observation Semantics:
If there are zero valid observations ($N_{valid} = 0$):
$$Q_{obs} = 0.0, \quad M_{res} = 0.0, \quad C_{synced} \equiv 0.0$$
Registry observability coverage ($O_{obsv}$) is clamped to $0.0$ when no observations are present, preventing artificial confidence.

### General Formula ($N_{valid} > 0$):
$$C_{synced} = w_{obs} \cdot Q_{obs} + w_{res} \cdot M_{res} + w_{obsv} \cdot O_{obsv}$$
where:
* $w_{obs} = 0.40$ (Observation availability: ratio of valid channels to expected channels).
* $w_{res} = 0.35$ (Residual agreement factor: calculated **ONLY over valid channels**):
  $$M_{res} = \exp\left(-\frac{1}{N_{valid}} \sum_{i \in valid} |r_{norm,i}|\right)$$
* $w_{obsv} = 0.25$ (Architectural coverage factor: $O_{obsv} = 0.50$).
* Normalization constraint: $w_{obs} + w_{res} + w_{obsv} = 1.0$.
* Strictly bounded in $[0.0, 1.0]$.

> [!CAUTION]
> **Strict Non-Statistical Disclaimer**:
> $C_{synced}$ is an inspectable heuristic calibration metric. It is **never** presented as a statistically learned probability or Bayesian confidence interval.

---

## 11. Initialization and Operating Regime State Machine

The operating regime is governed by physical thermal and rotational thresholds:
* `COLD_START`: $T_{oil} < 40.0\ ^\circ\text{C}$ and $\text{RPM} < 1000$.
* `WARMING`: $T_{oil} < 50.0\ ^\circ\text{C}$ or $T_{cht} < 60.0\ ^\circ\text{C}$.
* `STEADY_OPERATION`: Nominal operating temperatures and speeds established.

---

## 12. Synchronization State Transitions

The twin's `sync_status` transitions deterministically:
1. `INITIALIZING`: Startup steps before filter settling ($step\_count < 3$).
2. `SYNCHRONIZED`: All expected channels valid and normalized residuals bounded ($|r_{norm}| \le 5.0$).
3. `PARTIALLY_SYNCHRONIZED`: Telemetry valid for a proper subset of expected channels.
4. `DEGRADED_OBSERVABILITY`: All channels present but significant model-observation disagreement ($|r_{norm}| > 5.0$).
5. `STALE`: Observation age exceeds $\tau_{stale}$ ($2.0\text{ s}$). Pure physics fallback.
6. `UNSYNCHRONIZED`: Zero valid observations or temporal packet rejection.

---

## 13. Known Limitations & Technical Disclosures

1. **Reduced-Order Physics**: Turbocharger is a reduced-order algebraic surrogate; cooling is an effective lumped single-node thermal surrogate.
2. **Estimator Calibration**: Filter gains $K_i$ and time constants $\tau_{sync}$ are hand-tuned model calibration parameters for bounded correction, not experimentally validated physical Rotax constants or optimal Kalman gains.
3. **No Experimental Flight Validation**: Physics calibrations are referenced to published OEM documentation (EASA TCDS E.122, Operator's Manual), not UAV flight-test data.
4. **No Airworthiness or Flight Certification**: The digital twin is a research and prototyping engineering model, not flight-certified avionics software.
5. **Observability Metric Scope**: $O_{obsv} = 0.5000$ reflects architectural registry coverage, not formal nonlinear observability-rank analysis.
6. **Benchmark Performance**: The reported ~0.925 ms/step (~1080 updates/s) reflects host-side software computational throughput under benchmark conditions; it is not a claim of 1 kHz avionics capability, flight certification, or guaranteed scheduler latency.
7. **Deterministic Replay Scope**: Replay determinism is guaranteed for identical runtime, configuration, and input conditions; no universal cross-platform IEEE bitwise claims across heterogeneous CPU architectures or compiler optimizations are asserted.
8. **Uncalibrated Statistical Uncertainty**: Residuals are deterministic errors, not covariance-calibrated Gaussian white noise.
9. **Scope Discipline**: Phase 3 does not perform fault diagnosis, classification, or prognostic RUL estimation.

---

## 14. Observability Summary

* **Directly Measured**: Crank RPM, CHT 1..4, EGT 1..4, Oil Pressure, Oil Temp, Fuel Flow, Coolant Temp, Vibration RMS.
* **Model Derived**: Engine Torque, Propeller Torque, MAP, Indicated Power, Radiator Heat Flux.
* **Unobserved Surrogates**: Turbo Spool Speed, Compressor Aerodynamic Efficiency.
* **Unavailable / Unmodeled**: Main bus battery voltage, alternator current, avionics bus voltage.

---

## 15. Scope Boundary: What is NOT Implemented

To protect architectural modularity, the following capabilities are explicitly deferred to Phase 4+:
* **NO Machine Learning**: No XGBoost, Random Forest, or neural network anomaly detection.
* **NO Forecasting**: No TimesFM foundation model or multi-step time-series forecasting.
* **NO Prognosis**: No Remaining Useful Life (RUL) estimation or degradation curves.
* **NO Fault Classification**: No sensor fault vs engine failure diagnostic classifier.
* **NO Hardware Communications**: No live MAVLink, CAN-bus, or UDP socket listener.
* **NO Dashboard Redesigns**: Existing dashboard interfaces are preserved.
