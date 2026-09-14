# Phase 4: Causal Fault Injection & Fault-Mechanism Modeling

## SIH26054 — Digital Twin System for Aero Piston Engines (Rotax 914 UL/F)

---

### 1. Executive Summary & Architectural Invariant

Phase 4 establishes the **causal fault-injection layer** for the Rotax 914 UL/F grey-box aero engine simulator. The core architectural invariant enforced across this phase is the strict separation between **physical plant mechanisms** and **observation-layer sensor faults**:

```
PHYSICAL FAULT (F1–F5)
  ↓
Physical mechanism degradation in subsystem ODEs / models
  ↓
True engine state (T_cht, T_egt, T_oil, P_oil, m_fuel, omega, a_rms)
  ↓
Telemetry generation (calibrated noise, units, physical clipping)
  ↓
Observed telemetry (TelemetryRecord)

SENSOR FAULT (F6–F7)
  ↓
Normal physical plant progression (TRUE ENGINE STATE UNCHANGED)
  ↓
Observation corruption (Bias, Drift, Stuck, Dropout)
  ↓
Observed telemetry (TelemetryRecord)
```

Physical faults (F1–F5) modify the internal physical dynamics, fluid balances, and thermal ODEs. They **never** directly overwrite or fabricate telemetry channels. Sensor faults (F6–F7) modify observation channels only, ensuring that true physical states remain 100% uncorrupted and finite.

---

### 2. Fault Taxonomy & Mapping

The fault taxonomy provides canonical Phase 4 naming while maintaining 100% backward compatibility with existing ML classifiers, pipelines, and schema records:

| Class ID | Canonical Phase 4 Name | Existing / Compatibility Enum | Affected Subsystem | Layer | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | `INJECTOR_DELIVERY_ABNORMALITY` | `FUEL_INJECTION_ABNORMALITY` | `FUEL`, `THERMAL` | Physical | Fuel delivery degradation $\dot{m}_{fuel,eff} = \dot{m}_{fuel,nominal}(1 - s \cdot d_{fuel})$; supports cylinder localization |
| **F2** | `LUBRICATION_DEGRADATION` | `LUBRICATION_DEGRADATION` | `LUBRICATION`, `DYNAMICS` | Physical | Pump pressure loss, mechanical friction increase, cooler heat rejection loss |
| **F3** | `COOLING_DEGRADATION` | `COOLING_DEGRADATION` | `COOLING`, `THERMAL` | Physical | Radiator convection loss, head fin conductance degradation; supports cylinder localization |
| **F4** | `COMBUSTION_MISFIRE` | `COMBUSTION_INSTABILITY` | `DYNAMICS`, `THERMAL`, `VIBRATION` | Physical | Indicated combustion efficiency degradation $\eta_{comb} = 1 - s \cdot d_{comb}$; torque ripple, 1X harmonic surge |
| **F5** | `MECHANICAL_DEGRADATION` | `MECHANICAL_DEGRADATION` | `VIBRATION`, `DYNAMICS` | Physical | Amplification of 1X/2X rotational harmonics, broadband noise surge, friction torque increase |
| **F6** | `SENSOR_BIAS`, `SENSOR_DRIFT` | `SENSOR_FAULT` (mode: `bias`/`drift`) | `SENSOR` | Observation | Additive measurement offset $y_{obs} = y_{true} + b$; linear time drift $b(t) = b_0 + r \cdot \Delta t$ |
| **F7** | `SENSOR_DROPOUT`, `SENSOR_STUCK` | `SENSOR_FAULT` (mode: `dropout`/`stuck`)| `SENSOR` | Observation | Signal loss $y_{obs} = \text{NaN}$; observation latching at $y_{obs}(t_{onset})$ |

---

### 3. Causal Physical Mechanisms & Mathematical Formulations

#### F1 — Injector / Fuel-Delivery Abnormality
* **Governing Equation**:
  $$\dot{m}_{fuel,eff,i} = \dot{m}_{fuel,nominal,i} \cdot (1 - s \cdot d_{fuel})$$
  where $d_{fuel} = 0.35$ is classified as `MODEL_CALIBRATION`.
* **Cylinder Localization**:
  When `affected_cylinder = k` ($k \in \{1, 2, 3, 4\}$), only injector $k$ experiences the delivery deficit. The global fuel flow reflects $\sum_{i=1}^4 \dot{m}_{fuel,eff,i}$.
* **Causal Downstream Propagation**:
  $$\dot{m}_{fuel,eff} \longrightarrow Q_{gen,i} = \dot{m}_{fuel,eff,i} \cdot \text{LHV} \cdot \eta_{th} \longrightarrow T_{cht,i} \text{ and } T_{egt,i}$$
  Under lean mixtures, individual runner EGT rises sharply; indicated brake power and total fuel consumption drop.

#### F2 — Lubrication Degradation
* **Governing Equations**:
  - Hydraulic Pressure Loss:
    $$P_{oil} = P_{oil,nominal} \cdot (1 - s \cdot d_{p\_loss})$$
    with $d_{p\_loss} = 0.55$ (`MODEL_CALIBRATION`).
  - Friction Torque Increase:
    $$T_{fric} = T_{fric,nominal} \cdot (1 + s \cdot d_{fric})$$
    with $d_{fric} = 0.08$ (`MODEL_CALIBRATION`).
  - Frictional Heat Dissipation to Oil:
    $$\dot{Q}_{oil,gen} = \dot{Q}_{oil,gen,nominal} \cdot (1 + s \cdot d_{heat})$$
    with $d_{heat} = 0.20$ (`MODEL_CALIBRATION`).
  - Oil Cooler Conductance Loss:
    $$h_{oil,cool} = h_{oil,cool,nominal} \cdot (1 - s \cdot d_{cool})$$
    with $d_{cool} = 0.20$ (`MODEL_CALIBRATION`).
* **Causal Downstream Propagation**:
  Lubrication degradation produces hydraulic pressure drops, elevated oil temperature, and slight crankshaft deceleration due to increased Coulomb/viscous friction.

#### F3 — Cooling Degradation
* **Governing Equations**:
  - Head Fin Convective Conductance:
    $$h_{cool,total} = h_{cool,nominal} \cdot (1 - s \cdot d_{cooling})$$
    with $d_{cooling} = 0.55$ (`MODEL_CALIBRATION`).
  - Localized Cylinder Head Conductance:
    $$h_{cool,i} = h_{cool,base} \cdot (1 - s_i \cdot d_{cooling})$$
  - Radiator Convective Rejection:
    $$h_{rad,eff} = h_{rad,nominal} \cdot (1 - s \cdot d_{cooling})$$
* **Causal Downstream Propagation**:
  Reduced fin and radiator conductance impedes thermal dissipation:
  $$C_{th} \frac{dT_{cht,i}}{dt} = Q_{gen,i} - h_{cool,i}(T_{cht,i} - T_{amb}) - h_{head\_coolant}(T_{cht,i} - T_{coolant})$$
  Both CHT and liquid coolant jacket temperatures rise continuously towards elevated equilibrium states.

#### F4 — Combustion / Misfire Instability
* **Governing Equations**:
  - Reduced-Order Indicated Combustion Efficiency:
    $$\eta_{comb,i} = \eta_{comb,nominal} \cdot (1 - s \cdot d_{comb})$$
    with $d_{comb} = 0.70$ (`MODEL_CALIBRATION`).
  - Indicated Power Deficit:
    $$P_{ind,i} = P_{chem,i} \cdot \eta_{comb,i} \cdot \eta_{ind,base}$$
  - Cylinder Heat Release Drop:
    $$Q_{gen,i} = Q_{gen,base} \cdot \eta_{comb,i}$$
  - Torque Ripple / Harmonic Imbalance:
    Cylinder misfire introduces a crankshaft torque deficit pulse:
    $$A_{1X} = A_{1X,base} + s \cdot \Delta A_{misfire}$$
    with $\Delta A_{misfire} = 0.55$ g (`MODEL_CALIBRATION`).
* **Causal Downstream Propagation**:
  Misfire drops indicated torque and unburnt reaction heat, lowering affected cylinder EGT while dynamically exciting 1X rotational vibration harmonics.

#### F5 — Mechanical / Vibration Degradation
* **Governing Equations**:
  - Crankshaft-Synchronous 1X Harmonic:
    $$A_{1X} = A_{1X,nominal} \cdot (1 + s \cdot 1.8)$$
    where $f_{1X} = \text{RPM} / 60$.
  - 2X Rotational Harmonic:
    $$A_{2X} = A_{2X,nominal} \cdot (1 + s \cdot 1.8)$$
    where $f_{2X} = 2 \cdot f_{1X}$.
  - Broadband Noise Scaling:
    $$\sigma_{vib} = \sigma_{vib,nominal} \cdot (1 + s \cdot 2.5)$$
  - Mechanical Friction Addition:
    $$T_{fric} = T_{fric,nominal} \cdot (1 + 0.08 \cdot s_{mech})$$
* **Causal Downstream Propagation**:
  Harmonic frequencies dynamically track actual engine RPM; vibration RMS rises sharply while mechanical friction slightly retards rotational acceleration.

#### F6 & F7 — Sensor Faults (Observation Layer Only)
* **Equations**:
  - Sensor Bias: $y_{obs} = y_{true} + s \cdot b_{max}$
  - Sensor Drift: $y_{obs}(t) = y_{true}(t) + s \cdot r_{drift} \cdot (t - t_{start})$
  - Sensor Dropout: $y_{obs} = \text{NaN}$
  - Sensor Stuck: $y_{obs}(t) = y_{obs}(t_{onset})$
* **Invariance**: Physical ODE integrators receive zero perturbation from sensor faults.

---

### 4. Cylinder Localization

Faults F1, F3, and F4 support cylinder localization via `affected_cylinder \in {1, 2, 3, 4}`:
- When specified, only the target cylinder receives the degradation severity $s_k = s$, while other cylinders operate at nominal baseline ($s_j = 0.0$).
- Heat balances and runner dynamics evaluate each cylinder independently:
  $$\Delta T_{cht,k} \gg \Delta T_{cht,j} \quad (j \ne k)$$
- Shared downstream effects (e.g. overall brake power, combined exhaust manifold pressure, cooling loop temperature) react naturally to the integrated multi-cylinder state.

---

### 5. Deterministic Temporal Fault Behavior

All temporal fault modulations are strictly deterministic with **zero uncontrolled RNG**:
- **Step / Windowed Fault**: Defined by `start_time` and optional `end_time`. Active if $t_{start} \le t \le t_{end}$.
- **Ramped Fault**: Controlled by `parameters["ramp_duration"]`:
  $$\text{effective\_severity}(t) = s \cdot \min\left(1.0, \max\left(0.0, \frac{t - t_{start}}{\tau_{ramp}}\right)\right)$$
- **Intermittent / Periodic Fault**: Modulated via modular arithmetic:
  $$\text{active} = (t - t_{start}) \pmod T < \delta_{duty} \cdot T$$
  Guarantees exact mathematical reproducibility across arbitrary simulation runs.

---

### 6. Fault Composition Rules

Compatible physical faults compose simultaneously without silent overwrites:
1. **F2 (Lubrication) + F5 (Mechanical)**:
   Both faults increase Coulomb/viscous mechanical friction additively:
   $$\text{friction\_factor} = 1.0 + 0.08 \cdot s_{lub} + 0.08 \cdot s_{mech}$$
2. **F3 (Cooling) + F5 (Mechanical)**:
   Cooling degradation increases CHT and coolant temperature; mechanical degradation elevates rotational harmonics and RMS vibration. Both operate concurrently on distinct physical mechanisms.
3. **Physical Fault + Sensor Fault**:
   Physical fault drives the true plant ODEs; sensor fault corrupts the resulting observation channel.

---

### 7. Label Leakage Prevention Contract

To prevent artificial diagnosis shortcuts in downstream machine learning:
- Ordinary telemetry (`TelemetryRecord`) and Digital Twin states (`CanonicalTwinState`) **never** expose ground-truth fault labels (`fault_type`, `fault_id`, `fault_severity`, `mechanism_description`) as observation channels or residual signals.
- Ground-truth fault tags in `TelemetryRecord.fault_type` and `TelemetryRecord.fault_severity` are strictly reserved for offline supervised dataset labeling.

---

### 8. Verification & Performance Summary

* **Phase 4 Validation Suite**: 22 tests passing (`tests/test_fault_injection_phase4.py`).
* **Full Repository Regression**: 411 passed, 2 skipped, 0 failed (`pytest tests/ -q`).
* **Zero Phase 3 Regressions**: All 389 baseline Phase 3 tests remain 100% intact and passing.
* **Simulation Latency**: Measured at **0.0925 ms/update** (throughput: ~10,800 steps/s), well within the Phase 4 target of `< 5 ms/update`.

---

### 9. Explicit Non-OEM / Non-Certified Assumptions

1. **Reduced-Order Calibration Constants**:
   - $d_{fuel} = 0.35$, $d_{p\_loss} = 0.55$, $d_{fric} = 0.08$, $d_{cooling} = 0.55$, $d_{comb} = 0.70$, $d_{vib} = 1.8$ are engineering calibration constants (`MODEL_CALIBRATION`), not certified Rotax failure limits.
2. **Combustion Misfire Model**:
   - Misfire is modeled via lumped thermodynamic and torque ripple approximations, not crank-angle-resolved 3D in-cylinder CFD.
3. **Cooling Circuit**:
   - Lumped 1-node liquid loop with air fins (`REDUCED_ORDER_COOLING_SURROGATE`), not multiphase radiator aerothermal CFD.
4. **Airworthiness Disclaimer**:
   - This simulator is a research-grade grey-box digital twin and is **not certified** by EASA, FAA, or BRP-Rotax for flight clearance or life-critical avionics deployment.
