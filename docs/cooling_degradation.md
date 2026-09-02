# SIH26054: Cooling Degradation Physics & Validation (Phase 4B)

## 1. Executive Summary & Objective

Phase 4B implements a physically grounded cooling-system degradation fault within the reduced-order grey-box aero piston engine simulator of the **SIH26054 Digital Twin System**.

> ### ⚠️ Engineering Reference & Scope Disclaimer:
> This cooling degradation model is a **reduced-order engineering fault model** designed for synthetic telemetry generation, digital twin residual tracking, and PHM diagnostic algorithm training. It is **NOT** a certified OEM failure model, classified MALE UAV flight-test recording, or 3D CFD engine solver.

---

## 2. Physical Mechanism & Causal Chain

Rather than artificially adding temperature offsets to output telemetry, cooling degradation is modeled directly inside the thermal heat-balance differential equations:

```text
Cooling System Degradation (e.g. radiator blockage, coolant loss)
                          ↓
        Cooling Conductance Reduction (h_cool ↓)
                          ↓
      Net Heat Balance Imbalance (Q_gen > Q_cool)
                          ↓
     Cylinder Head Temperature Elevation (CHT ↑)
                          ↓
   Thermal Conduction Coupling to Lubricant (Q_couple ↑)
                          ↓
     Delayed Secondary Oil Temperature Elevation (Oil Temp ↑)
```

---

## 3. Governing Equations

### 3.1 Nominal Convective Cooling Conductance
The baseline cooling conductance $h_{\text{cool\_nominal}}$ (W/K) accounts for airspeed ram-air ventilation and engine cooling fan airflow:
$$h_{\text{cool\_nominal}} = h_{\text{cool\_base}} + h_{\text{cool\_rpm}} \cdot \text{RPM} + h_{\text{cool\_airspeed}} \cdot v_{\text{airspeed}}$$

### 3.2 Effective Conductance under Fault Injection
When a cooling degradation fault is active with effective severity $\sigma \in [0.0, 1.0]$:
$$h_{\text{cool\_effective}} = \max\left(1.0, \; h_{\text{cool\_nominal}} \cdot \left(1.0 - k_{\text{cooling\_max\_loss}} \cdot \sigma\right)\right)$$

Where:
- $\sigma = 0.0$: Healthy baseline ($h_{\text{cool\_effective}} = h_{\text{cool\_nominal}}$).
- $\sigma = 1.0$: Maximum modeled cooling degradation level ($h_{\text{cool\_effective}} = (1 - k_{\text{cooling\_max\_loss}}) \cdot h_{\text{cool\_nominal}}$).
- $k_{\text{cooling\_max\_loss}} = 0.55$ (**Tier C/D Calibration Parameter**): Sets the maximum modeled conductance reduction to $55\%$ at $\sigma = 1.0$.

### 3.3 Lumped Capacitance CHT Differential Equation
$$C_{\text{th, CHT}} \frac{dT_{\text{cht}}}{dt} = Q_{\text{gen}} - h_{\text{cool\_effective}} \cdot (T_{\text{cht}} - T_{\text{ambient}})$$

Steady-state CHT target:
$$T_{\text{cht, ss}} = T_{\text{ambient}} + \frac{Q_{\text{gen}}}{h_{\text{cool\_effective}}}$$

Because $h_{\text{cool\_effective}}$ decreases as severity rises, $T_{\text{cht, ss}}$ increases inversely with conductance, naturally elevating cylinder head temperature.

### 3.4 Secondary Oil Temperature Coupling
Hot cylinder heads conduct heat directly into the circulating engine oil charge:
$$Q_{\text{couple}} = k_{\text{oil\_cht\_couple}} \cdot (T_{\text{cht}} - T_{\text{oil}})$$

As $T_{\text{cht}}$ rises due to degraded cooling, $Q_{\text{couple}}$ increases, physically pulling oil temperature upward with the characteristic slow thermal capacitance lag of the 3.5 L oil charge ($\tau_{\text{oil}} \approx 120\text{ s}$).

---

## 4. Parameter Classification

| Parameter Name | Tier | Value | Units | Physical / Telemetry Effect | Classification |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `h_cool_base` | **Tier C** | 16.0 | W/K | Baseline static convection | Empirically calibrated |
| `h_cool_rpm` | **Tier C** | 0.012 | W/(K·RPM) | Airflow gain with engine speed | Empirically calibrated |
| `h_cool_airspeed` | **Tier C** | 0.35 | W/(K·(m/s)) | Ram-air cooling conductance gain | Empirically calibrated |
| `k_cooling_max_loss` | **Tier C/D** | 0.55 | fraction | Max conductance reduction fraction at severity=1.0 | Engineering calibration assumption |
| `c_th_cht` | **Tier C** | 920.0 | J/K | Cylinder head lumped thermal capacity | Calibrated to thermal lag |
| `k_oil_cht_couple` | **Tier C** | 2.5 | W/K | CHT-to-oil thermal conduction coupling | Calibrated to oil temperature |

---

## 5. Severity Sweep & Steady-State Response

Under controlled cruise flight conditions (75% throttle, 2000 m altitude, 45 m/s airspeed, $T_{\text{ambient}} = 2.0^\circ\text{C}$):

| Fault Severity ($\sigma$) | Effective Conductance $h_{\text{eff}}$ | Steady-State CHT ($^\circ\text{C}$) | Steady-State Oil Temp ($^\circ\text{C}$) | Operational Status |
| :---: | :---: | :---: | :---: | :--- |
| **0.00** (Healthy) | 88.3 W/K | **66.5 °C** | **87.5 °C** | Nominal |
| **0.25** (Mild) | 76.1 W/K | **76.9 °C** | **88.5 °C** | Normal operating range |
| **0.50** (Moderate) | 64.0 W/K | **91.2 °C** | **89.9 °C** | Elevated CHT |
| **0.75** (Severe) | 51.8 W/K | **112.3 °C** | **91.9 °C** | Warning zone |
| **1.00** (Critical) | 39.7 W/K | **146.2 °C** | **95.1 °C** | Approaching 150°C CHT reference limit |

Both CHT and oil temperature exhibit **strict, monotonic, physically coherent responses** with zero artificial clamp steps.

---

## 6. Fault Activation, Scheduling & Recovery

The cooling degradation model supports full timeline scheduling via `FaultState`:
- **Gating**: Active only when $\text{fault\_type} == \text{COOLING\_DEGRADATION}$, $\text{active} == \text{True}$, $\text{severity} > 0$, and $\text{start\_time} \le t \le \text{end\_time}$.
- **Ramping**: Supports linear severity ramp-up via `parameters={"ramp_duration": ...}`.
- **Natural Recovery**: When the fault window ends ($t > \text{end\_time}$), $h_{\text{cool}}$ returns to nominal. The engine thermal state recovers **naturally** toward baseline according to its governing thermal differential equations without artificial state resets.

---

## 7. Validation Test Summary (`tests/test_cooling_degradation.py`)

| Test ID | Test Name | Acceptance Criteria | Result |
| :---: | :--- | :--- | :---: |
| **Test A** | Healthy Equivalence | No fault vs severity=0.0 produces identical physical trajectory | **PASS** |
| **Test B** | Fault Activation | Severity > 0 measurably elevates CHT | **PASS** |
| **Test C** | Severity Monotonicity | Severity sweep strictly increases steady-state CHT | **PASS** |
| **Test D** | Secondary Oil Response | Directional increase in oil temp via physical coupling | **PASS** |
| **Test E** | Fault Schedule Gating | Active strictly within [start_time, end_time] | **PASS** |
| **Test F** | Natural Recovery | CHT cools naturally back to nominal after deactivation | **PASS** |
| **Test G** | Numerical Stability | Zero NaNs, zero Infs, physically bounded values | **PASS** |
| **Test H** | Conductance Scaling | Direct subsystem conductance scaled by $(1 - k_{\text{loss}} \cdot \sigma)$ | **PASS** |

---

## 8. Interactive Validation Artifacts

Interactive Plotly HTML visualizations are stored in `docs/plots/`:
- [`8_cooling_degradation_transient.html`](file:///d:/NIRVANAA-SIH-SUBMISSION/docs/plots/8_cooling_degradation_transient.html): Time-domain trajectory of CHT, Oil Temperature, and Fault Severity during activation and natural recovery.
- [`9_cooling_severity_sweep.html`](file:///d:/NIRVANAA-SIH-SUBMISSION/docs/plots/9_cooling_severity_sweep.html): Steady-state CHT and oil temperature curves across severity $\sigma \in [0.0, 1.0]$.
