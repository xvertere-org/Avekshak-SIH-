# SIH26054: Fuel / Injection Abnormality Physics & Validation (Phase 4D)

## 1. Executive Summary & Objective

Phase 4D implements a physically grounded fuel/injection abnormality fault model within the reduced-order grey-box aero piston engine simulator of the **SIH26054 Digital Twin System**.

> ### ⚠️ Engineering Reference & Scope Disclaimer:
> This fuel/injection abnormality model is a **reduced-order engineering fault model** designed for synthetic telemetry generation, digital twin residual tracking, and PHM diagnostic algorithm training. It is **NOT** a certified OEM failure model, classified MALE UAV flight-test recording, or multi-zone CFD chemical kinetics combustion solver.

---

## 2. Physical Mechanism & Causal Chain

Rather than overwriting telemetry channels after simulation, fuel and injection abnormalities are modeled directly within the fuel delivery, combustion efficiency, and exhaust gas thermal balance equations:

```text
Fuel / Injection Abnormality (e.g. nozzle clog, rail pressure defect, leaking injector)
                                      ↓
                     Air-Fuel Mixture Ratio Shift (Lambda)
                                      ↓
            ┌─────────────────────────┴─────────────────────────┐
            ↓                                                   ↓
       LEAN ABNORMALITY                                   RICH ABNORMALITY
            ↓                                                   ↓
   Fuel Mass Flow (ṁ_fuel ↓)                           Fuel Mass Flow (ṁ_fuel ↑)
            ↓                                                   ↓
   Slower Flame Speed & Delayed Exhaust Burn          Incomplete Burn & Vaporization Quenching
            ↓                                                   ↓
   Exhaust Gas Temperature (EGT ↑)                    Exhaust Gas Temperature (EGT ↓)
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ↓
              Combustion Thermal Efficiency Reduction (η_comb ↓)
                                      ↓
             Natural Subtle Power & RPM Droop (Preserving Throttle)
```

---

## 3. Governing Equations

### 3.1 Fuel Mass Flow Degradation
Under nominal operation, fuel mass flow is computed via the Willans-line power approximation:
$$\dot{m}_{\text{fuel, nominal}} = a_{\text{fuel}} \cdot P_{\text{target}} + b_{\text{fuel}}$$

Under fuel injection abnormality with effective severity $\sigma \in [0.0, 1.0]$:
$$\dot{m}_{\text{fuel, effective}} = \begin{cases} 
\max\left(0.0, \; \dot{m}_{\text{fuel, nominal}} \cdot (1.0 - k_{\text{fuel\_flow\_lean}} \cdot \sigma)\right) & \text{if mode is LEAN} \\ 
\dot{m}_{\text{fuel, nominal}} \cdot (1.0 + k_{\text{fuel\_flow\_rich}} \cdot \sigma) & \text{if mode is RICH} 
\end{cases}$$

Where:
- $k_{\text{fuel\_flow\_lean}} = 0.25$ (**Tier C/D Calibration Parameter**): At severity $\sigma = 1.0$, fuel delivery drops by 25%.
- $k_{\text{fuel\_flow\_rich}} = 0.30$ (**Tier C/D Calibration Parameter**): At severity $\sigma = 1.0$, fuel delivery increases by 30%.

### 3.2 Exhaust Gas Temperature (EGT) Dynamics
Nominal steady-state EGT balances engine load, engine speed timing, and altitude air density:
$$T_{\text{egt, ss, base}} = T_{\text{egt, base}} + k_{\text{egt\_load}} \cdot \text{Load}_{\text{norm}} + k_{\text{egt\_rpm}} \cdot (\text{RPM} - \text{RPM}_{\text{idle}}) - k_{\text{egt\_density}} \cdot \sigma_{\text{density}}$$

Mixture shift alters the post-combustion gas enthalpy:
$$\Delta T_{\text{egt, mix}} = \begin{cases}
+k_{\text{egt\_lean\_gain}} \cdot \sigma & \text{if mode is LEAN (slower burn continuing into exhaust blowdown)} \\
-k_{\text{egt\_rich\_drop}} \cdot \sigma & \text{if mode is RICH (unburned fuel vaporization cooling / quenching)}
\end{cases}$$
$$T_{\text{egt, ss}} = \max\left(100.0, \; T_{\text{egt, ss, base}} + \Delta T_{\text{egt, mix}}\right)$$

The actual sensor telemetry evolves smoothly via the physical first-order gas transport lag:
$$\tau_{\text{egt}} \frac{dT_{\text{egt}}}{dt} = T_{\text{egt, ss}} - T_{\text{egt}}$$
Where $\tau_{\text{egt}} = 2.5\text{ s}$.

### 3.3 Combustion Efficiency & Power Availability
Off-stoichiometric combustion reduces thermal work conversion:
$$\eta_{\text{comb}} = \begin{cases}
1.0 - k_{\text{comb\_loss\_lean}} \cdot \sigma & \text{if mode is LEAN} \\
1.0 - k_{\text{comb\_loss\_rich}} \cdot \sigma & \text{if mode is RICH}
\end{cases}$$
$$P_{\text{combustion}} = P_{\text{max}} \cdot \text{Throttle}_{\text{norm}} \cdot \sigma_{\text{density}} \cdot \eta_{\text{rpm}} \cdot \eta_{\text{comb}}$$

This naturally reduces engine torque in `RotationalDynamics` ($T_{\text{eng}} = P_{\text{comb}} / \omega$), producing a realistic subtle RPM droop under propeller load without manipulating commanded throttle.

---

## 4. Parameter Classification

| Parameter Name | Tier | Value | Units | Physical / Telemetry Effect | Classification |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `k_fuel_flow_lean` | **Tier C/D** | 0.25 | fraction | Fuel flow reduction fraction at max lean fault | Engineering calibration assumption |
| `k_fuel_flow_rich` | **Tier C/D** | 0.30 | fraction | Fuel flow increase fraction at max rich fault | Engineering calibration assumption |
| `k_egt_lean_gain_c` | **Tier C/D** | 95.0 | °C | Steady-state EGT elevation at max lean fault | Engineering calibration assumption |
| `k_egt_rich_drop_c` | **Tier C/D** | 80.0 | °C | Steady-state EGT drop at max rich fault | Engineering calibration assumption |
| `k_comb_loss_lean` | **Tier C/D** | 0.08 | fraction | Power efficiency reduction at max lean fault | Engineering calibration assumption |
| `k_comb_loss_rich` | **Tier C/D** | 0.06 | fraction | Power efficiency reduction at max rich fault | Engineering calibration assumption |
| `tau_egt_s` | **Tier C** | 2.5 | s | Thermocouple dynamic lag time constant | Calibrated baseline |

---

## 5. Severity Sweep & Steady-State Response

Under controlled cruise flight conditions (75% throttle, 2000 m altitude, 45 m/s airspeed, $T_{\text{ambient}} = 2.0^\circ\text{C}$):

### 5.1 Lean Abnormality Sweep
| Severity ($\sigma$) | Fuel Flow (L/h) | Steady-State EGT (°C) | Steady-State RPM | Physical State |
| :---: | :---: | :---: | :---: | :--- |
| **0.00** (Healthy) | **12.95 L/h** | **692.5 °C** | **4348.0** | Baseline nominal |
| **0.25** (Mild Lean) | **12.01 L/h** | **711.5 °C** | **4312.4** | Mild mixture thinning |
| **0.50** (Moderate) | **11.02 L/h** | **731.4 °C** | **4276.7** | Elevated exhaust heat |
| **0.75** (Severe) | **10.01 L/h** | **751.2 °C** | **4241.1** | Near thermal warning |
| **1.00** (Critical) | **8.98 L/h** | **770.8 °C** | **4205.8** | Severe lean misfire risk; EGT < 800°C limit |

### 5.2 Rich Abnormality Sweep
| Severity ($\sigma$) | Fuel Flow (L/h) | Steady-State EGT (°C) | Steady-State RPM | Physical State |
| :---: | :---: | :---: | :---: | :--- |
| **0.00** (Healthy) | **12.95 L/h** | **692.5 °C** | **4348.0** | Baseline nominal |
| **0.25** (Mild Rich) | **13.89 L/h** | **669.1 °C** | **4321.3** | Mild fuel over-enrichment |
| **0.50** (Moderate) | **14.82 L/h** | **645.7 °C** | **4294.6** | High fuel consumption |
| **0.75** (Severe) | **15.74 L/h** | **622.3 °C** | **4267.9** | Heavy unburned exhaust quench |
| **1.00** (Critical) | **16.65 L/h** | **598.9 °C** | **4241.3** | Severe fuel penalty / carbon fouling risk |

---

## 6. Fault Scheduling, Gating & Natural Recovery

The fuel injection abnormality model supports timeline scheduling through `FaultState`:
- **Gating**: Active strictly when $\text{fault\_type} == \text{FUEL\_INJECTION\_ABNORMALITY}$, $\text{active} == \text{True}$, $\text{severity} > 0$, and $\text{start\_time} \le t \le \text{end\_time}$.
- **Mode Specification**: Specified via `parameters={"mode": "lean"}` or `{"mode": "rich"}` (or typed enum `FuelMixtureMode.LEAN` / `FuelMixtureMode.RICH`). Defaults to `"lean"` if omitted.
- **Natural Recovery**: Once the fault window ends ($t > \text{end\_time}$), fuel flow immediately returns to nominal, while EGT and RPM smoothly recover according to thermocouple thermal lag ($\tau_{\text{egt}} = 2.5\text{ s}$) and rotational inertia ($J = 0.085\text{ kg}\cdot\text{m}^2$). No artificial state resets are used.

---

## 7. Validation Test Summary (`tests/test_fuel_injection_abnormality.py`)

| Test ID | Test Name | Acceptance Criteria | Result |
| :---: | :--- | :--- | :---: |
| **Test A** | Healthy Equivalence | No fault vs severity=0.0 produces identical physical trajectory | **PASS** |
| **Test B** | Lean Activation | Active lean fault reduces fuel flow and elevates EGT | **PASS** |
| **Test C** | Rich Activation | Active rich fault increases fuel flow and reduces EGT | **PASS** |
| **Test D** | Lean EGT Direction | Lean severity sweep strictly increases steady-state EGT | **PASS** |
| **Test E** | Rich EGT Direction | Rich severity sweep strictly decreases steady-state EGT | **PASS** |
| **Test F** | Fuel-Flow Direction | Fuel flow hierarchy: Lean < Healthy < Rich | **PASS** |
| **Test G** | Severity Monotonicity | Strict monotonic response in both modes across $[0.0, 1.0]$ | **PASS** |
| **Test H** | Schedule Gating | Fault active strictly within $[t_{\text{start}}, t_{\text{end}}]$ | **PASS** |
| **Test I** | Natural Recovery | Telemetry recovers naturally toward nominal post-fault | **PASS** |
| **Test J** | Numerical Stability | Zero NaNs, zero Infs, bounded positive values in all channels | **PASS** |
| **Test K** | Subsystem Scaling | Direct verification of `FuelSystem` and `ThermalSystem` methods | **PASS** |

---

## 8. Interactive Validation Artifacts

Interactive Plotly HTML visualizations are stored in `docs/plots/`:
- [`12_fuel_injection_transient.html`](file:///d:/NIRVANAA-SIH-SUBMISSION/docs/plots/12_fuel_injection_transient.html): 4-panel time-domain trace of Fuel Flow, EGT, RPM, and Fault Severity demonstrating a sequential Lean Window $\to$ Recovery $\to$ Rich Window.
- [`13_fuel_injection_severity_sweep.html`](file:///d:/NIRVANAA-SIH-SUBMISSION/docs/plots/13_fuel_injection_severity_sweep.html): Dual-panel plot of steady-state Fuel Flow and EGT comparing Lean and Rich divergence across severity $\sigma \in [0.0, 1.0]$.
