# SIH26054: Physics-Informed Engine Simulator Specification & Physics Manual (Phase 2B)

## 1. Executive Summary & Engineering Anchor

This document details the reduced-order, lumped-parameter grey-box physics model developed for the **SIH26054 Aero Piston Engine Digital Twin System**.

> ### ⚠️ Critical Engineering Disclaimer:
> **The Rotax 912 ULS is used ONLY as a publicly documented engineering reference anchor.**
> This simulator does **NOT** represent an actual classified or proprietary MALE-UAV propulsion system, nor is it a certified OEM engine model or a Computational Fluid Dynamics (CFD) combustion simulation. It is a reduced-order lumped-parameter grey-box model designed for synthetic telemetry generation, state tracking, and downstream Prognostics & Health Management (PHM) research.

---

## 2. Parameter Tier Classification

To ensure complete transparency and eliminate hidden magic numbers, all parameters are segregated into four distinct tiers:

### Tier A — Public Reference Parameters
*Published, unclassified reference specifications from public engineering literature.*
- **Maximum RPM**: 5800 RPM (5-minute maximum limit)
- **Continuous RPM**: 5500 RPM
- **Continuous Power**: 58,000 W (58 kW / ~77.8 hp @ 5500 RPM)
- **Operational Envelopes**:
  - Maximum CHT: 150.0 °C (Rotax 912 ULS reference anchor limit)
  - Oil Temperature Range: 50.0 °C to 130.0 °C (Nominal: 90.0–110.0 °C)
  - Oil Pressure Range: 0.8 bar to 7.0 bar (Nominal: 2.0–5.0 bar)
- **ISA Constants**: Sea-level pressure $p_0 = 101325\text{ Pa}$, temperature $T_0 = 288.15\text{ K}$, density $\rho_0 = 1.225\text{ kg/m}^3$, lapse rate $L = 0.0065\text{ K/m}$, gas constant $R = 287.05\text{ J/(kg}\cdot\text{K)}$.

### Tier B — Physics-Derived Variables
*Direct analytical thermodynamic/mechanical conversions.*
- Geopotential barometric pressure: $p(h) = p_0 \cdot \left(\frac{T(h)}{T_0}\right)^{\frac{g}{L \cdot R}}$
- Air density & density ratio: $\rho(h) = \frac{p(h)}{R \cdot T(h)}$, $\sigma(h) = \frac{\rho(h)}{\rho_0}$
- Angular crankshaft velocity: $\omega = \frac{2\pi \cdot \text{RPM}}{60}$
- Brake torque: $\tau = \frac{P}{\omega}$

### Tier C — Calibration Parameters
*Empirically tuned values for reduced-order grey-box model dynamics.*
- Idle Speed: $\text{RPM}_{\text{idle}} = 1400\text{ RPM}$
- Rotating assembly + propeller equivalent inertia: $I = 0.28\text{ kg}\cdot\text{m}^2$
- Propeller load torque coefficient: $k_{\text{load}} = 3.036 \times 10^{-4}\text{ N}\cdot\text{m/(rad/s)}^2$
- Viscous & static friction torques: $k_{\text{fric}} = 0.015\text{ N}\cdot\text{m}\cdot\text{s/rad}$, $\tau_{\text{static}} = 3.5\text{ N}\cdot\text{m}$
- Willans-line fuel slope & intercept: $a_{\text{fuel}} = 6.8 \times 10^{-8}\text{ kg/J}$ (~245 g/kWh), $b_{\text{fuel}} = 0.00032\text{ kg/s}$
- CHT lumped thermal capacitance: $C_{\text{th}} = 920\text{ J/K}$, cylinder head heat absorption fraction $\eta_{\text{head}} = 0.048$
- Convective cooling conductances: $h_{\text{base}} = 16.0\text{ W/K}$, $h_{\text{rpm}} = 0.012\text{ W/(K}\cdot\text{RPM)}$, $h_{\text{air}} = 0.35\text{ W/(K}\cdot\text{(m/s))}$
- EGT steady-state gains: $T_{\text{base}} = 520^\circ\text{C}$, $k_{\text{load}} = 210^\circ\text{C}$, $k_{\text{rpm}} = 0.018^\circ\text{C/RPM}$, $\tau_{\text{egt}} = 2.5\text{ s}$
- Oil system capacitance & coupling: $C_{\text{oil}} = 1350\text{ J/K}$, $h_{\text{oil\_cool}} = 7.5\text{ W/K}$, $k_{\text{couple}} = 2.2\text{ W/K}$
- Oil pressure coefficients: $P_{\text{base}} = 1.0\text{ bar}$, $k_{\text{rpm}} = 0.00075\text{ bar/RPM}$, $k_{\text{temp}} = 0.022\text{ bar/}^\circ\text{C}$
- Vibration base amplitudes: 1x order $A_1 = 0.32\text{ g}$, 2x order $A_2 = 0.22\text{ g}$, process noise $\sigma_{\text{proc}} = 0.06\text{ g}$
- Sensor noise standard deviations: RPM ($\pm 4$), CHT ($\pm 0.4^\circ\text{C}$), EGT ($\pm 1.8^\circ\text{C}$), Oil Temp ($\pm 0.3^\circ\text{C}$), Oil Press ($\pm 0.025\text{ bar}$), Fuel Flow ($\pm 0.15\text{ L/h}$), Vibration ($\pm 0.015\text{ g}$)

### Tier D — Engineering Assumptions
*Functional approximations and structural simplifications.*
- Combustion/mechanical efficiency quadratic curve: $\eta_{\text{rpm}} = -0.75 \left(\frac{\text{RPM}}{5500}\right)^2 + 1.65 \left(\frac{\text{RPM}}{5500}\right) + 0.10$
- Propeller absorbing torque follows square law ($\tau \propto \omega^2$)
- Airspeed proxy for cooling airflow follows standard UAV flight phases (25–48 m/s)
- Vibration signature dominated by 1x and 2x crankshaft orders

---

## 3. Subsystem Mathematical Models

### 3.1 Atmosphere (ISA)
$$\begin{aligned}
T(h) &= T_0 - L \cdot h + \Delta T_{\text{offset}} \\
p(h) &= p_0 \left(\frac{T_0 - L \cdot h}{T_0}\right)^{\frac{g}{L \cdot R}} \\
\rho(h) &= \frac{p(h)}{R \cdot T(h)} \\
\sigma(h) &= \frac{\rho(h)}{\rho_0}
\end{aligned}$$

### 3.2 Rotational Dynamics & RK4 Numerical Integration
$$\begin{aligned}
P_{\text{target}} &= P_{\text{max\_cont}} \cdot \left(\frac{\text{throttle}}{100}\right) \cdot \sigma(h) \cdot \eta_{\text{rpm}}(\text{RPM}) \\
\tau_{\text{engine}} &= \frac{P_{\text{target}}}{\max(\omega, 10.0)} + \tau_{\text{idle\_assist}} \cdot \max(0, 1 - 2 \cdot \text{throttle}_{\text{norm}}) \\
\tau_{\text{load}} &= k_{\text{load}} \cdot \omega^2 \\
\tau_{\text{friction}} &= k_{\text{fric\_linear}} \cdot \omega + \tau_{\text{fric\_static}} \\
\frac{d\omega}{dt} &= \frac{\tau_{\text{engine}} - \tau_{\text{load}} - \tau_{\text{friction}}}{I}
\end{aligned}$$
*Integrated using 4th-Order Runge-Kutta (RK4) with sub-stepping ($\delta t \le 0.02\text{ s}$) to guarantee unconditional numerical stability for any external sampling rate.*

### 3.3 Fuel System (Willans Line)
$$\begin{aligned}
\dot{m}_{\text{fuel}} &= a_{\text{fuel}} \cdot P_{\text{target}} + b_{\text{fuel}} \quad [\text{kg/s}] \\
\dot{V}_{\text{fuel}} &= \frac{\dot{m}_{\text{fuel}}}{\rho_{\text{fuel}}} \times 3600 \quad [\text{L/h}] \\
\text{BSFC} &= \frac{\dot{m}_{\text{fuel}}}{P_{\text{target}}} \times 3.6 \times 10^9 \quad [\text{g/kWh}]
\end{aligned}$$

### 3.4 Thermal Dynamics (EGT & CHT)
$$\begin{aligned}
T_{\text{egt, ss}} &= T_{\text{base}} + k_{\text{load}} \cdot \text{load}_{\text{norm}} + k_{\text{rpm}} \cdot (\text{RPM} - \text{RPM}_{\text{idle}}) - k_{\text{density}} \cdot \sigma \\
\frac{dT_{\text{egt}}}{dt} &= \frac{T_{\text{egt, ss}} - T_{\text{egt}}}{\tau_{\text{egt}}} \\
Q_{\text{gen, cht}} &= \dot{m}_{\text{fuel}} \cdot \text{LHV} \cdot \eta_{\text{head}} \\
h_{\text{cool}} &= h_{\text{base}} + h_{\text{rpm}} \cdot \text{RPM} + h_{\text{air}} \cdot v_{\text{air}} \\
C_{\text{th}} \frac{dT_{\text{cht}}}{dt} &= Q_{\text{gen, cht}} - h_{\text{cool}} \cdot (T_{\text{cht}} - T_{\text{ambient}})
\end{aligned}$$

### 3.5 Lubrication System (Oil Temperature & Pressure)
$$\begin{aligned}
C_{\text{oil}} \frac{dT_{\text{oil}}}{dt} &= Q_{\text{gen, oil}} - h_{\text{oil\_cool}} \cdot (T_{\text{oil}} - T_{\text{ambient}}) + k_{\text{couple}} \cdot (T_{\text{cht}} - T_{\text{oil}}) \\
P_{\text{oil}} &= P_{\text{base}} + k_{\text{p, rpm}} \cdot \text{RPM} - k_{\text{p, temp}} \cdot \max(0, T_{\text{oil}} - 50.0)
\end{aligned}$$

### 3.6 Vibration Synthesis
$$\begin{aligned}
f_{\text{rot}} &= \frac{\text{RPM}}{60}, \quad f_{1\times} = f_{\text{rot}}, \quad f_{2\times} = 2 \cdot f_{\text{rot}} \\
v(t) &= A_1 \sin(2\pi f_{1\times} t + \phi_1) + A_2 \sin(2\pi f_{2\times} t + \phi_2) + w(t) \\
\text{RMS}_v &= \sqrt{\frac{1}{2} A_1^2 + \frac{1}{2} A_2^2 + \sigma_{\text{proc}}^2}
\end{aligned}$$

---

## 4. Validation Results Summary

The physics engine underwent automated test validation (`pytest -v` across 28 test cases):
1. **Atmosphere**: Validated sea-level values and monotonic pressure/density drops up to 10,000 m.
2. **Operating Point**: Validated monotonic throttle-power curves and density derating.
3. **RPM Dynamics**: Validated smooth transient response from idle (1400 RPM) to max power (5500 RPM) with RK4 convergence.
4. **Thermal**: Validated CHT steady-state convergence (85–115 °C) and EGT tracking (550–780 °C).
5. **Lubrication**: Validated oil pressure scaling with RPM and viscosity drop with temperature.
6. **Vibration**: Validated FFT spectral peaks matching 1x ($50\text{ Hz}$) and 2x ($100\text{ Hz}$) orders at 3000 RPM.
7. **Reproducibility**: Validated deterministic seeded noise outputs.
8. **Multi-Timestep Stability**: Verified stability across $dt \in [0.05, 0.1, 0.2, 0.5]\text{ s}$.

Interactive visualization artifacts are located in `docs/plots/`.
