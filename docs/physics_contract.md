# Aero Piston Engine Digital Twin — Physics Contract

**Project**: AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs (SIH26054)  
**Governing Phase**: Phase 1 — Engine Identity & Physics Contract Correction  
**Authoritative Reference Engine**: **Rotax 914 UL/F**  
**Version**: 1.0.0  

---

## 1. Executive Summary & Engine Identity Decision

This document establishes the binding **Physics Contract** for all subsystems across the SIH26054 project. Following an adversarial forensic audit, this contract establishes one authoritative, internally consistent engine identity and defines exact boundaries between:

1. **`REFERENCE_ENGINE`**: The **Rotax 914 UL/F** turbocharged aero piston engine. All geometry, operational limits, continuous ratings, and sensor channels defined in [`configs/engine_reference/rotax_914_ul_f.json`](file:///d:/SIH%20Drone/configs/engine_reference/rotax_914_ul_f.json) are derived directly from verified OEM manuals and Type Certificates.
2. **`CURRENT_MODEL_FIDELITY`**: The actual mathematical implementation present in the codebase today — a reduced-order, 0D/1D lumped-parameter grey-box prototype with naturally aspirated altitude scaling.
3. **`MISSING_PHYSICS`**: Physical mechanisms required for full Rotax 914 UL/F fidelity (turbocharger compressor/turbine, TCU wastegate servo loop, manifold absolute pressure, reduction gearbox ratio, multi-cylinder head/exhaust distribution, and electrical systems) that are **NOT** yet implemented.

> [!IMPORTANT]
> **No Fictitious Capabilities**: The project does NOT claim that the existing simulator is a complete Rotax 914 F digital twin. The Rotax 914 UL/F is our authoritative reference specification, while the current simulator is explicitly documented as a reduced-order prototype.

---

## 2. Claim Taxonomy

To prevent misleading claims, the project strictly uses the following four unambiguous statuses:

| Status | Definition | Current Project Application |
|---|---|---|
| **`IMPLEMENTED`** | Executable functionality exists in the repository source code. | Rotational ODE, ISA atmosphere, lumped CHT/oil ODEs, Willans fuel flow, synthetic vibration harmonics. |
| **`VERIFIED`** | Executable tests and automated benchmarks demonstrate that the implementation behaves as mathematically intended. | Reference specification verified against OEM manuals (`test_engine_reference_contract.py`). Simulator equations verified against internal physical tests (`test_physics_validation.py`). |
| **`VALIDATED`** | Compared against an independent authoritative model, certified simulator, or regulatory reference dataset. | **NOT CLAIMED** for the simulator dynamics. Reference limits match published EASA TCDS E.122, but simulator state transitions are not cross-validated with certified tools. |
| **`EXPERIMENTALLY VALIDATED`** | Validated against physical engine test-cell dynamometer recordings, hardware-in-the-loop (HIL) testbeds, or operational UAV flight data. | **STRICTLY NOT CLAIMED**. The repository contains zero physical flight recordings or OEM dynamometer logs. |

---

## 3. Reference Engine vs. Simulator Calibration Separation

The authoritative OEM/EASA specifications of the Rotax 914 UL/F must never be conflated with the configuration and calibration parameters of the reduced-order simulator:

### 3.1 Parameter Separation Matrix

| Parameter | Authoritative OEM / EASA Reference | Selected Model Configuration | Classification | Technical Rationale & Provenance |
|---|---|---|---|---|
| **Takeoff Power** | 84.5 kW (115 hp) @ 5800 RPM (OM-914 §2.1; EASA TCDS E.122 Issue 06 §III.6) | 84.5 kW @ 5800 RPM | `OEM_REFERENCE_VALUE` / `MODEL_CONFIGURATION_VALUE` | Source-verified reference target from EASA TCDS E.122 Issue 06 and OM-914 Section 2.1 (5 min limit). |
| **Continuous Power** | 73.5 kW (100 hp) @ 5500 RPM (OM-914 §2.1; EASA TCDS E.122 Issue 06 §III.6) | 73.5 kW @ 5500 RPM | `OEM_REFERENCE_VALUE` / `MODEL_CONFIGURATION_VALUE` | Maximum continuous power ceiling under standard rating. |
| **Continuous MAP Target** | EASA TCDS E.122: 1.150 bar (TCU v4.3) / 1.180 bar (TCU v4.6); OM-914 §2.1 / EASA max: 1.200 bar | 1.200 bar (1200 hPa / 35.4 in.Hg) | `MODEL_CONFIGURATION_VALUE` | Selected nominal simulator operating target corresponding to OM-914 §2.1 and EASA TCDS E.122 continuous upper limit. Not asserted as a universal OEM value. |
| **Continuous MAP Acceptance Band** | N/A (Simulation tolerance) | 1.185 – 1.215 bar (±15 hPa) | `MODEL_ACCEPTANCE_BAND` | Numerical simulation acceptance band for test verification. Strictly NOT an OEM operating limit. |
| **Takeoff MAP Target** | EASA TCDS E.122: 1.300 bar (TCU v4.3) / 1.320 bar (TCU v4.6); OM-914 §2.1 / EASA max: 1.350 bar | 1.350 bar (1350 hPa / 39.9 in.Hg) | `MODEL_CONFIGURATION_VALUE` | Selected nominal simulator takeoff boost setpoint corresponding to OM-914 §2.1 and EASA TCDS E.122 takeoff upper limit. |
| **Takeoff MAP Acceptance Band** | N/A (Simulation tolerance) | 1.335 – 1.365 bar (±15 hPa) | `MODEL_ACCEPTANCE_BAND` | Numerical simulation acceptance band for test verification. Strictly NOT an OEM operating limit. |
| **Continuous Critical Altitude** | EASA TCDS E.122: 16,000 ft (4875 m) for TCU v4.3/v4.6; OM-914 §2.1: 15,000 ft (4572 m) | 4572.0 m (15,000 ft) | `MODEL_CONFIGURATION_VALUE` | Selected surrogate operating configuration from OM-914 §2.1. EASA TCDS E.122 certified rating is 4875 m. Documented configuration divergence. |
| **Takeoff Critical Altitude** | EASA TCDS E.122: 8,000 ft (2450 m) for TCU v4.3/v4.6; OM-914: ~2500 m | 2500.0 m | `MODEL_CONFIGURATION_VALUE` | Selected surrogate takeoff boost threshold. EASA TCDS E.122 certified rating is 2450 m. |
| **Reduction Gearbox Ratio** | 2.42857:1 (51:21 teeth; OM-914 §2.1; EASA TCDS E.122) | 2.42857:1 (spur reduction model) | `OEM_REFERENCE_VALUE` | Implemented in Phase 2 with mechanical power conservation ($P_{eng\_out} \ge P_{prop}$). |
| **Gearbox Efficiency** | ~97-98% typical spur mesh | 0.975 (97.5%) | `MODEL_CALIBRATION` | Surrogate gear mesh transmission efficiency with elastomeric dog-clutch damper. |
| **Cooling Thermal Capacitance** | System coolant capacity: ~4.0 L (liquid circuit) | 4500.0 J/K | `MODEL_CALIBRATION` | Effective lumped thermal capacitance of cylinder head water jackets and head coolant volume (~1.2 kg fast-response thermal mass). **NOT the 4 L total system coolant inventory**. |
| **Cylinder Bank Thermal Variation** | Boxer engine spatial distribution | (0.98, 1.00, 1.03, 0.99) | `MODEL_ASSUMPTION` | Deterministic bank variation multipliers (front heads 1, 2 run cooler via ram air; rear heads 3, 4 run warmer), normalized to mean 1.000. |

### 3.2 Authoritative EASA TCDS E.122 Issue 06 TCU Reconciliation Table

| TCU Variant | Applicable Serial Range | Continuous MAP | Continuous Alt | Takeoff MAP | Takeoff Alt | Continuous Power | Takeoff Power |
|---|---|---|---|---|---|---|---|
| **TCU Version 4.3 (Rotax 21)** | Up to s/n 4,420.199 | 1150 hPa (1.150 bar / 34.0 in.Hg) | 16,000 ft (4875 m) | 1300 hPa (1.300 bar / 38.4 in.Hg) | 8,000 ft (2450 m) | 73.5 kW @ 5500 RPM | 84.5 kW @ 5800 RPM |
| **TCU Version 4.6 (Rotax 99)** | From s/n 4,420.200 onwards | 1180 hPa (1.180 bar / 34.9 in.Hg) | 16,000 ft (4875 m) | 1320 hPa (1.320 bar / 39.0 in.Hg) | 8,000 ft (2450 m) | 73.5 kW @ 5500 RPM | 84.5 kW @ 5800 RPM |
| **EASA E.122 Certified Limits (§IV.3.3)** | All Rotax 914 F variants | 1150 – 1200 hPa (1.15 – 1.20 bar) | Cert. envelope | 1300 – 1350 hPa (1.30 – 1.35 bar) | Cert. envelope | 73.5 kW continuous | 84.5 kW (5 min) |
| **Simulator Model Configuration** | Grey-Box Reduced-Order Twin | 1200 hPa (1.200 bar target) | 4572 m (15,000 ft) | 1350 hPa (1.350 bar target) | 2500 m | 73.5 kW reference | 84.5 kW reference |


---

## 4. Detailed Subsystem Contract

Each subsystem is cataloged with its executable status:

### Subsystem A: Environment & Atmosphere
- **Status**: `PARTIALLY_IMPLEMENTED`
- **Implemented in Code**: Standard ISA altitude model in `simulator/engine_simulator.py`. Evaluates ambient pressure $P(h)$ and ambient temperature $T(h)$ with standard troposphere lapse rate ($0.0065\text{ K/m}$), ideal gas density $\rho = P / (R \cdot T)$, and relative density $\sigma = \rho / \rho_0$.
- **Missing / Unmodeled**: Non-standard atmosphere temperature deltas ($\Delta\text{ISA}$), dynamic ram-air recovery pressure, relative humidity, wind gusts.

### Subsystem B: Intake & Turbocharger Boost
- **Status**: `IMPLEMENTED` (Phase 2 Upgrade — `NUMERICALLY_VERIFIED`)
- **Implemented in Code**: Coupled reduced-order thermodynamic loop (`simulator/subsystems/turbocharger.py`):
  1. TCU Closed-Loop Surrogate (`TCUSurrogate`): Regulates wastegate position to target MAP (1.200 bar continuous, 1.350 bar takeoff).
  2. Turbine Enthalpy Extraction (`TurbineSurrogate`): Computes $\dot{W}_t$ expanding from $P_3$ toward ambient.
  3. Mechanical Shaft Transfer: $\dot{W}_c = \dot{W}_t \cdot \eta_{mech}$.
  4. Compressor Thermodynamics (`CompressorSurrogate`): Computes pressure ratio $PR \ge 1.0$, discharge temperature $T_2$, and intercooled charge air temperature $T_{charge}$.
  5. Downstream Intake Manifold Dynamics: Lag ODE $d(MAP)/dt = (P_{target} - MAP)/\tau_{map}$ with exact exponential decay. Preserves strict distinction between compressor boost capability ($PR \ge 1.0$) and downstream throttle restriction ($MAP < P_{amb}$).
- **Surrogate Disclosure**: Uses calibrated surrogate maps (`MODEL_CALIBRATION`/`MODEL_ASSUMPTION`); proprietary OEM compressor/turbine maps are explicitly disclosed as non-public.

### Subsystem C: Combustion & Fuel Consumption
- **Status**: `IMPLEMENTED` (Phase 2 Causally Coupled Reduced-Order Combustion Power Chain)
- **Implemented in Code**: Causal power chain (`simulator/subsystems/dynamics.py`):
  $$\dot{m}_{air} \to \dot{m}_{fuel} = \dot{m}_{air}/\text{AFR} \to P_{chem} = \dot{m}_{fuel} \cdot \text{LHV} \to P_{ind} = P_{chem} \cdot \eta_{ind} \to P_{brake} = P_{ind} - P_{fric}$$
  Strict power hierarchy invariant enforced: $P_{chem} > P_{ind} > P_{brake} \ge 0$ at all operating speeds above idle (`POWER_HIERARCHY_INVARIANT`).
  *Note*: This verifies the mechanical power inequality hierarchy; it does NOT claim a full closed energy balance with complete loss accounting.

### Subsystem D: Crankshaft Rotational Dynamics
- **Status**: `IMPLEMENTED` (Phase 2 Upgrade — `NUMERICALLY_VERIFIED`)
- **Implemented in Code**: Sub-stepped 4th-Order Runge-Kutta (RK4) integration with unconditional numerical stability across arbitrary external timesteps.
  $$J_{eq} \frac{d\omega_{eng}}{dt} = T_{comb} + T_{idle\_assist} - T_{load,eng} - T_{fric}$$
  where $J_{eq} = J_{eng} + J_{prop} / i^2$.

### Subsystem E: Reduction Gearbox & Propeller Drivetrain
- **Status**: `IMPLEMENTED` (Phase 2 Upgrade — `NUMERICALLY_VERIFIED`)
- **Implemented in Code**:
  1. Gearbox Kinematics: $i = 2.42857$ (51/21 spur gear ratio), $\omega_{prop} = \omega_{eng} / i$.
  2. Propeller Load Reflection: $T_{load,eng} = T_{prop} / (i \cdot \eta_{gb})$, with $\eta_{gb} = 0.975$.
  3. Power Invariant: $P_{eng\_out} = T_{load,eng} \cdot \omega_{eng} \ge P_{prop} = T_{prop} \cdot \omega_{prop}$. The gearbox never creates energy.
  4. Transmission Loss: $P_{gb\_loss} = P_{eng\_out} - P_{prop} \ge 0$.

### Subsystem F: Multi-Cylinder Thermal System (CHT & EGT)
- **Status**: `IMPLEMENTED` (Phase 2 Upgrade — `NUMERICALLY_VERIFIED`)
- **Implemented in Code**: 4 discrete cylinder heads and 4 exhaust runners (`simulator/subsystems/thermal.py`):
  1. Firing Sequence: 1-4-3-2 boxer layout.
  2. Deterministic Bank Variation: Front cylinders (1, 2) run slightly cooler via direct ram air (0.98, 1.00); rear cylinders (3, 4) run slightly warmer (1.03, 0.99), normalized to mean 1.000 (`MODEL_ASSUMPTION`).
  3. Local Cylinder-State Independence with Shared-System Coupling: Perturbing cylinder 1 directly affects only cylinder 1 at that instant; other cylinders remain decoupled instantaneously, while shared crankshaft torque, exhaust enthalpy, and liquid coolant loops couple them dynamically over time.
  4. Arithmetic Mean Invariant: Aggregate $T_{cht} = \frac{1}{4} \sum_{i=1}^4 T_{cht,i}$ and $T_{egt} = \frac{1}{4} \sum_{i=1}^4 T_{egt,i}$ within $10^{-5}\ ^\circ\text{C}$.

### Subsystem G: Lubrication System (Oil Temp & Pressure)
- **Status**: `PARTIALLY_IMPLEMENTED`
- **Implemented in Code**: Lumped oil thermal capacitance ODE coupled with CHT and ambient radiator dissipation. Dynamic oil pressure with RPM gain and temperature-viscosity proxy drop.

### Subsystem H: Vibration Dynamics
- **Status**: `PARTIALLY_IMPLEMENTED`
- **Implemented in Code**: Analytical synthesis of 1x and 2x harmonics with load scaling, broadband process noise, and fault gain multipliers.

### Subsystem I: Electrical & Ignition System
- **Status**: `NOT_IMPLEMENTED`
- **Implemented in Code**: None. Deferred to future electrical phase.

### Subsystem K: Liquid Cooling Loop Surrogate
- **Status**: `IMPLEMENTED` (Phase 2 Addition — `NUMERICALLY_VERIFIED`)
- **Implemented in Code**: Designated as `REDUCED_ORDER_COOLING_SURROGATE` (`simulator/subsystems/cooling.py`):
  1. Effective lumped cooling thermal capacitance $C_{coolant} = 4500\text{ J/K}$ (`MODEL_CALIBRATION`, representing the effective fast-response thermal mass of the cylinder head water jackets and head coolant volume; not full 4 L system inventory).
  2. Thermostat regulation with nominal opening threshold at 80 °C (OM-914 §13.1).
  3. Radiator heat rejection scaling with flight airspeed ($k_{rad} = 0.025\text{ (m/s)}^{-1}$).
  4. Heat exchange coupling with the 4 discrete cylinder heads ($h_{head} = 12.0\text{ W/K}$ per head).

---

## 5. Architectural Integrity & Validation Contract

1. **No Fictitious Parameters**: Every parameter carries explicit provenance metadata (`provenance_metadata` in config classes) categorized into `OEM_REFERENCE_VALUE`, `MODEL_ASSUMPTION`, or `MODEL_CALIBRATION`.
2. **Numerical Verification Only**: Phase 2 claims `NUMERICALLY_VERIFIED` status backed by:
   - 24 automated unit and integration tests in `tests/test_physics_validation_phase2.py`.
   - 8-point canonical operating matrix in `evidence/phase2_operating_matrix.json`.
3. **Zero Flight Validation Claims**: The repository contains zero physical flight recordings or OEM dynamometer logs. No flight or certification validation is claimed.

