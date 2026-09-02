# SIH26054 Architecture Specification (Phase 2B)

## Project Title
**AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.**

---

## 1. System Overview & Architecture

The SIH26054 Digital Twin platform follows a modular, decoupled pipeline architecture designed to ingest real-time and synthetic telemetry, track baseline physics states, detect operational anomalies, predict Remaining Useful Life (RUL), generate explainable diagnostic reports, and deliver real-time operator situational awareness.

```
┌─────────────────┐
│  MissionConfig  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   Physics-Informed Engine Simulator     │  (Phase 2B/3: Calibrated & Validated Grey-Box)
│   - Atmosphere (ISA)                    │
│   - Rotational Dynamics (RK4)           │
│   - Fuel Flow (Willans-line)            │
│   - Thermal Dynamics (CHT & EGT)        │
│   - Lubrication & Oil Pressure          │
│   - Order Vibration Synthesis (1x, 2x)  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Telemetry Layer & Buffer         │  (Typed channels + Sensor Noise + Provenance)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           Digital Twin Core             │  (Nominal state estimation & residuals)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Prognostics & Health Mgmt (PHM)     │  (Anomaly detection & fault classification)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       Forecasting / RUL Engine          │  (Degradation trajectory & RUL bounds)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Explainable AI (XAI) Engine        │  (Feature attribution & diagnostic summaries)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Dashboard & Telemetry Display      │  (Operator payload & situational UI)
└─────────────────────────────────────────┘
```

---

## 2. Module Responsibilities

### 2.1 Simulator (`simulator/` & `validation/`)
- **Responsibility**: Simulates aero piston engine thermal, mechanical, and fluid dynamic responses across flight envelopes (Takeoff, Climb, Cruise, Loiter, Descent, Landing).
- **Phase 2B/3 Status**: Fully operational, calibrated, and validated grey-box physical engine simulator.
- **Reference Anchor Disclaimer**: The simulator uses the **Rotax 912 ULS** strictly as a publicly documented engineering reference anchor. It is **NOT** a computational fluid dynamics (CFD) solver, certified engine model, or actual classified UAV engine.
- **Validation Framework (`validation/`)**:
  - 8 automated validation suites (Physical bounds, monotonicity, transient lag hierarchy, empirical timestep sweep, steady-state stability, cross-channel coherence, vibration order tracking, representative mission).
  - Recommended operating timestep: $dt = 0.1\text{ s}$ (10 Hz).
  - Golden baseline reference: `data/golden_baseline_summary.json` (17,200 samples).
- **Fault Degradation & Physics (`simulator/fault_interface.py` & Subsystems)**:
  - Phase 4A establishes typed fault contracts (`FaultType`, `FaultSubsystem`, `FaultState`, `FaultSchedule`).
  - Phase 4B implements physics-grounded **Cooling Degradation** in `ThermalSystem`, reducing effective convective cooling conductance $h_{\text{cool\_effective}} = h_{\text{cool}} \cdot (1 - k_{\text{loss}} \cdot \sigma)$. Produces natural CHT elevation and secondary delayed oil temperature rise via conduction coupling.
  - Phase 4C implements physics-grounded **Lubrication Degradation** in `LubricationSystem`, reducing hydraulic delivery pressure $P_{\text{oil}} = P_{\text{nom}} \cdot (1 - k_{\text{p\_loss}} \cdot \sigma)$, increasing frictional heat generation, and reducing radiator heat rejection. Produces compound pressure drop and monotonic oil temperature rise.
  - Phase 4D implements physics-grounded **Fuel / Injection Abnormality** supporting typed `FuelMixtureMode.LEAN` and `FuelMixtureMode.RICH`. Lean mode produces fuel flow drop and EGT elevation; rich mode produces fuel flow increase and EGT quench; both modes incorporate subtle combustion efficiency power droop.
- **Subsystem Architecture**:
  - `simulator/config.py`: Segregated parameter tiers (Tier A: Public reference, Tier B: Physics-derived, Tier C: Calibration, Tier D: Engineering assumptions).
  - `simulator/subsystems/atmosphere.py`: ISA troposphere pressure, temperature, and density lapse model.
  - `simulator/subsystems/mission.py`: Multi-phase flight plan generator and step interpolator.
  - `simulator/subsystems/dynamics.py`: 4th-Order Runge-Kutta (RK4) rotational dynamics with torque balance.
  - `simulator/subsystems/fuel.py`: Willans-line fuel consumption and BSFC estimation.
  - `simulator/subsystems/thermal.py`: Lumped capacitance CHT model and first-order lagging EGT.
  - `simulator/subsystems/lubrication.py`: Coupled oil thermal model and temperature/viscosity oil pressure.
  - `simulator/subsystems/vibration.py`: 1x and 2x crankshaft order harmonics, broadband process noise, and FFT analytics.
  - `simulator/telemetry_generator.py`: Packaging of physical states into `TelemetryRecord` with calibrated sensor noise.
  - `simulator/engine_simulator.py`: Batch and streaming execution orchestrator.

### 2.2 Telemetry (`telemetry/`)
- **Responsibility**: Defines strict data contracts, unit representations, and stream buffering.
- **Key Channels**:
  - `altitude` (m), `ambient_temp` (°C), `throttle` (%), `load` (%)
  - `rpm` (RPM), `cht` (°C), `egt` (°C), `oil_temp` (°C), `oil_pressure` (bar), `fuel_flow` (L/h), `vibration` (g)
- **Provenance Tracking**:
  - `source`: Generator / dataset tag (e.g. `simulator_v1_physics`, `test_bench_data`)
  - `source_type`: Origin classification (`synthetic`, `simulated`, `test_bench`, `flight_test`)
  - `simulation_version`: Codebase/model iteration tag

### 2.3 Digital Twin (`digital_twin/`)
- **Responsibility**: Tracks expected nominal engine physics states for any given operating point and computes dynamic residuals (e.g., $\Delta \text{CHT} = \text{CHT}_{\text{observed}} - \text{CHT}_{\text{nominal}}$).

### 2.4 Prognostics & Health Management (`phm/`)
- **Responsibility**: Evaluates residuals and telemetry trends to generate an overall Health Index (0.0–1.0), flag anomaly events, and categorize operational faults.

### 2.5 Forecasting & RUL (`forecasting/`)
- **Responsibility**: Predicts Remaining Useful Life (RUL) in operational hours with upper/lower uncertainty bounds and degradation trajectories.

### 2.6 Explainability (`explainability/`)
- **Responsibility**: Attributes anomalous behavior to root physical drivers (residuals, temperature trends, pressure drops) for human-in-the-loop engineering trust.

### 2.7 Dashboard (`dashboard/`)
- **Responsibility**: Packages real-time telemetry, twin state, diagnostics, RUL, and explanations into clean payloads for Streamlit/Plotly operator views.

---

## 3. Parameter Tier Structure

All simulator parameters are structured in `simulator/config.py`:
- **Tier A (Public Reference)**: Rotax 912 ULS anchor specifications (58 kW continuous power @ 5500 RPM, 5800 max RPM, CHT limit 135 °C, oil temp/pressure envelopes, ISA constants).
- **Tier B (Physics-Derived)**: Analytical conversions for atmospheric density factor $\sigma$, $\omega$, torque, and power derating.
- **Tier C (Calibration Parameters)**: Inertia ($I=0.28\text{ kg}\cdot\text{m}^2$), propeller load constant $k_{\text{load}}$, friction parameters, thermal conductances, Willans fuel slope/intercept, and sensor noise variances.
- **Tier D (Engineering Assumptions)**: Combustion efficiency curve approximation, airspeed proxies, and vibration order weights.
