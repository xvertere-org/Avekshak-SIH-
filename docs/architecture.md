# SIH26054 Architecture Specification (Phase 1)

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
│   Physics-Informed Engine Simulator     │  (Reduced-order grey-box model)
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Telemetry Layer & Buffer         │  (Typed channels + Provenance)
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

### 2.1 Simulator (`simulator/`)
- **Responsibility**: Simulates aero piston engine thermal, mechanical, and fluid dynamic responses across flight envelopes (Takeoff, Climb, Cruise, Loiter, Descent, Landing).
- **Phase 1 Scope**: Lightweight abstract base interface (`BaseEngineSimulator`) and schema-compliant execution stub (`EngineSimulator`).
- **Reduced-Order Physics Disclaimer**: The future simulator is designed as a reduced-order grey-box model combining lumped thermal-fluid equations and empirical parameters. It is **NOT** a computational fluid dynamics (CFD) solver or an authoritative certified OEM engine model.

### 2.2 Telemetry (`telemetry/`)
- **Responsibility**: Defines strict data contracts, unit representations, and stream buffering.
- **Key Channels**:
  - `altitude` (m), `ambient_temp` (°C), `throttle` (%), `load` (%)
  - `rpm` (RPM), `cht` (°C), `egt` (°C), `oil_temp` (°C), `oil_pressure` (bar), `fuel_flow` (L/h), `vibration` (g)
- **Provenance Tracking**:
  - `source`: Generator / dataset tag (e.g. `simulator_v1_stub`, `test_bench_data`)
  - `source_type`: Origin classification (`simulated`, `synthetic`, `test_bench`, `flight_test`)
  - `simulation_version`: Codebase/model iteration tag

### 2.3 Digital Twin (`digital_twin/`)
- **Responsibility**: Tracks expected nominal engine physics states for any given operating point and computes dynamic residuals (e.g., $\Delta \text{CHT} = \text{CHT}_{\text{observed}} - \text{CHT}_{\text{nominal}}$).
- **Phase 1 Scope**: Interface (`DigitalTwin`) outputting `DigitalTwinState`.

### 2.4 Prognostics & Health Management (`phm/`)
- **Responsibility**: Evaluates residuals and telemetry trends to generate an overall Health Index (0.0–1.0), flag anomaly events, and categorize operational faults.
- **Target Fault Categories**:
  - `cooling_degradation`
  - `injector_fuel_abnormality`
  - `lubrication_issue`
  - `mechanical_vibration_fault`
  - `sensor_drift_failure`
  - `none` (nominal)

### 2.5 Forecasting & RUL (`forecasting/`)
- **Responsibility**: Predicts Remaining Useful Life (RUL) in operational hours with upper/lower uncertainty bounds and degradation trajectories.
- **Phase 1 Scope**: Interface (`RULPredictor`) providing `RULPrediction` contracts.

### 2.6 Explainability (`explainability/`)
- **Responsibility**: Attributes anomalous behavior to root physical drivers (residuals, temperature trends, pressure drops) for human-in-the-loop engineering trust.
- **Phase 1 Scope**: Interface (`ExplainabilityEngine`) providing `ExplanationReport`.

### 2.7 Dashboard (`dashboard/`)
- **Responsibility**: Packages real-time telemetry, twin state, diagnostics, RUL, and explanations into clean payloads for Streamlit/Plotly operator views.

---

## 3. Configuration Management (`configs/`)

Configuration is cleanly isolated from code logic:
- `default_mission.json`: Defines flight parameters, target altitude, ambient temperature, mission phase, duration, and fault injection tags.
- `default_engine.json`: Defines generic engine parameters. All specifications are non-authoritative templates.
- `telemetry_settings.json`: Declares channel metadata, units, and provenance standards.
- `config_loader.py`: Safe parsing utilities with dataclass validation.

---

## 4. Multi-Engine & Multi-UAV Scalability

Although the Phase 1 MVP targets a single UAV and single aero piston engine (`ENGINE_UAV_01`), all schemas and interfaces enforce explicit `mission_id`, `engine_id`, timestamps, and provenance metadata to enable multi-engine and fleet-wide monitoring without refactoring core contracts.
