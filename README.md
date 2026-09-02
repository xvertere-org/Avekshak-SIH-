# NIRVANAA — SIH26054 Digital Twin System

> **AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs**

---

## 1. Problem Overview (SIH26054)

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs) rely heavily on aero piston propulsion systems for extended ISR (Intelligence, Surveillance, and Reconnaissance) missions. Engine health degradation during critical mission phases (e.g., thermal runaway, oil pressure drop, injector clogging) can jeopardize mission success and asset survivability.

This project delivers a modular, real-time Digital Twin and Prognostics & Health Management (PHM) system that combines reduced-order physics modeling with machine learning to provide real-time state estimation, early fault detection, Remaining Useful Life (RUL) estimation, and human-interpretable diagnostic explanations.

---

## 2. MVP Objective & Current Status

- **Phase 1: Project Setup + Architecture** — Complete ✅
- **Phase 2B: Physics-Informed Engine Simulator** — Complete ✅
  - Lumped-parameter grey-box engine simulator operating across all flight phases.
  - Rotational dynamics with RK4 numerical integration and torque balance.
  - Willans-line fuel flow and BSFC model.
  - Lumped thermal CHT capacitance and lagging EGT models.
  - Coupled oil thermal dynamics and viscosity-dependent oil pressure.
  - Order-based vibration synthesis (1x, 2x crankshaft harmonics + broadband noise).
  - ISA atmosphere model with altitude power derating.
  - Calibrated sensor measurement noise and full provenance tracking.

> **Engineering Reference Anchor & Disclaimer:**  
> The engine simulator uses the **Rotax 912 ULS** strictly as a publicly documented engineering anchor (58 kW continuous power @ 5500 RPM, max 5800 RPM). It is a **reduced-order physics-informed / grey-box model**, **NOT** a CFD solver, certified OEM engine model, or actual classified UAV engine. Synthetic telemetry is never represented as actual UAV flight data.

---

## 3. High-Level Architecture & Data Flow

```
MissionConfig / FlightPhase (TAKEOFF, CLIMB, CRUISE, LOITER, DESCENT, LANDING)
     │
     ▼
Atmosphere Layer (ISA Lapse, Pressure, Density Factor)
     │
     ▼
Rotational Dynamics (P_target, Load Torque, Friction, RK4 Engine Speed)
     │
     ▼
Fuel & Thermal Subsystems (Willans Fuel, CHT Lumped Capacitance, EGT Lag)
     │
     ▼
Lubrication & Vibration Subsystems (Oil Temp/Pressure, 1x & 2x Orders)
     │
     ▼
TelemetryRecord (Typed Channels, Calibrated Sensor Noise, Provenance Metadata)
     │
     ▼
DigitalTwin Interface (State Estimation & Residual Engine)
     │
     ▼
PHM Interface (Anomaly Detection & Fault Categorization)
     │
     ▼
Forecasting / RUL Interface (Degradation Trajectory & Uncertainty Bounds)
     │
     ▼
Explainability Interface (Feature Attribution & Diagnostic Summary)
     │
     ▼
Dashboard Interface (Operator Situational Payload)
```

---

## 4. Project Directory Structure

```
NIRVANAA-SIH-SUBMISSION/
├── simulator/
│   ├── __init__.py
│   ├── base.py                 # Abstract base class BaseEngineSimulator
│   ├── config.py               # Tiered parameter definitions (Tiers A, B, C, D)
│   ├── engine_simulator.py      # Simulator orchestrator (batch & streaming)
│   ├── telemetry_generator.py  # Telemetry conversion & calibrated sensor noise
│   └── subsystems/
│       ├── __init__.py
│       ├── atmosphere.py       # ISA standard atmosphere model
│       ├── mission.py          # Mission profile generator & step interpolator
│       ├── dynamics.py         # Power target, load torque, friction, RK4 rotational dynamics
│       ├── fuel.py             # Willans-line fuel consumption model
│       ├── thermal.py          # CHT lumped thermal capacitance & EGT lag
│       ├── lubrication.py      # Coupled oil thermal model & oil pressure
│       └── vibration.py        # Order-based vibration synthesis (1x, 2x) & FFT
├── telemetry/                  # Schemas, provenance tracking, and buffer
├── digital_twin/               # Digital Twin state tracking & residual engine
├── phm/                        # Prognostics & Health Management
├── forecasting/                # RUL and time-series forecasting interfaces
├── explainability/             # Explainable AI (XAI) feature attribution
├── dashboard/                  # Operator dashboard interface
├── configs/                    # Mission, engine, and telemetry configurations
├── data/                       # Data storage hierarchy (raw, processed, external)
├── docs/                       # Specifications, physics manual & validation plots
│   ├── architecture.md
│   ├── simulator_physics.md
│   └── plots/
├── scripts/
│   └── generate_validation_plots.py  # Diagnostic Plotly visualization generator
├── tests/                      # Automated test suite (28 test cases)
│   ├── test_schemas.py
│   ├── test_imports.py
│   ├── test_interfaces.py
│   ├── test_physics_checkpoint_rpm.py
│   └── test_physics_validation.py
├── requirements.txt
└── main.py
```

---

## 5. Technology Stack

- **Core Runtime**: Python 3.10+
- **Data & Scientific Computing**: `numpy`, `scipy`, `pandas`
- **Machine Learning & Modeling**: `scikit-learn`, `xgboost`
- **Visualization & UI**: `plotly`, `streamlit`, `matplotlib`
- **Explainability**: `shap`
- **Experiment Tracking**: `mlflow`
- **Testing**: `pytest`

---

## 6. Installation & Setup

```bash
git clone https://github.com/Yashuuuu02/NIRVANAA-SIH-SUBMISSION.git
cd NIRVANAA-SIH-SUBMISSION
pip install -r requirements.txt
```

---

## 7. Running Verification & Validation

### Run Automated Tests (28 tests)
```bash
pytest -v
```

### Run End-to-End Pipeline Dry-Run
```bash
python main.py --dry-run
```

### Generate Interactive Physics Validation Plots
```bash
python scripts/generate_validation_plots.py
```
*(Interactive HTML validation artifacts are generated in `docs/plots/`)*
