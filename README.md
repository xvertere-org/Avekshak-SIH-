# NIRVANAA — SIH26054 Digital Twin System

> **AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs**

---

## 1. Problem Overview (SIH26054)

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs) rely heavily on aero piston propulsion systems for extended ISR (Intelligence, Surveillance, and Reconnaissance) missions. Engine health degradation during critical mission phases (e.g., thermal runaway, oil pressure drop, injector clogging) can jeopardize mission success and asset survivability.

This project delivers a modular, real-time Digital Twin and Prognostics & Health Management (PHM) system that combines reduced-order physics modeling with machine learning to provide real-time state estimation, early fault detection, Remaining Useful Life (RUL) estimation, and human-interpretable diagnostic explanations.

---

## 2. MVP Objective

The primary objective of the initial MVP is to establish an end-to-end monitoring pipeline for a simulated MALE UAV aero piston engine:
1. Simulating engine thermal/fluid/mechanical parameters across mission profiles.
2. Generating typed telemetry streams with provenance tracking.
3. Tracking nominal physical baselines with a Digital Twin residual engine.
4. Detecting anomalies and diagnosing fault categories (cooling degradation, lubrication issues, injector abnormalities, mechanical vibration, sensor drift).
5. Forecasting degradation trends and Remaining Useful Life (RUL).
6. Delivering explainable diagnostic insights to operators via an interactive dashboard.

---

## 3. High-Level Architecture & Phase 1 Flow

```
MissionConfig
     │
     ▼
Simulator Interface (Reduced-Order Grey-Box)
     │
     ▼
TelemetryRecord (Channels + Units + Provenance)
     │
     ▼
DigitalTwin Interface (State Estimation & Residuals)
     │
     ▼
PHM Interface (Anomaly Detection & Fault Categorization)
     │
     ▼
Forecasting / RUL Interface (Trajectory & Bounds)
     │
     ▼
Explainability Interface (Feature Attribution & Summary)
     │
     ▼
Dashboard Interface (Operator Situational Payload)
```

---

## 4. Phase 1 Scope

Phase 1 establishes the foundational software architecture, typed contracts, modular interfaces, configuration management, and testing infrastructure:
- **Zero Placeholder Math**: Pure interface and contract definitions without uncalibrated mock physics or stub ML models.
- **Strict Separation of Concerns**: Each pipeline stage is an independent module with clean abstractions.
- **Provenance & Telemetry Standards**: Complete telemetry schema with physical units and source metadata.
- **Testing & Verification**: Automated unit and interface tests verifying end-to-end connectivity.

> **Important Engineering Disclaimer:**  
> The engine simulator in this project is designed as a **reduced-order physics-informed / grey-box model**. It is **NOT** a computational fluid dynamics (CFD) solver or an authoritative certified OEM engine model. Engine configurations provided in templates are non-authoritative baseline parameters.

---

## 5. Project Directory Structure

```
NIRVANAA-SIH-SUBMISSION/
├── simulator/            # Physics-Informed Engine Simulator interfaces and stubs
│   ├── __init__.py
│   ├── base.py
│   └── engine_simulator.py
├── telemetry/            # Telemetry schema, provenance, and stream buffer
│   ├── __init__.py
│   ├── schema.py
│   └── streamer.py
├── digital_twin/         # Digital Twin state tracking and residual engine
│   ├── __init__.py
│   └── twin_model.py
├── phm/                  # Prognostics and Health Management
│   ├── __init__.py
│   └── detector.py
├── forecasting/          # RUL and Time-series forecasting interfaces
│   ├── __init__.py
│   └── rul_predictor.py
├── explainability/       # Explainable AI (XAI) feature attribution
│   ├── __init__.py
│   └── explainer.py
├── dashboard/            # Operator dashboard interface
│   ├── __init__.py
│   └── app.py
├── configs/              # Mission, Engine, and Telemetry configurations
│   ├── __init__.py
│   ├── config_loader.py
│   ├── default_mission.json
│   ├── default_engine.json
│   └── telemetry_settings.json
├── data/                 # Data storage hierarchy
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/                 # Architectural specifications
│   └── architecture.md
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_schemas.py
│   ├── test_imports.py
│   └── test_interfaces.py
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

---

## 6. Technology Stack

- **Core Runtime**: Python 3.10+
- **Data & Scientific Computing**: `numpy`, `scipy`, `pandas`
- **Machine Learning & Modeling**: `scikit-learn`, `xgboost`
- **Visualization & UI**: `plotly`, `streamlit`
- **Explainability**: `shap`
- **Experiment Tracking**: `mlflow`
- **Testing**: `pytest`

---

## 7. Installation & Setup

### Clone Repository
```bash
git clone https://github.com/Yashuuuu02/NIRVANAA-SIH-SUBMISSION.git
cd NIRVANAA-SIH-SUBMISSION
```

### Create Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Running Verification & Tests

### Run Automated Tests
```bash
pytest -v
```

### Run Phase 1 Pipeline Dry-Run
```bash
python main.py --dry-run
```

---

## 9. Next Steps (Phase 2 Roadmap)

- Implement reduced-order thermal-fluid grey-box equations in `simulator/`.
- Calibrate nominal operational curves for aero piston engine profiles.
- Implement fault injection modes for cooling, fuel, lubrication, mechanical, and sensor anomalies.
- Connect digital twin state estimator to live telemetry streams.
