# NIRVANAA — SIH26054 Digital Twin System

> **AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs**

---

## 1. Problem Overview (SIH26054)

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs) rely heavily on aero piston propulsion systems for extended ISR (Intelligence, Surveillance, and Reconnaissance) missions. Engine health degradation during critical mission phases (e.g., thermal runaway, oil pressure drop, injector clogging) can jeopardize mission success and asset survivability.

This project delivers a modular Digital Twin and Prognostics & Health Management (PHM) system for **controlled physics-based simulation**. It combines reduced-order physics modelling with machine learning to provide engineering health indicators, anomaly assessment, conditional RUL estimates, and interpretable diagnostic evidence.

> **Validation boundary:** The simulator and its outputs are engineering demonstrations based on synthetic telemetry. They are not a complete Rotax 914 UL/F validation, OEM calibration, airworthiness assessment, or real-flight validation.

---

## 2. System Status — Complete ✅

| Phase | Component | Status |
| :--- | :--- | :--- |
| **Phase 1** | Project Setup & Architecture | ✅ Complete |
| **Phase 2B** | Physics-Informed Engine Simulator (Reduced-Order Grey-Box Prototype; Rotax 914 UL/F Reference) | ✅ Complete |
| **Phase 3** | Simulator Calibration & Validation (8 suites, golden baseline) | ✅ Complete |
| **Phase 4A** | Fault & Degradation Interface (typed contracts) | ✅ Complete |
| **Phase 4B** | Cooling Degradation Physics (conductance degradation) | ✅ Complete |
| **Phase 4C** | Lubrication Degradation Physics (pressure/friction) | ✅ Complete |
| **Phase 4D** | Fuel / Injection Abnormality Physics (lean/rich) | ✅ Complete |
| **Phase 4E** | Mechanical Degradation Physics (bearing wear, vibration) | ✅ Complete |
| **Phase 4F** | Sensor Fault Physics (bias, drift, stochastic noise) | ✅ Complete |
| **Phase 5** | Telemetry Pipeline & Canonical Ingestion | ✅ Complete |
| **Phase 6** | Digital Twin State Estimation & Residual Generation | ✅ Complete |
| **Phase 7** | Hybrid Anomaly Detection (Threshold + EWMA + Persistence + Isolation Forest) | ✅ Complete |
| **Phase 8** | Multiclass Fault Diagnosis (XGBoost 6-Class Classifier) | ✅ Complete |
| **Phase 9** | Health Index & Causal Degradation Tracking | ✅ Complete |
| **Phase 10** | TimesFM-3 Future Telemetry Forecasting (Gated/Baseline Fallback) | ✅ Complete |
| **Phase 11** | Authoritative RUL & Prognostics (Theil–Sen + MC Uncertainty) | ✅ Complete |
| **Phase 12** | Explainability & Multi-Modal Evidence Fusion (SHAP + Physics + Temporal) | ✅ Complete |
| **Phase 13** | Unified System Pipeline Orchestrator | ✅ Complete |

Test totals evolve with the active branch. Use the commands below to collect and run the current suite; do not rely on historical test-count claims.

---

## 3. End-to-End Pipeline Architecture

```
Mission Configuration & Fault Scenario
                  ↓
Physics-Informed Engine Simulator (Reduced-Order Grey-Box Prototype; Rotax 914 UL/F Reference)
                  ↓
Canonical Telemetry Ingestion (with quality & dropout handling)
                  ↓
Phase 6: Physics-Informed Digital Twin & Dynamic Residuals
                  ↓
Phase 7: Hybrid Anomaly Detection (Threshold + EWMA + Persistence + Isolation Forest)
                  ↓
Phase 8: Multiclass Supervised Fault Diagnosis (XGBoost 6-Class)
                  ↓
Phase 9: Health Index & Causal Degradation Tracking (HI + Rate + Trend + Sensor Isolation)
                  ↓
Phase 10: TimesFM-3 Future Telemetry Forecasting (with Gated/Baseline Fallback)
                  ↓
Phase 11: Authoritative Prognostics & RUL (Theil–Sen + MC Uncertainty + Weakest Link EOL)
                  ↓
Phase 12: Explainability & Multi-Modal Evidence Fusion (SHAP + Physics + Temporal + RUL)
                  ↓
DashboardStatePayload (Unified System State)
                  ↓
Streamlit UI & Operator Decision Support Advisory
```

---

## 4. Key Performance Metrics

| Metric | Value | Budget |
| :--- | :--- | :--- |
| **Mean Inference Latency** | ~53 ms | < 200 ms |
| **Median (P50) Latency** | ~55 ms | < 200 ms |
| **P95 Latency** | ~84 ms | < 200 ms |
| **P99 Latency** | ~88 ms | < 200 ms |
| **Real-Time Margin** | >16× | > 1× |
| **Forecast Mode** | Causal EWMA Baseline | (TimesFM gated) |
| **Orchestrator Init (Bootstrap)** | ~4.8 s | One-time |

*Standard aero telemetry at 1.0 Hz (1000 ms budget). Pipeline processes in ~55 ms.*

---

## 5. Project Directory Structure

```
NIRVANAA-SIH-SUBMISSION/
├── simulator/            # Physics-Informed Engine Simulator & Subsystems
├── validation/           # Validation suites, metrics, and calibration sweeps
├── telemetry/            # Schemas, provenance tracking, and buffer
├── digital_twin/         # Digital Twin state tracking & residual engine
├── anomaly_detection/    # Phase 7: Hybrid anomaly detection
├── fault_diagnosis/      # Phase 8: XGBoost multiclass fault classification
├── health_index/         # Phase 9: Health index & degradation tracking
├── forecasting/          # Phase 10: TimesFM forecasting (gated/baseline)
├── prognostics/          # Phase 11: RUL & prognostic estimation
├── explainability/       # Phase 12: Multi-modal evidence fusion (SHAP+Physics+Temporal)
├── orchestrator/         # Phase 13: Unified system pipeline orchestrator
├── phm/                  # Legacy PHM interface (Phase 1)
├── dashboard/            # Operator dashboard interface
├── configs/              # Mission, engine, and telemetry configurations
├── data/                 # Data storage & Golden Baseline summary
├── docs/                 # Architecture docs, physics manual, validation reports
│   └── plots/            # Interactive Plotly validation figures (13 plots)
├── evidence/             # Final validation & evidence package
├── scripts/              # Validation plot & evidence generation scripts
├── tests/                # Automated test suite (340 test cases, 23 test files)
├── requirements.txt
└── main.py               # Phase 13 production entrypoint
```

---

## 6. Technology Stack

- **Core Runtime**: Python 3.10+
- **Data & Scientific Computing**: `numpy`, `scipy`, `pandas`
- **Machine Learning & Modeling**: `scikit-learn`, `xgboost`
- **Visualization & UI**: `plotly`, `streamlit`, `matplotlib`
- **Explainability**: `shap`
- **Testing**: `pytest`

---

## 7. Installation & Setup

```bash
git clone https://github.com/Yashuuuu02/NIRVANAA-SIH-SUBMISSION.git
cd NIRVANAA-SIH-SUBMISSION
pip install -r requirements.txt
```

---

## 8. Running the System

### Run the test suite
```bash
python -m pytest tests/ -q
```

### Run the dashboard
```bash
python -m streamlit run dashboard/app.py
```

### Run Production Pipeline (Phase 13)
```bash
python main.py                                    # Healthy scenario (35s)
python main.py --scenario cooling --duration 120  # Cooling fault injection
python main.py --scenario cooling --benchmark     # With latency profiling
```

### Run Legacy Phase 1 Dry-Run
```bash
python main.py --legacy-phase1
```

### Generate Evidence Package
```bash
python scripts/generate_evidence_package.py
```

### Generate Phase 4 health and prognostics reports
```bash
python scripts/generate_phase4_health_prognostics_results.py
```

### Generate Interactive Validation Plots
```bash
python scripts/generate_validation_plots.py
```

### Run Simulator Validation Runner
```bash
python validation/validation_runner.py
```

---

## 9. Architectural Declarations

### Algorithm Freezing Statement
Phase 13 does not redesign, retune, replace, or modify the algorithms, thresholds, schemas, or training procedures of Phases 1–12. For runtime inference, Phase 13 deterministically bootstraps Phase 7 Isolation Forest and Phase 8 XGBoost model instances using the existing training procedures and synthetic simulator-generated data. This is synthetic bootstrap model fitting, not external-dataset training or algorithm redesign.

### Preserved Distinctions
- **Algorithm & Training Procedure Freezing**: Feature schemas, classifier configurations, EWMA thresholds, Theil–Sen estimator rules, and multi-modal fusion equations from Phases 1–12 remain unmodified.
- **Runtime Model Fitting**: Deterministic synthetic bootstrap fitting is executed on synthetic simulator data with fixed seeds during orchestrator startup.
- **Pretrained TimesFM Weights**: The TimesFM integration requires an accessible gated checkpoint and compatible local runtime. When unavailable, the application reports the condition and uses its labelled baseline forecast path; it does not fabricate pretrained weights.

### Engineering Reference Anchor & Fidelity Boundary
- **Authoritative Reference Engine**: **Rotax 914 UL/F** (4-cylinder, 1211.2 cc, turbocharged, 84.5 kW takeoff / 73.5 kW continuous rating, 2.4286:1 reduction gearbox). Specification and parameter provenance are maintained in [`configs/engine_reference/rotax_914_ul_f.json`](file:///d:/SIH%20Drone/configs/engine_reference/rotax_914_ul_f.json).
- **Current Simulator Fidelity**: Reduced-order lumped-parameter 0D/1D grey-box prototype. The simulator uses naturally aspirated density derating, a 1:1 direct propeller load simplification ($J=0.28\text{ kg}\cdot\text{m}^2$), and lumped thermal nodes.
- **Missing Physics**: Exhaust gas turbocharger, compressor map, turbine expansion, wastegate actuator, electronic Turbo Control Unit (TCU), manifold absolute pressure (MAP), charge-air heating, 2.43:1 reduction gearbox dynamics, 4-cylinder individual thermal/exhaust runner networks, and electrical/ignition systems. Detailed in [`docs/physics_contract.md`](file:///d:/SIH%20Drone/docs/physics_contract.md).
- **Prohibited Claims**: The system does **NOT** claim to be a "full Rotax 914 F digital twin", "production-ready", "airworthiness validated", or "experimentally validated on real UAV flight recordings".

### Claim Taxonomy
The project adheres to a four-tier verification and validation taxonomy:
1. **`IMPLEMENTED`**: Executable functionality exists in the repository codebase.
2. **`VERIFIED`**: Executable tests/evidence demonstrate that the implementation behaves as mathematically intended. (Reference specifications are verified against official OEM manuals; simulator equations are verified against internal unit tests).
3. **`VALIDATED`**: Compared against an independent authoritative model, certified simulator, or regulatory reference dataset (NOT claimed for the simulator dynamics).
4. **`EXPERIMENTALLY VALIDATED`**: Validated against physical engine test-cell dynamometer recordings or operational flight data (**STRICTLY NOT CLAIMED**; all telemetry is synthetic).

---

## 10. Documentation

| Document | Description |
| :--- | :--- |
| `docs/physics_contract.md` | System-wide Physics Contract, fidelity boundary & claim taxonomy |
| `configs/engine_reference/rotax_914_ul_f.json` | Authoritative Rotax 914 UL/F reference spec with source provenance |
| `docs/architecture.md` | System architecture specification |
| `docs/simulator_physics.md` | Physics subsystem equations & parameters |
| `docs/simulator_validation.md` | Calibration & validation report |
| `docs/anomaly_detection.md` | Phase 7: Anomaly detection design |
| `docs/fault_diagnosis.md` | Phase 8: Fault diagnosis design |
| `docs/health_index.md` | Phase 9: Health index design |
| `docs/forecasting.md` | Phase 10: Forecasting design |
| `docs/prognostics_rul.md` | Phase 11: RUL & prognostics design |
| `docs/explainability.md` | Phase 12: Explainability design |
| `docs/system_orchestrator.md` | Phase 13: System orchestrator design |
| `docs/cooling_degradation.md` | Cooling fault physics |
| `docs/lubrication_degradation.md` | Lubrication fault physics |
| `docs/fuel_injection_abnormality.md` | Fuel injection fault physics |
| `docs/fault_interface.md` | Fault interface contracts |

