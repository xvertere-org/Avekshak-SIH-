# NIRVANAA — SIH26054 Rotax 914 Aero Piston Engine Digital Twin

> **Physics-Informed Reduced-Order Grey-Box Digital Twin & Prognostics System for Health Monitoring, Fault Diagnosis, and Mission Reliability of MALE UAV Propulsion Systems**

---

## 1. Problem Overview & Scope (SIH26054)

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs) rely on turbocharged aero piston propulsion systems for extended Intelligence, Surveillance, and Reconnaissance (ISR) missions. Undetected in-flight propulsion degradation—such as thermal conductance loss, oil pressure drops, injector delivery abnormalities, combustion misfires, or sensor calibration drift—can jeopardize mission success and asset survivability.

**NIRVANAA** delivers a modular, physics-informed grey-box Digital Twin and Prognostics & Health Management (PHM) system calibrated against the **turbocharged BRP-Rotax 914 UL/F** aero engine architecture. It synchronizes incoming flight telemetry with 1D lumped-parameter thermodynamic and rotational state estimators, tracks physical residuals, isolates fault signatures through directional subsystem coupling, projects model-defined degradation horizons ($D_\text{EOL}$), and evaluates counterfactual what-if mission completion risk.

> **Validation boundary:** The simulator and its outputs are engineering demonstrations based on synthetic telemetry. They are not a complete Rotax 914 UL/F validation, OEM calibration, airworthiness assessment, or real-flight validation.

---

## 2. System Capabilities & Authoritative Evidence Matrix

Every claimed capability in NIRVANAA is tied directly to an auditable, deterministic evidence artifact and automated regression suite:

| System Capability | Implementation Paradigm | Evidence Artifact | Automated Verification Suite |
| :--- | :--- | :--- | :--- |
| **Engine State Estimation** | 1D lumped-parameter thermal, lubrication, and rotational ODE network | [`evidence/phase2_operating_matrix.json`](evidence/phase2_operating_matrix.json) | [`tests/test_phase2_operating_matrix.py`](tests/test_phase2_operating_matrix.py) |
| **Simulator Validation & Calibration** | 8 validation suites against OEM steady-state & transient limits | [`evidence/final_validation_report.md`](evidence/final_validation_report.md) | [`tests/test_simulator_validation.py`](tests/test_simulator_validation.py) |
| **Multi-Fault Physics Simulation** | 6 physical fault modes (cooling, lubrication, injector, misfire, friction, sensor) | [`evidence/evidence_package.json`](evidence/evidence_package.json) | [`tests/test_fault_physics_phase4.py`](tests/test_fault_physics_phase4.py) |
| **Dynamic Residual Generation** | Physical & normalized residuals against synchronized nominal twin | [`evidence/phase6_fault_diagnosis_matrix.json`](evidence/phase6_fault_diagnosis_matrix.json) | [`tests/test_fault_diagnosis_phase6.py`](tests/test_fault_diagnosis_phase6.py) |
| **Health Index Assessment (HI)** | Dynamically renormalized weighted linear sum of piecewise-linear residual evidence | [`evidence/phase5_health_assessment_matrix.json`](evidence/phase5_health_assessment_matrix.json) | [`tests/test_health_assessment_phase5.py`](tests/test_health_assessment_phase5.py) |
| **Fault Diagnosis & Isolation** | Residual signature lookup & directional physical coupling rules | [`evidence/phase6_fault_diagnosis_matrix.json`](evidence/phase6_fault_diagnosis_matrix.json) | [`tests/test_fault_diagnosis_phase6.py`](tests/test_fault_diagnosis_phase6.py) |
| **Synthetic Fleet Population** | Parameterized Monte Carlo variations across synthetic engine cohorts | [`evidence/phase7_population_matrix.json`](evidence/phase7_population_matrix.json) | [`tests/test_phase7_population.py`](tests/test_phase7_population.py) |
| **Prognostics & RUL Estimation** | Robust Theil–Sen linear regression with deterministic empirical quantile bounds | [`evidence/phase8_rul_matrix.json`](evidence/phase8_rul_matrix.json) | [`tests/test_phase8_rul.py`](tests/test_phase8_rul.py) |
| **Telemetry Ingestion & Replay** | Canonical schema, unit conversions, clock skew detection, multi-rate sync | [`evidence/phase9_telemetry_matrix.json`](evidence/phase9_telemetry_matrix.json) | [`tests/test_phase9_real_telemetry.py`](tests/test_phase9_real_telemetry.py) |
| **Mission Reliability & What-If** | Counterfactual scenario simulation & mission risk scoring ($R_\text{mission}$) | [`evidence/phase10_mission_matrix.json`](evidence/phase10_mission_matrix.json) | [`tests/test_phase10_mission.py`](tests/test_phase10_mission.py) |
| **Engineering Explainability** | Structured attribution tracing decisions to residuals, sensors, and rules | [`evidence/phase11_explainability_matrix.json`](evidence/phase11_explainability_matrix.json) | [`tests/test_phase11_explainability.py`](tests/test_phase11_explainability.py) |
| **Claims & Submission Integrity** | Automated two-pass lexical & contextual disclaimer linter | [`evidence/phase12_claims_matrix.json`](evidence/phase12_claims_matrix.json) | [`tests/test_phase12_claims_audit.py`](tests/test_phase12_claims_audit.py) |

---

## 3. End-to-End System Pipeline Architecture

```
Telemetry Stream / Replay Buffer (CanonicalTelemetryPacket)
                  ↓
Telemetry Preprocessing & Validation (Units, Clock Skew, Multi-Rate Sync)
                  ↓
Synchronized Physics-Informed Digital Twin (1D Lumped-Parameter Grey-Box State)
                  ↓
Residual Generation Engine (Physical Residuals & Normalized Z-Scores)
                  ↓
Physics-Informed Health Assessment (Dynamically Renormalized Piecewise-Linear Evidence)
                  ↓
Hypothesis Isolation & Diagnosis (Residual Signatures & Sensor Disambiguation)
                  ↓
Degradation Tracking & RUL Estimation (Theil–Sen Regression with Empirical Quantile Bounds)
                  ↓
Counterfactual What-If Simulation (Mission Risk Index R_mission under Stress Scenarios)
                  ↓
Auditable Evidence & Explainability Layer (Provenance Attribution to Residuals & Rules)
                  ↓
Operator Advisory Dashboard (Streamlit UI & Decision Support)
```

---

## 4. Transparent 6-Tier AI/ML Runtime Inventory

To ensure absolute clarity regarding machine learning utilization, every analytical component is classified into exactly one operational tier:

| Computational Tier | Components Included | Implementation Method | Operational Digital Twin Role |
| :--- | :--- | :--- | :--- |
| **1. OPERATIONAL_RUNTIME** | Synchronizer, Residuals, Health, Diagnosis, Degradation, RUL, What-If | 1D ODE physics, piecewise-linear evidence deadbands, directional signature matching, Theil–Sen robust linear regression. | **Active Core Runtime**: Powers all real-time state estimation, health tracking, and diagnostic isolation. **Contains ZERO black-box neural networks.** |
| **2. VALIDATION_ONLY** | `fault_diagnosis/classifier.py`, `evaluation.py` | XGBoost & Random Forest multi-class classifiers. | Evaluated in validation test suites as comparative models; **NOT invoked in operational runtime**. |
| **3. QUARANTINED** | `forecasting/timesfm_wrapper.py` | TimesFM zero-shot foundation time-series neural network. | Quarantined/bypassed in production orchestrator due to heavy dependencies and non-deterministic execution. |
| **4. OFFLINE_EXPERIMENT** | `prognostics/uncertainty.py`, `telemetry/features.py` | Monte Carlo trajectory sampling ($M=500$ realizations), FFT spectral band analytics. | Offline parameter sensitivity sweeps and dataset feature engineering. |
| **5. LEGACY** | `phase14/` | Early proof-of-concept prototypes. | Retained for historical development context; not part of active runtime. |
| **6. UNUSED** | Orphaned experimental scripts | N/A | Excluded from test suites and production handoffs. |

---

## 5. Measured Host-Side Latency Benchmark

Latency was empirically measured across 1,000 continuous digital twin update steps (state synchronization, residual evaluation, anomaly detection, health calculation) on commodity desktop hardware (Python 3.11, Windows/x86_64 CPU).

| Latency Metric | Measured Value | Operational Processing Budget | Status |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | **0.388 ms** | 20.0 ms (50 Hz streaming budget) / 1000 ms (1 Hz telemetry) | ✅ Well within budget |
| **Median (P50) Latency** | **0.328 ms** | 20.0 ms | ✅ Sub-millisecond steady-state |
| **P95 Latency** | **0.559 ms** | 20.0 ms | ✅ Sub-millisecond steady-state |
| **P99 Latency** | **0.763 ms** | 20.0 ms | ✅ Sub-millisecond steady-state |
| **Maximum Latency (Spike)**| **28.994 ms** | 20.0 ms | ⚠️ OS scheduling / GC spike |
| **Real-Time Classification**| **Soft Real-Time** | 1.0–10.0 Hz telemetry ingestion | Host-side soft real-time suitable; hard real-time explicitly disclaimed |

*Authoritative Source: [`evidence/phase6_latency_benchmark.json`](file:///d:/SIH%20Drone/evidence/phase6_latency_benchmark.json).*

---

## 6. Authoritative Rotax 914 UL/F OEM Reference Specifications

Key engine parameters are cross-checked against official type certification and manufacturer documentation:

| Parameter | Authoritative Value | Source Document & Section | Engineering Provenance & Classification |
| :--- | :--- | :--- | :--- |
| **Takeoff Power (5 min)** | **84.5 kW** (5800 RPM) | EASA TCDS E.122 / Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Takeoff limit ($115\,\text{HP}$ gross metric; brochure cites $84.8\,\text{kW}$, authoritative certification standard is $84.5\,\text{kW}$). |
| **Continuous Power** | **73.5 kW** (5500 RPM) | EASA TCDS E.122 / Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Max continuous cruising power ($100\,\text{HP}$). |
| **Critical Altitude** | **4875 m** ($16,000\,\text{ft}$) | EASA TCDS E.122 / Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Critical turbo boost ceiling ($115\,\text{kPa}$). *Simulator test envelope sweeps up to $4500\,\text{m}$ as `MODEL_IMPLEMENTATION` test ceiling.* |
| **Idle Oil Pressure** | **0.8 bar** | Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Minimum permissible oil pressure below $3500\,\text{RPM}$. |
| **Normal Oil Pressure** | **2.0 – 5.0 bar** | Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Operational oil pressure envelope above $3500\,\text{RPM}$. |
| **Cold-Start Oil Pressure** | **7.0 bar** | Rotax 914 OM Sec 2.1 / EASA TCDS E.122 | `AUTHORITATIVE_REFERENCE`: Maximum permissible transient pressure during cold start. |
| **Max CHT Limit** | **135 °C** | EASA TCDS E.122 / Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Maximum allowable cylinder head temperature. |
| **Gearbox Ratio** | **2.4286:1** (51/21 teeth) | Rotax 914 OM Sec 2.1 | `AUTHORITATIVE_REFERENCE`: Propeller gearbox reduction ratio. |
| **Health Deadband** | **1.5 $\sigma$** | Phase 5 Health Specification (`health_index/schema.py`) | `ENGINEERING_HEURISTIC`: Normalized residual deadband ($\tau_\text{nominal}$) — *NOT a bar pressure unit*. |

---

## 7. Preserved Technical Limitations & Disclaimers

The following fundamental engineering limitations govern the system and its outputs:

1. **Synthetic Grey-Box Simulator**: The simulation engine is a 1D lumped-parameter model. It does not replace 3D CFD, combustion acoustics, or finite-element structural modeling.
2. **Absence of Real Engine Fleet Validation**: All degradation trajectories and fault responses are validated on synthetic, simulated, or replayed benchmarks. The system has **not** been validated against an operational fleet of physical Rotax 914 engines.
3. **Non-Certification**: NIRVANAA is a research and engineering competition prototype (SIH 2026). It is **not** certified under FAA DO-178C, FAA Part 33, or EASA CS-E airworthiness standards.
4. **Model-Defined RUL**: Remaining Useful Life represents the projected time for modeled degradation state $D(t)$ to reach threshold $D_\text{EOL} = 0.50$ under the estimator's modeled operating/stress assumptions. It does **not** predict certified mechanical Time Between Overhaul (TBO).
5. **Heuristic Mission Risk Index ($R_\text{mission}$)**: The mission risk score is an engineering metric ($[0, 1]$) combining envelope excursions and degradation rates. It is **strictly not** a frequentist or Bayesian failure probability.
6. **Host-Side Execution Latency**: Real-time throughput is demonstrated on desktop hardware (mean $<0.4\,\text{ms}$). OS scheduling spikes reach $\sim 29\,\text{ms}$. Hard real-time determinism and avionics RTOS execution are not provided.
7. **Single-Fault Dominance Assumption**: Fault diagnosis evaluates residual signatures assuming a primary physical failure mode. Complex cascading multi-fault interactions may yield ambiguous hypotheses.
8. **Synthetic Population Assumptions**: Fleet variability is generated via parameterized Monte Carlo perturbations of model coefficients, not empirical manufacturing tolerances.
9. **Absence of Calibrated Failure Probabilities**: The system outputs deterministic health indices and heuristic risk scores; it does not output calibrated failure probabilities ($P(\text{failure})$).

For complete technical specifications, see [`docs/claims_and_limitations.md`](file:///d:/SIH%20Drone/docs/claims_and_limitations.md).

---

## 8. Automated Verification & Testing

The repository maintains full regression test coverage across all subsystems:

```bash
# Run the automated claims and submission integrity audit
python scripts/audit_claims.py

# Generate Phase 12 claims and SIH coverage evidence matrices
python scripts/generate_phase12_claims_audit.py

# Run Phase 12 claims audit test suite
pytest tests/test_phase12_claims_audit.py -v

# Run full repository test suite (>730 automated tests)
pytest tests/ -q
```
---

## 9. Running the System

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

## 10. Architectural Declarations

### Algorithm Freezing Statement
Phase 13 does not redesign, retune, replace, or modify the algorithms, thresholds, schemas, or training procedures of Phases 1–12. For runtime inference, Phase 13 deterministically bootstraps Phase 7 Isolation Forest and Phase 8 XGBoost model instances using the existing training procedures and synthetic simulator-generated data. This is synthetic bootstrap model fitting, not external-dataset training or algorithm redesign.

### Preserved Distinctions
- **Algorithm & Training Procedure Freezing**: Feature schemas, classifier configurations, EWMA thresholds, Theil–Sen estimator rules, and multi-modal fusion equations from Phases 1–12 remain unmodified.
- **Runtime Model Fitting**: Deterministic synthetic bootstrap fitting is executed on synthetic simulator data with fixed seeds during orchestrator startup.
- **Pretrained TimesFM Weights**: The TimesFM integration requires an accessible gated checkpoint and compatible local runtime. When unavailable, the application reports the condition and uses its labelled baseline forecast path; it does not fabricate pretrained weights.

### Engineering Reference Anchor & Fidelity Boundary
- **Authoritative Reference Engine**: **Rotax 914 UL/F** (4-cylinder, 1211.2 cc, turbocharged, 84.5 kW takeoff / 73.5 kW continuous rating, 2.4286:1 reduction gearbox). Specification and parameter provenance are maintained in [`configs/engine_reference/rotax_914_ul_f.json`](configs/engine_reference/rotax_914_ul_f.json).
- **Current Simulator Fidelity**: Reduced-order lumped-parameter 0D/1D grey-box prototype. The simulator uses naturally aspirated density derating, a 1:1 direct propeller load simplification ($J=0.28\text{ kg}\cdot\text{m}^2$), and lumped thermal nodes.
- **Missing Physics**: Exhaust gas turbocharger, compressor map, turbine expansion, wastegate actuator, electronic Turbo Control Unit (TCU), manifold absolute pressure (MAP), charge-air heating, 2.43:1 reduction gearbox dynamics, 4-cylinder individual thermal/exhaust runner networks, and electrical/ignition systems. Detailed in [`docs/physics_contract.md`](docs/physics_contract.md).
- **Prohibited Claims**: The system does **NOT** claim to be a "full Rotax 914 F digital twin", "production-ready", "airworthiness validated", or "experimentally validated on real UAV flight recordings".

### Claim Taxonomy
The project adheres to a four-tier verification and validation taxonomy:
1. **`IMPLEMENTED`**: Executable functionality exists in the repository codebase.
2. **`VERIFIED`**: Executable tests/evidence demonstrate that the implementation behaves as mathematically intended. (Reference specifications are verified against official OEM manuals; simulator equations are verified against internal unit tests).
3. **`VALIDATED`**: Compared against an independent authoritative model, certified simulator, or regulatory reference dataset (NOT claimed for the simulator dynamics).
4. **`EXPERIMENTALLY VALIDATED`**: Validated against physical engine test-cell dynamometer recordings or operational flight data (**STRICTLY NOT CLAIMED**; all telemetry is synthetic).

---

## 11. Documentation

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

