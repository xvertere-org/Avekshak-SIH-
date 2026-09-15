# SIH26054 Rotax 914 Digital Twin: Canonical Claims, Provenance & Technical Limitations

> [!IMPORTANT]
> **GOVERNING SPECIFICATION FOR EXTERNAL CLAIMS & SUBMISSION INTEGRITY**
> This document establishes the authoritative boundaries of what the NIRVANAA SIH26054 Aero Piston Engine Digital Twin does, what it does not do, and the exact evidentiary basis for all technical statements.
> Every claim made in documentation, research reports, presentations, and user interfaces must conform to the classifications and limitations defined herein.

---

## 1. Authoritative vs. Synthetic Scope

The NIRVANAA system is a **Physics-Informed Reduced-Order Grey-Box Digital Twin prototype** targeting the turbocharged BRP-Rotax 914 UL/F aero piston engine for Medium-Altitude Long-Endurance (MALE) UAV applications.

### 1.1 What the System Is
- A **lumped-parameter 1D thermodynamic and rotational state estimator** synchronized with incoming engine telemetry.
- A **deterministic residual generation engine** comparing physical telemetry against nominal physics predictions (thermal, lubrication, manifold dynamics).
- A **physics-informed diagnostic and isolation system** evaluating residual signatures, directional subsystem coupling, and sensor fault hypotheses.
- An **uncertainty-aware prognostic estimator** projecting model-defined degradation states ($D(t)$) toward an engineering end-of-life horizon ($D_\text{EOL}$) using robust Theil–Sen regression with deterministic empirical quantile bounds.
- A **counterfactual what-if simulation framework** evaluating mission completion margins under alternate operating profiles and simulated faults.
- A **fully auditable evidence architecture** generating deterministic JSON matrices tracing every output to sensor channels, physics residuals, and engineering rules.

### 1.2 What the System Is NOT
- It is **NOT** a flight-certified software system under FAA DO-178C, EASA CS-E, or MIL-STD-882E.
- It is **NOT** an airworthiness diagnostic authority; it does not issue certified airworthiness directives or mandatory grounding orders.
- It is **NOT** a 3D computational fluid dynamics (CFD) or finite-element structural model.
- It is **NOT** an empirical fleet reliability database; it does not compute certified MTBF, MTTF, or Weibull component survival probabilities.
- It is **NOT** a black-box autonomous deep learning controller; the operational digital twin runtime does not rely on opaque deep neural networks for state estimation.
- It is **NOT** validated against real-world operational UAV flight test data; all operational demonstrations and test matrices are established on synthetic, simulated, or replayed benchmarks.

---

## 2. Authoritative Rotax 914 UL/F OEM Numerical Reference Table

All numerical engine parameters utilized across configurations, models, and documentation are cross-checked against official OEM technical publications.

| Parameter | Authoritative Value | Unit | Source Document | Section / Page | Engine Applicability | Provenance Classification & Semantic Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Takeoff Power (5 min limit)** | **84.5** | kW | EASA Type Certificate Data Sheet E.122 / Rotax 914 Operators Manual | OM Sec 2.1 (p. 2-1) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Max takeoff performance at $5800\,\text{RPM}$ ($115\,\text{HP}$ gross metric; commercial brochure cites $84.8\,\text{kW}$ / $115\,\text{HP}$, authoritative certification standard is $84.5\,\text{kW}$). |
| **Max Continuous Power** | **73.5** | kW | EASA TCDS E.122 / Rotax 914 Operators Manual | OM Sec 2.1 (p. 2-1) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Continuous operational limit at $5500\,\text{RPM}$ ($100\,\text{HP}$). |
| **Max Engine Speed** | **5800** | RPM | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-1) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: 5-minute takeoff redline speed. |
| **Continuous Engine Speed** | **5500** | RPM | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-1) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Max continuous cruising speed. |
| **Idle Engine Speed** | **1400** | RPM | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-1) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Minimum nominal ground idle speed. |
| **Critical Altitude** | **4875** | m | EASA TCDS E.122 / Rotax 914 Operators Manual | OM Sec 2.1 (p. 2-1) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Critical altitude ($16,000\,\text{ft}$) up to which TCU maintains max continuous boost pressure ($115\,\text{kPa}$). *Note: Simulator test envelope sweeps up to $4500\,\text{m}$ as `MODEL_IMPLEMENTATION` test ceiling.* |
| **Gearbox Reduction Ratio** | **2.4286:1** | ratio | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-1) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Propeller gearbox reduction ratio (51/21 teeth). |
| **Max Cylinder Head Temp (CHT)** | **135** | °C | EASA TCDS E.122 / Rotax 914 Operators Manual | OM Sec 2.1 (p. 2-2) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Maximum permissible cylinder head temperature. |
| **Max Exhaust Gas Temp (EGT)** | **1000** | °C | EASA TCDS E.122 / Rotax 914 Operators Manual | OM Sec 2.1 (p. 2-2) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Maximum permissible exhaust gas temperature at takeoff. |
| **Min Oil Pressure (Idle)** | **0.8** | bar | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-2) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Minimum allowable oil pressure below $3500\,\text{RPM}$. |
| **Normal Oil Pressure (Min)** | **2.0** | bar | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-2) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Lower operational limit above $3500\,\text{RPM}$. |
| **Normal Oil Pressure (Max)** | **5.0** | bar | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-2) | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Upper operational limit above $3500\,\text{RPM}$. |
| **Max Oil Pressure (Cold Start)** | **7.0** | bar | BRP-Rotax 914 Operators Manual Ed. 4 / Rev. 0 | OM Sec 2.1 (p. 2-2) / TCDS E.122 Sec A.III | Rotax 914 UL / F | `AUTHORITATIVE_REFERENCE`: Maximum permissible transient oil pressure during cold start. |
| **Health Residual Deadband** | **1.5** | $\sigma$ | Phase 5 Health Assessment Specification | `health_index/schema.py` | Grey-box Twin | `ENGINEERING_HEURISTIC`: Deadband threshold ($\tau_\text{nominal}$) in normalized residual standard deviations ($\sigma$) — *NOT a bar pressure unit*. |
| **Degraded Pressure Clamp** | **0.5** | bar | Phase 4 Lubrication Physics Model | `simulator/subsystems/lubrication.py` | Simulator | `MODEL_IMPLEMENTATION`: Numerical floor preventing negative/zero oil pressure in simulation loop. |

---

## 3. Explicit 6-Tier AI/ML Runtime Inventory

To eliminate ambiguity regarding machine learning utilization, every analytical and learning component in the repository is audited and classified into exactly one of six operational tiers:

```
                                AI/ML COMPUTATIONAL INVENTORY
                                
  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
  │   OPERATIONAL RUNTIME   │     │     VALIDATION ONLY     │     │       QUARANTINED       │
  │ • Grey-box ODE physics  │     │ • XGBoost 6-class       │     │ • TimesFM foundation    │
  │ • Residual generator    │     │   classifier in         │     │   model in              │
  │ • Directional coupling  │     │   fault_diagnosis/      │     │   forecasting/          │
  │ • Theil-Sen regression  │     │ • Random Forest         │     │   (gated/bypassed       │
  │ • Empirical quantiles   │     │   comparisons           │     │    due to dependencies) │
  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
  │   OFFLINE EXPERIMENT    │     │         LEGACY          │     │         UNUSED          │
  │ • Feature extraction    │     │ • Early stage 1 prototypes│   │ • Orphaned scripts      │
  │   scripts in telemetry/ │     │   in phase14/           │     │   not invoked in tests  │
  │ • MC trajectory scripts │     │                         │     │   or runtime            │
  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

| Component / Subsystem | Repository Path | Classification Tier | Implemented Mathematical / Analytical Method | Role in Operational Digital Twin |
| :--- | :--- | :--- | :--- | :--- |
| **Rotational Dynamics & Engine Core** | `simulator/subsystems/dynamics.py`, `digital_twin/twin_model.py` | `OPERATIONAL_RUNTIME` | First-principles torque balance ODE: $J \dot{\omega} = \tau_\text{ind} - \tau_\text{fric} - \tau_\text{load}$. | Deterministic state estimation synchronized with telemetry. |
| **Thermal Subsystem** | `simulator/subsystems/thermal.py`, `digital_twin/twin_model.py` | `OPERATIONAL_RUNTIME` | 4-cylinder lumped-parameter heat transfer ODE network (combustion heat input, convective cooling, coolant coupling). | Computes expected CHT/EGT temperatures and physical residuals. |
| **Lubrication Subsystem** | `simulator/subsystems/lubrication.py`, `digital_twin/twin_model.py` | `OPERATIONAL_RUNTIME` | Empirical Vogel-Fulcher-Tammann oil viscosity curve coupled to positive-displacement pump delivery. | Computes expected oil pressure and temperature residuals. |
| **Health Aggregator** | `health_index/calculator.py`, `digital_twin/health.py` | `OPERATIONAL_RUNTIME` | Dynamically renormalized weighted linear sum of piecewise-linear residual evidence functions ($e_i \in [0, 1]$). | Generates continuous engine and subsystem Health Index ($[0, 1]$). |
| **Diagnostic Hypothesis Isolation** | `digital_twin/diagnosis.py` | `OPERATIONAL_RUNTIME` | Deterministic physical residual signature compatibility scoring and sensor fault disambiguation logic. | Ranks fault hypotheses based on directional physical coupling. |
| **Degradation & RUL Estimator** | `digital_twin/degradation.py`, `digital_twin/rul.py` | `OPERATIONAL_RUNTIME` | Robust non-parametric Theil–Sen linear regression with deterministic empirical quantile bounds (15th/85th percentiles of pairwise slopes). | Projects time-to-horizon under modeled operating/stress assumptions. |
| **Multi-Class Supervised Classifier** | `fault_diagnosis/classifier.py`, `fault_diagnosis/pipeline.py` | `VALIDATION_ONLY` | Gradient-boosted decision trees (XGBoost) and Random Forests trained on synthetic population datasets. | Evaluated in validation test suites as benchmark comparisons; **NOT invoked in operational runtime**. |
| **Foundation Forecasting Model** | `forecasting/timesfm_wrapper.py`, `forecasting/` | `QUARANTINED` | TimesFM zero-shot foundation time-series neural network. | Quarantined/bypassed in production orchestrator due to heavy dependencies and non-deterministic execution. |
| **Monte Carlo Trajectory Sampler** | `prognostics/uncertainty.py` | `OFFLINE_EXPERIMENT` | Stochastic Monte Carlo perturbation ($M=500$ realizations) across anchor HI, slope, and threshold distributions. | Offline research exploration of parameter sensitivity; operational twin uses deterministic Theil–Sen quantiles. |
| **Vibration & Spectral Analytics** | `telemetry/features.py` | `OFFLINE_EXPERIMENT` | Fast Fourier Transform (FFT) spectral band energy, RMS, and kurtosis calculation. | Offline signal processing and dataset feature engineering. |

---

## 4. Preserved Engineering Limitations Catalog

The following nine fundamental boundaries are preserved as core engineering disclosures. They must never be softened, hidden, or omitted:

1. **Synthetic Grey-Box Simulator Scope:**
   The engine simulator is a 1D lumped-parameter reduced-order model. While calibrated against Rotax 914 thermodynamic limits, it does not simulate 3D cylinder gas dynamics, turbulent flame propagation, acoustic manifold resonances, or localized structural stress concentrations.
2. **Absence of Real Engine Fleet Validation:**
   The degradation trajectories and fault models are synthetic physics representations calibrated against open engineering literature and nominal OEM specifications. The system has **NOT** been validated against an operational fleet of physical Rotax 914 engines or physical UAV flight logs.
3. **Non-Certification & Non-Airworthiness:**
   This software is an academic and engineering competition prototype (SIH 2026). It is **NOT** certified for airborne use under FAA DO-178C (Software Considerations in Airborne Systems), DO-254 (Design Assurance for Airborne Electronic Hardware), EASA CS-E, or MIL-STD-882E.
4. **Model-Defined Remaining Useful Life (RUL):**
   RUL values generated by the system represent the projected mathematical duration required for the modeled degradation state $D(t)$ to reach an engineering threshold ($D_\text{EOL} = 0.50$) under the estimator's modeled operating/stress assumptions. They do **NOT** predict real mechanical Time Between Overhaul (TBO) or sudden catastrophic component fatigue.
5. **Heuristic Mission Risk Index ($R_\text{mission}$):**
   The mission risk score ($R_\text{mission} \in [0, 1]$) is a heuristic engineering metric combining envelope excursion penalties, rate-of-degradation trends, and health deficits. It is **NOT** a calibrated statistical probability of engine failure ($P(\text{failure})$) or loss-of-aircraft probability.
6. **Host-Side Execution Latency & Soft Real-Time Boundary:**
   Measured execution latency is established on commodity desktop hardware (Python 3.11, Windows/x86_64). While steady-state streaming execution takes $<1.0\,\text{ms}$ (mean $0.388\,\text{ms}$, p99 $0.763\,\text{ms}$), host operating system scheduling and garbage collection introduce spikes up to $28.994\,\text{ms}$. The system is suitable for **soft real-time** streaming at 1–10 Hz, but provides **no hard real-time determinism** or avionics RTOS guarantees.
7. **Single / Localized Fault Dominance Assumption:**
   The diagnostic residual signature matrix assumes that an anomaly is driven by a predominant physical degradation mechanism. Complex cascading failures or concurrent sensor failures may produce overlapping residual signatures, resulting in unisolated or ambiguous hypotheses.
8. **Synthetic Population Parameterization:**
   Fleet variability models (wear coefficients, friction variations, sensor offsets) are generated via parameterized Monte Carlo perturbation of model parameters, rather than empirical manufacturing tolerances or fleet teardown measurements.
9. **Absence of Calibrated Failure Probabilities:**
   The system emits deterministic health states, compatibility rankings, and risk scores. It does **NOT** output calibrated frequentist or Bayesian probabilities of failure.

---

## 5. Mandatory Claim-Strength Evaluation Protocol

When writing or evaluating any technical claim regarding NIRVANAA, authors and auditors must apply the **Strongest Reasonable Interpretation Test**:
> *"What is the strongest reasonable interpretation a technically knowledgeable evaluator could take from this wording?"*
> *If that interpretation exceeds the repository's verified evidence, the claim must be rewritten.*

### Audited High-Risk Terms & Bounded Usage

| High-Risk Term | Prohibited Interpretations (Overstatement) | Permitted & Evidence-Bounded Usage |
| :--- | :--- | :--- |
| `predict` / `prediction` | Forecasting actual physical mechanical breakdowns in real engines. | Deterministic trend extrapolation of grey-box degradation states toward model threshold $D_\text{EOL}$. |
| `detect` / `diagnose` | Flight-certified avionics warning system isolating physical hardware failures. | Rule-based and residual-threshold hypothesis isolation on simulated or replayed telemetry. |
| `reliable` / `reliability` | Airworthiness-certified MTBF, MTTF, or Weibull survival functions. | Heuristic mission risk index ($R_\text{mission}$) quantifying envelope margins during simulated missions. |
| `real-time` | Hard real-time deterministic execution on certified RTOS microcontrollers. | Host-side benchmarked execution throughput ($<1\,\text{ms}$ mean) suitable for soft real-time ingestion at 1 Hz. |
| `AI` / `ML` / `intelligent` | Autonomous black-box neural networks making automated flight decisions. | Physics-informed grey-box digital twin; supervised classifiers evaluated in validation suites only. |
| `digital twin` | Complete CFD/FEM structural replica of an entire propulsion system. | 1D lumped-parameter thermodynamic and rotational state estimator synchronized with telemetry. |
| `validated` | Full-scale dynamometer, environmental chamber, or flight test validation. | Parameter cross-check against EASA TCDS and synthetic verification against simulation benchmarks. |
| `RUL` / `remaining life` | Certified maintenance hours remaining before physical overhaul. | Model-defined projection to $D_\text{EOL}$ under modeled operating/stress assumptions. |
| `confidence` | Calibrated Bayesian credible interval or statistical p-value. | Engineering metric reflecting residual variance, sample count, or anomaly persistence. |
| `causal` / `causality` | Formal Pearlian causal DAG discovery or do-calculus identification. | Directional forward propagation of physical coupling across modeled thermodynamic subsystems. |
| `flight-ready` / `production-ready` | Software certified for installation in operational aircraft. | Fully tested research and competition prototype implemented in Python. |

---

## 6. SIH26054 Requirements Coverage Audit

The system's compliance against the SIH26054 problem statement is audited based strictly on executed, demonstrated code behavior:

| SIH26054 Requirement | Audited Status | Demonstrated Implementation & Evidence Basis |
| :--- | :--- | :--- |
| **Physics-Informed Digital Twin** | `IMPLEMENTED` | Lumped-parameter 1D engine state estimator synchronized with telemetry; tracks CHT, EGT, RPM, MAP, oil pressure. Verified in Phase 2, 3, 6 suites (`evidence/phase2_operating_matrix.json`, `tests/test_phase2_operating_matrix.py`). |
| **Real-Time Telemetry Processing** | `PARTIALLY_IMPLEMENTED` | Soft real-time host-side execution verified ($<1\,\text{ms}$ mean latency for 1000 steps); hard real-time RTOS deployment is out of scope. Verified in `evidence/phase6_latency_benchmark.json`. |
| **Telemetry Ingestion Infrastructure** | `IMPLEMENTED` | Canonical schema, unit normalization, clock skew detection, sequence gap handling, and multi-rate interpolation. Verified in Phase 9 suites (`tests/test_phase9_real_telemetry.py`). |
| **Real Engine Flight Validation** | `NOT_IMPLEMENTED` | All operational testing and validation conducted on synthetic/replayed streams. No physical flight test logs claimed (`evidence/phase9_telemetry_matrix.json`). |
| **Multi-Fault Physics Simulation** | `IMPLEMENTED` | 6 fault classes: cooling degradation, oil pressure loss/wear, injector clogging, misfire, mechanical friction, and sensor faults. Verified in Phase 4 suites (`tests/test_fault_physics_phase4.py`). |
| **Health Index Assessment (HI)** | `IMPLEMENTED` | Continuous health index $\text{HI} \in [0, 1]$ computed across subsystems via dynamically renormalized weighted linear sums. Verified in Phase 5 suites (`tests/test_health_assessment_phase5.py`). |
| **Fault Diagnosis & Isolation** | `IMPLEMENTED` | Physics-informed hypothesis isolation via residual signatures and directional coupling rules. Verified in Phase 6 suites (`tests/test_fault_diagnosis_phase6.py`). |
| **Prognostics & RUL Estimation** | `IMPLEMENTED` | Robust Theil–Sen linear trend projection to model horizon $D_\text{EOL}$ with empirical quantile bounds. Verified in Phase 8 suites (`tests/test_phase8_rul.py`). |
| **Mission Reliability & What-If** | `IMPLEMENTED` | Counterfactual scenario simulation evaluating mission completion risk under alternate operating profiles and fault injections. Verified in Phase 10 suites (`tests/test_phase10_mission.py`). |
| **Engineering Explainability** | `IMPLEMENTED` | Provenance-traced explanations linking outputs to sensor channels, physics residuals, and engineering heuristics. Verified in Phase 11 suites (`tests/test_phase11_explainability.py`). |
| **Interactive Operator Dashboard** | `IMPLEMENTED` | Streamlit operational UI displaying twin state, health indices, telemetry replay, and scenario what-if controls (`dashboard/app.py`). |
| **Supervised ML Classification** | `VALIDATION_ONLY` | XGBoost/Random Forest classifiers trained and evaluated on synthetic population benchmarks; not part of operational runtime (`fault_diagnosis/`). |
| **DO-178C / Airworthiness Certification** | `OUT_OF_SCOPE` | Academic and competition prototype; airworthiness certification explicitly disclaimed. |
