# SIH26054 Phase 3 Technical Report: Physics–ML Residual Integration

**Generated:** 2026-09-14T21:13:07.172251  
**System Architecture:** Grey-Box Digital Twin (Physics Primary, ML Residual Correction)  
**Reference Engine:** Rotax 914 UL/F Turbocharged Aero Piston Engine  

---

## 1. Executive Summary

Phase 3 integrates deterministic thermodynamic/fluid physics with machine-learning residual correction to establish the Grey-Box Digital Twin core.

### Key Principles & Implementation Facts
- **Physics Baseline:** Deterministic nominal expected state generation across all canonical aero-piston channels (`cht`, `egt`, `oil_temp`, `oil_pressure`, `fuel_flow`, `rpm`, `map_bar`, `coolant_temp`, `vibration`).
- **Residual Learning:** Mathematically exact residual computation $r = y_{\text{observed}} - \hat{y}_{\text{physics}}$ and corrected prediction $\hat{y}_{\text{corrected}} = \hat{y}_{\text{physics}} + \hat{r}_{\text{ML}}$.
- **Leakage Safety:** Scalers, imputers, and regressors are fitted strictly on training flight partitions; test flight partitions remain completely unseen.
- **Target-Dependent Performance:** Evaluation shows that model performance is target-dependent. The residual model improves several channels, but does not outperform the physics-only baseline on every channel.
- **Operator-Facing Terminology:** The dashboard and telemetry interfaces expose clear operational terminology (*Physics estimate*, *Sensor-informed correction*, *Corrected prediction*, *Detected deviation*, *Model confidence*) with zero internal engineering jargon.

---

## 2. Three-Way Model Comparison Results

Evaluated on held-out test flight profiles (600 records across 2 independent degradation flight profiles):

| Target Channel | Unit | Physics-Only MAE | Pure ML MAE | Grey-Box MAE | vs Physics Status | vs Pure ML Status | ΔMAE vs Physics | ΔMAE vs Pure ML |
|---|---|---|---|---|---|---|---|---|
| **cht** | °C | 6.0733 | 6.3040 | **6.5381** | `DEGRADED` | `DEGRADED` | **-0.4648** | **-0.2341** |
| **egt** | °C | 8.8504 | 29.5363 | **12.6465** | `DEGRADED` | `IMPROVED` | **-3.7961** | **+16.8898** |
| **oil_temp** | °C | 8.8024 | 7.9804 | **5.1454** | `IMPROVED` | `IMPROVED` | **+3.6569** | **+2.8350** |
| **oil_pressure** | bar | 0.3582 | 0.4521 | **0.4891** | `DEGRADED` | `DEGRADED` | **-0.1309** | **-0.0370** |
| **fuel_flow** | L/h | 0.3510 | 1.0972 | **0.5193** | `DEGRADED` | `IMPROVED` | **-0.1683** | **+0.5780** |
| **rpm** | RPM | 61.5075 | 188.8911 | **147.8079** | `DEGRADED` | `IMPROVED` | **-86.3004** | **+41.0832** |
| **map_bar** | bar | 0.0055 | 0.0185 | **0.0183** | `DEGRADED` | `SIMILAR` | **-0.0127** | **+0.0002** |
| **coolant_temp** | °C | 0.9921 | 1.1004 | **0.7939** | `IMPROVED` | `IMPROVED` | **+0.1983** | **+0.3066** |
| **vibration** | g | 0.0178 | 0.0416 | **0.0206** | `DEGRADED` | `IMPROVED` | **-0.0028** | **+0.0209** |

### Detailed Channel-Level Findings:
- **Oil Temperature (`oil_temp`):** Grey-Box achieves substantial improvement over both baselines (MAE: 5.1454 °C vs Physics: 8.8024 °C and Pure ML: 7.9804 °C; $\Delta$MAE vs Physics = +3.6570 °C).
- **Coolant Temperature (`coolant_temp`):** Grey-Box achieves improvement over both baselines (MAE: 0.7939 °C vs Physics: 0.9921 °C and Pure ML: 1.1004 °C; $\Delta$MAE vs Physics = +0.1982 °C).
- **Exhaust Gas Temp (`egt`), Fuel Flow (`fuel_flow`), RPM (`rpm`), Vibration (`vibration`):** Grey-Box significantly outperforms Pure ML, but does not beat the physics-only baseline in this synthetic cruise regime.
- **Cylinder Head Temp (`cht`) & Oil Pressure (`oil_pressure`):** Grey-Box performs slightly worse than both baselines, showing that adding linear residual regression adds variance when the underlying degradation is localized.
- **Manifold Pressure (`map_bar`):** The physics-only baseline already has near-zero error (MAE: 0.0055 bar). ML residuals do not improve upon this baseline.

---

## 3. Explicit Target & Channel Mapping

| Role | Features / Channels | Description |
|---|---|---|
| **Physics Inputs** | `throttle`, `altitude`, `ambient_temp`, `airspeed` | Primary observable operating context driving baseline thermodynamics |
| **Observed Targets** | `cht`, `egt`, `oil_temp`, `oil_pressure`, `fuel_flow`, `rpm`, `map_bar`, `coolant_temp`, `vibration` | Ground-truth sensor telemetry measurements |
| **Physics Predictions** | `cht_expected`, `egt_expected`, `oil_temp_expected`, etc. | Output of `DigitalTwinModel.step_expected()` |
| **Residual Targets** | $r = y_{\text{observed}} - \hat{y}_{\text{physics}}$ | Discrepancy between sensor observation and nominal model |
| **ML Features** | Observable context + $\Delta$ rates + physics expected states | Inputs supplied to residual regressor $\hat{r}_{\text{ML}}$ |
| **Corrected Predictions** | $\hat{y}_{\text{corrected}} = \hat{y}_{\text{physics}} + \hat{r}_{\text{ML}}$ | Final sensor-informed grey-box estimate |
| **Health Indicators** | Residual magnitude and z-scores | Monitored deviation signaling operational degradation |

---

## 4. Controlled Benchmark & Surrogate Distinction Policy

> [!IMPORTANT]
> **Rotax Engine Claim Policy:**
> External benchmark datasets (Paderborn, CWRU, FEMTO, C-MAPSS) are auxiliary component benchmarks used solely for algorithm verification (e.g. bearing fault classification and run-to-failure prognostics). They **do not** validate the complete Rotax 914 UL/F aero-piston engine.
> - Paderborn results are reported over *independent experiment files*, with *no exact duplicate file-level mean feature vectors under the selected summary features*.
> - Aero-piston thermodynamics, cooling loops, and rotational dynamics are governed strictly by the Rotax 914 UL/F physics contract.
> - Zero real-world flight hours, latency, or airworthiness certifications are fabricated.

---

## 5. Explainability Architecture

Local feature attribution separates:
1. **Physics Drivers:** Commanded throttle position and ambient atmospheric conditions determining base operational point.
2. **Residual Drivers:** High-order operational dynamics, thermal lag, and localized load deviations learned by the residual model.
3. **Health Indicators:** Magnitude and direction of unmodeled residual deviations indicating degradation (e.g. cooling loop degradation elevates CHT and coolant temperature).

*Note: Feature attributions describe model feature importance and sensitivity, and do not establish physical causation.*

---

## 6. Official Conclusion

Phase 3 is implementation-complete and verified for the current simulated aero-piston workflow. The physics-only, pure-ML, and physics-plus-ML residual paths are implemented. Residual calculation, corrected prediction, leakage protection, deterministic inference, explainability structure, dashboard integration, and automated tests are in place. Evaluation shows target-dependent performance. The grey-box model improves oil temperature, coolant temperature, exhaust-gas temperature relative to pure ML, fuel flow relative to pure ML, and vibration relative to pure ML. It does not outperform the physics-only baseline on every channel, so the results should be presented as a controlled technical demonstration rather than universal performance superiority. The current implementation uses simulated aero-piston telemetry and controlled benchmark datasets. It does not constitute certified validation of the complete Rotax 914 engine and must not be presented as OEM, FAA, or real-flight validation.

---

## 7. Verification and Reproduction Commands

```powershell
# Run Phase 3 generation script
python scripts/generate_phase3_residual_results.py

# Run automated Phase 3 verification suite
pytest -v tests/test_physics_ml_residual.py

# Run full regression suite
pytest -q tests/test_ml_baselines.py tests/test_ml_data_layer.py tests/test_data_pipeline.py
```