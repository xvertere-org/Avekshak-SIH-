# SIH26054 Phase 2: Dataset-Specific ML Baselines Summary

**Generated:** 2026-09-14T20:26:46.592347  
**System Architecture:** Grey-Box Digital Twin (Physics Primary, ML Auxiliary)  
**Rotax Engine Claim Policy:** Benchmark results strictly isolated from Rotax flight claims. Zero fabricated physics.  

---

## 1. Executive Summary Table

| Dataset ID | Task | Primary Model | Status | Key Metric | Reference Metric (Dummy) | Split Strategy |
|---|---|---|---|---|---|---|
| `cwru` | Fault Classification | `RandomForest` | **`controlled_demo`** | Acc: `0.000` (Macro F1: `0.000`) | Acc: `0.000` | File Holdout (`97.mat`, `130.mat` -> `105.mat`) |
| `paderborn` | Fault Classification | `RandomForest` | **`completed`** | Acc: `1.000` (Macro F1: `1.000`) | Acc: `0.333` | Source File Group (168 train, 36 val, 36 test) |
| `nust` | Fault Classification | `RandomForest` | **`completed`** | Acc: `0.980` (Macro F1: `0.978`) | Acc: `0.665` | Source File Group (94 train, 20 val, 20 test) |
| `cwru` | Anomaly Detection | `IsolationForest` | **`controlled_demo`** | F1: `1.000`, FPR: `0.000` | Baseline Centroid F1: `1.000` | File Holdout (Trained on Normal `97.mat`) |
| `paderborn` | Anomaly Detection | `IsolationForest` | **`completed`** | F1: `0.970`, FPR: `0.060` | Baseline Centroid F1: `0.986` | Source File Group (Trained on Healthy files) |
| `nust` | Anomaly Detection | `IsolationForest` | **`completed`** | F1: `0.150`, FPR: `0.045` | Baseline Centroid F1: `0.161` | Source File Group (Trained on Healthy files) |
| `femto` | Anomaly Detection | N/A | **`skipped`** | No explicit binary healthy reference | N/A | Documented Skip |
| `femto` | Bearing RUL | `RandomForest` | **`completed`** | MAE: `710.2` steps (R²: `-0.330`) | Dummy MAE: `778.9` | Entity Held-Out (4 train bearings, 2 test) |
| `nasa_battery` | Battery SOH | `RandomForest` | **`completed`** | MAE: `0.0603` (R²: `0.817`) | Dummy MAE: `0.1706` | Entity Held-Out (24 train cells, 10 test) |
| `cmapss` | Turbofan RUL | `RandomForest` | **`completed`** | MAE: `13.4` cycles (R²: `0.803`) | Dummy MAE: `37.0` | Entity Held-Out (70 train units, 30 test) |
| `basic` | UAV Telemetry | N/A | **`skipped`** | Manual download required | N/A | Unavailable |

---

## 2. Dataset-Specific Baseline Findings & Limitations

### A. Fault Classification
1. **CWRU (`cwru`)**: Controlled demonstration on 3 files (`97.mat`, `130.mat`, `105.mat`). Because each file represents a single fault mode, full closed-set generalization is not statistically claimed.
2. **Paderborn (`paderborn`)**: Evaluated across 240 distinct measurement files. RandomForest achieved **99.98% accuracy** across 3 classes (`healthy`, `outer_race_fault`, `inner_race_fault`) with strictly zero file overlap.
3. **NUST (`nust`)**: Evaluated across 134 experimental runs. RandomForest achieved **98.03% accuracy** on `bearing_condition` with zero run overlap.

### B. Semi-Supervised Anomaly Detection
- Trained **strictly on healthy reference data** in training groups; evaluated on held-out test groups containing both healthy and damaged bearings.
- **Paderborn**: Isolation Forest achieved **0.970 F1 score** (FPR: 0.060).
- **NUST**: Isolation Forest achieved **0.150 F1 score** (FPR: 0.045).
- **FEMTO**: Explicitly documented as **skipped** because it is a run-to-failure trajectory dataset without an explicit binary healthy reference class.

### C. Degradation and RUL Regression
- Temporal ordering preserved within every entity; zero shuffling across time steps.
- **FEMTO**: 6 bearings partitioned into 4 training bearings and 2 test bearings (`Bearing1_1`, `Bearing1_2`). RandomForest achieved MAE of **710.2 time steps** (vs Dummy 778.9).
- **NASA Battery**: 34 lithium-ion cells partitioned into 24 training and 10 test cells. RandomForest achieved SOH MAE of **0.0603**.
- **NASA C-MAPSS**: 100 turbofan units partitioned into 70 training and 30 test units. RandomForest achieved RUL MAE of **13.4 cycles**.

---

## 3. Grey-Box Guardrails & Core Domain Protection
- **Zero Domain Modifications**: Verified that zero files were modified in `physics/`, `simulator/`, `telemetry/`, `digital_twin/`, `health_index/`, `anomaly_detection/`, `fault_diagnosis/`, `frontend/`, `backend/`.
- **Zero Fabricated Physics**: No mock residuals, health scores, or Rotax labels were injected.
- **Recommended Next Step**: Connect real digital-twin physics residuals (`map_residual`, `cht_residual`, `cooling_residual`, `subsystem_health_scores`) to the ML feature layer.