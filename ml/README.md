# Grey-Box ML Integration Layer (`ml/`)

This directory implements **Phase 1 of Grey-Box ML Integration** for the **SIH26054 Aero-Piston Engine Grey-Box Digital Twin** project.

It provides a unified, leakage-safe dataset registry, loader, feature schema, and group-aware splitting framework for all processed benchmark and component datasets.

---

## 1. Architectural Philosophy & Context

The SIH26054 system employs a strict **grey-box** hierarchy:

```text
Telemetry
   │
   ▼
Data Quality Validation
   │
   ▼
Physics-Based Digital Twin (Rotax 914 Engine Model)
   │
   ▼
Physics Predictions & Residual Calculation
   │
   ▼
Subsystem Health Index
   │
   ▼
Machine Learning Models (Anomaly Detection, Fault Diagnosis, RUL)
   │
   ▼
Final Health Decision
```

### Core Principles:
1. **Physics is Primary**: The Rotax 914 thermodynamic/mechanical digital twin captures known first-principles behavior (turbocharger wastegate dynamics, 4-cylinder heat rejection, propeller reduction ratio).
2. **ML is Auxiliary**: ML models learn non-linear patterns and degradation dynamics that cannot be fully represented by the physics model. ML **never** replaces or bypasses the physics twin.
3. **Strict Physical Segregation**:
   - **C-MAPSS** & **NASA Battery** are **methodology benchmark datasets** for RUL and degradation filtering. They are **never** treated as direct Rotax telemetry.
   - **CWRU, FEMTO, Paderborn, NUST** are **bearing and component vibration datasets**. They are never cast into Rotax engine channels (e.g. CHT, EGT, MAP).
4. **Leakage-Safe by Design**: Overlapping windows from the same unit, bearing, battery, or experiment run are never randomly partitioned between training and evaluation splits.

---

## 2. Directory Structure

```text
ml/
├── __init__.py                # Top-level exports
├── dataset_registry.py        # Central registry, metadata, and task definitions
├── data_loader.py             # Parquet-only loader, schema validation, incompatible-data guard
├── feature_schema.py          # Feature group taxonomy (vibration, degradation, reserved physics)
├── split_strategy.py          # Leakage-safe group splitter with CWRU guards
├── validation.py              # Numerical integrity, leakage, and protected domain checks
├── configs/
│   └── dataset_config.json    # Declarative configuration of all datasets
├── artifacts/                 # Serialized schemas and runtime outputs
└── README.md                  # This documentation
```

---

## 3. Dataset Registry & Taxonomy

| Dataset ID | Full Name | Availability | Integration Role | Target Tasks | Grouping Key | Known Limitations |
|---|---|---|---|---|---|---|
| `cmapss` | NASA C-MAPSS Turbofan | AVAILABLE | RUL_METHODOLOGY_BENCHMARK | Degradation Trend, RUL | `unit_number` | Gas turbine physics; not piston engine telemetry. |
| `femto` | FEMTO / PRONOSTIA Bearing | AVAILABLE | BEARING_DEGRADATION_RUL | Bearing Degradation, RUL | `bearing_id` | Accelerated wear under constant load. Test set lacks ground truth RUL in feature table. |
| `cwru` | Case Western Reserve Univ | AVAILABLE | BEARING_FAULT_CLASSIFICATION | Fault Classification, Feature Validation | `source_file` / `bearing_id` | **Only 3 source files** (`105.mat`, `130.mat`, `97.mat`). Conventional 3-way split is statistically invalid; use for feature verification and controlled demos. |
| `paderborn` | Paderborn Bearing | AVAILABLE | BEARING_FAULT_DIAGNOSIS | Fault Diagnosis, Feature Validation | `bearing_id` / `source_file` | Motor test rig with artificially accelerated fatigue. |
| `nasa_battery`| NASA PCoE Battery Aging | AVAILABLE | SOH_RUL_BENCHMARK | SOH & RUL Methodology | `battery_id` | Lithium-ion chemistry; strictly a filtering/RUL benchmark. |
| `nust` | NUST Bearing Vibration | AVAILABLE | VIBRATION_FAULT_FEATURE_LEARNING| Feature Learning, Anomaly Detection | `source_file` | Multi-channel rig across temperature and humidity variations. |
| `basic` | BASiC UAV Telemetry | UNAVAILABLE | UNAVAILABLE | N/A | None | Requires manual download from Zenodo DOI 10.5281/zenodo.8195068. |

---

## 4. Usage Quickstart

### A. Inspecting the Registry
```python
from ml import DatasetRegistry

registry = DatasetRegistry()
print("Available datasets:", registry.list_available_datasets())

meta = registry.get("cmapss")
print(f"Role: {meta.integration_role}, Group Candidates: {meta.candidate_grouping_columns}")
```

### B. Loading Processed Parquet Data
```python
from ml import DataLoader

loader = DataLoader()
df = loader.load_dataset("femto", split="train")
summary = loader.summarize_dataset("femto", split="train")

print(f"Loaded {summary.num_rows} rows, {summary.num_features} features.")
print(f"Grouping column: {summary.grouping_column}")
```

### C. Leakage-Safe Group Splitting
```python
from ml import DataLoader, GroupSplitter

loader = DataLoader()
df = loader.load_dataset("paderborn")

splitter = GroupSplitter()
train_df, val_df, test_df, result = splitter.split(
    df=df,
    dataset_id="paderborn",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

assert not result.leakage_detected
print(f"Train groups: {result.train_groups}, Test groups: {result.test_groups}")
```

### D. Feature Classification & Reserved Physics Features
```python
from ml import FeatureSchema

columns = ["h_rms", "h_kurtosis", "spectral_centroid", "rul_label", "bearing_id"]
classification = FeatureSchema.classify_columns(columns, dataset_name="femto")

print("Vibration features:", classification.vibration_features)
print("Degradation features:", classification.degradation_features)

# Verification that no mock physics was fabricated in Phase 1
FeatureSchema.assert_no_fabricated_physics(columns)
```

---

## 5. Verification & Quality Gates

Run the verification test suite:
```bash
python -m pytest -q tests/test_ml_data_layer.py
```
This suite verifies:
- 100% adherence to parquet inputs (rejection of raw zip/mat/csv inputs).
- Absence of train/test group leakage.
- Absence of unhandled NaNs/Infs in feature matrices.
- Protection of core domain directories (`physics/`, `simulator/`, `digital_twin/`, etc.).
