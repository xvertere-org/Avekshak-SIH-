# SIH26054 ML Dataset Registry & Provenance Report

**Generated:** 2026-09-14T19:46:55.228093  
**Available Datasets:** 6 / 7  
**Total Processed Parquet Files:** 14  
**Total Processed Records:** 715,548  

## Executive Summary

| Dataset ID | Display Name | Role | Status | Files | Rows | Grouping Key | Groups |
|---|---|---|---|---|---|---|---|
| `cmapss` | NASA C-MAPSS | `RUL_METHODOLOGY_BENCHMARK` | AVAILABLE | 8 | 265,256 | `unit_number` | 260 |
| `femto` | FEMTO Bearing | `BEARING_DEGRADATION_RUL` | AVAILABLE | 2 | 21,493 | `bearing_id` | 17 |
| `cwru` | CWRU Bearing | `BEARING_FAULT_CLASSIFICATION` | AVAILABLE | 1 | 1,179 | `source_file` | 3 |
| `paderborn` | Paderborn Bearing | `BEARING_FAULT_DIAGNOSIS` | AVAILABLE | 1 | 29,792 | `bearing_id` | 3 |
| `nasa_battery` | NASA Battery | `SOH_RUL_BENCHMARK` | AVAILABLE | 1 | 7,565 | `battery_id` | 34 |
| `nust` | NUST Vibration | `VIBRATION_FAULT_FEATURE_LEARNING` | AVAILABLE | 1 | 390,263 | `source_file` | 134 |
| `basic` | BASiC UAV Telemetry | `UNAVAILABLE` | UNAVAILABLE | 0 | 0 | `NONE` | 0 |

---

## Detailed Dataset Profiles & Limitations

### C-MAPSS Turbofan Degradation Dataset (`cmapss`)
- **Role**: `RUL_METHODOLOGY_BENCHMARK`
- **Target Tasks**: DEGRADATION_TREND_BENCHMARK, RUL_BENCHMARK
- **Status**: `AVAILABLE`
- **Grouping Column**: `unit_number` (260 unique groups)
- **Processed Records**: 265,256
- **Limitations & Boundaries**:
  * Simulated turbofan engine (gas turbine) physics, not spark-ignition aero-piston engine.
  * Must not be used as direct Rotax 914 engine telemetry.
  * Sensor channels (cmapss_s_1..21) are normalized gas turbine parameters; must never be mapped to piston channels (e.g. CHT, EGT, MAP).

### FEMTO / PRONOSTIA Bearing Degradation Dataset (`femto`)
- **Role**: `BEARING_DEGRADATION_RUL`
- **Target Tasks**: BEARING_DEGRADATION_MODELING, RUL_MODELING
- **Status**: `AVAILABLE`
- **Grouping Column**: `bearing_id` (17 unique groups)
- **Processed Records**: 21,493
- **Limitations & Boundaries**:
  * Run-to-failure bearing test bench under constant operating loads (artificial accelerated wear).
  * Test set does not include ground-truth RUL targets in feature table.
  * Must not be confused with complete aero-engine mechanical assembly.

### Case Western Reserve University Bearing Dataset (`cwru`)
- **Role**: `BEARING_FAULT_CLASSIFICATION`
- **Target Tasks**: BEARING_FAULT_CLASSIFICATION, FEATURE_VALIDATION, CONTROLLED_DEMONSTRATION
- **Status**: `AVAILABLE`
- **Grouping Column**: `source_file` (3 unique groups)
- **Processed Records**: 1,179
- **Limitations & Boundaries**:
  * CRITICAL LIMITATION: Current processed dataset contains only 3 independent source files (105.mat, 130.mat, 97.mat).
  * Conventional random train/validation/test splitting across records from the same file creates catastrophic data leakage.
  * Must be used primarily for vibration feature validation, baseline feature extraction verification, and controlled fault-pattern demonstrations until more independent source files are acquired.
  * Not representative of piston engine reciprocating dynamics.

### Paderborn University Bearing Dataset (`paderborn`)
- **Role**: `BEARING_FAULT_DIAGNOSIS`
- **Target Tasks**: BEARING_FAULT_DIAGNOSIS, VIBRATION_FEATURE_VALIDATION
- **Status**: `AVAILABLE`
- **Grouping Column**: `bearing_id` (3 unique groups)
- **Processed Records**: 29,792
- **Limitations & Boundaries**:
  * Laboratory motor-bearing test rig under controlled speed/torque/radial force conditions.
  * Artificially damaged and accelerated fatigue bearings; not a whole-engine vibroacoustic profile.

### NASA PCoE Battery Aging Dataset (`nasa_battery`)
- **Role**: `SOH_RUL_BENCHMARK`
- **Target Tasks**: SOH_BENCHMARK, RUL_BENCHMARK
- **Status**: `AVAILABLE`
- **Grouping Column**: `battery_id` (34 unique groups)
- **Processed Records**: 7,565
- **Limitations & Boundaries**:
  * Electrochemical lithium-ion cell degradation benchmark, not thermo-mechanical piston engine data.
  * Used strictly for testing SOH and RUL methodology and filtering algorithms.
  * Must never be treated as Rotax engine electrical system telemetry.

### NUST Bearing and Fault Dataset (`nust`)
- **Role**: `VIBRATION_FAULT_FEATURE_LEARNING`
- **Target Tasks**: VIBRATION_FAULT_FEATURE_LEARNING, ANOMALY_DETECTION_VALIDATION
- **Status**: `AVAILABLE`
- **Grouping Column**: `source_file` (134 unique groups)
- **Processed Records**: 390,263
- **Limitations & Boundaries**:
  * Multi-channel industrial test rig with variable temperature and humidity conditions.
  * Used for vibration feature representation learning across diverse environmental conditions.

### BASiC: Biomisa Arducopter Sensory Critique Dataset (`basic`)
- **Role**: `UNAVAILABLE`
- **Target Tasks**: UNAVAILABLE
- **Status**: `UNAVAILABLE`
- **Grouping Column**: `NONE` (0 unique groups)
- **Processed Records**: 0
- **Limitations & Boundaries**:
  * DATASET UNAVAILABLE: Requires manual download of DataFlash Text Logs (Zenodo DOI 10.5281/zenodo.8195068) into data/raw/basic/.
  * No processed parquet files currently exist in data/processed/basic/.
