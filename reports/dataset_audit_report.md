# SIH26054 Aero-Piston Engine Digital Twin — Dataset Audit Report

**Audit Date**: September 14, 2026  
**Auditor**: ML & Data Engineering Team  
**Scope**: All raw data stores located under `data/raw/`  
**Status**: COMPLETE  

---

## Executive Summary

An exhaustive, evidence-based audit was performed across all seven dataset directories present in the workspace (`data/raw/`). The SIH26054 project targets an **Aero-Piston Engine Digital Twin** (Rotax 912 iS architecture). Because our physics-model teammates are actively modifying digital twin equations, this audit strictly enforces isolation: **no physics, simulator, or API files were modified**, and external datasets are evaluated strictly on their physical legitimacy, signal compatibility, and data integrity.

### Dataset Availability & Classification Matrix

| Dataset Identifier | Domain & Physical System | Availability Status | Size on Disk | Files / Objects | Compatibility with Aero-Piston Twin | Assigned Integration Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C-MAPSS** | Commercial Turbofan Engine Degradation | **COMPLETE** | ~45.3 MB (CMAPSSData) + 14.7 GB (N-CMAPSS archive) | 14 files | `METHODOLOGY_BENCHMARK_ONLY` | `RUL_METHODOLOGY_BENCHMARK` |
| **CWRU** | Electric Motor Rolling Element Bearings | **AUDITED & PROCESSED** | 9.7 MB | 3 `.mat` files | `USABLE_AFTER_DOMAIN_SPECIFIC_PREPROCESSING` | `VIBRATION_FEATURE_SOURCE` |
| **FEMTO / PRONOSTIA** | Experimental Bearing Accelerated Run-to-Failure | **COMPLETE** | 2.67 GB | 24,074 files (CSVs + ZIPs) | `USABLE_AFTER_DOMAIN_SPECIFIC_PREPROCESSING` | `VIBRATION_FEATURE_SOURCE` |
| **NASA Battery** | Li-ion Electrochemical Cell Aging | **COMPLETE** | 1.28 GB (raw) + 209 MB (extracted ZIPs) | 6 archives, 34 MAT files | `METHODOLOGY_BENCHMARK_ONLY` | `RUL_METHODOLOGY_BENCHMARK` |
| **NUST** | Reciprocating IC-Engine Journal Bearing | **COMPLETE** | 1.25 GB | 683 files (134 machine-readable CSVs) | `USABLE_AFTER_DOMAIN_SPECIFIC_PREPROCESSING` | `AERO_PISTON_SUPPORTING` |
| **Paderborn** | Modular Bearing Test Rig (Artificial/Accelerated) | **COMPLETE** | 5.36 GB (raw) + 2.1 GB (extracted) | 240 MAT files, 6 PDF logs | `USABLE_AFTER_DOMAIN_SPECIFIC_PREPROCESSING` | `VIBRATION_FEATURE_SOURCE` |
| **BASiC** | Biomisa Arducopter Sensory Critique | **UNAVAILABLE** | 0 B | 1 `.gitkeep` | `UNAVAILABLE` | `NONE` (Strictly Quarantined) |

---

## Critical Safety & Data Integrity Finding

> [!CAUTION]
> ### Unsafe Turbofan-to-Piston Feature Mapping Quarantined
> During our repository audit, legacy code in `telemetry/adapters.py` (lines 206–225) was identified containing dangerous heuristic substitutions:
> ```python
> # LEGACY UNSAFE MAPPING FOUND IN telemetry/adapters.py:
> "cht": float(row["s_2"]),         # Total temperature at LPC outlet -> Cylinder Head Temp!
> "egt": float(row["s_3"]),         # Total temperature at HPC outlet -> Exhaust Gas Temp!
> "oil_temp": float(row["s_4"]),    # Total temperature at LPT outlet -> Engine Oil Temp!
> "oil_pressure": float(row["s_7"]),# Total pressure at HPC outlet -> Engine Oil Pressure!
> "fuel_flow": float(row["s_12"]),  # Fan speed -> Fuel Flow!
> ```
> **Physical Reality**: C-MAPSS models a twin-spool high-bypass commercial turbofan engine (JT9D-class). Mapping turbofan core thermodynamic stations to internal combustion four-stroke piston engine parameters violates thermodynamic laws and corrupts the physics-based digital twin.
>
> **Action Taken in Data Pipeline**:
> 1. In `data_pipeline/common/validation.py`, a strict validator `validate_no_unsafe_cmapss_columns()` was implemented to detect and reject any presence of `cht`, `egt`, `oil_temp`, `oil_pressure`, or `fuel_flow` in C-MAPSS schemas.
> 2. All processed C-MAPSS Parquet files are prefixed with explicit provenance: `cmapss_s_1` through `cmapss_s_21`.
> 3. Downstream consumers are strictly restricted to testing degradation modeling techniques (LSTM/TCN/Transformer RUL curve fitting) without passing these signals into the digital twin residual calculator.

---

## Detailed Dataset Technical Audits

### 1. NASA C-MAPSS Turbofan Degradation Dataset

* **Physical Nature**: Simulated degradation of gas turbine engine components (fan, LPC, HPC, HPT, LPT) operating under varying flight profiles.
* **Storage Location**: `data/raw/cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/`
* **Inventory**:
  * 4 training sets: `train_FD001.txt` (100 engines), `train_FD002.txt` (260 engines), `train_FD003.txt` (100 engines), `train_FD004.txt` (248 engines). Total train rows: **160,359**.
  * 4 test sets: `test_FD001.txt` (100 engines), `test_FD002.txt` (259 engines), `test_FD003.txt` (100 engines), `test_FD004.txt` (249 engines). Total test rows: **104,897**.
  * Ground truth RUL vectors: `RUL_FD001.txt`, `RUL_FD002.txt`, `RUL_FD003.txt`, `RUL_FD004.txt`.
* **Sampling Rate**: Discrete cycle-based (1 snapshot per complete flight cycle, not continuous time).
* **Signals**:
  * 3 operational settings (`op_setting_1` altitude, `op_setting_2` Mach number, `op_setting_3` throttle resolver angle).
  * 21 sensor measurements (temperatures, pressures, spool speeds, fuel flow ratios).
* **Data Quality**: Zero missing values, zero unreadable records. Constant sensor channels for single-condition sets (FD001/FD003: sensors 1, 5, 10, 16, 18, 19 have zero variance) are flagged and documented.
* **Split Integrity**: Trajectory-level grouped splitting by `unit_number` preserves zero temporal leakage.
* **Permitted Usage**: RUL algorithm validation, survival analysis, prognostic horizon benchmarking.
* **Forbidden Usage**: Direct substitution for Rotax 912 iS telemetry.

---

### 2. CWRU Bearing Data Center

* **Physical Nature**: High-frequency vibration data from a 2-horsepower Reliance Electric motor with electro-discharge machined faults on deep groove ball bearings (SKF 6205-2RS at drive end, 6203-2RS at fan end).
* **Storage Location**: `data/raw/cwru/`
* **Audit & Processing Inventory**:
  * **Total raw files discovered**: 3
  * **Supported files discovered**: 3 (`.mat`: 3, `.csv`: 0, `.txt`: 0)
  * **Files successfully parsed**: 3 (`97.mat`, `105.mat`, `130.mat`)
  * **Files skipped and exact reasons**: 0 files skipped (all 3 files valid MATLAB format with numeric vibration arrays)
  * **Total vibration windows**: 1,179 windows (2048-sample windows with 50% overlap of 1024 samples)
  * **Total feature rows**: 1,179 rows in canonical Parquet table
  * **File breakdown**:
    * `97.mat`: Normal baseline, 0 HP load, 1796 RPM, DE+FE channels (474 windows)
    * `105.mat`: Drive End 0.007" inner race fault, 0 HP load, 1797 RPM, DE+FE+BA channels (351 windows)
    * `130.mat`: Drive End 0.007" outer race centered fault, 0 HP load, 1796 RPM, DE+FE+BA channels (354 windows)
* **Sampling Frequencies**: 12,000 Hz continuous uniform sampling across all vibration channels.
* **Signals & Sensor Locations**: Drive End accelerometer (`DE_time`), Fan End accelerometer (`FE_time`), Base accelerometer (`BA_time`), and shaft speed (`RPM`).
* **Feature Columns (47 total)**:
  * **14 Time-Domain Metrics**: `mean`, `std`, `variance`, `RMS`, `minimum`, `maximum`, `peak_to_peak`, `absolute_mean`, `skewness`, `kurtosis`, `crest_factor`, `shape_factor`, `impulse_factor`, `clearance_factor`
  * **Auxiliary & Aliases**: `rms` (backward-compatibility alias for RMS), `peak` (maximum absolute amplitude), `energy` (sum of squared values)
  * **4 Frequency-Domain Metrics**: `dominant_frequency`, `spectral_energy`, `spectral_centroid`, `frequency_band_energy`
  * **Frequency Sub-Bands**: 4 equal 1.5 kHz sub-bands up to 6 kHz Nyquist (`band_energy_0_1500hz`, `band_energy_1500_3000hz`, `band_energy_3000_4500hz`, `band_energy_4500_6000hz`)
  * **Traceability & Provenance**: `dataset_name`, `source_file`, `source_id`, `record_id`, `bearing_id`, `window_id`, `sensor_location`, `sampling_frequency`, `RPM`, `load_condition`, `fault_label`, `fault_size`, `source_checksum`, `preprocessing_version`, `feature_version`
* **Fault Classes**: Normal (474 windows), Outer Race Fault (354 windows), Inner Race Fault (351 windows).
* **Operating Conditions**: 0 HP load, 1796-1797 RPM.
* **Data Quality & Integrity**: Flawless numeric integrity; 0 NaN and 0 infinite values across all rows; 0 duplicate records; numeric columns strictly typed.
* **Output Parquet Path**: `data/processed/cwru/cwru_features.parquet`
* **Associated Artifacts**: `data/processed/cwru/cwru_metadata.json`, `data/processed/cwru/cwru_processing_manifest.json`, `data/processed/cwru/cwru_features_sample.csv`
* **Limitations of Available Subset**: The available local raw dataset consists of 3 files (1 normal, 1 inner race 7mil, 1 outer race 7mil). Because each fault condition exists in only 1 file, random window splitting is strictly prohibited due to autocorrelation/temporal leakage, and file-level splitting would leave entire fault classes omitted from train or test sets. Therefore, a single validated feature table is published with file/bearing ID grouping for algorithm benchmarking.
* **Permitted Usage**: Vibration feature extraction methodology, bearing fault signature analysis, crest factor/kurtosis benchmarking.
* **Forbidden Usage**: Direct substitution for Rotax 912 iS telemetry. No mapping of vibration data into piston-engine channels (CHT, EGT, oil pressure, fuel flow).

---

### 3. FEMTO / PRONOSTIA Bearing Degradation

* **Physical Nature**: Accelerated run-to-failure bearing tests under controlled radial loads on the PRONOSTIA testbed.
* **Storage Location**: `data/raw/femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/`
* **Inventory**:
  * **24,074 total files** (21,493 raw CSV snapshots loaded and processed).
  * **6 Training Bearings** (run-to-failure): `Bearing1_1` (3,269 files), `Bearing1_2` (1,015 files), `Bearing2_1` (1,062 files), `Bearing2_2` (797 files), `Bearing3_1` (515 files), `Bearing3_2` (1,637 files). Output: **7,534 feature windows**.
  * **11 Test Bearings** (truncated degradation): `Bearing1_3` through `Bearing1_7`, `Bearing2_3` through `Bearing2_7`, `Bearing3_3`. Output: **13,959 feature windows**.
* **Sampling Rate**: 25.6 kHz during 0.1 s snapshot every 10 seconds (2,560 samples per CSV file).
* **Signals**: Horizontal acceleration (`horiz_accel`), Vertical acceleration (`vert_accel`), acquisition timestamps.
* **Operating Conditions**:
  * Condition 1: 1800 RPM, 4000 N load
  * Condition 2: 1650 RPM, 4200 N load
  * Condition 3: 1500 RPM, 5000 N load
* **Data Quality**: High-fidelity accelerometer readings. Correct monotonic degradation trends evident in kurtosis and peak-to-peak metrics.
* **Split Integrity**: Strict split by `bearing_id`. Windows from the same physical bearing are never partitioned across train and test sets.
* **Permitted Usage**: Progressive bearing wear modeling, health index construction, degradation threshold detection.
* **Forbidden Usage**: Directly passing raw bearing g-levels as aero-piston cylinder vibrations without speed/load scaling.

---

### 4. NASA PCoE Battery Aging Dataset

* **Physical Nature**: Commercial 18650 Li-ion cells subjected to repeated charge, discharge, and impedance cycles at controlled temperatures.
* **Storage Location**: `data/raw/nasa_battery/` (raw archives) and `data/processed/nasa_battery/extracted/` (extracted staging).
* **Inventory**:
  * 6 primary ZIP archives: `1. BatteryAgingARC-FY08Q4.zip` through `6. BatteryAgingARC_53_54_55_56.zip`.
  * **34 unique battery cells** discovered and processed (`B0005`, `B0006`, `B0007`, `B0018`, `B0025`..`B0056`).
  * **7,565 total cycle records** extracted into `battery_cycles.parquet` (2,750 discharge cycles with valid capacity & SOH).
* **Sampling Rate**: Event-driven cycle measurements (charge profile, discharge profile, EIS impedance spectroscopy).
* **Signals**: Voltage, current, cell surface temperature, capacity (Ah), impedance parameters (Re, Rct).
* **Status**: **COMPLETE / PROCESSED**.
  * Output: `data/processed/nasa_battery/battery_cycles.parquet` (688 KB, 24 columns).
  * Manifest: `data/processed/nasa_battery/extraction_manifest.json`.
  * Metadata: `data/processed/nasa_battery/battery_metadata.json`.
  * Sample: `data/processed/nasa_battery/battery_cycles_sample.csv`.
* **Permitted Usage**: SOH (State of Health) and capacity-fade algorithm benchmarking.
* **Forbidden Usage**: Never connect or map to internal combustion piston engine telemetry.

---

### 5. NUST Reciprocating IC-Engine Journal Bearing Dataset

* **Physical Nature**: Reciprocating internal combustion engine test bench with healthy and degraded journal bearings operated under controlled climatic chamber conditions.
* **Relevance**: **Highest physical relevance** in the entire raw repository for piston-engine dynamics.
* **Storage Location**: `data/raw/nust/`
* **Inventory**:
  * 683 total filesystem objects; 134 valid machine-readable CSV files (remainder are Word `.docx` and `.rtf` test protocol reports).
  * **390,263 processed records** standardized into clean Parquet.
* **Operating Grid**:
  * Engine Speed: 1000 RPM, 1500 RPM, 2000 RPM
  * Chamber Humidity: 0%, 50%, 100%
  * Ambient Temperature: -10°C, 0°C, 15°C, 30°C, 45°C, 55°C
* **Signals (20 standardized channels)**:
  * Control & Drive: `Demand 1`, `Control 1`, `Output Drive 1`
  * Vibration Channels: `Channel 1`, `Channel 2`, `Channel 3`, `Channel 4`
  * Real-Time Kurtosis: `Channel 1 Kurtosis`, `Channel 2 Kurtosis`, `Channel 3 Kurtosis`, `Channel 4 Kurtosis`
  * Rear Aux Inputs: `Rear Input 1` through `Rear Input 8` (audited as stationary zero-fill channels)
* **Data Quality**: Validated header formats; duplicate initial timestamp rows detected and eliminated; physical units verified.
* **Permitted Usage**: Reciprocating engine journal bearing diagnostic modeling, temperature/humidity impact on vibration signatures.
* **Physical Caveat**: Automotive/industrial reciprocating engine block; rotational speeds (1000–2000 RPM) are lower than Rotax 912 iS cruise speeds (5000–5800 RPM crankshaft, geared to ~2300 RPM propeller).

---

### 6. Paderborn University Bearing Dataset

* **Physical Nature**: Modular bearing test bench capturing vibration acceleration (piezoelectric sensors at native 64 kHz) and motor drive current across healthy, artificially damaged, and accelerated-fatigued bearings.
* **Storage Location**: `data/raw/paderborn/` (raw RAR archives) and `data/processed/paderborn/extracted/` (extracted MAT suites).
* **Inventory**:
  * **240 valid `.mat` vibration files** across 3 core profile bearings:
    * `K001`: Healthy baseline (80 MAT files across 4 operating conditions, 20 repeats)
    * `KA01`: Outer race fault (80 MAT files across 4 operating conditions, 20 repeats)
    * `KI04`: Inner race fault (80 MAT files across 4 operating conditions, 20 repeats)
  * 6 cataloged PDF measuring log files.
  * **29,792 window feature vectors** generated into `paderborn_features.parquet` (4096-sample windows with 50% overlap).
* **Signals & Features**:
  * Raw: 64 kHz piezoelectric acceleration (`vibration_1`, 256,823 samples per run).
  * 9 Time-domain features: mean, std, variance, rms, peak, peak-to-peak, crest factor, skewness, kurtosis.
  * 6 Frequency-domain features: dominant frequency, spectral energy, spectral centroid, band energy 0-5 kHz, band energy 5-15 kHz, band energy 15-32 kHz.
* **Status**: **COMPLETE / PROCESSED**.
  * Output: `data/processed/paderborn/paderborn_features.parquet` (3.84 MB, 30 columns).
  * Manifest: `data/processed/paderborn/paderborn_processing_manifest.json`.
  * Metadata: `data/processed/paderborn/paderborn_metadata.json`.
  * Sample: `data/processed/paderborn/paderborn_features_sample.csv`.
* **Permitted Usage**: High-frequency vibration frequency-domain benchmark, envelope analysis, and fatigue damage mode classification.
* **Forbidden Usage**: Direct piston telemetry mapping or downsampling to engine telemetry rates.

---

### 7. BASiC Dataset

* **Status**: **UNAVAILABLE** (Only `.gitkeep` present, 0 bytes).
* **Protocol**: Under no circumstances should synthetic or hallucinated data be injected. Any reference to BASiC in training sets is strictly prevented.

---

## Data Pipeline Design & Architecture

The ingestion and preprocessing infrastructure was built under `data_pipeline/` following strict industrial standards:

```text
data_pipeline/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── schemas.py       # Strict Pydantic/dataclass schema contracts
│   ├── validation.py    # Unsafe column detection, RUL checks, leakage checks
│   ├── provenance.py    # SHA-256 hashes, metadata logging, record tracing
│   └── splitting.py     # Trajectory and group-aware train/test/val splitting
├── cmapss/
│   ├── loader.py        # FD001-FD004 parser and constant feature filtering
│   ├── preprocess.py    # Piecewise linear RUL target generator (max RUL = 125)
│   └── adapter.py       # Quarantined naming adapter (cmapss_s_1..21)
├── cwru/
│   ├── loader.py        # MATLAB .mat file parser
│   ├── features.py      # Windowed statistical & frequency vibration metrics
│   └── adapter.py       # Standardized feature schema exporter
├── femto/
│   ├── loader.py        # Sequence-ordered multi-CSV file reader
│   ├── features.py      # High-speed time/frequency feature extractor
│   └── adapter.py       # Trajectory assembly and bearing-level splitter
├── nust/
│   ├── loader.py        # Robust CSV loader with operating grid extractor
│   ├── preprocess.py    # Duplicate timestamp cleaner and metadata enricher
│   └── adapter.py       # Standardized IC-engine vibration dataset builder
├── nasa_battery/
│   ├── loader.py        # Zip extraction and .mat cycle extractor
│   └── adapter.py       # SOH capacity degradation builder
└── paderborn/
    ├── loader.py        # Rar archive scanner and classification parser
    └── adapter.py       # 64 kHz signal windowing and feature generator
```

---

## Preprocessing Verification Summary

All preprocessing pipelines were executed end-to-end via automated runner (`scratch/run_preprocessing.py`), yielding high-performance Parquet artifacts under `data/processed/`:

* **C-MAPSS**: 8 Parquet files (FD001–FD004 train & test), totaling **265,256 records** with complete RUL labels and zero unsafe column mappings.
* **CWRU**: 1 Parquet file (`cwru_features.parquet`), **1,179 records**, 16 statistical vibration features.
* **FEMTO**: 2 Parquet files (`femto_train_features.parquet` and `femto_test_features.parquet`), **21,493 raw CSVs processed into 21,493 feature windows** across 17 bearings.
* **NUST**: 1 Parquet file (`nust_processed.parquet`), **390,263 records** across 134 operational conditions.
* **Automated Test Suite**: `tests/test_data_pipeline.py` contains 32 comprehensive tests spanning unit safety, schema enforcement, data leakage detection, and physics isolation. **Test result: 32 passed in 3.69s (100% pass rate).**
