# SIH26054 ML Data Layer & Split Validation Report

**Generated:** 2026-09-14T19:46:57.200299  
**Leakage-Free Guarantee:** PASSED (Zero Leakage)  
**Evaluated Datasets:** 6  

## 1. Group-Level Split Audit Table

| Dataset / Partition | Grouping Column | Total Groups | Train Groups | Val Groups | Test Groups | Train Rows | Val Rows | Test Rows | Leakage Detected |
|---|---|---|---|---|---|---|---|---|---|
| `cmapss_FD001_train` | `unit_number` | 100 | 70 | 15 | 15 | 14,407 | 3,160 | 3,064 | **False** |
| `femto_train_features` | `bearing_id` | 6 | 4 | 0 | 2 | 3,860 | 0 | 3,674 | **False** |
| `cwru_features` | `source_file` | 3 | 2 | 0 | 1 | 828 | 0 | 351 | **False** |
| `paderborn_features` | `source_file` | 240 | 168 | 36 | 36 | 20,857 | 4,467 | 4,468 | **False** |
| `nasa_battery_cycles` | `battery_id` | 34 | 24 | 5 | 5 | 4,307 | 1,726 | 1,532 | **False** |
| `nust_processed` | `source_file` | 134 | 94 | 20 | 20 | 252,612 | 65,170 | 72,481 | **False** |

---

## 2. Dataset-by-Dataset Split Details & Group Assignments

### cmapss_FD001_train
- **Grouping Column**: `unit_number`
- **Total Rows**: 20,631
- **Train Groups (70)**: `[1, 2, 3, 4, 5, 6, 8, 10, 11, 16]`...
- **Val Groups (15)**: `[7, 12, 31, 35, 45, 50, 55, 59, 68, 75]`...
- **Test Groups (15)**: `[9, 13, 14, 15, 20, 23, 36, 37, 42, 48]`...
- **Notes**: Standard leakage-safe group partition.

### femto_train_features
- **Grouping Column**: `bearing_id`
- **Total Rows**: 7,534
- **Train Groups (4)**: `['Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']`
- **Val Groups (0)**: `[]`
- **Test Groups (2)**: `['Bearing1_1', 'Bearing1_2']`
- **Notes**: Standard leakage-safe group partition.

### cwru_features
- **Grouping Column**: `source_file`
- **Total Rows**: 1,179
- **Train Groups (2)**: `['97.mat', '130.mat']`
- **Val Groups (0)**: `[]`
- **Test Groups (1)**: `['105.mat']`
- **Notes**: CWRU controlled demonstration split (2 train files, 1 test file; 0 val).
- **Train Class Distribution**: `{'normal': 474, 'outer_race': 354}`
- **Test Class Distribution**: `{'inner_race': 351}`

### paderborn_features
- **Grouping Column**: `source_file`
- **Total Rows**: 29,792
- **Train Groups (168)**: `['N09_M07_F10_K001_1.mat', 'N09_M07_F10_K001_11.mat', 'N09_M07_F10_K001_12.mat', 'N09_M07_F10_K001_13.mat', 'N09_M07_F10_K001_14.mat', 'N09_M07_F10_K001_16.mat', 'N09_M07_F10_K001_17.mat', 'N09_M07_F10_K001_18.mat', 'N09_M07_F10_K001_19.mat', 'N09_M07_F10_K001_20.mat']`...
- **Val Groups (36)**: `['N09_M07_F10_K001_10.mat', 'N09_M07_F10_K001_2.mat', 'N09_M07_F10_K001_5.mat', 'N09_M07_F10_K001_7.mat', 'N09_M07_F10_KA01_12.mat', 'N09_M07_F10_KA01_2.mat', 'N09_M07_F10_KI04_1.mat', 'N09_M07_F10_KI04_11.mat', 'N09_M07_F10_KI04_12.mat', 'N09_M07_F10_KI04_3.mat']`...
- **Test Groups (36)**: `['N09_M07_F10_K001_15.mat', 'N09_M07_F10_K001_4.mat', 'N09_M07_F10_K001_6.mat', 'N09_M07_F10_K001_9.mat', 'N09_M07_F10_KA01_11.mat', 'N09_M07_F10_KA01_19.mat', 'N09_M07_F10_KA01_4.mat', 'N09_M07_F10_KA01_5.mat', 'N09_M07_F10_KA01_6.mat', 'N09_M07_F10_KI04_10.mat']`...
- **Notes**: Standard leakage-safe group partition.
- **Train Class Distribution**: `{'outer_race_fault': 7703, 'inner_race_fault': 6825, 'healthy': 6329}`
- **Test Class Distribution**: `{'healthy': 1612, 'inner_race_fault': 1492, 'outer_race_fault': 1364}`

### nasa_battery_cycles
- **Grouping Column**: `battery_id`
- **Total Rows**: 7,565
- **Train Groups (24)**: `['B0005', 'B0018', 'B0026', 'B0027', 'B0028', 'B0030', 'B0031', 'B0033', 'B0038', 'B0039']`...
- **Val Groups (5)**: `['B0007', 'B0032', 'B0036', 'B0044', 'B0055']`
- **Test Groups (5)**: `['B0006', 'B0025', 'B0029', 'B0034', 'B0054']`
- **Notes**: Standard leakage-safe group partition.
- **Train Class Distribution**: `{'charge': 1653, 'discharge': 1639, 'impedance': 1015}`
- **Test Class Distribution**: `{'charge': 540, 'discharge': 536, 'impedance': 456}`

### nust_processed
- **Grouping Column**: `source_file`
- **Total Rows**: 390,263
- **Train Groups (94)**: `['1ST AT -10 c 2022May27-0325-0011.csv', '1st at -10 2022Jun04-2239-0005.csv', '1st at -10c 2022May26-2336-0006.csv', '1st at -10c 2022May26-2336-0014.csv', '1st at -10c 2022May27-0007-0024.csv', '1st at -10c 2022May27-0007-0030.csv', '1st at -10c 2022May27-0109-0049.csv', '1st at -10c 2022May27-0219-0013.csv', '1st at -10c 2022May27-0338-0059.csv', '1st at -10c 2022May27-0421-0053.csv']`...
- **Val Groups (20)**: `['1st at -10c 2022May27-0219-0014.csv', '1st at -10c 2022May27-2221-0031.csv', '1st at 0c 2022May27-2254-0013.csv', '2nd at -10c 2022May27-2254-0015.csv', '2nd at 0c 2022May27-0141-0043.csv', '2nd at 0c 2022May27-0219-0019.csv', '2nd at 0c 2022May27-0338-0068.csv', '3rd at 15c 2022May27-0007-0044.csv', '3rd at 15c 2022May27-0219-0024.csv', '3th at 15c 2022May26-2304-0037.csv']`...
- **Test Groups (20)**: `['1st at -10c 2022May27-0109-0039.csv', '1st at -10c 2022May27-0338-0064.csv', '1st at -10c 2022May27-0528-0021.csv', '2ND AT 0C 2022Jun05-2156-0002.csv', '2nd  at 0c 2022May27-0325-0018.csv', '2nd at 0c 2022May27-0421-0045.csv', '2nd at 0c 2022May27-0421-0062.csv', '3rd at 15c 2022Jun04-2302-0006.csv', '3rd at 15c 2022May27-0141-0021.csv', '3rd at 15c 2022May27-0219-0058.csv']`...
- **Notes**: Standard leakage-safe group partition.
- **Train Class Distribution**: `{'faulty': 143513, 'healthy': 109099}`
- **Test Class Distribution**: `{'healthy': 41654, 'faulty': 30827}`

---

## 3. Data Integrity & Domain Protection Audit

- **Parquet Format Enforcement**: 100% of canonical inputs are Parquet files. Raw `.zip`, `.mat`, `.txt`, `.csv` loading attempts are blocked with `InvalidFileFormatError`.
- **Channel Protection**: Zero collision between benchmark feature sets and core Rotax 914 channels (CHT, EGT, MAP, RPM, Oil, Coolant).
- **Physics Features**: Reserved physics features (`physics_residuals`, `normalized_residuals`, `subsystem_health_scores`) remain unpopulated/unfabricated in Phase 1 as required.
- **Core Domain Boundaries**: Verified zero modifications to `physics/`, `simulator/`, `telemetry/`, `digital_twin/`, `health_index/`, `anomaly_detection/`, `fault_diagnosis/`, `frontend/`, `backend/`.