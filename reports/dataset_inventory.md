# SIH26054 — Dataset Inventory Report

**Generated**: 2026-09-14T11:40:39.619403

**Root**: `data/raw/`

## Summary

| Dataset | Status | Files | Dirs | Size | Extensions | Archives |
|---------|--------|-------|------|------|------------|----------|
| cmapss | PRESENT | 16 | 3 | 14.72 GB | (1), .zip(1), .pdf(1), .txt(13) | 1 |
| cwru | AUDITED_AND_PROCESSED | 3 | 0 | 9.3 MB | .mat(3) | 0 |
| femto | PRESENT | 24074 | 24 | 2.49 GB | (1), .zip(4), .csv(24069) | 4 |
| nasa_battery | PRESENT | 3 | 0 | 1.19 GB | (1), .zip(2) | 2 |
| nust | PRESENT | 683 | 367 | 1.16 GB | (1), .docx(202), .rtf(346), .csv(134) | 0 |
| paderborn | PRESENT | 33 | 0 | 4.99 GB | (1), .rar(32) | 32 |
| basic | PRESENT | 1 | 0 | 0.0 MB | (1) | 0 |

## CMAPSS

- **Path**: `data/raw/cmapss`
- **Status**: PRESENT
- **Total files**: 16
- **Total directories**: 3
- **Total size**: 14.72 GB (15,805,793,295 bytes)
- **Extensions**: {'': 1, '.zip': 1, '.pdf': 1, '.txt': 13}
- **Archives**: 1
  - `cmapss/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip` (15030.3 MB)
- **Sample files**:
  - `cmapss/.gitkeep`
  - `cmapss/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/Damage Propagation Modeling.pdf`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/readme.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/RUL_FD001.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/RUL_FD002.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/RUL_FD003.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/RUL_FD004.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/test_FD001.txt`
  - `cmapss/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/test_FD002.txt`

## CWRU

- **Path**: data/raw/cwru
- **Status**: AUDITED_AND_PROCESSED (All available files audited, parsed, and processed)
- **Total raw files discovered**: 3
- **Supported files discovered**: 3 (.mat: 3)
- **Files successfully parsed**: 3 (97.mat, 105.mat, 130.mat)
- **Files skipped**: 0 (no corrupt or unsupported files)
- **Total vibration windows**: 1,179
- **Total feature rows**: 1,179
- **Feature columns**: 47 (14 time-domain + rms compatibility alias + 4 frequency-domain with subbands + condition & provenance)
- **Sampling frequencies**: 12,000 Hz
- **Fault classes**: normal (474 windows), outer_race (354 windows), inner_race (351 windows)
- **Operating conditions**: 0 HP load, 1796–1797 RPM
- **Output Parquet path**: data/processed/cwru/cwru_features.parquet
- **Artifacts**: cwru_features.parquet, cwru_metadata.json, cwru_processing_manifest.json, cwru_features_sample.csv
- **Limitations of available subset**: The local raw repository contains 3 files representing distinct fault conditions (97.mat: Normal, 105.mat: Inner race 7mil, 130.mat: Outer race 7mil). Random window splitting is prohibited due to temporal leakage; splitting by file would cause zero-shot evaluation or missing classes. A single canonical feature table is generated with documented source file / bearing ID grouping.

## FEMTO

- **Path**: `data/raw/femto`
- **Status**: PRESENT
- **Total files**: 24074
- **Total directories**: 24
- **Total size**: 2.49 GB (2,668,379,631 bytes)
- **Extensions**: {'': 1, '.zip': 4, '.csv': 24069}
- **Archives**: 4
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Training_set.zip` (133.9 MB)
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Validation_Set.zip` (302.1 MB)
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Training_set.zip` (133.9 MB)
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Validation_Set.zip` (302.1 MB)
- **Sample files**:
  - `femto/.gitkeep`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Training_set.zip`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Validation_Set.zip`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Training_set.zip`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Validation_Set.zip`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set/Bearing1_3/acc_00001.csv`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set/Bearing1_3/acc_00002.csv`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set/Bearing1_3/acc_00003.csv`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set/Bearing1_3/acc_00004.csv`
  - `femto/10.+FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set/Bearing1_3/acc_00005.csv`

## NASA_BATTERY

- **Path**: `data/raw/nasa_battery`
- **Status**: PRESENT
- **Total files**: 3
- **Total directories**: 0
- **Total size**: 1.19 GB (1,275,529,765 bytes)
- **Extensions**: {'': 1, '.zip': 2}
- **Archives**: 2
  - `nasa_battery/11.+Randomized+Battery+Usage+Data+Set.zip` (1016.4 MB)
  - `nasa_battery/5.+Battery+Data+Set.zip` (200.0 MB)
- **Sample files**:
  - `nasa_battery/.gitkeep`
  - `nasa_battery/11.+Randomized+Battery+Usage+Data+Set.zip`
  - `nasa_battery/5.+Battery+Data+Set.zip`

## NUST

- **Path**: `data/raw/nust`
- **Status**: PRESENT
- **Total files**: 683
- **Total directories**: 367
- **Total size**: 1.16 GB (1,250,791,830 bytes)
- **Extensions**: {'': 1, '.docx': 202, '.rtf': 346, '.csv': 134}
- **Sample files**:
  - `nust/.gitkeep`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/-10 deg Celsius/3th Auto engine at -10C2022May22-2335-0019.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/-15 deg Celsius/2nd Auto engine at -15C 2022May22-2335-0021.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/-20 deg Celsius/1st Auto engine at -20C 2022May22-2335-0024.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/-5 deg Celsius/4th Auto engine at -5C2022May22-2335-0016.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/0 deg Celsius/5th Auto engine at 0C 2022May22-2335-0012.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/10 deg Celsius/7th Auto engine at+10C 2022May22-2335-0007.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/15 deg Celsius/8th Auto engine at+15C 2022May22-2335-0005.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/20 deg Celsius/9th Auto engine at+20C 2022May22-2312-0005.docx`
  - `nust/Engine Journal Bearings Dataset/Additional Datasets/30% Humidity/Faulty Bearings/1000 RPM/25 deg Celsius/10th Auto engine at+25C 2022May22-2253-0019.docx`

## PADERBORN

- **Path**: `data/raw/paderborn`
- **Status**: PRESENT
- **Total files**: 33
- **Total directories**: 0
- **Total size**: 4.99 GB (5,357,708,539 bytes)
- **Extensions**: {'': 1, '.rar': 32}
- **Archives**: 32
  - `paderborn/K001.rar` (165.8 MB)
  - `paderborn/K002.rar` (154.5 MB)
  - `paderborn/K003.rar` (165.2 MB)
  - `paderborn/K004.rar` (155.8 MB)
  - `paderborn/K005.rar` (159.6 MB)
  - `paderborn/K006.rar` (167.8 MB)
  - `paderborn/KA01.rar` (158.9 MB)
  - `paderborn/KA03.rar` (168.0 MB)
  - `paderborn/KA04.rar` (172.5 MB)
  - `paderborn/KA05.rar` (156.1 MB)
  - `paderborn/KA06.rar` (156.5 MB)
  - `paderborn/KA07.rar` (152.9 MB)
  - `paderborn/KA08.rar` (161.4 MB)
  - `paderborn/KA09.rar` (163.5 MB)
  - `paderborn/KA15.rar` (156.7 MB)
  - `paderborn/KA16.rar` (159.4 MB)
  - `paderborn/KA22.rar` (151.7 MB)
  - `paderborn/KA30.rar` (156.0 MB)
  - `paderborn/KB23.rar` (163.0 MB)
  - `paderborn/KB24.rar` (178.0 MB)
  - `paderborn/KB27.rar` (154.9 MB)
  - `paderborn/KI01.rar` (167.1 MB)
  - `paderborn/KI03.rar` (151.8 MB)
  - `paderborn/KI04.rar` (164.6 MB)
  - `paderborn/KI05.rar` (152.9 MB)
  - `paderborn/KI07.rar` (153.2 MB)
  - `paderborn/KI08.rar` (156.2 MB)
  - `paderborn/KI14.rar` (156.0 MB)
  - `paderborn/KI16.rar` (158.6 MB)
  - `paderborn/KI17.rar` (159.6 MB)
  - `paderborn/KI18.rar` (153.8 MB)
  - `paderborn/KI21.rar` (157.2 MB)
- **Sample files**:
  - `paderborn/.gitkeep`
  - `paderborn/K001.rar`
  - `paderborn/K002.rar`
  - `paderborn/K003.rar`
  - `paderborn/K004.rar`
  - `paderborn/K005.rar`
  - `paderborn/K006.rar`
  - `paderborn/KA01.rar`
  - `paderborn/KA03.rar`
  - `paderborn/KA04.rar`

## BASIC

- **Path**: `data/raw/basic`
- **Status**: PRESENT
- **Total files**: 1
- **Total directories**: 0
- **Total size**: 0.0 MB (0 bytes)
- **Extensions**: {'': 1}
- **Sample files**:
  - `basic/.gitkeep`
