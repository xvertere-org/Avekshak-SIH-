# SIH26054 Complete Project Audit

**Audit date:** 14 September 2026  
**Workspace audited:** `D:\NIRVANAA-SIH-SUBMISSION`  
**Audit method:** Read-only inspection of source, configuration, data and evidence; direct runtime check of the TimesFM adapter; pytest collection. No architecture, training, data, or model changes were made for this audit.

## 1. Executive summary

SIH26054 (currently branded **Avekshak** in the dashboard) is an executable, synthetic, physics-informed aero-piston-engine PHM demonstration. Its live path is a Python/Streamlit application that simulates a mission, removes simulator fault labels before inference, computes Digital Twin residuals, runs anomaly detection and a bootstrapped XGBoost diagnosis model, computes health/degradation/forecast/RUL/explainability outputs, and sends one `DashboardStatePayload` to the dashboard.

The core synthetic pipeline is implemented and connected. The dashboard runs locally through `dashboard/app.py`. The production CLI entry point is `main.py`. Mission replay, reporting, and what-if simulation are implemented in `phase14/` and exposed in the dashboard.

Important boundaries:

- This is a reduced-order grey-box simulator referenced to a Rotax 914 UL/F configuration. It is **not** a complete OEM-calibrated or experimentally validated Rotax digital twin.
- The live pipeline uses synthetic simulator telemetry. Public datasets are present locally, but the canonical Phase 13 runtime does not load them.
- Isolation Forest and XGBoost are fitted deterministically at orchestrator bootstrap from synthetic simulator data; no persisted trained classifier artifact was found.
- In this environment, `TimesFM3ModelAdapter` directly reported `LOADED_PRETRAINED`, CUDA, and a usable forecaster. The repository requirements do not declare `timesfm`/`timesfm3` or a Hugging Face client dependency, so reproducibility from `requirements.txt` alone is not established.
- Dataset manifests are stale: `data/acquisition_manifest.json` says several sources have zero local files, while the local `data/raw/` and `data/processed/` folders contain those datasets.
- Hugging Face Space/Bucket deployment is a plan, not a repository-proven implementation. No Space metadata, Dockerfile, or deployment configuration was found.

## 2. Repository structure actually present

| Area | Purpose and principal files | Status |
|---|---|---|
| Entry points | `main.py:28` CLI pipeline; `dashboard/app.py:156` Streamlit app | Implemented |
| Simulator | `simulator/engine_simulator.py:35`; `simulator/subsystems/`; `simulator/fault_interface.py` | Implemented, reduced-order |
| Telemetry | `telemetry/ingestion.py`, `preprocessing.py`, `quality.py`, `features.py`, `pipeline.py` | Implemented |
| Digital Twin | `digital_twin/twin_model.py:69`, `residuals.py:288`, `quality.py`, `synchronizer.py` | Implemented |
| Detection/diagnosis | `anomaly_detection/`; `fault_diagnosis/pipeline.py:29`, `classifier.py:60` | Implemented |
| Health/prognostics | `health_index/`; `forecasting/`; `prognostics/` | Implemented with gating/fallbacks |
| Explainability | `explainability/pipeline.py:34`, `shap_explainer.py`, `physics_evidence.py`, `fusion.py` | Implemented |
| Orchestration/advisory | `orchestrator/pipeline.py:56`, `adapter.py`, `advisor.py`, `bootstrap.py:29` | Implemented |
| Presentation/Phase 14 | `dashboard/`; `phase14/replay.py`, `reporting.py`, `what_if.py` | Implemented |
| External-data tooling | `data_pipeline/{cmapss,cwru,femto,nasa_battery,nust,paderborn}/`; `ml/` | Present; not the canonical live path |
| Docs/evidence | `README.md`, `docs/`, `evidence/`, `reports/` | Extensive, but not fully current |

The working tree was dirty at audit start. Uncommitted work includes changed detector/dashboard/test files plus an untracked `ml/tasks/` workstream, generation scripts, test files, and ML reports. Those files are included below as **present in the workspace**, but are not treated as a committed release baseline.

## 3. Implemented versus planned feature matrix

| Capability | Evidence | Runtime connection | Assessment |
|---|---|---|---|
| Mission scenario simulation | `SystemPipelineOrchestrator.run_simulation()` at `orchestrator/pipeline.py:764`; sidebar controls in `dashboard/app.py` | Yes | Implemented |
| Thermal, lubrication, fuel, vibration and sensor faults | `simulator/fault_interface.py`; subsystem files | Yes | Implemented synthetic fault mechanisms |
| Canonical seven-channel telemetry | `telemetry/schema.py`; dashboard channel grid | Yes | Implemented |
| Twin expected state and residuals | `PipelineHandoffAdapter.step_digital_twin_streaming()` called at `orchestrator/pipeline.py:276` | Yes | Implemented |
| Threshold/EWMA/persistence/Isolation Forest detection | `anomaly_detection/detectors.py`; `pipeline.py:34` | Yes | Implemented |
| Six-class XGBoost diagnosis | `fault_diagnosis/classifier.py:60`; bootstrap at `orchestrator/bootstrap.py:36` | Yes | Implemented, synthetic-bootstrap fitted |
| Health index/degradation tracking | `health_index/pipeline.py:38` | Yes | Implemented |
| TimesFM-3 forecasting | `forecasting/models.py:79`, `pipeline.py:24` | Yes when model is accessible; EWMA fallback otherwise | Implemented, external-weight dependent |
| RUL | `prognostics/pipeline.py:20`, Theil-Sen/Monte Carlo helpers | Yes | Implemented, withheld until adequate history |
| SHAP/physics/temporal explanation | `explainability/pipeline.py:34` | Yes | Implemented; SHAP depends on active classifier |
| Replay/report/what-if | `phase14/replay.py:19`, `reporting.py:19`, `what_if.py:24` | Yes | Implemented |
| Real aero-piston training or flight validation | No such data/model path found | No | Not established |
| Persisted model registry/checkpoints | No XGBoost/Isolation Forest artifacts found | No | Not implemented for core models |
| Hugging Face Space/Bucket deployment | No deployment files found | No | Planned only |

## 4. End-to-end runtime flow

1. `main.py:28` creates a scenario and invokes `SystemPipelineOrchestrator`; `dashboard/app.py:44` caches an orchestrator for Streamlit.
2. `orchestrator/pipeline.py:764` simulates telemetry using `EngineSimulator` and feeds every step to `step()`.
3. `step()` validates timestamp/session boundaries (`orchestrator/pipeline.py:157-270`) and strips injected fault metadata using `PipelineHandoffAdapter.sanitize_telemetry_for_inference()` (`orchestrator/pipeline.py:273`).
4. The adapter feeds the Digital Twin and obtains expected telemetry plus raw/normalised residuals (`orchestrator/pipeline.py:276-283`).
5. `HybridAnomalyDetector` receives the residual frame (`orchestrator/pipeline.py:286-292`).
6. `FaultDiagnosisPipeline` consumes the residual features and anomaly context (`orchestrator/pipeline.py:295-305`).
7. `HealthIndexPipeline` computes health/degradation state (`orchestrator/pipeline.py:308-315`).
8. `ForecastingPipeline` buffers causal telemetry and uses TimesFM-3 if available or an explicitly labelled baseline (`orchestrator/pipeline.py:318-333`).
9. `RULPipeline` uses health and forecast outputs, with warm-up/quality gates (`orchestrator/pipeline.py:336-341`; `prognostics/pipeline.py:118-134`).
10. `ExplainabilityPipeline` fuses SHAP, physics, temporal and RUL evidence (`orchestrator/pipeline.py:344-358`); `OperatorActionAdvisor` builds decision-support text (`:361-374`).
11. A typed `DashboardStatePayload` is assembled (`:377-557`), adapted by `dashboard/services/adapter.py:49`, and rendered by one of eight dashboard pages.

The dashboard has Overview, Live Telemetry, Diagnostics, Prognostics, System Status, Mission Replay, Mission Report, and What-If Comparison navigation in `dashboard/app.py`. Its “demo scenarios” use `dashboard/services/demo_provider.py`; the app separately runs a mapped authoritative simulation to enable replay/reporting.

## 5. ML and analytics inventory

| Model/component | Training/evaluation code | Data and schema | Inference status and limitations |
|---|---|---|---|
| Residual threshold, EWMA, CUSUM-style persistence | `anomaly_detection/detectors.py` (`ResidualThresholdDetector`, `EWMADetector`, `PersistenceGate`) | Twin residuals | Deterministic rules; no fitted artifact |
| Isolation Forest | `anomaly_detection/detectors.py:321`; bootstrap `orchestrator/bootstrap.py:36-127` | Synthetic healthy residual feature population | Fitted at application bootstrap, not saved; `fast_mode` uses 50 estimators (`bootstrap.py:78`) |
| XGBoost six-class diagnosis | `fault_diagnosis/classifier.py:28-137`; synthetic generator `fault_diagnosis/dataset.py:133` | 16 engineered residual/telemetry features from `fault_diagnosis/features.py`; canonical labels are generated from simulator `FaultType` | Fitted at bootstrap with 30 estimators in current orchestrator config (`orchestrator/schema.py:314`), rather than persisted. Classifier default config separately specifies 200 estimators (`classifier.py:35`); report/model configuration must state which was used. |
| Vibration fault work | CWRU/Paderborn feature pipelines plus uncommitted `ml/tasks/classification.py` | Component-level bearing data | Not wired into Phase 13 diagnosis; not a validated aero-piston vibration model |
| Health index | `health_index/pipeline.py`, `calculator.py`, `degradation.py` | Residuals, diagnosis status, sensor-isolation information | Rule/aggregation and causal smoothing, not a separately trained ML artifact |
| Grey-box residual correction | Uncommitted `ml/tasks/residual_correction.py`; `reports/phase3_physics_ml_residual_report.md` | Simulated and benchmark records | Present but not imported by the canonical orchestrator; report calls it a controlled technical demonstration |
| TimesFM-3 | `forecasting/models.py:79-190`, `forecasting/pipeline.py` | Seven canonical channels; context 32, horizon 16 (`forecasting/schema.py:76-80`) | Direct audit runtime check: `LOADED_PRETRAINED`, CUDA, available. External gated checkpoint `google/timesfm-3.0-pytorch`; cannot be reproduced from requirements alone. |
| Forecast baselines | `forecasting/baselines.py` | Same causal buffer | Persistence and Causal EWMA fallback are implemented and labelled. |
| RUL | `prognostics/pipeline.py`, `trajectory.py`, `uncertainty.py`, `threshold.py` | Health history, forecast and EOL rules | Theil-Sen trend plus Monte Carlo/weakest-link logic; not a trained RUL checkpoint. Numerical RUL is withheld during insufficient history. |
| SHAP explainability | `explainability/shap_explainer.py`; `explainer.py` | Active XGBoost feature vector | Runtime attribution for an active classifier; not independent proof of physical causation. |

No `.joblib`, `.pkl`, `.pt`, `.pth`, `.onnx`, `.safetensors`, XGBoost JSON model, or equivalent core-model artifact was found. JSON files under `configs/`, `data/`, `evidence/`, and `reports/` are configuration/evidence rather than trained core classifiers.

## 6. Dataset and preprocessing inventory

| Dataset | Local evidence | Processed form | Permitted conclusion |
|---|---|---|---|
| C-MAPSS | `data/raw/cmapss/`: 16 files, ~15.07 GB; raw includes more than the nominal C-MAPSS archive | FD001–FD004 train/test parquet: 20,631–61,249 train rows and 13,096–41,214 test rows; 25 or 31 columns | Simulated turbofan degradation methodology only; do not map it to piston CHT/EGT/oil channels. |
| CWRU | 3 MAT files, 9.29 MB | `cwru_features.parquet`: 1,179 x 47 | Bearing vibration feature benchmark only. Three-file/class coverage is insufficient for broad deployed classification claims. |
| FEMTO/PRONOSTIA | 24,074 raw files, ~2.54 GB | train 7,534 x 29; test 13,959 x 29 | Bearing run-to-failure/RUL-methodology benchmark, not engine RUL validation. |
| NASA battery | 3 raw archives, ~1.22 GB; many extracted MATs | `battery_cycles.parquet`: 7,565 x 25 | Electrochemical SOH/RUL methodology only. |
| NUST IC-engine bearings | 682 raw files, ~1.19 GB | `nust_processed.parquet`: 390,263 x 30 | Reciprocating IC-engine bearing evidence; not an aero-piston flight engine dataset. |
| Paderborn | 33 raw archives, ~5.11 GB | `paderborn_features.parquet`: 29,792 x 30 | Electric-motor bearing rig, component-level only. |
| BASiC UAV | one raw item listed | no processed dataset found | SITL flight/sensor context; not aero-piston propulsion validation. |

The source/provenance statements are in `data/dataset_manifest.json`. `data/acquisition_manifest.json` is outdated relative to the actual local directories and should not be used as current availability evidence. A historical `reports/dataset_audit_report.json` also contains an unsafe C-MAPSS-to-piston channel mapping description; the live Phase 13 pipeline does not call the benchmark adapters, but this report is a documentation/claim risk.

## 7. Simulator and Digital Twin audit

`simulator/engine_simulator.py:35` and `simulator/subsystems/` implement atmosphere, mission, rotational dynamics, fuel, thermal, cooling, lubrication, vibration, turbocharger/TCU surrogate, and sensor-fault code. Engine reference facts/provenance live in `configs/engine_reference/rotax_914_ul_f.json`; the system-wide stated fidelity boundary is `docs/physics_contract.md` and `configs/physics_contract.json`.

The Digital Twin’s canonical state and expected state use `digital_twin/twin_model.py:69`, state estimation/synchronisation, quality validation, and quality-aware residual generation in `digital_twin/residuals.py:288`. Fault paths are typed in `simulator/fault_interface.py` and include cooling conductance loss, lubrication pressure/friction change, fuel abnormality, mechanical/vibration degradation, and sensor bias/drift/noise/dropout.

Physically grounded reference parameters and operating constraints are documented, but implementation is a lumped reduced-order model. The repository itself explicitly excludes a complete turbocharger/compressor-map/wastegate/gearbox/cylinder-network/ignition representation. There is no physical-engine test-cell calibration, UAV flight recording calibration, certified simulator comparison, or airworthiness validation in the repository. Therefore, the valid claim is **internally tested synthetic grey-box behaviour**, not experimentally validated engine performance.

## 8. Runtime and latency audit

The only directly inspected machine-readable benchmark is `evidence/phase6_latency_benchmark.json`: 1,000 streaming steps, mean 0.3882 ms, median 0.3284 ms, p95 0.5592 ms, p99 0.7631 ms, maximum 28.9939 ms. The evidence labels the steady-state result as passing a 20 ms budget but notes the strict worst case fails the same budget. This is a component/streaming benchmark, not proof of browser rendering, model load, network, or end-to-end deployment latency.

`main.py:94-108` has an end-to-end steady-state reporting path that reads `DashboardStatePayload.execution_latency_ms` after the first five steps. README claims approximately 53/55/84/88 ms (mean/P50/P95/P99); the earlier audit claims ~58 ms. Neither number was re-measured in this audit, so the values should be treated as **historical, hardware- and configuration-specific claims**, not a current deployment SLA. The browser/dashboard server was not included in either observed benchmark.

At 1 Hz, the historical core paths appear comfortably below the 1,000 ms interval, but the following remain unverified for deployment: cold start, CUDA/TensorFM load, browser rendering, external-network delay, concurrent users, and Hugging Face CPU hardware.

## 9. Deployment readiness: Hugging Face

No `README` Space metadata, `Dockerfile`, `runtime.txt`, `packages.txt`, `pyproject.toml`, `app.yaml`, workflow, bucket-mount code, or deployment script was found at the repository root. `requirements.txt` declares NumPy, pandas, scikit-learn, XGBoost, torch, Plotly, Streamlit, and pytest, but omits `shap`, `scipy`, and the TimesFM packages imported by source. `forecasting/models.py:170-188` requires a local checkpoint or `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` and accepted access to the gated repository.

Current assessment: a local Streamlit run is proven; a public Hugging Face Space, private artifact bucket integration, authenticated secret configuration, cold-start/memory plan, and a reproducible dependency lock are **not proven**. Do not claim public no-install deployment or private-weight protection until those files and a deployed Space exist.

## 10. Tests and quality status

`pytest --collect-only -q` completed during this audit and collected **865 tests**. That differs from the README’s “340+” and `evidence/phase15_validation.json`’s historical 386-pass result at commit `a1e0d8c` on 7 September 2026.

A full `pytest -q` execution was attempted during this audit but did not finish. After 39 progress symbols (including one `F`), the Python process terminated with **Windows fatal exception: access violation** while PyTorch/safetensors was loading TimesFM-3. The recorded stack reaches `forecasting/models.py:182` from `phase14/tests/test_phase14_replay.py:149` (`test_replay_deterministic_execution`). Because abrupt native-process termination prevented pytest from printing a normal failure summary, the identity/root cause of the earlier ordinary failure is not available from this run. Current test result: **865 collected; suite aborted; pass/fail/skip totals unavailable**.

The test suite covers simulator physics, telemetry integrity, Digital Twin, anomaly detection, diagnosis, health/RUL, forecasting, explainability, timestamp resilience, dashboard no-fabrication behaviour, and Phase 14 replay/report/what-if. New uncommitted tests include `tests/test_ml_baselines.py`, `test_physics_ml_residual.py`, and `test_phase4_health_prognostics.py`; their results should be reported separately until committed.

## 11. Documentation consistency, bugs, and risks

1. **Test-count and execution drift (high):** README says 340+; historical evidence says 386; the current collector finds 865. The current full run aborts in native TimesFM/PyTorch/safetensors loading. Update release statements only after a captured clean run.
2. **Dataset-manifest drift (high):** acquisition manifest says several datasets are absent, but local raw and processed data exist. Rebuild the acquisition/inventory manifest from disk.
3. **Deployment reproducibility gap (high):** requirements omit imported dependencies and no Space configuration exists.
4. **No persisted core model artifacts (medium):** runtime bootstrap makes results dependent on code, environment and startup data/configuration rather than versioned release artifacts.
5. **External-data domain mismatch (high claim risk):** public bearing, turbofan, battery and SITL data cannot validate complete aero-piston behaviour. Preserve the strict domain boundary in presentation material.
6. **Unsafe historical adapter/report claim (high):** `reports/dataset_audit_report.json` documents C-MAPSS sensor-to-piston mapping. The live path is isolated from it, but it should be corrected or quarantined before public review.
7. **Model configuration ambiguity (medium):** XGBoost class defaults use 200 estimators while bootstrap uses 30. Published metrics must identify the exact runtime model settings.
8. **TimesFM native-runtime risk (high):** current CUDA/pretrained availability was verified locally, but the full test run fatally crashed inside PyTorch/safetensors during another TimesFM load. The gated token, accepted terms, packages, compatible CUDA stack and GPU are not encoded in reproducible deployment configuration.
9. **Safety boundary (high):** advisory text must remain decision support. No certification, command authority, or real-aircraft operational claim is supported.

## 12. Prioritized next steps

1. Reproduce and fix/isolate the native TimesFM PyTorch/safetensors access violation, then capture a full 865-test result with commit SHA, Python/dependency versions and hardware; update README/evidence test counts.
2. Generate a single current dataset inventory from the actual filesystem and reconcile/archive stale acquisition manifests and unsafe mapping text.
3. Add a locked deployment specification for Hugging Face (SDK, entry point, Python version, all dependencies, secret names, model cache/storage strategy, hardware tier, startup test).
4. Version the synthetic-bootstrap training dataset/configuration and save model artifacts plus metadata if reproducible releases are required.
5. Keep external datasets in isolated benchmark workflows; add explicit no-cross-domain training assertions for all new `ml/tasks` pipelines.
6. Obtain or document a real aero-piston test-cell/flight-data validation plan before advancing any performance, RUL, or diagnostic claim beyond synthetic verification.

# MASTER CONTEXT FOR CHATGPT

SIH26054/Avekshak is a Python Streamlit demonstration of a physics-informed, reduced-order grey-box Digital Twin for a Rotax 914 UL/F-referenced MALE-UAV aero-piston propulsion system. The working live path is `dashboard/app.py` → `orchestrator/pipeline.py` → `DashboardStatePayload` and has eight UI views, including replay, reports and what-if comparison. The pipeline simulates a mission, strips simulator fault labels before inference, produces twin residuals, runs threshold/EWMA/persistence/Isolation Forest detection, bootstrapped six-class XGBoost diagnosis, health/degradation tracking, TimesFM-3 or EWMA forecasting, Theil-Sen/Monte-Carlo RUL, SHAP/physics/temporal explanation, and a decision-support advisory.

The live engine telemetry is synthetic. The core Isolation Forest and XGBoost are fitted deterministically at orchestrator bootstrap from synthetic simulator data, not loaded from persisted artifacts. TimesFM-3 was directly verified in the current environment as pretrained and CUDA-available, but it is a gated external model and the repository dependencies/deployment configuration do not make this reproducible by themselves. Health and RUL are algorithmic/rule-based pipelines with quality and history gates; RUL is intentionally withheld when history is insufficient.

Local external datasets include C-MAPSS, CWRU, FEMTO, NASA battery, NUST and Paderborn data, with processed parquet outputs. They are component/methodology benchmarks only and are not used by the canonical live Phase 13 path. They do not validate a complete aero-piston engine; C-MAPSS is turbofan simulation, CWRU/Paderborn/FEMTO are bearing datasets, NASA battery is electrochemical, and BASiC is UAV SITL. No real aero-piston flight/test-cell validation, airworthiness certification, or OEM calibration is established.

Current audit facts: pytest collects 865 tests; an older evidence artifact recorded 386 passing at an earlier commit, but the current full run aborted with a Windows access violation in PyTorch/safetensors while TimesFM-3 loaded during Phase 14 replay testing. No current full-suite pass total should be claimed. `evidence/phase6_latency_benchmark.json` records a 1,000-step streaming benchmark (mean 0.3882 ms, p95 0.5592 ms, worst 28.9939 ms); that is not browser/deployment latency. Historical 53–58 ms end-to-end claims were not re-measured here.

Deployment to Hugging Face is planned, not implemented in the repository: no Space/Docker/runtime configuration or model bucket integration was found. `requirements.txt` is incomplete for the imported runtime. Highest-priority work is to capture current test results, reconcile stale data manifests, create reproducible deployment/model packaging, preserve dataset-domain boundaries, and acquire a credible real-engine validation plan.
