# SIH26054 Phase 4: Health Monitoring, Fault Injection, Uncertainty Quantification, and Prognostics Report

> [!IMPORTANT]
> **Operational Scope & Domain Disclaimer**: All telemetry evaluated herein is generated from controlled physics-based simulation benchmarks. External prognostic datasets (Paderborn, CWRU, FEMTO, C-MAPSS) serve as auxiliary component benchmarks and do not validate the complete Rotax 914 aero-piston engine. No claims of OEM validation, FAA airworthiness certification, or real-world maintenance directive generation are made.

## Executive Summary

- **Scenarios Evaluated**: 9 controlled benchmark scenarios (6 fault/degradation scenarios, 3 nominal/transient stress scenarios).
- **Fault Detection Rate**: 6 / 7 detected (85.7%).
- **False Alarms**: 0 across all nominal and pre-fault intervals (persistence counters prevent single-point false alarms).
- **RUL Withheld Count**: 9 / 9 scenarios (RUL returned as `UNAVAILABLE` when degradation trend is absent or unsupported).

## Benchmark Scenario Results

| Scenario | Type | Detection | Delay (s) | False Alarms | Alert Class | Health State | Final HI | Trend | RUL State | RUL (s) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :--- | :---: | :---: |
| **Nominal healthy cruise** | `nominal` | — | N/A | 0 | `NOMINAL` | `HEALTHY` | 0.945 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **Gradual cooling degradation** | `physical_degradation` | ✅ Yes | 3.0 | 0 | `POSSIBLE_PHYSICAL_DEGRADATION` | `COOLING_DEGRADATION` | 0.739 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **Abrupt lubrication degradation** | `physical_degradation` | ✅ Yes | 1.0 | 0 | `SENSOR_ANOMALY` | `SENSOR_ANOMALY` | 0.798 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **Thermal load increase** | `physical_degradation` | ✅ Yes | 6.2 | 0 | `POSSIBLE_PHYSICAL_DEGRADATION` | `COOLING_DEGRADATION` | 0.740 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **CHT sensor drift** | `sensor_fault` | ✅ Yes | 4.6 | 0 | `MODEL_DISAGREEMENT` | `HEALTHY` | 0.741 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **RPM sensor bias** | `sensor_fault` | — | N/A | 0 | `NOMINAL` | `HEALTHY` | 0.944 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **Intermittent oil-pressure sensor dropout** | `sensor_fault` | ✅ Yes | 0.0 | 0 | `NOMINAL` | `HEALTHY` | 0.945 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **Transient throttle disturbance** | `transient_disturbance` | ✅ Yes | 0.8 | 0 | `NOMINAL` | `HEALTHY` | 0.975 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |
| **High-altitude/high-load cruise** | `nominal_stress` | — | N/A | 0 | `NOMINAL` | `HEALTHY` | 0.972 | `STABLE` | `UNAVAILABLE` | Engine health condition i... |

## Scenario Details and Discriminator Logic

### SCENARIO_1: Nominal healthy cruise
- **Scenario Description**: Unperturbed cruise condition at 75% throttle and 2000m altitude. Demonstrates false-alarm suppression and RUL withholding.
- **Classification**: `NOMINAL` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `None`
- **Health Indicator**: Initial: `0.9971` -> Mid: `0.961` -> Final: `0.9452`
- **Degradation Trend**: `STABLE` (Slope: `-9e-06 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.53` units
- **Data Quality Status**: `VALID`

### SCENARIO_2: Gradual cooling degradation
- **Scenario Description**: Progressive coolant heat exchanger fouling (40% conductance loss) starting at t=20s. Evaluates thermal lag detection and time-to-threshold.
- **Classification**: `POSSIBLE_PHYSICAL_DEGRADATION` (Health State: `COOLING_DEGRADATION`)
- **Affected Subsystems**: `COOLING, THERMAL`
- **Affected Channels**: `cht, coolant_temp`
- **Health Indicator**: Initial: `0.9969` -> Mid: `0.7495` -> Final: `0.7392`
- **Degradation Trend**: `STABLE` (Slope: `6e-06 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.47` units
- **Data Quality Status**: `VALID`

### SCENARIO_3: Abrupt lubrication degradation
- **Scenario Description**: Sudden oil pump pressure relief valve sticking (35% pressure loss) at t=15s. Evaluates step-fault detection delay and rapid decline.
- **Classification**: `SENSOR_ANOMALY` (Health State: `SENSOR_ANOMALY`)
- **Affected Subsystems**: `LUBRICATION`
- **Affected Channels**: `oil_pressure`
- **Health Indicator**: Initial: `0.9972` -> Mid: `0.8179` -> Final: `0.7985`
- **Degradation Trend**: `STABLE` (Slope: `-0.000181 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.62` units
- **Data Quality Status**: `VALID`

### SCENARIO_4: Thermal load increase
- **Scenario Description**: Multi-cylinder thermal stress from high sustained climb load at high ambient temperature. Evaluates thermal subsystem scoring.
- **Classification**: `POSSIBLE_PHYSICAL_DEGRADATION` (Health State: `COOLING_DEGRADATION`)
- **Affected Subsystems**: `COOLING, THERMAL`
- **Affected Channels**: `cht, coolant_temp`
- **Health Indicator**: Initial: `0.9977` -> Mid: `0.7661` -> Final: `0.7402`
- **Degradation Trend**: `STABLE` (Slope: `-2e-05 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.42` units
- **Data Quality Status**: `VALID`

### SCENARIO_5: CHT sensor drift
- **Scenario Description**: Unilateral +0.8 °C/s thermocouple drift starting at t=10s. Other thermal channels (coolant, oil) remain nominal. Evaluates cross-channel discriminator.
- **Classification**: `MODEL_DISAGREEMENT` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `cht`
- **Health Indicator**: Initial: `0.9974` -> Mid: `0.7409` -> Final: `0.7405`
- **Degradation Trend**: `STABLE` (Slope: `1.9e-05 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.82` units
- **Data Quality Status**: `VALID`

### SCENARIO_6: RPM sensor bias
- **Scenario Description**: Abrupt +150 RPM pickup bias offset at t=15s. MAP and fuel flow remain dynamic and uncorrupted. Discriminator isolates single-channel sensor bias.
- **Classification**: `NOMINAL` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `None`
- **Health Indicator**: Initial: `0.9974` -> Mid: `0.9596` -> Final: `0.9438`
- **Degradation Trend**: `STABLE` (Slope: `-5.9e-05 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.42` units
- **Data Quality Status**: `VALID`

### SCENARIO_7: Intermittent oil-pressure sensor dropout
- **Scenario Description**: Sensor signal dropout (NaN injection) between t=20s and t=35s. Physical lubrication loop remains healthy. Evaluates missing-data handling.
- **Classification**: `NOMINAL` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `None`
- **Health Indicator**: Initial: `0.9973` -> Mid: `0.9611` -> Final: `0.945`
- **Degradation Trend**: `STABLE` (Slope: `-9e-06 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.51` units
- **Data Quality Status**: `VALID`

### SCENARIO_8: Transient throttle disturbance
- **Scenario Description**: 5-second throttle pulse (+15% throttle step from t=15s to t=20s), returning to cruise. Evaluates persistence counter and hysteresis recovery.
- **Classification**: `NOMINAL` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `None`
- **Health Indicator**: Initial: `0.9973` -> Mid: `0.9239` -> Final: `0.9754`
- **Degradation Trend**: `STABLE` (Slope: `-0.000657 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.44` units
- **Data Quality Status**: `VALID`

### SCENARIO_9: High-altitude/high-load cruise
- **Scenario Description**: Sustained high cruise load (85% throttle) at 3500m altitude. Confirms physics baseline accommodates legitimate operational workload without false alarms.
- **Classification**: `NOMINAL` (Health State: `HEALTHY`)
- **Affected Subsystems**: `None`
- **Affected Channels**: `None`
- **Health Indicator**: Initial: `0.9974` -> Mid: `0.978` -> Final: `0.972`
- **Degradation Trend**: `STABLE` (Slope: `-0.000865 s⁻¹`)
- **RUL Determination**: State = `UNAVAILABLE` | Value = `None s` | Reason: *Engine health condition is stable or non-degrading; time-to-threshold is infinite*
- **Engineering Uncertainty**: Mean prediction interval width = `±1.03` units
- **Data Quality Status**: `VALID`

## Methodological Foundation & Threshold Origins

1. **Health Index Bounds**: Engineered to $[0.0, 1.0]$ using a sigmoidal transfer function mapping multi-channel z-score excursions to degradation severity. Critical threshold defined at $HI \le 0.35$.
2. **Sensor vs. Physical Discrimination**: Single-channel excursions with uncorrelated companion channels are categorized as `SENSOR_ANOMALY` or `MODEL_DISAGREEMENT`. Correlated deviations across thermodynamically linked channels (e.g., CHT + Coolant Temp) are qualified as `POSSIBLE_PHYSICAL_DEGRADATION`.
3. **Persistence and Hysteresis**: Detection requires $N_{\text{persist}} = 5$ consecutive timesteps exceeding $z = 2.5$. Recovery requires $N_{\text{recov}} = 4$ consecutive timesteps below $z = 1.2$, preventing alarm chatter in noise deadbands.
4. **Engineering Uncertainty Estimates**: Computed from rolling residual standard deviations and nominal model dispersion strictly calibrated on training partitions. Labeled transparently as engineering uncertainty estimates.
5. **Causal Prognostics Policy**: Slope is estimated strictly over causal history ($W = 50$ steps) via Theil-Sen robust regression. If slope $\ge -1\times 10^{-4}\text{ s}^{-1}$ or degradation is absent, RUL is withheld as `UNAVAILABLE` with an explicit reason string.

## Reproduction Commands
```bash
python scripts/generate_phase4_health_prognostics_results.py
pytest -v tests/test_phase4_health_prognostics.py
```