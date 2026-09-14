# Phase 7 — Physics-Constrained Synthetic Engine Population

## 1. Motivation
Previous validation phases for the Rotax 914 UL/F grey-box propulsion model operated primarily on a single deterministic nominal engine configuration under nominal operating conditions. This introduced an architectural vulnerability: algorithms tuned exclusively on a single configuration could overfit to nominal thresholds, masking sensitivity to manufacturing tolerances, environmental shifts, sensor calibration variances, and cylinder-to-cylinder imbalances.

Phase 7 addresses this by introducing a **Physics-Constrained Synthetic Engine Population Layer**.

> **MANDATORY DISCLAIMER:**  
> This population is synthetic and generated from the project's grey-box simulator. It is not representative of measured Rotax fleet statistics unless independently demonstrated. It is intended for software verification, sensitivity analysis, and controlled algorithm development.

---

## 2. Population Architecture & Layering
The population layer sits strictly **ABOVE** the existing grey-box simulator and Digital Twin pipeline without modifying underlying physical equations or forking the simulator:

```text
Population Configuration (PopulationConfig)
        ↓
   EngineProfile (Typed, Immutable, Bounded, with Provenance)
        ↓
 SimulatorConfig (Tier A locked, Tier C/D overridden)
        ↓
  EngineSimulator (Existing Rotax 914 Reduced-Order Simulator)
        ↓
 TelemetryRecord (Clean separation: observable runtime vs offline truth)
        ↓
   DigitalTwin (Deterministic Synchronization & Replay)
        ↓
 Phase 5 Residual & Health Evaluator (H_sub, H_phys, HI_raw)
        ↓
 Phase 6 Temporal Detector & Diagnoser (S_anom, Hypothesis Ranking)
```

The population module is located in `simulator/population/`:
- `parameter_distributions.py`: Parameter definitions, bounded distributions, provenance, engineering rationales, and rejection sampling.
- `engine_profile.py`: Typed immutable engine profile dataclasses, serialization, and `.to_simulator_config()` translation.
- `mission_profiles.py`: 8 canonical mission profiles, ISA atmospheric coupling, and deterministic seed jitter.
- `population_generator.py`: Streaming and batch generator with engine-level train/val/test split and execution pipeline.
- `validation.py`: Physical conservation constraints, certified invariants, and population auditing.

---

## 3. Parameter Distributions & Bounded Variations
Synthetic variations are restricted to calibrated Tier C parameters and surrogate model coefficients. **Tier A certified reference specifications remain strictly fixed.**

### Certified Locked Constants (Zero Randomization)
- **Reference Engine:** Rotax 914 UL/F
- **Architecture:** 4 cylinders, 4-stroke boxer
- **Displacement:** 1211.2 cc (Bore 79.5 mm, Stroke 61.0 mm)
- **Compression Ratio:** 9.0:1
- **Gearbox Reduction Ratio:** 2.42857:1 (51:21 spur gears)
- **Max Takeoff RPM:** 5800 RPM (5-minute ceiling)
- **Max Continuous RPM:** 5500 RPM
- **Firing Order:** 1-4-3-2

### Bounded Parameter Catalog
All 27 varied parameters are bounded by physical limits and assigned explicit provenance:

| Subsystem | Parameter | Nominal | Bounds | Std Dev | Units | Provenance | Rationale |
|---|---|---|---|---|---|---|---|
| **Thermal** | `c_th_cht` | 920.0 | [800.0, 1050.0] | 45.0 | J/K | `MODEL_CALIBRATION` | Casting mass and cooling fin surface area tolerances |
| | `h_cool_base` | 16.0 | [13.5, 18.5] | 1.0 | W/K | `MODEL_CALIBRATION` | Cowling airflow baffling fit and ducting tolerances |
| | `q_gen_fraction` | 0.048 | [0.042, 0.054] | 0.002 | — | `MODEL_CALIBRATION` | Head heat transfer fraction from combustion |
| | `t_egt_base_c` | 520.0 | [490.0, 550.0] | 12.0 | °C | `MODEL_CALIBRATION` | Exhaust manifold fabrication tolerances |
| | `k_egt_load` | 210.0 | [190.0, 230.0] | 8.0 | °C | `MODEL_CALIBRATION` | EGT sensitivity to engine load |
| | `tau_egt_s` | 2.5 | [2.0, 3.2] | 0.22 | s | `MODEL_CALIBRATION` | Thermocouple sheath thermal conduction lag |
| | `c_coolant_j_per_k` | 4500.0 | [3800.0, 5200.0] | 250.0 | J/K | `MODEL_CALIBRATION` | Lumped liquid coolant jacket thermal capacitance |
| | `h_rad_base` | 18.0 | [15.0, 21.0] | 1.2 | W/K | `MODEL_CALIBRATION` | Radiator core air-side base thermal conductance |
| **Lubrication** | `c_oil` | 1350.0 | [1150.0, 1550.0] | 75.0 | J/K | `MODEL_CALIBRATION` | Oil reservoir and passage effective thermal mass |
| | `h_oil_cool` | 20.0 | [17.0, 23.0] | 1.2 | W/K | `MODEL_CALIBRATION` | Oil cooler radiator heat dissipation conductance |
| | `oil_press_base_bar` | 1.0 | [0.85, 1.15] | 0.06 | bar | `MODEL_CALIBRATION` | Pressure relief valve spring preload tolerance |
| | `k_oil_p_rpm` | 0.00085 | [0.00075, 0.00095] | 4e-5 | bar/RPM | `MODEL_CALIBRATION` | Positive-displacement oil pump delivery slope |
| | `k_oil_p_temp` | 0.008 | [0.006, 0.010] | 8e-4 | bar/°C | `MODEL_CALIBRATION` | Oil viscosity thinning temperature sensitivity |
| **Combustion** | `a_fuel_kg_per_j` | 6.8e-8 | [6.2e-8, 7.4e-8] | 2.4e-9 | kg/J | `MODEL_CALIBRATION` | Brake specific fuel consumption slope (Willans line) |
| | `b_fuel_kg_per_s` | 0.00032 | [0.00028, 0.00036] | 1.5e-5 | kg/s | `MODEL_CALIBRATION` | Carburetor idle jet metering tolerance |
| | `combustion_efficiency_multiplier` | 1.0 | [0.95, 1.05] | 0.02 | — | `SYNTHETIC_VARIATION` | Overall combustion efficiency variation |
| **Turbocharger** | `tcu_continuous_map_target_bar` | 1.200 | [1.170, 1.230] | 0.012 | bar | `MODEL_CALIBRATION` | Surrogate TCU boost pressure regulation target |
| | `tau_map_s` | 0.35 | [0.28, 0.45] | 0.03 | s | `MODEL_CALIBRATION` | Intake manifold pneumatic filling lag |
| | `eta_turb_nominal` | 0.65 | [0.60, 0.70] | 0.02 | — | `MODEL_CALIBRATION` | Surrogate radial turbine isentropic efficiency |
| | `eta_comp_nominal` | 0.70 | [0.65, 0.75] | 0.02 | — | `MODEL_CALIBRATION` | Surrogate centrifugal compressor isentropic efficiency |
| | `intercooler_efficiency` | 0.60 | [0.52, 0.68] | 0.03 | — | `MODEL_CALIBRATION` | Charge-air cooler thermal effectiveness |
| **Mechanical** | `inertia_kg_m2` | 0.28 | [0.25, 0.32] | 0.012 | kg·m² | `MODEL_CALIBRATION` | Rotating assembly plus propeller moment of inertia |
| | `k_load` | 3.036e-4 | [2.70e-4, 3.40e-4] | 1.4e-5 | N·m/(rad/s)² | `MODEL_CALIBRATION` | Propeller aerodynamic absorption coefficient |
| | `k_fric_linear` | 0.015 | [0.012, 0.018] | 0.0011 | N·m·s/rad | `MODEL_CALIBRATION` | Journal bearing viscous friction coefficient |
| | `torque_fric_static` | 3.5 | [2.8, 4.2] | 0.25 | N·m | `MODEL_CALIBRATION` | Piston ring sliding and static friction torque |
| | `vib_order1_base_g` | 0.32 | [0.26, 0.38] | 0.024 | g | `MODEL_CALIBRATION` | 1x rotational unbalance baseline amplitude |
| | `vib_order2_base_g` | 0.22 | [0.18, 0.26] | 0.016 | g | `MODEL_CALIBRATION` | 2x reciprocating unbalance baseline amplitude |
| | `vib_noise_std_g` | 0.06 | [0.045, 0.075] | 0.006 | g | `MODEL_CALIBRATION` | Broadband process vibration noise std dev |
| **Sensors** | `sensor_noise_*` | Nominal | [0.6x, 1.5x] | — | Various | `SYNTHETIC_VARIATION` | Channel-by-channel sensor noise variations |
| | `sensor_bias_*` | 0.0 | Bounded | Small | Various | `SYNTHETIC_VARIATION` | Persistent calibration zero-offsets |

---

## 4. Physical Correlations & Cylinder Variations

### Physical Correlations
Parameters are not randomized as independent white noise. Physically justified correlations are enforced:
1. **Thermal mass $\leftrightarrow$ Coolant circuit mass ($r \approx 0.45$):** Engines with higher cylinder head thermal mass have correspondingly sized cooling systems.
2. **Static friction $\leftrightarrow$ Oil heat dissipation ($r \approx 0.40$):** Higher friction torque generates more frictional heat in the oil, necessitating correlated oil cooling.
3. **Combustion efficiency $\leftrightarrow$ Fuel consumption slope ($r \approx -0.35$):** More efficient combustion slightly reduces specific fuel consumption slope $a_{fuel}$.

### Cylinder-to-Cylinder Imbalance
Discrete cylinders 1–4 are varied independently within bounded $\pm 2.5\%$ tolerances while enforcing strict engine-level conservation:
$$\frac{1}{4} \sum_{i=1}^4 k_{bank, i} = 1.0 \quad \text{and} \quad \frac{1}{4} \sum_{i=1}^4 \eta_{comb, i} = 1.0$$
This preserves total engine power, displacement, and heat rejection while introducing controlled synthetic runner-to-runner thermal and exhaust spreads.

---

## 5. Canonical Mission Profiles & Atmospheric Coupling
Eight canonical mission trajectories cover the full flight envelope:
1. `GROUND_IDLE`: 120s, throttle 0–10%, sea level, warmup.
2. `TAKEOFF_CLIMB`: 240s, throttle 100% $\to$ 90%, altitude 0 $\to$ 2500m.
3. `CRUISE`: 240s, throttle 75%, altitude 2500m steady.
4. `HIGH_ALTITUDE_CRUISE`: 240s, throttle 82%, altitude 4500m (high-altitude synthetic operating point near the Rotax 914 OM §2.1 continuous operating ceiling of 4572m / 15,000 ft, and below EASA TCDS E.122 critical altitude of 4875m / 16,000 ft).
5. `DESCENT`: 200s, throttle 40%, altitude 3000m $\to$ 300m.
6. `RAPID_THROTTLE_TRANSITION`: 150s, throttle steps (25% $\to$ 90% $\to$ 40% $\to$ 95% $\to$ 50%).
7. `ENDURANCE`: 400s, throttle 65%, altitude 2000m steady loiter.
8. `HOT_DAY_OPERATION`: 250s, altitude 1200m, ambient temperature ISA + 23 K (~38°C at sea level).

**ISA Atmospheric Coupling:** Ambient barometric pressure is derived strictly via the barometric troposphere equation $P = P_0 (1 - L \cdot h / T_0)^{\frac{g}{RL}}$, preventing unphysical decoupled pressure-altitude jumps.

---

## 6. Train / Validation / Test Engine-Level Isolation
To ensure future machine learning models cannot leak time-series information:
- Partitioning is performed **strictly by `engine_instance_id`** (e.g. 70% Train, 15% Validation, 15% Test).
- An engine appearing in the training set **NEVER appears in evaluation or test sets**.
- Row-level random splitting within the same engine trajectory is mathematically impossible.

---

## 7. Zero Leakage Architecture
The Digital Twin execution pipeline operates exclusively on observable telemetry fields:
- `timestamp`, `altitude`, `ambient_temp`, `throttle`, `rpm`, `cht`, `egt`, `oil_temp`, `oil_pressure`, `fuel_flow`, `vibration`, `map_bar`, `charge_air_temp`, etc.
- Offline ground-truth fields (`fault_type`, `fault_severity`, `fault_scenario_id`) are quarantined strictly for offline scoring and validation. The runtime Digital Twin never receives or consumes ground-truth metadata.

---

## 8. Phase 5 & 6 Robustness Evaluation & Findings

### Healthy Population Stability
Evaluating 15 diverse engines across 1,800 healthy simulation steps revealed:
- **Mean Health Index ($HI_{raw}$):** 0.9969 (5th percentile: 0.9721, median: 1.0000, 95th percentile: 1.0000)
- **Overall Per-Step False Alarm Rate (FAR):** **1.11%** (20 out of 1,800 steps)
- **Steady-State Per-Step FAR ($t \ge 5.0\text{s}$):** **0.00%** (0 out of 1,650 steps)
- **Per-Mission FAR:** **80.0%** overall (12/15 missions had $\ge 1$ startup transient alert) vs **0.0%** post-warmup ($t \ge 5.0\text{s}$)
- **Mean Anomaly Score ($S_{anom} = 1 - HI_{raw}$):** 0.0031

The healthy population does **NOT** collapse into false alarms in steady flight despite substantial engine-to-engine physical diversity (e.g., CHT spread $\pm 6^\circ\text{C}$, oil pressure spread $\pm 0.35$ bar, fuel flow spread $\pm 1.0$ L/h).

### Fault Population Detectability & Diagnosability
Testing F1–F7 across varied engines exposed critical architectural insights:
1. **F2 (Lubrication Degradation), F4 (Combustion Misfire), F5 (Mechanical Degradation):**
   - **Detection Rate:** **100.0%**
   - **Detection Latency:** 2.50s (F2), 3.20s (F4), 2.50s (F5)
   - **Active-Window Top-1 Diagnosis:** 100.0% (F2), 100.0% (F4), 80.0% (F5)
2. **F3 (Cooling Degradation):**
   - **Detection Rate:** 20.0% under short test durations (60s).
   - **Causal Mechanism:** Thermal inertia (the $4500\text{ J/K}$ liquid coolant mass and $920\text{ J/K}$ head capacitance) is a major modeled contributor to the observed detection delay; controlled sensitivity tests show latency decreasing from $>60\text{s}$ to steady detection only after extended durations ($60\text{s}+$ to $120\text{s}+$).
3. **F1 (Injector Abnormality localized to Cylinder 1) & Single-Channel Sensor Faults (F6/F7):**
   - **Architectural Principle:** **Engine-level anomaly detection is intentionally not equivalent to channel-level fault observability.**
   - In accordance with the Phase 6 architectural requirement, single-cylinder runner spread does **NOT** trigger engine-level anomaly votes ($S_{anom} = 1 - HI_{raw} < 0.018$).
   - When a sensor bias or localized cylinder lean condition affects only 1 channel, the arithmetic mean across 6 active subsystems produces an engine-level anomaly score below the trip threshold ($S_{anom} \approx 0.015 < 0.018$).
   - However, the physics-informed diagnoser successfully evaluates the runner spread and sensor quality status, achieving **80.0% Active Top-1 accuracy for F1 (injector delivery)** and **80.0% Active Top-1 accuracy for F7 (sensor dropout)** without inducing engine-level false alarms.

---

## 9. Performance & Memory Benchmarks
- **Engine Profile Generation:** **0.18 ms** per engine profile (>5,500 profiles/sec).
- **Simulator Step Time:** **0.322 ms** per step (**3,106 steps/sec**).
- **Digital Twin Step Time:** **0.682 ms** per step (**1,466 steps/sec**).
- **Streaming Iterator Memory:** `iter_profiles()` operates with constant $O(1)$ memory scaling ($\approx 5\text{ KB}$ peak heap differential across 10,000 generated profiles).
- **Dataset Accumulation Memory:** In-memory collection of complete simulated telemetry streams scales linearly $O(N)$ ($\approx 2.77\text{ KB}$ per telemetry record, or $\approx 26\text{ MB}$ per 100 full mission trajectories).
