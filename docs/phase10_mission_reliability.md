# Phase 10: Mission Reliability & What-If Simulation

## 1. Executive Summary & Objective

Phase 10 implements a deterministic, mission-level simulation and counterfactual scenario-comparison layer above the existing grey-box Rotax 914 propulsion model and Digital Twin pipeline:
- Grey-box physics (`EngineSimulator`, `Atmosphere`, `TurbochargerSubsystem`, `CoolingSubsystem`)
- Dynamic residual generation & quality assessment (Phase 5)
- Physics-based health assessment (Phase 5)
- Temporal anomaly detection & physics-informed fault diagnosis (Phase 6)
- Robust degradation modeling & uncertainty-aware RUL estimation (Phase 8)
- Telemetry normalization, schema enforcement, and replay (Phase 9)

The Phase 10 mission orchestration enables deterministic "what-if" engineering evaluations:
- What is the impact of elevated altitude on engine thermal margins?
- How does hot-day operation affect oil pressure and cooling capacity?
- How do transient vs persistent subsystem faults alter health trajectories?
- How does estimated RUL evolve under aggressive flight regimes?

### Non-Claims & Mandatory Disclaimers
1. **Model Scenario Results Only**: All mission results are synthetic, model-based counterfactual outputs of the reduced-order grey-box simulation.
2. **No Certified Reliability**: This system does NOT provide certified flight safety, fleet reliability statistics, failure probabilities, survival probabilities, or Mean Time Between Failures (MTBF).
3. **No Airworthiness / Certification Compliance**: No airworthiness, flight clearance, or DO-178C certification compliance is claimed or implied.
4. **4500 m Synthetic Model Scenario**: The 4500 m scenario is strictly a synthetic simulation point; it does NOT represent a certified service ceiling or fleet operational envelope.
5. **Mission Risk Index Disclaimer**: The `MissionRiskIndex` is strictly an engineering heuristic index $[0.0, 1.0]$ combining health loss, envelope excursions, degraded state duration, and RUL margin. It is **NOT** a probability of mission failure.

---

## 2. Mission Model & Specification (`MissionSpec`)

The mission specification is a typed, immutable dataclass (`digital_twin.mission_types.MissionSpec`) providing complete parameterization and execution reproducibility:

```python
@dataclass(frozen=True)
class MissionSpec:
    mission_id: str
    duration_s: float
    dt_s: float
    environment: EnvironmentProfile
    controls: ControlProfile
    fault_schedule: Optional[FaultSchedule] = None
    random_seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Supported Flight Phases
Simulation scenarios support standard flight segments:
- `START`: Ground engine cranking and initial ignition.
- `TAXI`: Low-power ground taxi maneuvering.
- `TAKEOFF`: High-power takeoff run (up to 5 minutes at takeoff rating).
- `CLIMB`: Sustained climb to operational ceiling.
- `CRUISE`: Steady-state cruise at configured airspeed proxy.
- `DESCENT`: Reduced power descent profile.
- `LANDING`: Approach and touchdown power regime.

---

## 3. Environment & Atmosphere Model

The environment layer reuses the authoritative ISA troposphere model (`simulator.subsystems.atmosphere.Atmosphere`) without code duplication or competing formulations.

### Hot-Day Temperature Delta Formulation
To prevent double-counting of environmental effects, hot-day scenarios apply an explicit temperature offset $\Delta T_{\text{hot}}$ exactly once to the standard ISA temperature:

$$T_{\text{amb}}(h) = T_{\text{ISA}}(h) + \Delta T_{\text{hot}}$$

where:
- $T_{\text{ISA}}(h) = T_0 - L \cdot h$ ($T_0 = 15.0\,^\circ\text{C}$, $L = 0.0065\,\text{K/m}$).
- Pressure $p_{\text{amb}}(h)$ and standard density factor $\sigma(h)$ are evaluated once through `Atmosphere.compute(altitude_m=h)`.

---

## 4. Control Profiles

The actuation profile is parameterized by `ControlProfile`:
- `throttle_schedule`: Deterministic schedule $u_{\text{thr}}(t) \in [0.0, 100.0]\,\%$.
- `load_schedule`: Engine power demand schedule $u_{\text{load}}(t) \in [0.0, 100.0]\,\%$.
- Profiles support continuous ramps, smooth transitions, and step changes without numerical instability.

---

## 5. Fault Injection & Dynamic Recovery

Fault scenarios reuse the Phase 4 `FaultSchedule` and `FaultState` infrastructure without modifying engine equations:
- **F1 (Fuel System)**: Injector delivery abnormality / mixture imbalance (`mode = lean/rich`).
- **F2 (Lubrication)**: Oil pressure loss / lubrication degradation.
- **F3 (Cooling)**: Coolant pump / heat rejection degradation.
- **F4 (Combustion)**: Cylinder misfire / combustion efficiency loss.
- **F5 (Mechanical)**: Bearing degradation / rotational harmonic order amplification.
- **F6 / F7 (Sensors)**: Observation layer bias, drift, and dropouts.

### Fault Onset, Offset & Thermal Inertia
- Pre-fault ($t < t_{\text{start}}$): Healthy baseline operation.
- Fault-on ($t_{\text{start}} \le t \le t_{\text{end}}$): Injected physical disturbance modulates subsystem dynamics.
- Post-fault ($t > t_{\text{end}}$): The fault command is deactivated. Thermal and lubrication states recover smoothly toward nominal according to physical capacitance time constants ($C_{\text{th,cht}}$, $\tau_{\text{egt}}$, $C_{\text{oil}}$), proving that the mission layer introduces zero artificial permanent degradation.

---

## 6. Authoritative Envelope Monitoring

The simulator continuously monitors physical channels against authoritative operating thresholds. Every threshold records strict provenance:

| Channel | Limit | Unit | Direction | Classification | Source Document & Section | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rpm` | 5800.0 | RPM | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Takeoff rating (max 5 min) |
| `rpm` | 5500.0 | RPM | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Max continuous speed |
| `rpm` | 1400.0 | RPM | MIN | `MODEL_ENVELOPE` | Rotax OM 914, Sec. 2.1 | Nominal idle floor |
| `map` | 1.350 | bar | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Takeoff MAP limit |
| `map` | 1.200 | bar | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Continuous MAP limit |
| `cht` | 135.0 | °C | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Water/glycol coolant limit |
| `coolant_temp` | 120.0 | °C | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Coolant exit temp limit |
| `oil_temp` | 130.0 | °C | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Max oil inlet temp |
| `oil_temp` | 50.0 | °C | MIN | `OEM_REFERENCE_LIMIT` | Rotax OM 914, Sec. 2.1 | Min temp for takeoff power |
| `oil_pressure` | 0.8 | bar | MIN | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Minimum idle oil pressure |
| `oil_pressure` | 2.0 | bar | MIN | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Normal operating minimum |
| `oil_pressure` | 5.0 | bar | MAX | `OEM_REFERENCE_LIMIT` | EASA TCDS E.122, Sec. IV | Normal operating maximum |
| `egt` | 950.0 | °C | MAX | `OEM_REFERENCE_LIMIT` | Rotax OM 914, Sec. 2.1 | Takeoff EGT limit (5 min) |
| `egt` | 910.0 | °C | MAX | `OEM_REFERENCE_LIMIT` | Rotax OM 914, Sec. 2.1 | Continuous EGT limit |
| `vibration` | 1.20 | g RMS | MAX | `MODEL_ENVELOPE` | SIH26054 Greybox Contract | **NOT AN OEM LIMIT** |

---

## 7. Downstream Pipeline Integration

### Health Assessment (Phase 5)
`MissionSimulator` extracts overall engine health ($HI_{\text{smooth}}$, $HI_{\text{raw}}$) and individual subsystem health without modifying Phase 5 algorithms. Threshold times are tracked:
- Time below Watch: $HI < 0.85$.
- Time below Degraded: $HI < 0.70$.
- Time below Critical: $HI < 0.50$.

### Degradation & RUL (Phase 8)
- Consumes `DegradationAssessment` and `RULAssessment` strictly as read-only outputs.
- Does not modify Theil-Sen regression, history windows, $D_{\text{EOL}}$ horizon, or uncertainty bounds.
- Explicit regression test `test_phase8_rul_methodology_unaltered` proves bit-for-bit numerical invariance.

---

## 8. Heuristic Mission Risk Index (`MissionRiskIndex`)

The mission risk score is defined as an auditable, deterministic engineering heuristic:

$$\text{RiskScore} = 0.40 \cdot C_{\text{health}} + 0.30 \cdot C_{\text{duration}} + 0.20 \cdot C_{\text{envelope}} + 0.10 \cdot C_{\text{rul}}$$

where:
- $C_{\text{health}} = \max(0, \min(1, 1 - HI_{\min}))$: Peak health degradation.
- $C_{\text{duration}} = \min\left(1, \frac{t_{\text{degraded}} + 2 \cdot t_{\text{critical}}}{T_{\text{mission}}}\right)$: Cumulative time in compromised states.
- $C_{\text{envelope}} = \min(1, N_{\text{excursions}} / 10)$: Saturated envelope excursion count.
- $C_{\text{rul}} = \max\left(0, \min\left(1, 1 - \frac{RUL_{\text{final}}}{2000\,\text{h}}\right)\right)$ if $RUL$ is valid, else $0.0$.

All 4 constituent components are retained in `MissionMetrics.risk_index` for full auditability.

---

## 9. 12 Golden Scenarios & Comparative Deltas

The suite evaluates 12 canonical mission scenarios over a 150 s simulation timeline ($dt = 1.0$ s). All values below reflect exact simulation outputs:

| ID | Scenario Name | Primary Parameter / Injection | Min HI | Mean HI | Final HI | Max CHT (°C) | Max EGT (°C) | Min Oil P (bar) | Max Vib (g) | Risk Score | $\Delta \text{Min HI}$ | $\Delta \text{Max CHT}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `NOMINAL_CRUISE` | Baseline: 1000m, ISA normal day, 75% throttle | 0.9042 | 0.9938 | 1.0000 | 82.44 | 706.71 | 2.640 | 0.657 | 0.0383 | 0.0000 | 0.00 |
| 2 | `HIGH_ALTITUDE` | 4500m synthetic model scenario | 0.8880 | 0.9926 | 1.0000 | 80.85 | 766.26 | 2.320 | 0.763 | 0.0448 | -0.0162 | -1.59 |
| 3 | `HOT_DAY` | ISA + 20 K temperature offset (1000m) | 0.9084 | 0.9940 | 1.0000 | 90.33 | 690.18 | 2.544 | 0.627 | 0.0366 | +0.0042 | +7.89 |
| 4 | `HOT_DAY_HIGH_ALTITUDE` | 4500m synthetic + 20 K offset | 0.9065 | 0.9936 | 1.0000 | 81.88 | 761.93 | 2.247 | 0.763 | 0.0374 | +0.0023 | -0.56 |
| 5 | `HIGH_LOAD` | Sustained 95% continuous throttle | 0.7383 | 0.9823 | 0.9963 | 122.68 | 781.79 | 3.371 | 0.763 | 0.3047 | -0.1659 | +40.24 |
| 6 | `AGGRESSIVE_THROTTLE` | Throttle cycling [60% $\leftrightarrow$ 95%] | 0.7383 | 0.9841 | 0.9992 | 118.92 | 781.79 | 3.149 | 0.763 | 0.3047 | -0.1659 | +36.48 |
| 7 | `F1_INJECTOR` | Cyl 1 fuel injector abnormality ($t=100\text{--}200$s) | 0.9042 | 0.9938 | 1.0000 | 82.44 | 710.42 | 2.640 | 0.657 | 0.0383 | 0.0000 | 0.00 |
| 8 | `F2_LUBRICATION` | Lubrication degradation ($t=100\text{--}200$s) | 0.9042 | 0.9875 | 0.9770 | 82.44 | 706.71 | 2.640 | 0.657 | 0.1382 | 0.0000 | 0.00 |
| 9 | `F3_COOLING` | Coolant pump degradation ($t=100\text{--}200$s) | 0.9042 | 0.9938 | 0.9999 | 98.51 | 706.71 | 2.640 | 0.657 | 0.0383 | 0.0000 | +16.07 |
| 10 | `F4_MISFIRE` | Cyl 2 combustion misfire ($t=100\text{--}200$s, sev=0.60) | 0.8791 | 0.9593 | 0.8815 | 82.44 | 706.71 | 2.640 | 0.727 | 0.1484 | -0.0251 | 0.00 |
| 11 | `F5_MECHANICAL` | Mechanical bearing degradation ($t=100\text{--}200$s) | 0.9042 | 0.9790 | 0.9526 | 82.44 | 706.71 | 2.640 | 1.145 | 0.1382 | 0.0000 | 0.00 |
| 12 | `COMBINED_ENVIRONMENT_FAULT`| 4500m synthetic + Hot Day (+20K) + F3 fault | 0.9065 | 0.9932 | 0.9961 | 89.81 | 761.93 | 2.247 | 0.763 | 0.0374 | +0.0023 | +7.37 |

---

### 9.1 Forensic Investigation: Causal Propagation & Health Mechanics

#### 1. Why Full-Mission `Min HI` is 0.9042 Across Fault Scenarios
At $t = 0.0$ s, the engine simulator initializes at idle ($1400$ RPM) and receives a step throttle command to $75\%$. During the dynamic engine spin-up transient ($t = 1 \to 4$ s), rotational and fuel tracking residuals temporarily dip overall health to an instantaneous minimum of $HI = 0.9042$ at $t = 3$ s. By $t = 10$ s, the engine settles into nominal steady state ($HI = 1.0000$). When a localized subsystem fault is injected at $t = 100$ s (e.g., F2 drops $HI$ to $0.9770$, F5 drops $HI$ to $0.9503$), the fault-induced health reduction does not dip below the initial startup transient value of $0.9042$. Therefore, the whole-mission scalar metric $\min_{t \in [0, 150]} HI(t)$ identically records $0.9042$. With F4 misfire aligned with Phase 6 validated baseline (severity 0.60), $HI$ drops to $0.8791$, breaking below the startup transient ($\Delta \min \text{HI} = -0.0251$). The real fault degradation is clearly visible in `mean_hi`, `final_hi`, the in-fault health trajectories, and individual subsystem health scores.

#### 2. Why Engine-Level HI Remains High During Single-Subsystem Faults
Phase 5 explicitly evaluates engine health via weighted linear aggregation across 6 subsystems:
$$\text{HI}_{\text{raw}} = 0.25 \cdot H_{\text{thermal}} + 0.20 \cdot H_{\text{lubrication}} + 0.20 \cdot H_{\text{fuel}} + 0.15 \cdot H_{\text{combustion}} + 0.10 \cdot H_{\text{mechanical}} + 0.10 \cdot H_{\text{rotational}}$$
When an isolated physical fault occurs:
- **F2 (Lubrication)**: Lubrication health drops to $0.8526$ ($\Delta H_{\text{lub}} = -0.1474$). Weighted contribution: $0.20 \times (-0.1474) = -0.0295$. Overall $\text{HI}_{\text{raw}}$ drops to $0.9705$ and $\text{HI}_{\text{smooth}}$ drops to $0.9770$.
- **F5 (Mechanical)**: Mechanical health drops to $0.7214$ ($\Delta H_{\text{mech}} = -0.2786$). Weighted contribution: $0.10 \times (-0.2786) = -0.0279$. Overall $\text{HI}_{\text{raw}}$ drops to $0.9721$ and $\text{HI}_{\text{smooth}}$ drops to $0.9518$.
- **F4 (Misfire)**: With severity 0.60 (Phase 6 validated baseline), Combustion drops to $0.6562$ and Rotational drops to $0.6629$. Overall $\text{HI}_{\text{smooth}}$ drops to $0.8851$. Average EGT residual drops by $-69.5$°C ($z = -2.78 < -2.00$), confirming `COMBUSTION_MISFIRE`.
- **F1 (Injector)**: Single-cylinder 35% lean imbalance elevates cylinder 1 EGT by $+30.3$°C ($z = +0.57$). Because all normalized residuals $|z| < 1.50$ ($\tau_{\text{nom}}$), Phase 5 assigns zero penalty and HI remains $1.0000$.
- **F3 (Cooling)**: CHT rises by $+15.5$°C ($z = +1.33$), remaining just below the $\tau_{\text{nom}} = 1.50$ penalty boundary.

#### 3. Causal Propagation Matrix at In-Fault Operating Point ($t = 125.0$ s)

| Scenario | Intervention ($\Delta$ Input) | Primary Telemetry $\Delta$ | Key Residual $z$-Score | Subsystem Health $\Delta$ | $\text{HI}_{\text{smooth}}$ | Diagnosis Hypothesis |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `NOMINAL_CRUISE` | Baseline ($1000$m, $75\%$) | None (RPM 4356, CHT 80.9°C, OilP 4.27 bar) | All $\|z\| < 0.1$ | None (All 1.0000) | 1.0000 | `HEALTHY` (1.00) |
| `HIGH_LOAD` | Throttle: $75\% \to 95\%$ | RPM $+1403$, CHT $+40.7$°C, MAP $+0.14$ bar | $\text{CHT } z=+3.31$ | Thermal: $-0.0245$ | 0.9959 | `UNKNOWN` (Excursion) |
| `AGGRESSIVE_THROTTLE` | Throttle: $60\% \leftrightarrow 95\%$ | RPM $+1416$, CHT $+19.9$°C, OilP $+1.00$ bar | Rapid dynamic swings | Nominal | 0.9990 | `HEALTHY` (Dynamic) |
| `F1_INJECTOR` | Cyl 1 Fuel delivery $-35\%$ | Cyl 1 EGT $+30.3$°C, Cyl 1 CHT $-4.6$°C | $\text{EGT}_{\text{cyl1}} z=+0.57$ | Nominal ($\|z\| < 1.5$) | 1.0000 | `HEALTHY` (Sub-threshold) |
| `F2_LUBRICATION` | Lubrication degradation $-40\%$ | Oil Pressure $-0.98$ bar, Oil Temp $+5.1$°C | $\text{OilP } z=-1.93$ | Lubrication: $-0.1240$ | 0.9790 | `LUBRICATION_DEGRADATION` (0.95) |
| `F3_COOLING` | Cooling pump $-50\%$ | CHT $+15.4$°C (peak $98.5$°C), CoolT $+2.2$°C | $\text{CHT } z=+1.33$ | Thermal: $-0.0040$ | 0.9999 | `HEALTHY` (Sub-threshold) |
| `F4_MISFIRE` | Cyl 2 Combustion misfire $-60\%$ | Cyl 2 EGT $-145.1$°C, RPM $-389$, EGT $-69.5$°C | $\text{EGT}_{\text{cyl2}} z=-4.15$, $\text{EGT } z=-2.70$ | Comb: $-0.344$, Rot: $-0.337$ | 0.8851 | `COMBUSTION_MISFIRE` (0.56) |
| `F5_MECHANICAL` | Mechanical bearing degradation | Vibration $+0.500$ g (0.61g $\to$ 1.11g) | $\text{Vib } z=+2.48$ | Mechanical: $-0.2786$ | 0.9518 | `MECHANICAL_DEGRADATION` (0.96) |
| `HOT_DAY` | $\Delta T_{\text{hot}} = +20.0$ K ($1000$m) | AmbT $+20.0$°C, CHT $+8.4$°C, OilT $+5.3$°C | $\text{CHT } z=+0.72$ | Nominal ($\|z\| < 1.5$) | 1.0000 | `HEALTHY` (Nominal Env) |
| `HIGH_ALTITUDE` | Altitude: $1000\text{m} \to 4500\text{m}$ | AmbT $-22.8$°C, CHT $-17.6$°C, EGT $+59.9$°C | $\text{EGT } z=+0.76$ | Nominal ($\|z\| < 1.5$) | 1.0000 | `HEALTHY` (Cold High Alt) |
| `HOT_DAY_HIGH_ALTITUDE` | $4500\text{m} + 20\text{K offset}$ | AmbT $+20.0$°C vs HA, CHT $+9.6$°C vs HA | $\text{CHT } z=+0.82$ | Nominal ($\|z\| < 1.5$) | 1.0000 | `COOLING_DEGRADATION` (Env Stress) |
| `COMBINED_ENVIRONMENT_FAULT`| $4500\text{m} + 20\text{K} + \text{F3}$ | CHT $+23.9$°C vs HA ($87.3$°C), EGT $+56.8$°C | $\text{CHT } z=+1.08$ | Thermal: $-0.0083$ | 0.9995 | `COOLING_DEGRADATION` (0.88) |

---

## 10. 6-Way Causal Decomposition

To decouple environmental stress from subsystem fault injection, the 6-way causal decomposition experiment evaluates:
1. `Baseline` (1000m, normal ISA)
2. `Altitude only` (4500m synthetic)
3. `Hot-Day only` (ISA + 20K)
4. `Altitude + Hot-Day` (4500m, ISA + 20K)
5. `F3 Cooling only` (1000m, normal ISA, F3 fault)
6. `Combined` (Altitude + Hot-Day + F3 Cooling)

**Key Finding**: Elevated ambient temperature reduces radiator temperature differential ($\Delta T_{\text{rad}} = T_{\text{coolant}} - T_{\text{amb}}$), exacerbating cooling pump degradation non-additively.

---

## 11. Performance & Memory Profiling

Benchmarks executed on Windows host environment:
- **1 Mission (Full Trajectory)**: 0.421 s
- **1 Mission (Streaming Mode)**: 0.435 s
- **10 Missions (Streaming)**: 4.300 s (mean: 0.430 s/mission)
- **Mean Step Execution Latency**: 2.68 ms per step (well below the 200 ms soft real-time threshold)
- **Memory Boundedness**: In streaming mode (`full_trajectory=False`), raw `TelemetryRecord` histories are omitted, keeping historical storage $O(1)$ relative to duration.

---

## 12. Verification & Integrity Audits

- **Unit & Integration Suite**: 40 tests in `tests/test_phase10_mission.py` passing 100%.
- **Phase 8 Regression**: 50 tests in `tests/test_phase8_rul.py` passing 100%.
- **Phase 9 Regression**: 49 tests in `tests/test_phase9_real_telemetry.py` passing 100%.
- **State Isolation**: Proven via `test_strong_state_isolation_a_alone_vs_after_b` ($\text{result}(B \text{ alone}) == \text{result}(B \text{ after } A)$).
- **Anti-Leakage Audit**: Static source inspection confirms `twin.update(record)` receives zero ground-truth fault labels or scenario IDs.
- **Anti-Circularity Audit**: Source inspection confirms that `simulator/` and `digital_twin/` core modules have zero imports of `mission_simulator` or `mission_types`.
