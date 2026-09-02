# SIH26054: Fault & Degradation State Interface (Phase 4A)

## 1. Executive Summary & Objective

Phase 4A establishes a clean, extensible, typed contract and timeline scheduling mechanism for engine faults and degradation states in the **SIH26054 Aero Piston Engine Digital Twin System**.

> ### ⚠️ Explicit Phase 4A Scope Disclaimer:
> **Phase 4A defines the fault/degradation contract only; it does not claim that fault behavior has been physically modeled yet.**  
> The validated healthy physics engine remains **100% unchanged**. When no fault is active, healthy physical trajectories (RPM, CHT, EGT, oil pressure, fuel flow, vibration) are identical to the Phase 3 baseline.

---

## 2. Design Architecture: Separation of Concerns

To maintain modularity and avoid monolithic conditional blocks, the fault architecture enforces strict separation of concerns across four layers:

```text
┌────────────────────────────────────────┐
│ 1. Fault Definition (FaultState)       │  (Phase 4A: Typed enums, normalized severity, metadata)
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 2. Fault Scheduling (FaultSchedule)    │  (Phase 4A: Multi-fault timeline & ramp resolution)
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 3. Subsystem Fault Physics             │  (Phase 4B - 4E: Specific subsystem degradation dynamics)
│    - Cooling Degradation               │  [Phase 4B]
│    - Lubrication Degradation           │  [Phase 4C]
│    - Fuel Injection Abnormality        │  [Phase 4D]
│    - Mechanical Degradation            │  [Phase 4E]
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 4. Telemetry Effects & Sensor Noise    │  (Phase 4F: Sensor corruption, bias, drift, dropout)
└────────────────────────────────────────┘
```

---

## 3. Supported Fault Categories

The interface categorizes faults into five planned operational failure modes via the `FaultType` enum:

| `FaultType` Enum Value | Targeted Subsystem | Physical / Data Impact | Implementation Phase |
| :--- | :--- | :--- | :---: |
| `COOLING_DEGRADATION` | `FaultSubsystem.THERMAL` | Reduced cooling conductance, radiator fouling, coolant leak | Phase 4B |
| `LUBRICATION_DEGRADATION` | `FaultSubsystem.LUBRICATION` | Oil pump degradation, oil gallery leakage, elevated oil temp | Phase 4C |
| `FUEL_INJECTION_ABNORMALITY` | `FaultSubsystem.FUEL` | Clogged injector, fuel pressure drop, uneven mixture, BSFC drift | Phase 4D |
| `MECHANICAL_DEGRADATION` | `FaultSubsystem.DYNAMICS` / `VIBRATION` | Bearing wear, mechanical friction increase, elevated 1x/2x vibration | Phase 4E |
| `SENSOR_FAULT` | `FaultSubsystem.SENSOR` | Calibration bias, thermal drift, signal freezing, intermittent packet drop | Phase 4F |

---

## 4. Severity Convention & Semantics

Fault severity is strictly normalized:
- **`severity = 0.0`**: Nominal, healthy operation (zero physical or data distortion).
- **`severity = 1.0`**: Maximum modeled degradation level for the specific fault category.

> **Important**: `severity = 1.0` does **not** assume complete catastrophic engine structural disintegration. It represents the maximum parameterized limit defined for that specific degradation model (e.g., maximum radiator blockage, maximum modeled oil pressure drop).

Severity values outside `[0.0, 1.0]` are rejected at initialization with a descriptive `ValueError`.

---

## 5. Fault State & Scheduling Contracts

### `FaultState` Data Model
```python
@dataclass
class FaultState:
    fault_type: Union[FaultType, str] = FaultType.NONE
    severity: float = 0.0
    active: bool = True
    start_time: float = 0.0
    end_time: Optional[float] = None
    affected_subsystem: Union[FaultSubsystem, str] = FaultSubsystem.NONE
    parameters: Dict[str, Any] = field(default_factory=dict)
```

Key methods:
- `is_active_at(timestamp: float) -> bool`: Returns `True` iff `active=True`, `fault_type != NONE`, `severity > 0.0`, and `start_time <= t <= end_time`.
- `get_effective_severity(timestamp: float) -> float`: Evaluates the current severity level, supporting linear ramp-up if `parameters={"ramp_duration": ...}` is specified.
- `to_dict() / from_dict()`: Full dictionary serialization for JSON configuration compatibility.

### `FaultSchedule` Multi-Fault Timeline
```python
schedule = FaultSchedule([
    FaultState(FaultType.COOLING_DEGRADATION, severity=0.4, start_time=100.0, end_time=400.0),
    FaultState(FaultType.LUBRICATION_DEGRADATION, severity=0.7, start_time=250.0, end_time=600.0),
])
```
- `get_active_faults(timestamp)`: Retrieves all concurrent faults active at simulation time $t$.
- `get_primary_fault(timestamp)`: Identifies the dominant active fault based on effective severity.

---

## 6. Physical Faults vs. Sensor Faults

A fundamental principle of the digital twin architecture is the clean separation between **physical state degradation** and **instrumentation/sensor corruption**:

1. **Physical Engine Faults** (Phases 4B–4E):
   - Alter real underlying state variables (e.g., higher true CHT, lower true oil pressure, increased mechanical vibration).
   - Cross-channel physical relationships remain thermodynamically coherent (e.g., reduced cooling elevates both CHT and oil temp).
2. **Sensor Faults** (Phase 4F):
   - Inject artifacts solely at the measurement and telemetry layer (`TelemetryGenerator`).
   - The true underlying engine states remain healthy, while the transmitted telemetry channel exhibits bias, drift, or freezing.
   - Crucial for testing whether the Digital Twin and PHM algorithms can distinguish between an actual physical engine problem and an instrument sensor failure.

---

## 7. Backward Compatibility Guarantee

The healthy engine simulation behavior is guaranteed:
```python
# All three invocations produce identical physical engine telemetry:
rec_healthy1 = sim.step(mission_config)
rec_healthy2 = sim.step(mission_config, fault_state=None)
rec_healthy3 = sim.step(mission_config, fault_state=FaultState(active=False))
```
- **Automated Verification**: 12 dedicated regression tests in `tests/test_fault_interface.py` certify that healthy trajectories (RPM, CHT, EGT, oil pressure, fuel flow, vibration) remain identical.
- **Full Suite**: All 50 tests pass (`38 existing + 12 new Phase 4A tests`).
