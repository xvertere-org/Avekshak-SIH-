# Phase 9 — Real Telemetry Readiness Engineering Specification

## 1. Executive Summary & Authoritative Boundary

Phase 9 establishes the architectural capability for the SIH26054 Digital Twin pipeline to ingest, validate, normalize, timestamp, replay, and route telemetry from external data sources.

> [!IMPORTANT]
> **Explicit Real-World Disclaimer**:
> **No real Rotax 914 flight telemetry was used or validated in Phase 9.**
> All validation datasets, golden replays, and benchmark streams in this phase are strictly **synthetic**, **simulated**, or **replayed**. Phase 9 establishes real-telemetry ingestion readiness at the software boundary; it does not claim certified airworthiness, flight clearance, or operational validation on an aircraft.

---

## 2. Ingestion Pipeline Architecture

The telemetry ingestion layer decouples physical transport and data representation from engine physics:

```text
RAW EXTERNAL TELEMETRY (JSON / NDJSON / CSV)
            ↓
    SOURCE ADAPTER (JSONReplayAdapter, CSVReplayAdapter)
            ↓
    BOUNDARY & SCHEMA VALIDATOR (BoundaryValidator)
       ├── Non-numeric / NaN / Inf quarantine
       ├── Sequence number auditing (duplicate / skipped / wraparound)
       └── Clock skew detection (negative latency tracking)
            ↓
    EXPLICIT UNIT NORMALIZER (UnitConverter)
       ├── Pa / hPa / psi / inHg → bar
       ├── K / °F → °C
       ├── Mass-to-volumetric flow (requires explicit rho_fuel parameter)
       └── Acceleration g → m/s² (velocity conversion prohibited)
            ↓
    MULTI-RATE ALIGNER & QUALITY AUDITOR (MultiRateBuffer)
       ├── Channel-specific staleness timeout (tau_stale)
       ├── Held-last-value marked ESTIMATED / HELD_LAST_VALUE
       └── Bounded interpolation marked ESTIMATED / INTERPOLATED
            ↓
    CANONICAL TELEMETRY PACKET (CanonicalTelemetryPacket)
            ↓ [Lossless / Validated Interoperability Bridge]
    DIGITAL TWIN SYNCHRONIZER (StateEstimator.step)
            ↓
    PHYSICS / RESIDUALS / HEALTH / DIAGNOSIS / RUL
```

---

## 3. Canonical Telemetry Contract

### 3.1 Core Classes (`telemetry/canonical.py`)
- **`CanonicalMeasurement`**: Strongly typed observation of an individual channel:
  - `channel_name: str`
  - `value: Optional[float]` (in canonical SI/engineering unit)
  - `unit: str` (e.g. `bar`, `degC`, `L/h`, `g`, `m`, `RPM`)
  - `raw_value: Optional[Any]` (immutable original representation)
  - `raw_unit: Optional[str]`
  - `timestamp: float` (canonical synchronization coordinate)
  - `source_timestamp: Optional[float]` (sensor acquisition time)
  - `ingest_timestamp: Optional[float]` (boundary reception time)
  - `latency: Optional[float]` ($t_{\text{ingest}} - t_{\text{source}}$)
  - `sequence_num: Optional[int]`
  - `source_id: str`
  - `source_type: SourceType` (`SIMULATOR`, `REPLAY`, `REAL_SENSOR`, `DERIVED`, `DEFAULT_ASSUMED`)
  - `quality: DataQualityStatus` (`VALID`, `MISSING`, `NON_FINITE`, `OUT_OF_RANGE`, `STALE`, etc.)
  - `status: QuantityStatus` (`MEASURED`, `ESTIMATED`, `PREDICTED`, `DERIVED`, `DEFAULT_ASSUMED`, `UNAVAILABLE`)
  - `sensor_id: Optional[str]`
  - `calibration: Optional[CalibrationMetadata]`
  - `filtered_value: Optional[float]`
  - `notes: str`

- **`CanonicalTelemetryPacket`**: An atomic collection of synchronized or simultaneous channel measurements.
  - Bridge methods:
    - `to_telemetry_record() -> TelemetryRecord`: Lossless and validated bridge for downstream synchronizer consumption.
    - `from_telemetry_record(record: TelemetryRecord) -> CanonicalTelemetryPacket`: Ingests simulator records into canonical structure with zero physics loss.

### 3.2 Canonical Channels and Standard Units
| Channel Category | Channel Name | Canonical Unit | Alternate Inputs Supported |
|:---|:---|:---|:---|
| **Rotational** | `rpm`, `engine_rpm`, `propeller_rpm` | `RPM` | `rad/s`, `Hz` |
| **Air / Boost** | `map_bar`, `map`, `ambient_pressure` | `bar` | `Pa`, `hPa`, `kPa`, `psi`, `inHg` |
| **Thermal** | `cht`, `cht_cyl1`..`4`, `egt`, `oil_temp`, `coolant_temp` | `degC` | `K`, `degF` |
| **Lubrication** | `oil_pressure` | `bar` | `psi`, `kPa`, `Pa` |
| **Fuel** | `fuel_flow` (volumetric) | `L/h` | `m3/s`, `gpm`, or mass flow with explicit $\rho_{\text{fuel}}$ |
| **Vibration** | `vibration` (broadband RMS acceleration) | `g` | `m/s²`, `cm/s²` |
| **Environment** | `altitude` | `m` | `ft`, `km` |
| **Controls** | `throttle`, `load` | `%` | `fraction` (0.0 to 1.0) |

---

## 4. Dimensional Safety & Unit Conversions (`telemetry/units.py`)

### 4.1 Strict Dimensional Separation
Cross-dimensional conversions (e.g. converting `bar` to `degC`) raise `UnitConversionError`.

### 4.2 Vibration Dimensionality Protection
- Acceleration and velocity are distinct physical dimensions ($[L\cdot T^{-2}]$ vs $[L\cdot T^{-1}]$).
- Conversions between $g$ and $\text{m/s}^2$ are supported for acceleration.
- Direct scalar conversions between $g$ and $\text{mm/s}$ are strictly prohibited and raise `UnitConversionError`.

### 4.3 Fuel Density Assumption & Provenance
- Volumetric flow ($\text{L/h}$) is the canonical Rotax model representation.
- Ingestion of mass flow ($\text{kg/s}$ or $\text{kg/h}$) requires an explicit operational density parameter $\rho_{\text{fuel}}$ (e.g. $0.72\,\text{kg/L}$).
- The resulting measurement is tagged `QuantityStatus.DERIVED` with explicit provenance:
  ```text
  DERIVED_FROM_MASS_FLOW: assumed rho_fuel=0.7200 kg/L
  ```

---

## 5. Timestamp, Clock & Sequence Semantics (`telemetry/validator.py`)

### 5.1 Timestamp Epistemic Domains
- `source_timestamp`: Sensor acquisition time at source. Authoritative for Digital Twin synchronization when valid.
- `ingest_timestamp`: Time packet arrived at digital twin boundary.
- `latency`: $\Delta t = t_{\text{ingest}} - t_{\text{source}}$.

### 5.2 Clock Skew Handling
- If $\Delta t < -0.05\,\text{s}$ (ingest time precedes source time), it indicates clock skew or timestamp domain mismatch.
- Flagged as `ClockStatus.CLOCK_SKEW`.
- Clock diagnostics are strictly isolated from engine health; communication delays or clock anomalies are not diagnosed as engine mechanical failures.

### 5.3 Sequence Number Auditing
- Detects `NORMAL`, `DUPLICATE`, `SKIPPED`, and `OUT_OF_ORDER`.
- Wraparound is only evaluated if `sequence_modulus` is explicitly configured.
- If modulus is unknown and counter decrements, classified as `CONFIGURATION_UNKNOWN` and flagged as a discontinuity.

---

## 6. Multi-Rate Telemetry & Sample Alignment (`telemetry/multi_rate.py`)

### 6.1 Multi-Rate Buffer
Accommodates asynchronous channels arriving at distinct rates (e.g., $50\,\text{Hz}$ RPM, $5\,\text{Hz}$ CHT, $1\,\text{Hz}$ altitude).

### 6.2 Held-Last-Value Epistemic Status
- When a preceding sample is held during a faster alignment step, its status is explicitly marked:
  ```text
  QuantityStatus.ESTIMATED (notes: "HELD_LAST_VALUE")
  ```
- Held samples are **never** relabeled `MEASURED` at the query timestamp.

### 6.3 Maximum Sample Age (`tau_stale`)
- Reuses Phase 3 `tau_stale = 2.0 s` baseline.
- Channel-specific stale timeouts:
  - Dynamic channels (RPM, vibration): $0.5\,\text{s}$
  - Thermal channels (CHT, EGT): $2.0-3.0\,\text{s}$
  - Environmental channels (Altitude, ambient temp): $5.0\,\text{s}$
- Samples older than `tau_stale` evaluate to `DataQualityStatus.STALE` / `QuantityStatus.UNAVAILABLE`.

### 6.4 Bounded Interpolation
- Bounded linear interpolation is supported only if enabled and gap $\le \text{max\_interpolation\_gap}$ ($1.0\,\text{s}$).
- Prohibited across invalid samples, large dropouts, or fault boundaries.
- Interpolated values are marked `QuantityStatus.ESTIMATED (notes: "INTERPOLATED")`.

---

## 7. Replay Engine & Source Adapters (`telemetry/replay.py`, `telemetry/adapters.py`)

### 7.1 Replay Modes & Memory Architecture
1. **Streaming Mode (NDJSON / CSV Stream)**:
   - Line-by-line generator yielding packets sequentially.
   - Bounded $O(1)$ working buffer memory.
   - Configurable playback rate or offline deterministic stepping.
2. **Materialized Mode (JSON Array / DataFrame)**:
   - In-memory collection of all records for analytical slicing ($O(N)$ memory).

### 7.2 Source Adapters
- **`JSONReplayAdapter`**: Parses versioned external interchange JSON and NDJSON formats (`schema_version="1.0"`).
- **`CSVReplayAdapter`**: Maps arbitrary CSV columns, units, and sensor IDs to canonical channels.

### 7.3 Golden Replay Dataset (`tests/data/golden_telemetry.json`)
Deterministic synthetic reference mission containing:
- Steady cruise ($t=0..2$)
- Throttle step transient ($t=3$)
- Climb transient ($t=4$)
- Thermal lag transient ($t=5$)
- Intentional packet dropout ($t=6$ dropped, $t=7$ recovery)
- Transient vibration spike outlier ($t=8$, $18.5\,g$)
- Steady recovery and cruise ($t=9..10$)
Strictly labeled `dataset_class: "SYNTHETIC"`, `provenance: "SYNTHETIC_REPLAY"`.

---

## 8. Anti-Circularity & Anti-Label-Leakage Guardrails

### 8.1 Static Dependency Integrity
Telemetry ingestion modules (`telemetry/canonical.py`, `telemetry/validator.py`, `telemetry/units.py`, `telemetry/multi_rate.py`, `telemetry/replay.py`, `telemetry/adapters.py`) have **zero imports** from:
- `digital_twin.health`
- `digital_twin.diagnosis`
- `digital_twin.degradation`
- `digital_twin.rul`

### 8.2 Label Leakage Protection
The boundary validator and adapters never ingest or condition on simulated ground-truth labels (`fault_type`, `fault_severity`, `failure_time`). Measurements are processed purely on physical signal merit.

---

## 9. Limitations & Future Extensions

1. **No Real Flight Dataset**: As stated, all validation is performed using deterministic synthetic replays.
2. **Transport Decoupling**: Live physical transport protocols (MAVLink, CAN bus, UDP telemetry stream, serial RS-422) are transport-level concerns and are not implemented as fake network sockets. External adapters interface via file streams or standardized in-memory dictionaries.
3. **Hardware Clock Synchronization**: True hardware-in-the-loop multi-sensor clock synchronization (e.g., PTP IEEE 1588 or GPS PPS pulse) requires hardware DAQ support and is modeled here as software latency $\Delta t$.
