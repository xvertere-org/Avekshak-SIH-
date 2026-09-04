"""
Synthetic fault-diagnosis dataset generation for Phase 8.

Generates labeled mission runs from the existing physics-informed simulator,
producing ResidualFrame data with ground-truth fault_type labels and unique
mission_run_id per independent trajectory.

Generation matrix (121 runs total):
- NONE:                       6 runs (6 seeds)
- COOLING_DEGRADATION:        6 runs (3 severities x 2 seeds)
- LUBRICATION_DEGRADATION:    6 runs (3 severities x 2 seeds)
- FUEL_INJECTION_ABNORMALITY: 6 runs (3 severities x 2 seeds)
- MECHANICAL_DEGRADATION:     6 runs (3 severities x 2 seeds)
- SENSOR_FAULT:              91 runs (BIAS/DRIFT/NOISE: 7ch x 3sev,
                                       STUCK/DROPOUT: 7ch x 2seeds)

Sensor channel and mode are GENERATION FACTORS, never diagnosis labels.
The XGBoost target for all sensor-fault runs is "sensor_fault".
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultState, FaultType, FaultSchedule
from simulator.sensor_faults import SensorFaultMode, SensorChannel
from digital_twin.twin_model import DigitalTwin
from telemetry.ingestion import CanonicalTelemetryFrame


# Channels for sensor fault generation
SENSOR_CHANNELS: List[str] = [sc.value for sc in SensorChannel]

# Sensor modes
SENSOR_MODES: List[str] = [sm.value for sm in SensorFaultMode]

# Severity levels for physical faults
PHYSICAL_SEVERITIES: List[float] = [0.3, 0.5, 0.7]

# Seed pairs for physical faults (indexed by severity)
PHYSICAL_SEED_PAIRS: Dict[float, List[int]] = {
    0.3: [42, 100],
    0.5: [200, 300],
    0.7: [400, 500],
}

# Seeds for healthy runs
HEALTHY_SEEDS: List[int] = [42, 100, 200, 300, 400, 500]

# Severity-meaningful sensor modes (magnitude scales with severity)
SEVERITY_MEANINGFUL_MODES: List[str] = [
    SensorFaultMode.BIAS.value,
    SensorFaultMode.DRIFT.value,
    SensorFaultMode.NOISE.value,
]

# Binary-effect sensor modes (severity is on/off, varied by seed instead)
BINARY_EFFECT_MODES: List[str] = [
    SensorFaultMode.STUCK.value,
    SensorFaultMode.DROPOUT.value,
]

# Sensor severity levels for severity-meaningful modes
SENSOR_SEVERITIES: List[float] = [0.3, 0.5, 0.7]

# Seeds for binary-effect sensor modes (2 seeds per channel)
BINARY_SENSOR_SEEDS: List[int] = [42, 100]


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    dt: float = 1.0
    fault_onset_time: float = 0.0
    sensor_fault_severity_default: float = 0.7
    healthy_seeds: List[int] = field(default_factory=lambda: list(HEALTHY_SEEDS))
    physical_severities: List[float] = field(
        default_factory=lambda: list(PHYSICAL_SEVERITIES)
    )
    physical_seed_pairs: Dict[float, List[int]] = field(
        default_factory=lambda: {k: list(v) for k, v in PHYSICAL_SEED_PAIRS.items()}
    )
    sensor_severities: List[float] = field(
        default_factory=lambda: list(SENSOR_SEVERITIES)
    )
    binary_sensor_seeds: List[int] = field(
        default_factory=lambda: list(BINARY_SENSOR_SEEDS)
    )
    mission_profile: Optional[Any] = None


def _run_mission(
    seed: int,
    fault_state: Optional[FaultState] = None,
    dt: float = 1.0,
    mission_profile: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Run a single simulator mission and return the ResidualFrame as a DataFrame.

    Args:
        seed: Simulator random seed for this run.
        fault_state: Optional FaultState for fault injection.
        dt: Time step in seconds.
        mission_profile: Optional custom MissionProfile.

    Returns:
        DataFrame from ResidualFrame with all residual and context columns.
    """
    sim = EngineSimulator(seed=seed)
    twin = DigitalTwin()

    if fault_state is not None:
        records = sim.run(mission_profile=mission_profile, dt=dt, fault_schedule=fault_state)
    else:
        records = sim.run(mission_profile=mission_profile, dt=dt)

    df_tel = pd.DataFrame([r.to_dict() for r in records])
    frame = CanonicalTelemetryFrame(df_tel)
    rframe = twin.process_frame(frame)
    return rframe.to_dataframe()


def generate_fault_diagnosis_dataset(
    config: Optional[DatasetConfig] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate the complete Phase 8 fault-diagnosis dataset.

    Returns a single DataFrame containing all mission runs with columns:
    - All ResidualFrame columns (observed, expected, residuals, normalized residuals)
    - fault_type: ground-truth label (FaultType.value string)
    - mission_run_id: unique identifier for each independent mission run
    - generation metadata: sensor_channel, sensor_mode, severity (for analysis only)

    The fault_type column is the ONLY training target.
    sensor_channel, sensor_mode, and severity are generation metadata and
    must NEVER be used as classifier features.
    """
    config = config or DatasetConfig()
    all_runs: List[pd.DataFrame] = []
    run_count = 0

    # ---- 1. Healthy (NONE) runs ----
    for seed in config.healthy_seeds:
        run_id = f"none_s{seed}"
        if verbose:
            print(f"  Generating {run_id}...")
        df = _run_mission(seed=seed, dt=config.dt, mission_profile=config.mission_profile)
        df["mission_run_id"] = run_id
        df["fault_type"] = "none"
        df["generation_severity"] = 0.0
        df["generation_sensor_channel"] = ""
        df["generation_sensor_mode"] = ""
        all_runs.append(df)
        run_count += 1

    # ---- 2. Physical fault runs ----
    physical_fault_types = [
        FaultType.COOLING_DEGRADATION,
        FaultType.LUBRICATION_DEGRADATION,
        FaultType.FUEL_INJECTION_ABNORMALITY,
        FaultType.MECHANICAL_DEGRADATION,
    ]
    for ft in physical_fault_types:
        for severity in config.physical_severities:
            for seed in config.physical_seed_pairs[severity]:
                run_id = f"{ft.value}_sev{severity}_s{seed}"
                if verbose:
                    print(f"  Generating {run_id}...")

                fs = FaultState(
                    fault_type=ft,
                    severity=severity,
                    start_time=config.fault_onset_time,
                )
                # Fuel injection: add mode parameter
                if ft == FaultType.FUEL_INJECTION_ABNORMALITY:
                    fs.parameters["mode"] = "lean"

                df = _run_mission(seed=seed, fault_state=fs, dt=config.dt, mission_profile=config.mission_profile)
                df["mission_run_id"] = run_id
                df["fault_type"] = ft.value
                df["generation_severity"] = severity
                df["generation_sensor_channel"] = ""
                df["generation_sensor_mode"] = ""
                all_runs.append(df)
                run_count += 1

    # ---- 3. Sensor fault runs ----
    # 3a. Severity-meaningful modes (BIAS, DRIFT, NOISE)
    for mode in SEVERITY_MEANINGFUL_MODES:
        for ch_idx, channel in enumerate(SENSOR_CHANNELS):
            for sev_idx, severity in enumerate(config.sensor_severities):
                # Deterministic seed from (channel_idx, mode, severity_idx)
                seed = 1000 + ch_idx * 100 + SENSOR_MODES.index(mode) * 10 + sev_idx
                run_id = f"sensor_{channel}_{mode}_sev{severity}_s{seed}"
                if verbose:
                    print(f"  Generating {run_id}...")

                fs = FaultState(
                    fault_type=FaultType.SENSOR_FAULT,
                    severity=severity,
                    start_time=config.fault_onset_time,
                    parameters={
                        "sensor_channel": channel,
                        "sensor_mode": mode,
                    },
                )
                df = _run_mission(seed=seed, fault_state=fs, dt=config.dt, mission_profile=config.mission_profile)
                df["mission_run_id"] = run_id
                df["fault_type"] = "sensor_fault"
                df["generation_severity"] = severity
                df["generation_sensor_channel"] = channel
                df["generation_sensor_mode"] = mode
                all_runs.append(df)
                run_count += 1

    # 3b. Binary-effect modes (STUCK, DROPOUT)
    for mode in BINARY_EFFECT_MODES:
        for ch_idx, channel in enumerate(SENSOR_CHANNELS):
            for seed_idx, base_seed in enumerate(config.binary_sensor_seeds):
                seed = 2000 + ch_idx * 100 + SENSOR_MODES.index(mode) * 10 + seed_idx
                run_id = f"sensor_{channel}_{mode}_sev{config.sensor_fault_severity_default}_s{seed}"
                if verbose:
                    print(f"  Generating {run_id}...")

                fs = FaultState(
                    fault_type=FaultType.SENSOR_FAULT,
                    severity=config.sensor_fault_severity_default,
                    start_time=config.fault_onset_time,
                    parameters={
                        "sensor_channel": channel,
                        "sensor_mode": mode,
                    },
                )
                df = _run_mission(seed=seed, fault_state=fs, dt=config.dt, mission_profile=config.mission_profile)
                df["mission_run_id"] = run_id
                df["fault_type"] = "sensor_fault"
                df["generation_severity"] = config.sensor_fault_severity_default
                df["generation_sensor_channel"] = channel
                df["generation_sensor_mode"] = mode
                all_runs.append(df)
                run_count += 1

    if verbose:
        print(f"  Total runs generated: {run_count}")

    # Concatenate all runs
    dataset = pd.concat(all_runs, ignore_index=True)

    # Ensure fault_type is string
    dataset["fault_type"] = dataset["fault_type"].astype(str)

    return dataset


def get_dataset_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a summary of the generated dataset.

    Returns dict with per-class run counts, sample counts, severity/channel coverage.
    """
    summary: Dict[str, Any] = {
        "total_samples": len(dataset),
        "total_runs": dataset["mission_run_id"].nunique(),
        "classes": {},
    }

    for fault_type in sorted(dataset["fault_type"].unique()):
        class_data = dataset[dataset["fault_type"] == fault_type]
        class_runs = class_data["mission_run_id"].unique()
        class_info: Dict[str, Any] = {
            "runs": len(class_runs),
            "samples": len(class_data),
        }

        if fault_type == "sensor_fault":
            sensor_data = class_data[class_data["generation_sensor_channel"] != ""]
            if len(sensor_data) > 0:
                channels = sorted(sensor_data["generation_sensor_channel"].unique())
                modes = sorted(sensor_data["generation_sensor_mode"].unique())
                class_info["sensor_channels"] = channels
                class_info["sensor_modes"] = modes
                class_info["channel_count"] = len(channels)
                class_info["mode_count"] = len(modes)

        severities = class_data["generation_severity"].unique()
        class_info["severities"] = sorted([float(s) for s in severities if float(s) > 0])

        summary["classes"][fault_type] = class_info

    return summary
