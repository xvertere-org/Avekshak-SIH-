"""
Telemetry Generator and Sensor Noise Model for SIH26054.

Transforms internal physical subsystem states into schema-compliant TelemetryRecords
with calibrated sensor measurement noise and complete provenance metadata.

Phase 4F: Integrates SensorFaultProcessor for observation-layer sensor faults.
Sensor faults corrupt telemetry values WITHOUT modifying physical engine state.
"""

import math
import numpy as np
from typing import Optional, Dict, Any, List
from simulator.config import SimulatorConfig
from simulator.subsystems.dynamics import OperatingPoint
from simulator.subsystems.thermal import ThermalState
from simulator.subsystems.lubrication import LubricationState
from simulator.subsystems.fuel import FuelState
from simulator.subsystems.vibration import VibrationState
from simulator.subsystems.atmosphere import ISAState
from simulator.subsystems.mission import MissionStep
from simulator.sensor_faults import SensorFaultProcessor
from telemetry.schema import TelemetryRecord, FaultCategory


class TelemetryGenerator:
    """
    Synthesizes TelemetryRecord instances from physical subsystem states
    and applies calibrated sensor measurement noise.

    Phase 4F: Optionally applies sensor faults at the observation layer
    after normal sensor noise but before record construction.
    """

    def __init__(
        self,
        config: SimulatorConfig = SimulatorConfig(),
        rng: Optional[np.random.Generator] = None,
        apply_sensor_noise: bool = True,
    ):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.random_seed)
        self.apply_sensor_noise = apply_sensor_noise
        # Phase 4F: Sensor fault processor uses a SEPARATE RNG that does NOT
        # consume any draws from the main sensor-noise RNG. This guarantees
        # that golden healthy telemetry remains bitwise unchanged when no
        # sensor faults are active.
        # We derive a child seed from the main RNG's bit_generator state
        # using SeedSequence.spawn() so it's deterministic but independent.
        parent_ss = self.rng.bit_generator.seed_seq
        if parent_ss is not None:
            child_ss = parent_ss.spawn(1)[0]
        else:
            # Fallback: use a fixed offset from config seed
            child_ss = np.random.SeedSequence(config.random_seed + 7919)
        self._sensor_fault_rng = np.random.default_rng(child_ss)
        self.sensor_fault_processor = SensorFaultProcessor(
            rng=self._sensor_fault_rng
        )

    def generate(
        self,
        mission_step: MissionStep,
        atmo_state: ISAState,
        op_point: OperatingPoint,
        fuel_state: FuelState,
        thermal_state: ThermalState,
        lub_state: LubricationState,
        vib_state: VibrationState,
        engine_id: str = "ENGINE_UAV_01",
        mission_id: str = "MISSION_MALE_UAV_001",
        fault_type: str = FaultCategory.NONE.value,
        fault_severity: float = 0.0,
        sensor_faults: Optional[List[Any]] = None,
    ) -> TelemetryRecord:
        """
        Produce a single TelemetryRecord from subsystem states.

        Args:
            sensor_faults: Optional list of FaultState objects with
                           fault_type == SENSOR_FAULT. Applied after
                           normal sensor noise at the observation layer.
        """
        # Physical truth values
        rpm_val = op_point.rpm
        cht_val = thermal_state.cht_c
        egt_val = thermal_state.egt_c
        oil_temp_val = lub_state.oil_temp_c
        oil_press_val = lub_state.oil_pressure_bar
        fuel_flow_val = fuel_state.volumetric_flow_l_h
        vib_val = vib_state.rms_g

        # Apply calibrated sensor measurement noise (Tier C)
        if self.apply_sensor_noise:
            rpm_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_rpm))
            cht_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_cht))
            egt_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_egt))
            oil_temp_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_oil_temp))
            oil_press_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_oil_press))
            fuel_flow_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_fuel_flow))
            vib_val += float(self.rng.normal(0.0, self.config.tier_c.sensor_noise_vibration))

        # Phase 4F: Apply sensor faults at the observation layer
        # When sensor_faults is None or empty, this consumes zero RNG draws
        # and channel values are unchanged, preserving bitwise golden telemetry.
        if sensor_faults:
            channel_values = {
                "rpm": rpm_val,
                "cht": cht_val,
                "egt": egt_val,
                "oil_temp": oil_temp_val,
                "oil_pressure": oil_press_val,
                "fuel_flow": fuel_flow_val,
                "vibration": vib_val,
            }
            corrupted = self.sensor_fault_processor.apply(
                channel_values=channel_values,
                fault_states=sensor_faults,
                timestamp=mission_step.timestamp_s,
            )
            rpm_val = corrupted["rpm"]
            cht_val = corrupted["cht"]
            egt_val = corrupted["egt"]
            oil_temp_val = corrupted["oil_temp"]
            oil_press_val = corrupted["oil_pressure"]
            fuel_flow_val = corrupted["fuel_flow"]
            vib_val = corrupted["vibration"]

        # Clamp physical non-negatives (NaN-safe: dropout NaN must NOT become 0)
        if not math.isnan(rpm_val):
            rpm_val = max(0.0, rpm_val)
        if not math.isnan(oil_press_val):
            oil_press_val = max(0.0, oil_press_val)
        if not math.isnan(fuel_flow_val):
            fuel_flow_val = max(0.0, fuel_flow_val)
        if not math.isnan(vib_val):
            vib_val = max(0.0, vib_val)

        # Build telemetry record
        record = TelemetryRecord(
            timestamp=round(mission_step.timestamp_s, 3),
            mission_id=mission_id,
            engine_id=engine_id,
            mission_phase=mission_step.phase.value if hasattr(mission_step.phase, "value") else str(mission_step.phase),
            altitude=round(mission_step.altitude_m, 1),
            ambient_temp=round(atmo_state.temperature_c, 2),
            throttle=round(mission_step.throttle_pct, 1),
            load=round(op_point.engine_load_pct, 1),
            rpm=round(rpm_val, 1) if not math.isnan(rpm_val) else float('nan'),
            cht=round(cht_val, 2) if not math.isnan(cht_val) else float('nan'),
            egt=round(egt_val, 2) if not math.isnan(egt_val) else float('nan'),
            oil_temp=round(oil_temp_val, 2) if not math.isnan(oil_temp_val) else float('nan'),
            oil_pressure=round(oil_press_val, 3) if not math.isnan(oil_press_val) else float('nan'),
            fuel_flow=round(fuel_flow_val, 2) if not math.isnan(fuel_flow_val) else float('nan'),
            vibration=round(vib_val, 3) if not math.isnan(vib_val) else float('nan'),
            fault_type=fault_type,
            fault_severity=round(fault_severity, 3),
            source="simulator_v1_physics",
            source_type="synthetic",
            simulation_version=self.config.provenance_version,
            metadata={
                "power_kw": round(op_point.power_target_w / 1000.0, 2),
                "torque_nm": round(op_point.torque_engine_nm, 2),
                "density_factor": round(atmo_state.density_factor, 4),
                "order_1x_freq_hz": round(vib_state.order_1x_freq_hz, 2),
                "order_2x_freq_hz": round(vib_state.order_2x_freq_hz, 2),
                "bsfc_g_kwh": round(fuel_state.bsfc_g_kwh, 1),
            },
        )
        return record
