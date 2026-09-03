"""
Physics-Informed Aero Piston Engine Simulator Orchestrator for SIH26054.

Coordinates atmosphere, mission, rotational dynamics, fuel, thermal, lubrication,
and vibration subsystems to produce physically correlated telemetry streams.

DISCLAIMER:
Uses the Rotax 912 ULS ONLY as a publicly documented engineering reference anchor.
This simulator is a reduced-order lumped-parameter grey-box model and does NOT represent
a certified OEM engine model or actual classified MALE-UAV propulsion system.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
import pandas as pd

from simulator.base import BaseEngineSimulator
from simulator.config import SimulatorConfig
from simulator.subsystems.atmosphere import Atmosphere
from simulator.subsystems.mission import MissionProfile, MissionStep, FlightPhase
from simulator.subsystems.dynamics import RotationalDynamics
from simulator.subsystems.fuel import FuelSystem
from simulator.subsystems.thermal import ThermalSystem
from simulator.subsystems.lubrication import LubricationSystem
from simulator.subsystems.vibration import VibrationSystem
from simulator.telemetry_generator import TelemetryGenerator
from telemetry.schema import MissionConfig, EngineConfig, TelemetryRecord, FaultCategory


class EngineSimulator(BaseEngineSimulator):
    """
    Modular physics-informed grey-box engine simulator.
    Supports both discrete streaming steps and batch mission execution.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        sim_config: Optional[SimulatorConfig] = None,
        seed: Optional[int] = None,
    ):
        if engine_config is None:
            engine_config = EngineConfig()
        super().__init__(engine_config)

        self.sim_config = sim_config or SimulatorConfig()
        if seed is not None:
            self.sim_config.random_seed = seed

        self.rng = np.random.default_rng(self.sim_config.random_seed)
        self.current_time_s = 0.0

        # Instantiate physical subsystems
        self.atmosphere = Atmosphere(tier_a=self.sim_config.tier_a)
        self.dynamics = RotationalDynamics(
            tier_a=self.sim_config.tier_a,
            tier_c=self.sim_config.tier_c,
            tier_d=self.sim_config.tier_d,
            initial_rpm=self.sim_config.tier_c.rpm_idle,
        )
        self.fuel = FuelSystem(tier_c=self.sim_config.tier_c)
        self.thermal = ThermalSystem(
            tier_a=self.sim_config.tier_a,
            tier_c=self.sim_config.tier_c,
        )
        self.lubrication = LubricationSystem(
            tier_a=self.sim_config.tier_a,
            tier_c=self.sim_config.tier_c,
        )
        self.vibration = VibrationSystem(
            tier_c=self.sim_config.tier_c,
            tier_d=self.sim_config.tier_d,
            rng=self.rng,
        )
        self.telemetry_gen = TelemetryGenerator(
            config=self.sim_config,
            rng=self.rng,
            apply_sensor_noise=True,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal simulator states to initial baseline conditions."""
        if seed is not None:
            self.sim_config.random_seed = seed
        self.rng = np.random.default_rng(self.sim_config.random_seed)
        self.current_time_s = 0.0

        self.dynamics.set_rpm(self.sim_config.tier_c.rpm_idle)
        self.thermal.set_states(cht_c=85.0, egt_c=580.0)
        self.lubrication.set_oil_temp(65.0)
        self.vibration.phase_1 = 0.0
        self.vibration.phase_2 = 0.0

        # Phase 4F: Clear sensor fault stateful tracking (STUCK latches, etc.)
        self.telemetry_gen.sensor_fault_processor.reset()

    def step(
        self,
        mission_config: Optional[Union[MissionConfig, MissionStep]] = None,
        time_step: Optional[float] = None,
        mission_input: Optional[Union[MissionConfig, MissionStep]] = None,
        fault_state: Optional[Dict[str, Any]] = None,
    ) -> TelemetryRecord:
        """
        Advance the simulation by one discrete time step.

        Args:
            mission_config: MissionConfig or MissionStep with current flight demands.
            time_step: Simulation time increment (seconds). Defaults to config default_dt.
            mission_input: Alias for mission_config.
            fault_state: Placeholder hook for future Phase 4 fault injection.

        Returns:
            Fully populated and schema-compliant TelemetryRecord.
        """
        inp = mission_config if mission_config is not None else mission_input
        dt = time_step if time_step is not None else self.sim_config.default_dt
        self.current_time_s += dt

        # Normalize mission inputs
        if isinstance(inp, MissionStep):
            throttle_pct = inp.throttle_pct
            altitude_m = inp.altitude_m
            airspeed_ms = inp.airspeed_ms
            temp_offset_k = inp.temp_offset_k
            mission_phase_val = inp.phase.value if hasattr(inp.phase, "value") else str(inp.phase)
            mission_id_val = "MISSION_MALE_UAV_001"
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = inp.timestamp_s
        elif isinstance(inp, MissionConfig):
            throttle_pct = inp.throttle
            altitude_m = inp.altitude
            airspeed_ms = 45.0  # nominal proxy
            temp_offset_k = inp.ambient_temperature - 15.0
            mission_phase_val = inp.mission_phase
            mission_id_val = inp.mission_id
            # FIX #2 (Pre-4E Audit): MissionConfig.fault_type / fault_severity are Phase 1
            # schema stubs. Telemetry fault labels must reflect actual physics, not requested
            # labels. Physical fault injection is exclusively controlled via the fault_state
            # parameter using FaultState / FaultSchedule (Phase 4A contract).
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = self.current_time_s
        else:
            throttle_pct = 75.0
            altitude_m = 3000.0
            airspeed_ms = 45.0
            temp_offset_k = 0.0
            mission_phase_val = FlightPhase.CRUISE.value
            mission_id_val = "MISSION_MALE_UAV_001"
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = self.current_time_s

        cooling_severity = 0.0
        lubrication_severity = 0.0
        fuel_severity = 0.0
        mechanical_severity = 0.0
        mixture_mode = "lean"
        sensor_fault_list = []  # Phase 4F: sensor faults routed to telemetry layer
        # Resolve fault state and telemetry fault tags if provided
        if fault_state is not None:
            from simulator.fault_interface import FaultState, FaultSchedule, FaultType
            if isinstance(fault_state, FaultState):
                if fault_state.fault_type == FaultType.SENSOR_FAULT:
                    # Phase 4F: sensor faults bypass physics entirely
                    sensor_fault_list.append(fault_state)
                    if fault_state.is_active_at(step_time):
                        fault_type_val = fault_state.fault_type.value if hasattr(fault_state.fault_type, "value") else str(fault_state.fault_type)
                        fault_sev_val = fault_state.get_effective_severity(step_time)
                elif fault_state.is_active_at(step_time):
                    fault_type_val = fault_state.fault_type.value if hasattr(fault_state.fault_type, "value") else str(fault_state.fault_type)
                    fault_sev_val = fault_state.get_effective_severity(step_time)
                    if fault_state.fault_type == FaultType.COOLING_DEGRADATION:
                        cooling_severity = fault_sev_val
                    elif fault_state.fault_type == FaultType.LUBRICATION_DEGRADATION:
                        lubrication_severity = fault_sev_val
                    elif fault_state.fault_type == FaultType.FUEL_INJECTION_ABNORMALITY:
                        fuel_severity = fault_sev_val
                        mode_p = fault_state.parameters.get("mode", "lean") if fault_state.parameters else "lean"
                        mixture_mode = mode_p.value if hasattr(mode_p, "value") else str(mode_p)
                    elif fault_state.fault_type == FaultType.MECHANICAL_DEGRADATION:
                        mechanical_severity = fault_sev_val
            elif isinstance(fault_state, FaultSchedule):
                # Separate sensor faults from physical faults
                for f in fault_state.get_active_faults(step_time):
                    if f.fault_type == FaultType.SENSOR_FAULT:
                        sensor_fault_list.append(f)
                # Also include inactive sensor faults for STUCK latch tracking
                for f in fault_state._faults:
                    if f.fault_type == FaultType.SENSOR_FAULT and f not in sensor_fault_list:
                        sensor_fault_list.append(f)
                # Find primary non-sensor fault for telemetry labels
                non_sensor_active = [f for f in fault_state.get_active_faults(step_time)
                                     if f.fault_type != FaultType.SENSOR_FAULT]
                if non_sensor_active:
                    primary = max(non_sensor_active, key=lambda f: f.get_effective_severity(step_time))
                    fault_type_val = primary.fault_type.value if hasattr(primary.fault_type, "value") else str(primary.fault_type)
                    fault_sev_val = primary.get_effective_severity(step_time)
                elif sensor_fault_list:
                    # Only sensor faults active
                    active_sf = [f for f in sensor_fault_list if f.is_active_at(step_time)]
                    if active_sf:
                        primary_sf = max(active_sf, key=lambda f: f.get_effective_severity(step_time))
                        fault_type_val = primary_sf.fault_type.value if hasattr(primary_sf.fault_type, "value") else str(primary_sf.fault_type)
                        fault_sev_val = primary_sf.get_effective_severity(step_time)
                for f in fault_state.get_active_faults(step_time):
                    if f.fault_type == FaultType.COOLING_DEGRADATION:
                        cooling_severity = max(cooling_severity, f.get_effective_severity(step_time))
                    elif f.fault_type == FaultType.LUBRICATION_DEGRADATION:
                        lubrication_severity = max(lubrication_severity, f.get_effective_severity(step_time))
                    elif f.fault_type == FaultType.FUEL_INJECTION_ABNORMALITY:
                        fuel_severity = max(fuel_severity, f.get_effective_severity(step_time))
                        mode_p = f.parameters.get("mode", "lean") if f.parameters else "lean"
                        mixture_mode = mode_p.value if hasattr(mode_p, "value") else str(mode_p)
                    elif f.fault_type == FaultType.MECHANICAL_DEGRADATION:
                        mechanical_severity = max(mechanical_severity, f.get_effective_severity(step_time))
            elif isinstance(fault_state, dict):
                f_obj = FaultState.from_dict(fault_state)
                if f_obj.fault_type == FaultType.SENSOR_FAULT:
                    sensor_fault_list.append(f_obj)
                    if f_obj.is_active_at(step_time):
                        fault_type_val = f_obj.fault_type.value if hasattr(f_obj.fault_type, "value") else str(f_obj.fault_type)
                        fault_sev_val = f_obj.get_effective_severity(step_time)
                elif f_obj.is_active_at(step_time):
                    fault_type_val = f_obj.fault_type.value if hasattr(f_obj.fault_type, "value") else str(f_obj.fault_type)
                    fault_sev_val = f_obj.get_effective_severity(step_time)
                    if f_obj.fault_type == FaultType.COOLING_DEGRADATION:
                        cooling_severity = fault_sev_val
                    elif f_obj.fault_type == FaultType.LUBRICATION_DEGRADATION:
                        lubrication_severity = fault_sev_val
                    elif f_obj.fault_type == FaultType.FUEL_INJECTION_ABNORMALITY:
                        fuel_severity = fault_sev_val
                        mode_p = f_obj.parameters.get("mode", "lean") if f_obj.parameters else "lean"
                        mixture_mode = mode_p.value if hasattr(mode_p, "value") else str(mode_p)
                    elif f_obj.fault_type == FaultType.MECHANICAL_DEGRADATION:
                        mechanical_severity = fault_sev_val

        # 1. Atmosphere
        atmo_state = self.atmosphere.compute(altitude_m=altitude_m, temp_offset_k=temp_offset_k)

        # 2. Rotational Dynamics (RK4) with combustion efficiency factor and friction factor
        comb_eff = 1.0
        if fuel_severity > 0.0:
            if "rich" in mixture_mode.lower():
                comb_eff = 1.0 - getattr(self.sim_config.tier_c, "k_comb_loss_rich", 0.06) * fuel_severity
            else:
                comb_eff = 1.0 - getattr(self.sim_config.tier_c, "k_comb_loss_lean", 0.08) * fuel_severity

        # Phase 4E: mechanical degradation friction factor (Tier C/D assumption)
        friction_factor = 1.0 + self.sim_config.tier_c.k_mech_friction_gain * mechanical_severity

        op_point = self.dynamics.step(
            throttle_pct=throttle_pct,
            density_factor=atmo_state.density_factor,
            dt=dt,
            combustion_efficiency_factor=comb_eff,
            friction_factor=friction_factor,
        )

        # 3. Fuel System (Willans-line with abnormality scaling)
        fuel_state = self.fuel.compute(
            power_target_w=op_point.power_target_w,
            fuel_severity=fuel_severity,
            mixture_mode=mixture_mode,
        )

        # 4. Thermal System (EGT + CHT with mixture shift)
        thermal_state = self.thermal.step(
            rpm=op_point.rpm,
            load_pct=op_point.engine_load_pct,
            fuel_mass_flow_kg_s=fuel_state.mass_flow_kg_s,
            density_factor=atmo_state.density_factor,
            ambient_temp_c=atmo_state.temperature_c,
            airspeed_ms=airspeed_ms,
            dt=dt,
            cooling_severity=cooling_severity,
            fuel_severity=fuel_severity,
            mixture_mode=mixture_mode,
        )

        # 5. Lubrication System (Oil temp + pressure)
        lub_state = self.lubrication.step(
            rpm=op_point.rpm,
            cht_c=thermal_state.cht_c,
            fuel_mass_flow_kg_s=fuel_state.mass_flow_kg_s,
            ambient_temp_c=atmo_state.temperature_c,
            dt=dt,
            lubrication_severity=lubrication_severity,
        )

        # 6. Vibration System (1x, 2x orders + noise) with mechanical degradation
        # Phase 4E: mechanical_condition amplifies 1×/2× harmonics,
        #           mechanical_noise_factor amplifies broadband process noise only.
        mechanical_condition = 1.0 + self.sim_config.tier_c.k_mech_vib_gain * mechanical_severity
        mechanical_noise_factor = 1.0 + self.sim_config.tier_c.k_mech_noise_gain * mechanical_severity

        vib_state = self.vibration.step(
            rpm=op_point.rpm,
            load_pct=op_point.engine_load_pct,
            dt=dt,
            mechanical_condition=mechanical_condition,
            mechanical_noise_factor=mechanical_noise_factor,
        )

        # 7. Synthesize TelemetryRecord
        m_step_proxy = MissionStep(
            timestamp_s=step_time,
            phase=mission_phase_val,
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
            airspeed_ms=airspeed_ms,
            temp_offset_k=temp_offset_k,
            progress_pct=0.0,
        )

        record = self.telemetry_gen.generate(
            mission_step=m_step_proxy,
            atmo_state=atmo_state,
            op_point=op_point,
            fuel_state=fuel_state,
            thermal_state=thermal_state,
            lub_state=lub_state,
            vib_state=vib_state,
            engine_id=self.engine_config.engine_id,
            mission_id=mission_id_val,
            fault_type=fault_type_val,
            fault_severity=fault_sev_val,
            sensor_faults=sensor_fault_list if sensor_fault_list else None,
        )
        return record

    def run(
        self,
        mission_profile: Optional[MissionProfile] = None,
        dt: Optional[float] = None,
        fault_schedule: Optional[Any] = None,
    ) -> List[TelemetryRecord]:
        """
        Execute full mission profile simulation in batch mode with optional fault schedule.
        """
        profile = mission_profile or MissionProfile()
        step_dt = dt if dt is not None else self.sim_config.default_dt
        self.reset()

        schedule_obj = None
        if fault_schedule is not None:
            from simulator.fault_interface import FaultSchedule, FaultState
            if isinstance(fault_schedule, FaultSchedule):
                schedule_obj = fault_schedule
            elif isinstance(fault_schedule, list):
                schedule_obj = FaultSchedule(fault_schedule)
            elif isinstance(fault_schedule, FaultState):
                schedule_obj = FaultSchedule([fault_schedule])
            elif isinstance(fault_schedule, dict):
                schedule_obj = FaultSchedule([FaultState.from_dict(fault_schedule)])

        records: List[TelemetryRecord] = []
        for step in profile.generate_steps(dt=step_dt):
            rec = self.step(mission_config=step, time_step=step_dt, fault_state=schedule_obj)
            records.append(rec)

        return records

    def run_mission(
        self,
        mission_config: MissionConfig,
        time_step: float = 0.1,
    ) -> List[TelemetryRecord]:
        """
        Phase 1 compatibility interface for running a configured mission.
        """
        steps = max(1, int(mission_config.duration / time_step))
        self.reset()
        records: List[TelemetryRecord] = []
        for _ in range(steps):
            records.append(self.step(mission_config=mission_config, time_step=time_step))
        return records

    def run_to_dataframe(
        self,
        mission_profile: Optional[MissionProfile] = None,
        dt: Optional[float] = None,
        fault_schedule: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Execute simulation and return a pandas DataFrame for analysis and visualization.
        """
        records = self.run(mission_profile=mission_profile, dt=dt, fault_schedule=fault_schedule)
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)
        return df
