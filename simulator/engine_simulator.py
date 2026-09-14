"""
Physics-Informed Aero Piston Engine Simulator Orchestrator for SIH26054.

Coordinates atmosphere, mission, rotational dynamics, fuel, thermal, lubrication,
and vibration subsystems to produce physically correlated telemetry streams.

DISCLAIMER & FIDELITY CONTRACT:
Reference Engine Architecture: Rotax 914 UL/F (configs/engine_reference/rotax_914_ul_f.json).
Governing Physics Contract: docs/physics_contract.md.
This simulator is a reduced-order grey-box propulsion model featuring a coupled reduced-order
turbocharger surrogate loop, spur reduction gearbox (2.42857:1), 4 discrete cylinder thermal channels,
and lumped liquid cooling loop surrogate. It does NOT represent a certified OEM engine model,
a full CFD/combustion simulation, or actual classified MALE-UAV propulsion hardware.
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
from simulator.subsystems.turbocharger import TurbochargerSubsystem
from simulator.subsystems.cooling import CoolingSubsystem
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

        # Mass flow history for causal turbocharger iteration loop
        self._last_m_air: float = 0.045
        self._last_m_fuel: float = 0.003

        # Instantiate physical subsystems
        self.atmosphere = Atmosphere(tier_a=self.sim_config.tier_a)
        self.turbocharger = TurbochargerSubsystem(
            tier_a=self.sim_config.tier_a,
            config=self.sim_config.tier_c_turbo,
        )
        self.cooling = CoolingSubsystem(
            tier_a=self.sim_config.tier_a,
            config=self.sim_config.tier_c_cooling,
        )
        self.dynamics = RotationalDynamics(
            tier_a=self.sim_config.tier_a,
            tier_c=self.sim_config.tier_c,
            tier_d=self.sim_config.tier_d,
            tier_c_gearbox=self.sim_config.tier_c_gearbox,
            tier_c_turbo=self.sim_config.tier_c_turbo,
            initial_rpm=self.sim_config.tier_c.rpm_idle,
        )
        self.fuel = FuelSystem(tier_c=self.sim_config.tier_c)
        self.thermal = ThermalSystem(
            tier_a=self.sim_config.tier_a,
            tier_c=self.sim_config.tier_c,
            tier_c_cylinder=self.sim_config.tier_c_cylinder,
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
        self._last_m_air = 0.045
        self._last_m_fuel = 0.003

        self.turbocharger.reset(1.013)
        self.cooling.set_temperature(75.0)
        self.dynamics.set_rpm(self.sim_config.tier_c.rpm_idle)
        self.thermal.set_states(cht_c=85.0, egt_c=580.0)
        self.lubrication.set_oil_temp(65.0)
        self.vibration.phase_1 = 0.0
        self.vibration.phase_2 = 0.0
        self.vibration.rng = self.rng

        # Re-point telemetry generator RNG and clear sensor fault stateful tracking
        self.telemetry_gen.reset(rng=self.rng)


    def step(
        self,
        mission_config: Optional[Union[MissionConfig, MissionStep]] = None,
        time_step: Optional[float] = None,
        mission_input: Optional[Union[MissionConfig, MissionStep]] = None,
        fault_state: Optional[Any] = None,
        throttle_pct: Optional[float] = None,
        altitude_m: Optional[float] = None,
        dt: Optional[float] = None,
        airspeed_ms: Optional[float] = None,
        temp_offset_k: Optional[float] = None,
    ) -> TelemetryRecord:
        """
        Advance the simulation by one discrete time step.

        Args:
            mission_config: MissionConfig or MissionStep with current flight demands.
            time_step: Simulation time increment (seconds). Defaults to config default_dt.
            mission_input: Alias for mission_config.
            fault_state: FaultState, FaultSchedule, or dict defining active fault.
            throttle_pct: Direct throttle percentage override [0-100].
            altitude_m: Direct altitude override in meters.
            dt: Direct timestep override in seconds.
            airspeed_ms: Direct airspeed override in m/s.
            temp_offset_k: Direct temperature offset override in Kelvin.

        Returns:
            Fully populated and schema-compliant TelemetryRecord.
        """
        inp = mission_config if mission_config is not None else mission_input
        step_dt = dt if dt is not None else (time_step if time_step is not None else self.sim_config.default_dt)
        self.current_time_s += step_dt
        dt = step_dt

        # Normalize mission inputs
        if isinstance(inp, MissionStep):
            throttle_pct = inp.throttle_pct if throttle_pct is None else float(throttle_pct)
            altitude_m = inp.altitude_m if altitude_m is None else float(altitude_m)
            airspeed_ms = inp.airspeed_ms if airspeed_ms is None else float(airspeed_ms)
            temp_offset_k = inp.temp_offset_k if temp_offset_k is None else float(temp_offset_k)
            mission_phase_val = inp.phase.value if hasattr(inp.phase, "value") else str(inp.phase)
            mission_id_val = "MISSION_MALE_UAV_001"
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = inp.timestamp_s
        elif isinstance(inp, MissionConfig):
            throttle_pct = inp.throttle if throttle_pct is None else float(throttle_pct)
            altitude_m = inp.altitude if altitude_m is None else float(altitude_m)
            airspeed_ms = 45.0 if airspeed_ms is None else float(airspeed_ms)
            temp_offset_k = (inp.ambient_temperature - 15.0) if temp_offset_k is None else float(temp_offset_k)
            mission_phase_val = inp.mission_phase
            mission_id_val = inp.mission_id
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = self.current_time_s
        else:
            throttle_pct = 75.0 if throttle_pct is None else float(throttle_pct)
            altitude_m = 3000.0 if altitude_m is None else float(altitude_m)
            airspeed_ms = 45.0 if airspeed_ms is None else float(airspeed_ms)
            temp_offset_k = 0.0 if temp_offset_k is None else float(temp_offset_k)
            mission_phase_val = FlightPhase.CRUISE.value
            mission_id_val = "MISSION_MALE_UAV_001"
            fault_type_val = FaultCategory.NONE.value
            fault_sev_val = 0.0
            step_time = self.current_time_s

        cooling_severity = 0.0
        lubrication_severity = 0.0
        fuel_severity = 0.0
        combustion_severity = 0.0
        mechanical_severity = 0.0
        misfire_imbalance = 0.0
        mixture_mode = "lean"

        per_cyl_fuel_sev = [0.0, 0.0, 0.0, 0.0]
        per_cyl_cool_sev = [0.0, 0.0, 0.0, 0.0]
        per_cyl_comb_eff = [1.0, 1.0, 1.0, 1.0]
        has_cyl_fuel = False
        has_cyl_cool = False
        has_cyl_comb = False

        sensor_fault_list: List[Any] = []

        # Resolve fault state and telemetry fault tags if provided
        if fault_state is not None:
            from simulator.fault_interface import FaultState, FaultSchedule, FaultType, FaultSubsystem

            fault_objs: List[FaultState] = []
            if isinstance(fault_state, FaultState):
                fault_objs = [fault_state]
            elif isinstance(fault_state, FaultSchedule):
                active_list = fault_state.get_active_faults(step_time)
                fault_objs.extend(active_list)
                # Also include inactive sensor faults for STUCK latch tracking
                for f in fault_state._faults:
                    is_sf = (
                        f.fault_type in (
                            FaultType.SENSOR_FAULT,
                            FaultType.SENSOR_BIAS,
                            FaultType.SENSOR_DRIFT,
                            FaultType.SENSOR_DROPOUT,
                            FaultType.SENSOR_STUCK,
                        )
                        or getattr(f, "affected_subsystem", None) == FaultSubsystem.SENSOR
                    )
                    if is_sf and f not in sensor_fault_list:
                        sensor_fault_list.append(f)
            elif isinstance(fault_state, list):
                for item in fault_state:
                    if isinstance(item, FaultState):
                        fault_objs.append(item)
                    elif isinstance(item, dict):
                        fault_objs.append(FaultState.from_dict(item))
            elif isinstance(fault_state, dict):
                fault_objs = [FaultState.from_dict(fault_state)]

            # Process all faults
            for f in fault_objs:
                is_sf = (
                    f.fault_type in (
                        FaultType.SENSOR_FAULT,
                        FaultType.SENSOR_BIAS,
                        FaultType.SENSOR_DRIFT,
                        FaultType.SENSOR_DROPOUT,
                        FaultType.SENSOR_STUCK,
                    )
                    or getattr(f, "affected_subsystem", None) == FaultSubsystem.SENSOR
                )
                if is_sf:
                    if f not in sensor_fault_list:
                        sensor_fault_list.append(f)
                    continue

                if not f.is_active_at(step_time):
                    continue

                eff_sev = f.get_effective_severity(step_time)
                if eff_sev <= 0.0:
                    continue

                cyl = f.affected_cylinder

                # F1: Fuel delivery / injector abnormality
                if f.fault_type in (FaultType.INJECTOR_DELIVERY_ABNORMALITY, FaultType.FUEL_INJECTION_ABNORMALITY):
                    mode_p = f.parameters.get("mode", "lean") if f.parameters else "lean"
                    mixture_mode = mode_p.value if hasattr(mode_p, "value") else str(mode_p)
                    has_cyl_fuel = True
                    if cyl is not None:
                        per_cyl_fuel_sev[cyl - 1] = max(per_cyl_fuel_sev[cyl - 1], eff_sev)
                        # Average overall engine fuel severity is localized to 1 of 4 cylinders
                        fuel_severity = max(fuel_severity, eff_sev * 0.25)
                    else:
                        fuel_severity = max(fuel_severity, eff_sev)
                        for i in range(4):
                            per_cyl_fuel_sev[i] = max(per_cyl_fuel_sev[i], eff_sev)

                # F2: Lubrication degradation
                elif f.fault_type == FaultType.LUBRICATION_DEGRADATION:
                    lubrication_severity = max(lubrication_severity, eff_sev)

                # F3: Cooling degradation
                elif f.fault_type == FaultType.COOLING_DEGRADATION:
                    cooling_severity = max(cooling_severity, eff_sev)
                    has_cyl_cool = True
                    if cyl is not None:
                        per_cyl_cool_sev[cyl - 1] = max(per_cyl_cool_sev[cyl - 1], eff_sev)
                    else:
                        for i in range(4):
                            per_cyl_cool_sev[i] = max(per_cyl_cool_sev[i], eff_sev)

                # F4: Combustion misfire / instability
                elif f.fault_type in (FaultType.COMBUSTION_MISFIRE, FaultType.COMBUSTION_INSTABILITY):
                    combustion_severity = max(combustion_severity, eff_sev)
                    has_cyl_comb = True
                    d_comb = getattr(self.sim_config.tier_c, "k_comb_misfire_drop", 0.70)
                    if cyl is not None:
                        eff_drop = 1.0 - d_comb * eff_sev
                        per_cyl_comb_eff[cyl - 1] = min(per_cyl_comb_eff[cyl - 1], eff_drop)
                        misfire_imbalance = max(misfire_imbalance, eff_sev)
                    else:
                        eff_drop = 1.0 - d_comb * eff_sev
                        for i in range(4):
                            per_cyl_comb_eff[i] = min(per_cyl_comb_eff[i], eff_drop)
                        misfire_imbalance = max(misfire_imbalance, eff_sev * 0.5)

                # F5: Mechanical degradation
                elif f.fault_type == FaultType.MECHANICAL_DEGRADATION:
                    mechanical_severity = max(mechanical_severity, eff_sev)

            # Determine primary fault tag for ground-truth metadata in TelemetryRecord
            active_physical = [
                f for f in fault_objs
                if f.is_active_at(step_time)
                and f.fault_type not in (
                    FaultType.SENSOR_FAULT,
                    FaultType.SENSOR_BIAS,
                    FaultType.SENSOR_DRIFT,
                    FaultType.SENSOR_DROPOUT,
                    FaultType.SENSOR_STUCK,
                )
                and getattr(f, "affected_subsystem", None) != FaultSubsystem.SENSOR
            ]
            if active_physical:
                primary = max(active_physical, key=lambda f: f.get_effective_severity(step_time))
                fault_type_val = primary.fault_type.value if hasattr(primary.fault_type, "value") else str(primary.fault_type)
                fault_sev_val = primary.get_effective_severity(step_time)
            elif sensor_fault_list:
                active_sf = [f for f in sensor_fault_list if f.is_active_at(step_time)]
                if active_sf:
                    primary_sf = max(active_sf, key=lambda f: f.get_effective_severity(step_time))
                    fault_type_val = primary_sf.fault_type.value if hasattr(primary_sf.fault_type, "value") else str(primary_sf.fault_type)
                    fault_sev_val = primary_sf.get_effective_severity(step_time)

        # 1. Atmosphere & Ram-Air Dynamic Pressure Recovery
        atmo_state = self.atmosphere.compute(altitude_m=altitude_m, temp_offset_k=temp_offset_k)
        ram_pa = self.atmosphere.compute_ram_recovery_pa(atmo_state.density_kg_m3, airspeed_ms)

        # 2. Turbocharger & Boost Intake (Coupled Reduced-Order Loop)
        turbo_state = self.turbocharger.step(
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
            p_amb_pa=atmo_state.pressure_pa,
            t_amb_c=atmo_state.temperature_c,
            m_dot_air_kg_s=self._last_m_air,
            m_dot_fuel_kg_s=self._last_m_fuel,
            t_exh_c=self.thermal.egt_c,
            dt=dt,
            ram_pressure_recovery_pa=ram_pa,
        )

        # 3. Rotational Dynamics (RK4) with reduction gearbox (2.4286:1) and causal power chain
        comb_eff = 1.0
        if fuel_severity > 0.0:
            if "rich" in mixture_mode.lower():
                comb_eff = 1.0 - getattr(self.sim_config.tier_c, "k_comb_loss_rich", 0.06) * fuel_severity
            else:
                comb_eff = 1.0 - getattr(self.sim_config.tier_c, "k_comb_loss_lean", 0.08) * fuel_severity

        # Phase 4: friction factor composed from F2 (lubrication) and F5 (mechanical)
        friction_factor = (
            1.0
            + getattr(self.sim_config.tier_c, "k_lub_friction_gain", 0.08) * lubrication_severity
            + getattr(self.sim_config.tier_c, "k_mech_friction_gain", 0.08) * mechanical_severity
        )

        passed_comb_eff_cyl = per_cyl_comb_eff if has_cyl_comb else None

        op_point = self.dynamics.step(
            throttle_pct=throttle_pct,
            density_factor=atmo_state.density_factor,
            dt=dt,
            combustion_efficiency_factor=comb_eff,
            friction_factor=friction_factor,
            map_bar=turbo_state.map_bar,
            charge_air_temp_c=turbo_state.charge_air_temp_c,
            per_cylinder_combustion_efficiencies=passed_comb_eff_cyl,
        )
        self._last_m_air = op_point.air_mass_flow_kg_s
        self._last_m_fuel = op_point.fuel_mass_flow_kg_s

        # 4. Fuel System (Willans-line with abnormality scaling)
        passed_fuel_cyl = per_cyl_fuel_sev if has_cyl_fuel else None
        fuel_state = self.fuel.compute(
            power_target_w=op_point.power_target_w,
            fuel_severity=fuel_severity,
            mixture_mode=mixture_mode,
            per_cylinder_severities=passed_fuel_cyl,
        )

        # 5. Multi-Cylinder Thermal System (4 discrete cylinders, 1-4-3-2 firing order)
        passed_cool_cyl = per_cyl_cool_sev if has_cyl_cool else None
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
            coolant_temp_c=self.cooling.coolant_temp_c,
            per_cylinder_fuel_severities=passed_fuel_cyl,
            per_cylinder_combustion_efficiencies=passed_comb_eff_cyl,
            per_cylinder_cooling_severities=passed_cool_cyl,
        )

        # 6. Liquid Cooling Loop (REDUCED_ORDER_COOLING_SURROGATE)
        cooling_state = self.cooling.step(
            cylinder_cht_temps_c=self.thermal.cht_cyl,
            ambient_temp_c=atmo_state.temperature_c,
            airspeed_ms=airspeed_ms,
            dt=dt,
            cooling_fault_severity=cooling_severity,
        )

        # 7. Lubrication System (Oil temp + pressure)
        lub_state = self.lubrication.step(
            rpm=op_point.rpm,
            cht_c=thermal_state.cht_c,
            fuel_mass_flow_kg_s=fuel_state.mass_flow_kg_s,
            ambient_temp_c=atmo_state.temperature_c,
            dt=dt,
            lubrication_severity=lubrication_severity,
        )

        # 8. Vibration System (1x, 2x orders + noise) with mechanical degradation and misfire
        mechanical_condition = 1.0 + self.sim_config.tier_c.k_mech_vib_gain * mechanical_severity
        mechanical_noise_factor = 1.0 + self.sim_config.tier_c.k_mech_noise_gain * mechanical_severity

        vib_state = self.vibration.step(
            rpm=op_point.rpm,
            load_pct=op_point.engine_load_pct,
            dt=dt,
            mechanical_condition=mechanical_condition,
            mechanical_noise_factor=mechanical_noise_factor,
            misfire_imbalance_factor=misfire_imbalance,
        )

        # 9. Synthesize TelemetryRecord
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
            sensor_faults=sensor_fault_list,
            turbo_state=turbo_state,
            cooling_state=cooling_state,
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
