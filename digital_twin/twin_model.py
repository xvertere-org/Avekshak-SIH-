"""
Digital Twin State Estimation and Tracking Model for SIH26054.

Implements a physics-informed, reduced-order grey-box Digital Twin:
- Estimates nominal engine behavior from observable operating conditions:
    throttle, altitude (density ratio), ambient temperature, and mission phase
- Decoupled Operating Context -> Expected RPM / Load -> Thermal / Lubrication / Vibration
  (Crucially: observed RPM is NEVER used as an input to calculate expected states,
  ensuring complete sensor fault isolation)
- Strict unit consistency: fuel flow is calculated in L/h adhering to simulator contracts
- Deterministic, causal state tracking with bounded residual generation
"""

import math
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd

from telemetry.schema import TelemetryRecord, DigitalTwinState, EngineConfig
from telemetry.ingestion import CanonicalTelemetryFrame
from digital_twin.residuals import ResidualGenerator, ResidualFrame, SUPPORTED_RESIDUAL_CHANNELS
from simulator.config import SimulatorConfig, TierAParameters, TierCParameters, TierDParameters
from simulator.subsystems.atmosphere import Atmosphere


class DigitalTwinModel:
    """
    Independent physics-informed reduced-order Digital Twin engine model.

    Maintains internal nominal dynamic states (expected RPM, CHT, EGT, oil temperature)
    and estimates expected telemetry under healthy baseline physics.
    """

    def __init__(
        self,
        sim_config: Optional[SimulatorConfig] = None,
        engine_config: Optional[EngineConfig] = None,
    ):
        self.sim_config = sim_config or SimulatorConfig()
        self.engine_config = engine_config or EngineConfig()

        self.tier_a = self.sim_config.tier_a
        self.tier_c = self.sim_config.tier_c
        self.tier_d = self.sim_config.tier_d

        self.atmosphere = Atmosphere(tier_a=self.tier_a)

        # Internal state variables (initialized to healthy nominal baseline)
        self.expected_rpm = self.tier_c.rpm_idle
        self.omega = 2.0 * math.pi * self.expected_rpm / 60.0
        self.expected_cht = 85.0
        self.expected_egt = 580.0
        self.expected_oil_temp = 65.0
        self.last_timestamp: Optional[float] = None

        # Precalculate idle assist torque balance
        omega_idle = 2.0 * math.pi * self.tier_c.rpm_idle / 60.0
        t_load_idle = self.tier_c.k_load * (omega_idle ** 2)
        t_fric_idle = self.tier_c.k_fric_linear * omega_idle + self.tier_c.torque_fric_static
        self._idle_torque_balance = t_load_idle + t_fric_idle

    def reset(
        self,
        initial_rpm: Optional[float] = None,
        initial_cht: float = 85.0,
        initial_egt: float = 580.0,
        initial_oil_temp: float = 65.0,
    ) -> None:
        """Reset internal nominal state estimates to baseline initial conditions."""
        self.expected_rpm = initial_rpm if initial_rpm is not None else self.tier_c.rpm_idle
        self.omega = 2.0 * math.pi * self.expected_rpm / 60.0
        self.expected_cht = initial_cht
        self.expected_egt = initial_egt
        self.expected_oil_temp = initial_oil_temp
        self.last_timestamp = None

    def _compute_rpm_efficiency(self, rpm: float) -> float:
        """Tier D polynomial efficiency curve peaking near continuous rated speed (5500 RPM)."""
        norm_rpm = rpm / self.tier_a.rpm_max_continuous
        a, b, c = self.tier_d.rpm_eff_poly
        eff = a * (norm_rpm ** 2) + b * norm_rpm + c
        return max(0.35, min(1.05, eff))

    def _torque_derivatives(
        self,
        omega: float,
        throttle_pct: float,
        density_factor: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute torque balance derivatives for expected rotational dynamics."""
        rpm = max(0.0, omega * 60.0 / (2.0 * math.pi))
        throttle_norm = max(0.0, min(100.0, throttle_pct)) / 100.0
        eff = self._compute_rpm_efficiency(rpm)

        p_combustion = self.tier_a.power_max_continuous_w * throttle_norm * density_factor * eff
        omega_safe = max(omega, 10.0)
        t_combustion = p_combustion / omega_safe
        t_idle_assist = self._idle_torque_balance * max(0.0, (1.0 - throttle_norm * 2.0))
        torque_engine = t_combustion + t_idle_assist

        torque_load = self.tier_c.k_load * (omega ** 2)
        torque_friction = self.tier_c.k_fric_linear * omega + self.tier_c.torque_fric_static

        net_torque = torque_engine - torque_load - torque_friction
        domega_dt = net_torque / self.tier_c.inertia_kg_m2
        return p_combustion, torque_engine, torque_load, torque_friction, net_torque, domega_dt

    def step_expected(
        self,
        throttle_pct: float = 75.0,
        altitude_m: float = 2000.0,
        ambient_temp_c: float = 15.0,
        dt: float = 0.1,
        airspeed_ms: Optional[float] = None,
        mission_phase: str = "CRUISE",
    ) -> Dict[str, float]:
        """
        Advance the Digital Twin's nominal expected state by time step dt.

        Decoupled Flow:
        Observable conditions (throttle, altitude, ambient_temp)
            -> Expected RPM & Engine Load
            -> Expected Fuel Flow
            -> Expected CHT & EGT
            -> Expected Oil Temp & Pressure
            -> Expected Vibration
        """
        dt_safe = max(1e-4, float(dt))
        throttle_safe = max(0.0, min(100.0, float(throttle_pct)))
        alt_safe = max(-500.0, min(15000.0, float(altitude_m)))
        amb_safe = float(ambient_temp_c)

        # Determine airspeed proxy if not explicitly provided
        if airspeed_ms is not None:
            v_air = max(0.0, float(airspeed_ms))
        else:
            phase_upper = str(mission_phase).upper()
            if "TAKEOFF" in phase_upper:
                v_air = self.tier_d.airspeed_proxy_takeoff_ms
            elif "CLIMB" in phase_upper:
                v_air = self.tier_d.airspeed_proxy_climb_ms
            elif "LOITER" in phase_upper:
                v_air = self.tier_d.airspeed_proxy_loiter_ms
            elif "DESCENT" in phase_upper:
                v_air = self.tier_d.airspeed_proxy_descent_ms
            elif "LANDING" in phase_upper:
                v_air = self.tier_d.airspeed_proxy_landing_ms
            else:  # CRUISE default
                v_air = self.tier_d.airspeed_proxy_cruise_ms

        # 1. Atmosphere density factor
        atmo = self.atmosphere.compute(altitude_m=alt_safe)
        density_factor = atmo.density_factor

        # 2. Rotational Dynamics (RK4 step for expected RPM)
        omega_0 = self.omega
        p_target, _, _, _, _, k1 = self._torque_derivatives(omega_0, throttle_safe, density_factor)

        omega_k2 = max(0.0, omega_0 + 0.5 * dt_safe * k1)
        _, _, _, _, _, k2 = self._torque_derivatives(omega_k2, throttle_safe, density_factor)

        omega_k3 = max(0.0, omega_0 + 0.5 * dt_safe * k2)
        _, _, _, _, _, k3 = self._torque_derivatives(omega_k3, throttle_safe, density_factor)

        omega_k4 = max(0.0, omega_0 + dt_safe * k3)
        _, _, _, _, _, k4 = self._torque_derivatives(omega_k4, throttle_safe, density_factor)

        self.omega = max(0.0, omega_0 + (dt_safe / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        self.expected_rpm = self.omega * 60.0 / (2.0 * math.pi)

        # Expected engine load percentage proxy based on power output relative to density-adjusted max continuous
        max_available_p = max(100.0, self.tier_a.power_max_continuous_w * density_factor)
        load_pct = min(100.0, max(0.0, (p_target / max_available_p) * 100.0))
        load_norm = load_pct / 100.0

        # 3. Expected Fuel Flow (Willans-line power model matching simulator contract)
        m_dot_fuel = max(0.0, self.tier_c.a_fuel_kg_per_j * p_target + self.tier_c.b_fuel_kg_per_s)
        # Convert kg/s to L/h using tier_c.fuel_density_kg_per_l
        fuel_density = max(0.1, self.tier_c.fuel_density_kg_per_l)
        expected_fuel_flow = (m_dot_fuel / fuel_density) * 3600.0

        # 4. Expected Thermal (EGT & CHT)
        # Expected EGT steady state & thermocouple dynamic lag
        rpm_offset = max(0.0, self.expected_rpm - self.tier_c.rpm_idle)
        egt_ss = max(100.0, (
            self.tier_c.t_egt_base_c
            + self.tier_c.k_egt_load * load_norm
            + self.tier_c.k_egt_rpm * rpm_offset
            - self.tier_c.k_egt_density * density_factor
        ))
        tau_egt = max(0.1, self.tier_c.tau_egt_s)
        decay_egt = 1.0 - math.exp(-dt_safe / tau_egt)
        self.expected_egt += (egt_ss - self.expected_egt) * decay_egt

        # Expected CHT heat balance
        q_gen = max(0.0, m_dot_fuel * self.tier_c.fuel_lhv_j_per_kg * self.tier_c.q_gen_fraction)
        h_cool = max(1.0, (
            self.tier_c.h_cool_base
            + self.tier_c.h_cool_rpm * self.expected_rpm
            + self.tier_c.h_cool_airspeed * v_air
        ))
        cht_target_ss = amb_safe + (q_gen / h_cool)
        tau_cht = max(1.0, self.tier_c.c_th_cht / h_cool)
        decay_cht = 1.0 - math.exp(-dt_safe / tau_cht)
        self.expected_cht += (cht_target_ss - self.expected_cht) * decay_cht

        # 5. Expected Lubrication (Oil Temp & Pressure)
        q_oil_gen = max(0.0, m_dot_fuel * self.tier_c.fuel_lhv_j_per_kg * self.tier_c.q_oil_fraction)
        h_oil_cool = self.tier_c.h_oil_cool
        k_couple = self.tier_c.k_oil_cht_couple
        total_conductance = max(0.1, h_oil_cool + k_couple)
        oil_temp_ss = (q_oil_gen + h_oil_cool * amb_safe + k_couple * self.expected_cht) / total_conductance
        tau_oil = max(5.0, self.tier_c.c_oil / total_conductance)
        decay_oil = 1.0 - math.exp(-dt_safe / tau_oil)
        self.expected_oil_temp += (oil_temp_ss - self.expected_oil_temp) * decay_oil

        # Expected oil pressure (RPM gain minus temperature viscosity loss)
        temp_delta = max(0.0, self.expected_oil_temp - 50.0)
        p_oil = (
            self.tier_c.oil_press_base_bar
            + self.tier_c.k_oil_p_rpm * self.expected_rpm
            - self.tier_c.k_oil_p_temp * temp_delta
        )
        expected_oil_press = max(0.5, min(self.tier_a.oil_press_max_bar, p_oil))

        # 6. Expected Vibration (Nominal RMS from order harmonics + process noise)
        amp_1x = (self.tier_c.vib_order1_base_g + self.tier_c.vib_load_gain * load_norm) * self.tier_d.mechanical_condition
        amp_2x = (self.tier_c.vib_order2_base_g + self.tier_c.vib_load_gain * load_norm) * self.tier_d.mechanical_condition
        expected_vib_rms = math.sqrt(0.5 * (amp_1x ** 2) + 0.5 * (amp_2x ** 2) + (self.tier_c.vib_noise_std_g ** 2))

        # Analytical order frequencies
        f_order1 = self.expected_rpm / 60.0
        f_order2 = 2.0 * f_order1

        return {
            "rpm_expected": round(self.expected_rpm, 1),
            "cht_expected": round(self.expected_cht, 2),
            "egt_expected": round(self.expected_egt, 2),
            "oil_temp_expected": round(self.expected_oil_temp, 2),
            "oil_pressure_expected": round(expected_oil_press, 3),
            "fuel_flow_expected": round(expected_fuel_flow, 2),
            "vibration_expected": round(expected_vib_rms, 3),
            "load_expected": round(load_pct, 1),
            "power_expected_kw": round(p_target / 1000.0, 2),
            "order_1x_freq_hz": round(f_order1, 2),
            "order_2x_freq_hz": round(f_order2, 2),
        }


class DigitalTwin:
    """
    Digital Twin tracking and residual generation interface for SIH26054.

    Maintains backward compatibility with Phase 1 contracts (`twin.update(telemetry)`)
    while offering full causal batch expected state prediction and residual generation.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        sim_config: Optional[SimulatorConfig] = None,
    ):
        self.engine_config = engine_config or EngineConfig()
        self.sim_config = sim_config or SimulatorConfig()

        self.model = DigitalTwinModel(sim_config=self.sim_config, engine_config=self.engine_config)
        self.residual_generator = ResidualGenerator()
        self.history: List[DigitalTwinState] = []
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset Digital Twin internal dynamic states."""
        self.model.reset()
        self.history.clear()
        self.last_timestamp = None

    def update(self, telemetry: TelemetryRecord) -> DigitalTwinState:
        """
        Process a single streaming TelemetryRecord, estimate nominal states, and compute residuals.
        Maintains strict backward compatibility with Phase 1 schemas.
        """
        # Calculate dynamic time increment dt
        if self.last_timestamp is not None:
            if telemetry.timestamp > self.last_timestamp:
                dt = telemetry.timestamp - self.last_timestamp
                self.last_timestamp = telemetry.timestamp
            else:
                # Non-positive dt (duplicate or out-of-order): do not advance internal states
                dt = 0.0
        else:
            dt = self.sim_config.default_dt
            self.last_timestamp = telemetry.timestamp

        # Predict expected nominal states (observable conditions only)
        expected = self.model.step_expected(
            throttle_pct=telemetry.throttle,
            altitude_m=telemetry.altitude,
            ambient_temp_c=telemetry.ambient_temp,
            dt=dt,
            mission_phase=telemetry.mission_phase,
        )

        # Compute raw residuals (observed - expected)
        # Note: If observed is NaN (Phase 4F sensor dropout), residual is float('nan')
        cht_res = telemetry.cht - expected["cht_expected"] if not math.isnan(telemetry.cht) else float("nan")
        egt_res = telemetry.egt - expected["egt_expected"] if not math.isnan(telemetry.egt) else float("nan")
        oil_p_res = telemetry.oil_pressure - expected["oil_pressure_expected"] if not math.isnan(telemetry.oil_pressure) else float("nan")
        oil_t_res = telemetry.oil_temp - expected["oil_temp_expected"] if not math.isnan(telemetry.oil_temp) else float("nan")
        rpm_res = telemetry.rpm - expected["rpm_expected"] if not math.isnan(telemetry.rpm) else float("nan")
        fuel_res = telemetry.fuel_flow - expected["fuel_flow_expected"] if not math.isnan(telemetry.fuel_flow) else float("nan")
        vib_res = telemetry.vibration - expected["vibration_expected"] if not math.isnan(telemetry.vibration) else float("nan")

        residuals = {
            "cht_residual": round(cht_res, 4) if not math.isnan(cht_res) else float("nan"),
            "egt_residual": round(egt_res, 4) if not math.isnan(egt_res) else float("nan"),
            "oil_pressure_residual": round(oil_p_res, 4) if not math.isnan(oil_p_res) else float("nan"),
            "oil_temp_residual": round(oil_t_res, 4) if not math.isnan(oil_t_res) else float("nan"),
            "rpm_residual": round(rpm_res, 4) if not math.isnan(rpm_res) else float("nan"),
            "fuel_flow_residual": round(fuel_res, 4) if not math.isnan(fuel_res) else float("nan"),
            "vibration_residual": round(vib_res, 4) if not math.isnan(vib_res) else float("nan"),
        }

        nominal_estimates = {
            # Tier A reference anchor for audit compatibility (test_audit_cleanup.py)
            "nominal_cht": self.sim_config.tier_a.cht_nominal_c,
            "nominal_egt": expected["egt_expected"],
            "nominal_oil_pressure": expected["oil_pressure_expected"],
            "nominal_oil_temp": expected["oil_temp_expected"],
            # Dynamic Phase 6 expected nominal states
            "expected_cht": expected["cht_expected"],
            "expected_egt": expected["egt_expected"],
            "expected_oil_pressure": expected["oil_pressure_expected"],
            "expected_oil_temp": expected["oil_temp_expected"],
            "nominal_rpm": expected["rpm_expected"],
            "expected_rpm": expected["rpm_expected"],
            "nominal_fuel_flow": expected["fuel_flow_expected"],
            "expected_fuel_flow": expected["fuel_flow_expected"],
            "nominal_vibration": expected["vibration_expected"],
            "expected_vibration": expected["vibration_expected"],
            "nominal_load": expected["load_expected"],
        }

        twin_state = DigitalTwinState(
            timestamp=telemetry.timestamp,
            engine_id=telemetry.engine_id,
            observed_telemetry=telemetry,
            nominal_estimates=nominal_estimates,
            residuals=residuals,
            state_confidence=0.98,
            metadata={
                "power_expected_kw": expected["power_expected_kw"],
                "order_1x_freq_hz": expected["order_1x_freq_hz"],
                "order_2x_freq_hz": expected["order_2x_freq_hz"],
            },
        )
        self.history.append(twin_state)
        return twin_state

    def predict_expected(
        self,
        data: Union[CanonicalTelemetryFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Causal batch execution over time-series data, predicting expected state trajectories.

        Args:
            data: CanonicalTelemetryFrame or DataFrame of telemetry records.

        Returns:
            DataFrame containing '<channel>_expected' columns for all supported channels.
        """
        if isinstance(data, CanonicalTelemetryFrame):
            df = data.to_dataframe()
        else:
            df = data.copy()

        if len(df) == 0:
            return pd.DataFrame()

        self.reset()
        expected_rows: List[Dict[str, Any]] = []

        last_t = None
        for _, row in df.iterrows():
            t_curr = row.get("timestamp", 0.0)
            if last_t is not None and t_curr > last_t:
                dt = t_curr - last_t
            else:
                dt = self.sim_config.default_dt
            last_t = t_curr

            throttle = row.get("throttle", 75.0)
            altitude = row.get("altitude", 2000.0)
            amb_temp = row.get("ambient_temp", 15.0)
            phase = row.get("mission_phase", "CRUISE")

            exp_dict = self.model.step_expected(
                throttle_pct=throttle,
                altitude_m=altitude,
                ambient_temp_c=amb_temp,
                dt=dt,
                mission_phase=phase,
            )
            expected_rows.append(exp_dict)

        return pd.DataFrame(expected_rows, index=df.index)

    def process_frame(
        self,
        frame: Union[CanonicalTelemetryFrame, pd.DataFrame],
    ) -> ResidualFrame:
        """
        Complete batch workflow: predicts expected nominal states and calculates ResidualFrame.
        """
        expected_df = self.predict_expected(frame)
        return self.residual_generator.compute_residuals(frame, expected_df)
