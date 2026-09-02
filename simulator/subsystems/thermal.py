"""
Thermal System Model for SIH26054.

Implements:
1. Exhaust Gas Temperature (EGT) steady-state mapping and thermocouple dynamic lag.
2. Cylinder Head Temperature (CHT) lumped thermal capacitance heat-balance model.
"""

import math
from typing import NamedTuple, Optional
from simulator.config import TierAParameters, TierCParameters


class ThermalState(NamedTuple):
    """Calculated thermal state."""
    cht_c: float
    egt_c: float
    egt_steady_state_c: float
    q_gen_w: float
    h_cool_w_k: float
    dcht_dt: float


class ThermalSystem:
    """
    Thermal dynamics manager for CHT and EGT.
    """

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        tier_c: TierCParameters = TierCParameters(),
        initial_cht_c: Optional[float] = None,
        initial_egt_c: Optional[float] = None,
    ):
        self.tier_a = tier_a
        self.tier_c = tier_c

        # State variables
        self.cht_c = initial_cht_c if initial_cht_c is not None else 85.0
        self.egt_c = initial_egt_c if initial_egt_c is not None else 580.0

    def step(
        self,
        rpm: float,
        load_pct: float,
        fuel_mass_flow_kg_s: float,
        density_factor: float,
        ambient_temp_c: float,
        airspeed_ms: float,
        dt: float,
        cooling_severity: float = 0.0,
        fuel_severity: float = 0.0,
        mixture_mode: str = "lean",
    ) -> ThermalState:
        """
        Advance CHT and EGT states by time step dt.
        Supports Phase 4B cooling degradation and Phase 4D fuel/injection abnormality physics.
        """
        dt_safe = max(1e-4, float(dt))
        load_norm = max(0.0, min(100.0, load_pct)) / 100.0

        # --- 1. EGT Calculation ---
        # Steady-state target
        rpm_offset = max(0.0, rpm - self.tier_c.rpm_idle)
        egt_ss_base = (
            self.tier_c.t_egt_base_c
            + self.tier_c.k_egt_load * load_norm
            + self.tier_c.k_egt_rpm * rpm_offset
            - self.tier_c.k_egt_density * density_factor
        )

        # Phase 4D: Mixture shift on exhaust enthalpy / burn timing
        sev_fuel = max(0.0, min(1.0, float(fuel_severity)))
        mode_str = str(mixture_mode).lower().strip()
        delta_egt_mixture = 0.0
        if sev_fuel > 0.0:
            if "rich" in mode_str:
                delta_egt_mixture = -getattr(self.tier_c, "k_egt_rich_drop_c", 80.0) * sev_fuel
            else:  # lean
                delta_egt_mixture = getattr(self.tier_c, "k_egt_lean_gain_c", 95.0) * sev_fuel

        egt_ss = max(100.0, egt_ss_base + delta_egt_mixture)

        # Unconditionally stable exponential decay update for first-order lag
        tau_egt = max(0.1, self.tier_c.tau_egt_s)
        decay_egt = 1.0 - math.exp(-dt_safe / tau_egt)
        self.egt_c += (egt_ss - self.egt_c) * decay_egt

        # --- 2. CHT Lumped Capacitance Heat Balance ---
        # Heat generation from combustion: fuel energy rate * thermal fraction
        q_gen = max(0.0, fuel_mass_flow_kg_s * self.tier_c.fuel_lhv_j_per_kg * self.tier_c.q_gen_fraction)

        # Total nominal convective cooling conductance (W/K)
        h_cool_nominal = (
            self.tier_c.h_cool_base
            + self.tier_c.h_cool_rpm * rpm
            + self.tier_c.h_cool_airspeed * max(0.0, airspeed_ms)
        )
        h_cool_nominal = max(1.0, h_cool_nominal)

        # Phase 4B: Apply cooling degradation fault physics
        # h_cool_effective = h_cool_nominal * (1 - k_cooling_max_loss * severity)
        sev_clamped = max(0.0, min(1.0, float(cooling_severity)))
        degradation_factor = getattr(self.tier_c, "k_cooling_max_loss", 0.55) * sev_clamped
        h_cool = max(1.0, h_cool_nominal * (1.0 - degradation_factor))

        # Differential: C_th * dT_cht/dt = Q_gen - h_cool * (T_cht - T_ambient)
        cht_target_ss = ambient_temp_c + (q_gen / h_cool)
        tau_cht = max(1.0, self.tier_c.c_th_cht / h_cool)
        decay_cht = 1.0 - math.exp(-dt_safe / tau_cht)

        dcht_dt = (q_gen - h_cool * (self.cht_c - ambient_temp_c)) / self.tier_c.c_th_cht
        self.cht_c += (cht_target_ss - self.cht_c) * decay_cht

        return ThermalState(
            cht_c=self.cht_c,
            egt_c=self.egt_c,
            egt_steady_state_c=egt_ss,
            q_gen_w=q_gen,
            h_cool_w_k=h_cool,
            dcht_dt=dcht_dt,
        )

    def set_states(self, cht_c: float, egt_c: float) -> None:
        """Directly set thermal state values."""
        self.cht_c = cht_c
        self.egt_c = egt_c
