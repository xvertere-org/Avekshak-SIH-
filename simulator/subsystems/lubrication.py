"""
Lubrication and Oil System Model for SIH26054.

Implements:
1. Coupled lumped thermal model for oil temperature (slower dynamic response coupled to CHT).
2. RPM and oil temperature/viscosity dependent oil pressure model.
"""

import math
from typing import NamedTuple, Optional
from simulator.config import TierAParameters, TierCParameters


class LubricationState(NamedTuple):
    """Calculated lubrication state."""
    oil_temp_c: float
    oil_pressure_bar: float
    q_oil_gen_w: float
    q_oil_cool_w: float
    q_couple_w: float
    doil_temp_dt: float


class LubricationSystem:
    """
    Manages oil thermal dynamics and oil pressure.
    """

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        tier_c: TierCParameters = TierCParameters(),
        initial_oil_temp_c: Optional[float] = None,
    ):
        self.tier_a = tier_a
        self.tier_c = tier_c

        # State variable: oil temperature in °C
        self.oil_temp_c = initial_oil_temp_c if initial_oil_temp_c is not None else 65.0

    def step(
        self,
        rpm: float,
        cht_c: float,
        fuel_mass_flow_kg_s: float,
        ambient_temp_c: float,
        dt: float,
        lubrication_severity: float = 0.0,
    ) -> LubricationState:
        """
        Advance oil temperature and calculate instantaneous oil pressure.
        Supports Phase 4C lubrication degradation physics via lubrication_severity.
        """
        dt_safe = max(1e-4, float(dt))
        sev_clamped = max(0.0, min(1.0, float(lubrication_severity)))

        # --- 1. Oil Thermal Dynamics ---
        # Base heat generation from friction/combustion transferred into oil
        q_oil_gen_base = max(0.0, fuel_mass_flow_kg_s * self.tier_c.fuel_lhv_j_per_kg * self.tier_c.q_oil_fraction)
        # Phase 4C: Increased boundary/viscous frictional heat dissipation under degraded lubrication
        k_heat_gain = getattr(self.tier_c, "k_lub_heat_gain", 0.20)
        q_oil_gen = q_oil_gen_base * (1.0 + k_heat_gain * sev_clamped)

        # Convective cooling through oil radiator
        # Phase 4C: Reduced circulation flow through cooler under degraded lubrication
        k_cool_loss = getattr(self.tier_c, "k_lub_cool_loss", 0.20)
        h_oil_cool = max(1.0, self.tier_c.h_oil_cool * (1.0 - k_cool_loss * sev_clamped))
        q_oil_cool = h_oil_cool * (self.oil_temp_c - ambient_temp_c)

        # Thermal conduction coupling with cylinder head (CHT)
        k_couple = self.tier_c.k_oil_cht_couple
        q_couple = k_couple * (cht_c - self.oil_temp_c)

        # Net heat balance rate: C_oil * dT_oil/dt = Q_gen - Q_cool + Q_couple
        total_conductance = h_oil_cool + k_couple
        oil_temp_ss = (q_oil_gen + h_oil_cool * ambient_temp_c + k_couple * cht_c) / max(0.1, total_conductance)

        tau_oil = max(5.0, self.tier_c.c_oil / total_conductance)
        decay_oil = 1.0 - math.exp(-dt_safe / tau_oil)

        doil_dt = (q_oil_gen - q_oil_cool + q_couple) / self.tier_c.c_oil
        self.oil_temp_c += (oil_temp_ss - self.oil_temp_c) * decay_oil

        # --- 2. Oil Pressure Model ---
        # Positive pump displacement with RPM, negative drop due to viscosity loss with temperature
        temp_delta = max(0.0, self.oil_temp_c - 50.0)
        p_oil_nominal = (
            self.tier_c.oil_press_base_bar
            + self.tier_c.k_oil_p_rpm * rpm
            - self.tier_c.k_oil_p_temp * temp_delta
        )

        # Phase 4C: Hydraulic pressure reduction from pump degradation / gallery leakage
        k_p_loss = getattr(self.tier_c, "k_lub_p_loss", 0.55)
        p_oil_degraded = p_oil_nominal * (1.0 - k_p_loss * sev_clamped)

        # Realistic hydraulic pressure relief valve bounds [0.5, 7.0] bar
        oil_pressure_bar = max(0.5, min(self.tier_a.oil_press_max_bar, p_oil_degraded))

        return LubricationState(
            oil_temp_c=self.oil_temp_c,
            oil_pressure_bar=oil_pressure_bar,
            q_oil_gen_w=q_oil_gen,
            q_oil_cool_w=q_oil_cool,
            q_couple_w=q_couple,
            doil_temp_dt=doil_dt,
        )

    def set_oil_temp(self, temp_c: float) -> None:
        """Directly set oil temperature state."""
        self.oil_temp_c = temp_c
