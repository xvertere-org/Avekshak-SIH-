"""
Fuel Flow Model for SIH26054.

Implements Willans-line approximation for fuel consumption:
Mass_flow = a_fuel * Power + b_fuel
"""

from typing import NamedTuple
from simulator.config import TierCParameters


class FuelState(NamedTuple):
    """Calculated fuel consumption state."""
    mass_flow_kg_s: float
    volumetric_flow_l_h: float
    bsfc_g_kwh: float                       # Brake Specific Fuel Consumption (g/kWh)


class FuelSystem:
    """
    Fuel flow calculator based on Willans-line power approximation.
    """

    def __init__(self, tier_c: TierCParameters = TierCParameters()):
        self.tier_c = tier_c

    def compute(self, power_target_w: float) -> FuelState:
        """
        Calculate fuel mass and volumetric flow rates.

        Args:
            power_target_w: Mechanical power output in Watts (W)

        Returns:
            FuelState named tuple.
        """
        p = max(0.0, float(power_target_w))
        mass_flow_kg_s = max(0.0, self.tier_c.a_fuel_kg_per_j * p + self.tier_c.b_fuel_kg_per_s)

        # Convert to L/h: (kg/s / (kg/L)) * 3600 s/h
        vol_flow_l_h = (mass_flow_kg_s / self.tier_c.fuel_density_kg_per_l) * 3600.0

        # BSFC in g/kWh:
        # (mass_flow_kg_s * 1000 g/kg * 3600 s/h) / (power_target_w / 1000 kW) = (mass_flow_kg_s / power_target_w) * 3.6e9
        if p > 1000.0:
            bsfc = (mass_flow_kg_s / p) * 3.6e9
        else:
            bsfc = 0.0

        return FuelState(
            mass_flow_kg_s=mass_flow_kg_s,
            volumetric_flow_l_h=vol_flow_l_h,
            bsfc_g_kwh=bsfc,
        )
