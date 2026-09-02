"""
International Standard Atmosphere (ISA) Model for SIH26054.

Calculates pressure, temperature, density, and density ratio (sigma)
as a function of geopotential altitude and ambient temperature offset.
"""

import math
from typing import Dict, Any, NamedTuple
from simulator.config import TierAParameters


class ISAState(NamedTuple):
    """Atmospheric state at given altitude."""
    altitude_m: float
    temperature_k: float
    temperature_c: float
    pressure_pa: float
    pressure_bar: float
    density_kg_m3: float
    density_factor: float           # sigma = rho / rho0


class Atmosphere:
    """
    Standard ISA Atmosphere implementation with troposphere equations (0 to 11,000 m).
    """

    def __init__(self, tier_a: TierAParameters = TierAParameters()):
        self.p0 = tier_a.p0_sea_level
        self.t0 = tier_a.t0_sea_level
        self.rho0 = tier_a.rho0_sea_level
        self.l = tier_a.lapse_rate
        self.r = tier_a.gas_constant_air
        self.g = tier_a.gravity
        self.exponent = self.g / (self.l * self.r)

    def compute(self, altitude_m: float, temp_offset_k: float = 0.0) -> ISAState:
        """
        Compute atmospheric parameters at specified altitude with optional temperature deviation.

        Args:
            altitude_m: Altitude in meters (clamped physically to >= 0 m)
            temp_offset_k: Ambient temperature deviation from standard ISA (K or °C)

        Returns:
            ISAState named tuple with SI and aviation units.
        """
        h = max(0.0, float(altitude_m))
        
        # ISA standard temperature at altitude
        t_isa_k = self.t0 - self.l * h
        # Prevent non-physical negative temperatures
        t_isa_k = max(180.0, t_isa_k)

        # Ambient temperature including offset
        t_actual_k = t_isa_k + float(temp_offset_k)
        t_actual_k = max(180.0, t_actual_k)

        # Barometric pressure equation for troposphere
        temp_ratio = t_isa_k / self.t0
        pressure_pa = self.p0 * (temp_ratio ** self.exponent)

        # Density from ideal gas law: rho = p / (R * T)
        density_kg_m3 = pressure_pa / (self.r * t_actual_k)
        density_factor = density_kg_m3 / self.rho0

        return ISAState(
            altitude_m=h,
            temperature_k=t_actual_k,
            temperature_c=t_actual_k - 273.15,
            pressure_pa=pressure_pa,
            pressure_bar=pressure_pa / 100000.0,
            density_kg_m3=density_kg_m3,
            density_factor=density_factor,
        )

    def density_factor(self, altitude_m: float, temp_offset_k: float = 0.0) -> float:
        """Convenience method returning density factor sigma."""
        return self.compute(altitude_m, temp_offset_k).density_factor
