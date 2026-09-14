"""
Fuel Flow Model for SIH26054.

Implements Willans-line approximation for fuel consumption:
Mass_flow = a_fuel * Power + b_fuel
"""

from typing import NamedTuple, Optional, List
from simulator.config import TierCParameters


class FuelState(NamedTuple):
    """Calculated fuel consumption state."""
    mass_flow_kg_s: float
    volumetric_flow_l_h: float
    bsfc_g_kwh: float                       # Brake Specific Fuel Consumption (g/kWh)
    per_cylinder_mass_flow_kg_s: Optional[List[float]] = None


class FuelSystem:
    """
    Fuel flow calculator based on Willans-line power approximation.
    Supports causal fuel delivery degradation (d_fuel = 0.35, MODEL_CALIBRATION)
    and optional cylinder localization.
    """

    # Model calibration parameter: maximum fractional delivery loss at severity=1.0
    D_FUEL: float = 0.35

    def __init__(self, tier_c: TierCParameters = TierCParameters()):
        self.tier_c = tier_c

    def compute(
        self,
        power_target_w: float,
        fuel_severity: float = 0.0,
        mixture_mode: str = "lean",
        per_cylinder_severities: Optional[List[float]] = None,
    ) -> FuelState:
        """
        Calculate fuel mass and volumetric flow rates.
        Supports Phase 4 fuel/injection abnormality via fuel_severity, mixture_mode,
        and per_cylinder_severities.

        Args:
            power_target_w: Mechanical power output in Watts (W)
            fuel_severity: Normalized fault severity in [0.0, 1.0] across all cylinders
            mixture_mode: Abnormality mode ("lean" or "rich")
            per_cylinder_severities: Optional 4-element list of severities for cylinders 1..4

        Returns:
            FuelState named tuple.
        """
        p = max(0.0, float(power_target_w))
        mass_flow_nominal = max(0.0, self.tier_c.a_fuel_kg_per_j * p + self.tier_c.b_fuel_kg_per_s)
        m_nom_cyl = mass_flow_nominal / 4.0

        sev = max(0.0, min(1.0, float(fuel_severity)))
        mode_str = str(mixture_mode).lower().strip()
        k_rich = getattr(self.tier_c, "k_fuel_flow_rich", 0.30)
        k_lean = getattr(self.tier_c, "k_fuel_flow_lean", self.D_FUEL)

        if per_cylinder_severities is not None and len(per_cylinder_severities) == 4:
            cyl_flows: List[float] = []
            for s_i in per_cylinder_severities:
                s_clamped = max(0.0, min(1.0, float(s_i)))
                if s_clamped > 0.0:
                    if "rich" in mode_str:
                        cyl_flows.append(m_nom_cyl * (1.0 + k_rich * s_clamped))
                    else:
                        cyl_flows.append(max(0.0, m_nom_cyl * (1.0 - k_lean * s_clamped)))
                else:
                    cyl_flows.append(m_nom_cyl)
            mass_flow_kg_s = sum(cyl_flows)
        elif sev > 0.0:
            if "rich" in mode_str:
                mass_flow_kg_s = mass_flow_nominal * (1.0 + k_rich * sev)
                cyl_flows = [mass_flow_kg_s / 4.0] * 4
            else:  # lean
                mass_flow_kg_s = max(0.0, mass_flow_nominal * (1.0 - k_lean * sev))
                cyl_flows = [mass_flow_kg_s / 4.0] * 4
        else:
            mass_flow_kg_s = mass_flow_nominal
            cyl_flows = [m_nom_cyl] * 4

        # Convert to L/h: (kg/s / (kg/L)) * 3600 s/h
        vol_flow_l_h = (mass_flow_kg_s / self.tier_c.fuel_density_kg_per_l) * 3600.0

        # BSFC in g/kWh:
        if p > 1000.0:
            bsfc = (mass_flow_kg_s / p) * 3.6e9
        else:
            bsfc = 0.0

        return FuelState(
            mass_flow_kg_s=mass_flow_kg_s,
            volumetric_flow_l_h=vol_flow_l_h,
            bsfc_g_kwh=bsfc,
            per_cylinder_mass_flow_kg_s=cyl_flows,
        )
