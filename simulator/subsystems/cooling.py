"""
Liquid Cooling Loop Subsystem for SIH26054.

NOMENCLATURE & FIDELITY CONTRACT:
This implementation is explicitly designated and documented as:
REDUCED_ORDER_COOLING_SURROGATE.

It models the lumped liquid cooling circuit connecting the 4 cylinder heads
to a shared radiator with an electromechanical/wax thermostat bypass.
It does NOT model individual water pump impeller cavitation, expansion bottle
multiphase vapor lines, or dual-radiator aerodynamic airflow splits.
"""

from typing import NamedTuple, List, Optional
from simulator.config import TierAParameters, TierCCoolingLoopParameters


class CoolingState(NamedTuple):
    """Thermodynamic state of the REDUCED_ORDER_COOLING_SURROGATE liquid loop."""
    coolant_temp_c: float                  # Liquid coolant temperature (°C)
    thermostat_opening_fraction: float     # 0.05 (closed bypass) to 1.00 (fully open)
    q_heads_to_coolant_w: float            # Heat transferred from cylinder heads to coolant (W)
    q_radiator_dissipated_w: float         # Heat rejected to outside air by radiator (W)
    net_heat_flow_w: float                 # Net heat accumulation rate in coolant (W)


class CoolingSubsystem:
    """
    REDUCED_ORDER_COOLING_SURROGATE.
    Lumped liquid cooling loop ODE integrator with thermostat regulation
    and airspeed-dependent radiator forced convection.
    """
    MODEL_TYPE: str = "REDUCED_ORDER_COOLING_SURROGATE"

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        config: TierCCoolingLoopParameters = TierCCoolingLoopParameters(),
        initial_coolant_temp_c: Optional[float] = None,
    ):
        self.tier_a = tier_a
        self.config = config
        self.coolant_temp_c: float = (
            initial_coolant_temp_c if initial_coolant_temp_c is not None else 75.0
        )


    def compute_thermostat_fraction(self, t_coolant_c: float) -> float:
        """
        Thermostat opening fraction f_thermo in [0.05, 1.0].
        Nominal opening threshold at 80 °C (OM-914 Section 13.1).
        Below 75 °C: minimum bypass flow (0.05).
        75 °C to 85 °C: smooth progressive opening from 0.05 to 1.00.
        Above 85 °C: fully open to main radiator (1.00).
        """
        t_open = self.config.thermostat_temp_c  # 80.0 °C
        if t_coolant_c <= t_open - 5.0:
            return 0.05
        elif t_coolant_c >= t_open + 5.0:
            return 1.00
        else:
            norm = (t_coolant_c - (t_open - 5.0)) / 10.0
            return 0.05 + norm * 0.95

    def step(
        self,
        cylinder_cht_temps_c: List[float],
        ambient_temp_c: float,
        airspeed_ms: float,
        dt: float,
        cooling_fault_severity: float = 0.0,
    ) -> CoolingState:
        """
        Integrate liquid coolant temperature ODE across timestep dt.

        Args:
            cylinder_cht_temps_c: Temperatures of cylinders 1 to 4 [°C].
            ambient_temp_c: Outside Air Temperature (OAT) [°C].
            airspeed_ms: Flight airspeed proxy [m/s].
            dt: Time step [s].
            cooling_fault_severity: Degradation factor [0.0 to 1.0].
        """
        dt_safe = max(1e-4, float(dt))
        severity = max(0.0, min(1.0, float(cooling_fault_severity)))

        # 1. Heat transferred from all 4 cylinder heads into coolant jacket
        h_head = self.config.h_head_to_coolant
        q_heads = 0.0
        for cht in cylinder_cht_temps_c:
            q_heads += h_head * (cht - self.coolant_temp_c)

        # 2. Thermostat opening fraction
        f_thermo = self.compute_thermostat_fraction(self.coolant_temp_c)

        # 3. Radiator heat rejection to outside air
        # Radiator effectiveness degrades under cooling loss fault
        fault_rad_loss = 1.0 - severity * 0.55
        h_rad = self.config.h_rad_base * (1.0 + self.config.k_rad_airspeed * max(0.0, airspeed_ms)) * fault_rad_loss
        q_radiator = h_rad * f_thermo * (self.coolant_temp_c - ambient_temp_c)

        # 4. First-law thermal energy balance ODE on lumped coolant thermal mass:
        # C_coolant * d(T_coolant)/dt = Q_heads - Q_radiator
        net_heat = q_heads - q_radiator
        c_cool = max(100.0, self.config.c_coolant_j_per_k)
        dt_coolant = (net_heat / c_cool) * dt_safe

        self.coolant_temp_c = max(ambient_temp_c, self.coolant_temp_c + dt_coolant)

        return CoolingState(
            coolant_temp_c=self.coolant_temp_c,
            thermostat_opening_fraction=f_thermo,
            q_heads_to_coolant_w=q_heads,
            q_radiator_dissipated_w=q_radiator,
            net_heat_flow_w=net_heat,
        )

    def set_temperature(self, temp_c: float) -> None:
        """Directly set coolant temperature (e.g. for deterministic testing)."""
        self.coolant_temp_c = float(temp_c)
