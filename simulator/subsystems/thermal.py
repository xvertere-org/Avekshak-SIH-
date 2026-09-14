"""
Multi-Cylinder Thermal System Model for SIH26054 (Phase 2).

GOVERNING ARCHITECTURE:
Reference: BRP-Rotax 914 UL/F Engine Series.
Layout: 4-cylinder horizontally opposed boxer (1-4-3-2 firing order).
Cooling: Hybrid liquid cylinder heads (REDUCED_ORDER_COOLING_SURROGATE)
and ram-air cooled cylinder barrels.

INVARIANCE CONTRACT:
1. Exactly 4 discrete cylinder head temperatures (CHT 1-4) and runner temperatures (EGT 1-4).
2. Deterministic bank variation based on boxer geometry and ram-air ducting (MODEL_ASSUMPTION).
3. LOCAL CYLINDER-STATE INDEPENDENCE WITH SHARED-SYSTEM COUPLING: Direct local perturbation
   to cylinder 1 alters cylinder 1 state directly at that instant; cylinders 2-4 remain
   instantaneously decoupled, while shared crankshaft torque, exhaust enthalpy, and liquid
   coolant loops couple them dynamically over time.
4. Backward compatibility: Scalar cht and egt are exact arithmetic means of cylinders 1-4.
"""

import math
from typing import NamedTuple, Optional, List, Tuple
from simulator.config import TierAParameters, TierCParameters, TierCCylinderParameters


class ThermalState(NamedTuple):
    """Calculated multi-cylinder thermal state."""
    cht_c: float                           # Aggregate arithmetic mean CHT (°C)
    egt_c: float                           # Aggregate arithmetic mean EGT (°C)
    egt_steady_state_c: float              # Steady-state aggregate target (°C)
    q_gen_w: float                         # Total thermal heat generation (W)
    h_cool_w_k: float                      # Total convective cooling conductance (W/K)
    dcht_dt: float                         # Aggregate CHT derivative (°C/s)
    # Phase 2 Discrete Cylinder Channels
    cht_cyl1: float = 85.0                 # Cylinder 1 head temperature (°C)
    cht_cyl2: float = 85.0                 # Cylinder 2 head temperature (°C)
    cht_cyl3: float = 85.0                 # Cylinder 3 head temperature (°C)
    cht_cyl4: float = 85.0                 # Cylinder 4 head temperature (°C)
    egt_cyl1: float = 580.0                # Cylinder 1 runner exhaust gas temperature (°C)
    egt_cyl2: float = 580.0                # Cylinder 2 runner exhaust gas temperature (°C)
    egt_cyl3: float = 580.0                # Cylinder 3 runner exhaust gas temperature (°C)
    egt_cyl4: float = 580.0                # Cylinder 4 runner exhaust gas temperature (°C)


class ThermalSystem:
    """
    Four-Cylinder Thermal System Simulator.
    Integrates 4 independent cylinder CHT ODEs with 1-4-3-2 firing order,
    deterministic layout variations, and thermocouple lag filters for EGT.
    """

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        tier_c: TierCParameters = TierCParameters(),
        tier_c_cylinder: TierCCylinderParameters = TierCCylinderParameters(),
        initial_cht_c: Optional[float] = None,
        initial_egt_c: Optional[float] = None,
    ):
        self.tier_a = tier_a
        self.tier_c = tier_c
        self.cyl_cfg = tier_c_cylinder

        init_cht = initial_cht_c if initial_cht_c is not None else 85.0
        init_egt = initial_egt_c if initial_egt_c is not None else 580.0

        # Deterministic bank variation factors: front cylinders (1, 2) run slightly cooler
        # due to direct ram air, rear cylinders (3, 4) run slightly warmer.
        # Firing sequence: 1 - 4 - 3 - 2
        # Variations: Cyl 1 = 0.98, Cyl 2 = 1.00, Cyl 3 = 1.03, Cyl 4 = 1.01
        self.variation_factors = list(self.cyl_cfg.bank_variation_factors)

        # 4 discrete cylinder head temperature states (°C)
        self.cht_cyl: List[float] = [
            init_cht * self.variation_factors[0],
            init_cht * self.variation_factors[1],
            init_cht * self.variation_factors[2],
            init_cht * self.variation_factors[3],
        ]

        # 4 discrete exhaust gas temperature states (°C)
        self.egt_cyl: List[float] = [
            init_egt * self.variation_factors[0],
            init_egt * self.variation_factors[1],
            init_egt * self.variation_factors[2],
            init_egt * self.variation_factors[3],
        ]

        # Individual cylinder thermal capacitance (J/K)
        self.c_th_cyl = self.cyl_cfg.c_th_cylinder  # ~230 J/K per cylinder

        # External perturbation offset for cylinder isolation testing
        self.cylinder_perturbation_offsets = [0.0, 0.0, 0.0, 0.0]

    @property
    def cht_c(self) -> float:
        """Exact arithmetic mean of 4 cylinder head temperatures."""
        return sum(self.cht_cyl) / 4.0

    @property
    def egt_c(self) -> float:
        """Exact arithmetic mean of 4 cylinder exhaust gas temperatures."""
        return sum(self.egt_cyl) / 4.0

    def perturb_cylinder(self, cylinder_index: int, delta_cht_c: float) -> None:
        """
        Perturb a specific cylinder state directly (for local cylinder-state independence testing).
        cylinder_index: 0 to 3 for cylinders 1 to 4.
        """
        if 0 <= cylinder_index < 4:
            self.cht_cyl[cylinder_index] += float(delta_cht_c)

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
        coolant_temp_c: Optional[float] = None,
        per_cylinder_fuel_severities: Optional[List[float]] = None,
        per_cylinder_combustion_efficiencies: Optional[List[float]] = None,
        per_cylinder_cooling_severities: Optional[List[float]] = None,
    ) -> ThermalState:
        """
        Advance all 4 cylinder CHT ODEs and EGT thermocouple lag states by time step dt.
        Supports global and cylinder-localized causal faults (F1, F3, F4).
        """
        dt_safe = max(1e-4, float(dt))
        load_norm = max(0.0, min(100.0, load_pct)) / 100.0
        t_coolant = coolant_temp_c if coolant_temp_c is not None else 80.0

        # --- 1. Total & Per-Cylinder EGT Calculation ---
        rpm_offset = max(0.0, rpm - self.tier_c.rpm_idle)
        egt_ss_base = (
            self.tier_c.t_egt_base_c
            + self.tier_c.k_egt_load * load_norm
            + self.tier_c.k_egt_rpm * rpm_offset
            - self.tier_c.k_egt_density * density_factor
        )

        sev_fuel = max(0.0, min(1.0, float(fuel_severity)))
        mode_str = str(mixture_mode).lower().strip()
        delta_egt_mixture = 0.0
        if sev_fuel > 0.0:
            if "rich" in mode_str:
                delta_egt_mixture = -getattr(self.tier_c, "k_egt_rich_drop_c", 80.0) * sev_fuel
            else:  # lean
                delta_egt_mixture = getattr(self.tier_c, "k_egt_lean_gain_c", 95.0) * sev_fuel

        egt_ss_mean = max(100.0, egt_ss_base + delta_egt_mixture)
        tau_egt = max(0.1, self.tier_c.tau_egt_s)
        decay_egt = 1.0 - math.exp(-dt_safe / tau_egt)

        # Update 4 discrete EGT runners independently with deterministic runner variances
        # and causal per-cylinder combustion / fuel delivery shifts
        for i in range(4):
            # Per-cylinder fuel delivery factor
            if per_cylinder_fuel_severities is not None and len(per_cylinder_fuel_severities) == 4:
                s_f_i = max(0.0, min(1.0, float(per_cylinder_fuel_severities[i])))
                if "rich" in mode_str:
                    delta_egt_f_i = -getattr(self.tier_c, "k_egt_rich_drop_c", 80.0) * s_f_i
                else:
                    delta_egt_f_i = getattr(self.tier_c, "k_egt_lean_gain_c", 95.0) * s_f_i
            else:
                delta_egt_f_i = delta_egt_mixture

            # Per-cylinder combustion efficiency / misfire shift
            delta_egt_misfire_i = 0.0
            if per_cylinder_combustion_efficiencies is not None and len(per_cylinder_combustion_efficiencies) == 4:
                eta_c_i = per_cylinder_combustion_efficiencies[i]
                if eta_c_i < 1.0:
                    # Misfire causes lost reaction heat, dropping EGT strongly
                    delta_egt_misfire_i = -240.0 * (1.0 - eta_c_i)

            egt_target_i = max(100.0, egt_ss_base * self.variation_factors[i] + delta_egt_f_i + delta_egt_misfire_i)
            self.egt_cyl[i] += (egt_target_i - self.egt_cyl[i]) * decay_egt

        # --- 2. Per-Cylinder CHT Heat Balance ODEs ---
        # Total nominal convective cooling conductance (fins to ambient air)
        h_cool_nominal = (
            self.tier_c.h_cool_base
            + self.tier_c.h_cool_rpm * rpm
            + self.tier_c.h_cool_airspeed * max(0.0, airspeed_ms)
        )
        h_cool_nominal = max(1.0, h_cool_nominal)

        sev_clamped = max(0.0, min(1.0, float(cooling_severity)))
        degradation_factor = getattr(self.tier_c, "k_cooling_max_loss", 0.55) * sev_clamped
        h_cool_total = max(1.0, h_cool_nominal * (1.0 - degradation_factor))

        # Conductance and heat generation distributed to 4 individual heads
        h_cool_cyl_base = h_cool_nominal / 4.0
        q_gen_total = max(0.0, fuel_mass_flow_kg_s * self.tier_c.fuel_lhv_j_per_kg * self.tier_c.q_gen_fraction)
        q_gen_per_cyl_base = q_gen_total / 4.0
        h_head_to_coolant = 12.0  # W/K per cylinder head to liquid jacket

        dcht_dt_sum = 0.0
        for i in range(4):
            # Evaluate per-cylinder localized fuel factor
            f_fuel_i = 1.0
            if per_cylinder_fuel_severities is not None and len(per_cylinder_fuel_severities) == 4:
                s_f = max(0.0, min(1.0, float(per_cylinder_fuel_severities[i])))
                if s_f > 0.0:
                    if "rich" in mode_str:
                        f_fuel_i = 1.0 + getattr(self.tier_c, "k_fuel_flow_rich", 0.30) * s_f
                    else:
                        f_fuel_i = max(0.0, 1.0 - getattr(self.tier_c, "k_fuel_flow_lean", 0.35) * s_f)

            # Evaluate per-cylinder localized combustion efficiency factor
            eta_comb_i = 1.0
            if per_cylinder_combustion_efficiencies is not None and len(per_cylinder_combustion_efficiencies) == 4:
                eta_comb_i = max(0.0, min(1.0, float(per_cylinder_combustion_efficiencies[i])))

            # Cylinder heat generation: fuel delivery * combustion efficiency * bank factor
            q_gen_i = q_gen_per_cyl_base * self.variation_factors[i] * f_fuel_i * eta_comb_i

            # Evaluate per-cylinder localized cooling conductance
            h_cool_cyl_i = h_cool_cyl_base
            if per_cylinder_cooling_severities is not None and len(per_cylinder_cooling_severities) == 4:
                s_c = max(0.0, min(1.0, float(per_cylinder_cooling_severities[i])))
                if s_c > 0.0:
                    h_cool_cyl_i = max(0.2, h_cool_cyl_base * (1.0 - getattr(self.tier_c, "k_cooling_max_loss", 0.55) * s_c))
            elif sev_clamped > 0.0:
                h_cool_cyl_i = max(0.2, h_cool_cyl_base * (1.0 - degradation_factor))

            # Cylinder head energy balance:
            # C_th,i * dT_cht,i/dt = Q_gen,i - h_cool_i * (T_cht,i - T_amb) - h_coolant * (T_cht,i - T_coolant)
            q_fin_loss = h_cool_cyl_i * (self.cht_cyl[i] - ambient_temp_c)
            q_coolant_loss = h_head_to_coolant * (self.cht_cyl[i] - t_coolant)

            dcht_dt_i = (q_gen_i - q_fin_loss - q_coolant_loss) / self.c_th_cyl
            dcht_dt_sum += dcht_dt_i

            # Exponential decay integration
            h_eff_i = h_cool_cyl_i + h_head_to_coolant
            cht_ss_i = (q_gen_i + h_cool_cyl_i * ambient_temp_c + h_head_to_coolant * t_coolant) / max(0.1, h_eff_i)
            tau_cht_i = max(1.0, self.c_th_cyl / h_eff_i)
            decay_cht_i = 1.0 - math.exp(-dt_safe / tau_cht_i)

            self.cht_cyl[i] += (cht_ss_i - self.cht_cyl[i]) * decay_cht_i

        dcht_dt_mean = dcht_dt_sum / 4.0

        return ThermalState(
            cht_c=self.cht_c,
            egt_c=self.egt_c,
            egt_steady_state_c=egt_ss_mean,
            q_gen_w=q_gen_total,
            h_cool_w_k=h_cool_total,
            dcht_dt=dcht_dt_mean,
            cht_cyl1=self.cht_cyl[0],
            cht_cyl2=self.cht_cyl[1],
            cht_cyl3=self.cht_cyl[2],
            cht_cyl4=self.cht_cyl[3],
            egt_cyl1=self.egt_cyl[0],
            egt_cyl2=self.egt_cyl[1],
            egt_cyl3=self.egt_cyl[2],
            egt_cyl4=self.egt_cyl[3],
        )

    def set_states(self, cht_c: float, egt_c: float) -> None:
        """Directly set thermal state values uniformly across all 4 cylinders."""
        for i in range(4):
            self.cht_cyl[i] = cht_c * self.variation_factors[i]
            self.egt_cyl[i] = egt_c * self.variation_factors[i]


# Alias for subsystem naming convention
MultiCylinderThermalSubsystem = ThermalSystem

