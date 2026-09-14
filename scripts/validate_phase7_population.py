"""
Comprehensive Validation and Audit of Phase 7 Synthetic Engine Population.

Verifies:
1. Parameter distribution coverage and statistical moments
2. Physical constraints and conservation invariants
3. Operating-point envelope coverage across canonical missions
4. Physical engine-to-engine diversity (thermal, fuel, vibration, power)
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.population import (
    PopulationConfig,
    PopulationGenerator,
    CanonicalMission,
    simulate_engine_mission,
    validate_population,
    validate_engine_profile,
    POPULATION_DISTRIBUTIONS,
)


def main():
    parser = argparse.ArgumentParser(description="Audit and validate Phase 7 engine population.")
    parser.add_argument("--count", type=int, default=50, help="Number of engines to audit (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42).")
    args = parser.parse_args()

    print(f"=== Running Phase 7 Comprehensive Population Audit (N={args.count}) ===")
    config = PopulationConfig(
        population_id="AUDIT_ROTAX_914",
        num_engines=args.count,
        seed=args.seed,
    )
    gen = PopulationGenerator(config)
    profiles = gen.generate_population()

    # 1. Parameter Distribution & Physical Bounds Audit
    pop_report = validate_population(profiles)
    print(f"[1/4] Population Integrity: {'PASS' if pop_report.is_fully_valid else 'FAIL'}")
    print(f"      Valid engines: {pop_report.valid_profiles} / {pop_report.total_profiles}")
    print(f"      Total clipping occurrences: {sum(pop_report.clipping_counts.values())}")

    # Print sample parameter distributions
    print("\n--- Key Parameter Statistics ---")
    for param in ["c_th_cht", "h_cool_base", "c_oil", "oil_press_base_bar", "a_fuel_kg_per_j", "inertia_kg_m2"]:
        if param in pop_report.parameter_statistics:
            s = pop_report.parameter_statistics[param]
            dist = POPULATION_DISTRIBUTIONS[param]
            print(f"  {param:<20}: mean={s['mean']:.4g}, std={s['std']:.4g}, min={s['min']:.4g}, max={s['max']:.4g} (allowed: [{dist.lower_bound}, {dist.upper_bound}])")

    # 2. Operating Point Envelope Coverage
    print("\n[2/4] Testing Operating Envelope Coverage across Canonical Missions...")
    missions_to_test = [
        CanonicalMission.GROUND_IDLE,
        CanonicalMission.TAKEOFF_CLIMB,
        CanonicalMission.CRUISE,
        CanonicalMission.HIGH_ALTITUDE_CRUISE,
        CanonicalMission.DESCENT,
        CanonicalMission.RAPID_THROTTLE_TRANSITION,
        CanonicalMission.HOT_DAY_OPERATION,
    ]

    all_throttles = []
    all_altitudes = []
    all_rpms = []
    all_maps = []
    all_oats = []

    # Run a subset of engines across missions to assess envelope
    sample_engines = profiles[:5]
    for eng in sample_engines:
        for mis in missions_to_test:
            telemetry = simulate_engine_mission(
                profile=eng,
                mission=mis,
                dt=1.0,
                duration_scale=0.25,  # fast execution for audit
            )
            for rec in telemetry:
                all_throttles.append(rec.throttle)
                all_altitudes.append(rec.altitude)
                all_rpms.append(rec.rpm)
                if rec.map_bar is not None:
                    all_maps.append(rec.map_bar)
                all_oats.append(rec.ambient_temp)

    print(f"  Throttle envelope : [{min(all_throttles):.1f}%, {max(all_throttles):.1f}%]")
    print(f"  Altitude envelope : [{min(all_altitudes):.0f} m, {max(all_altitudes):.0f} m]")
    print(f"  RPM envelope      : [{min(all_rpms):.0f} RPM, {max(all_rpms):.0f} RPM]")
    if all_maps:
        print(f"  MAP envelope      : [{min(all_maps):.3f} bar, {max(all_maps):.3f} bar]")
    print(f"  Ambient Temp      : [{min(all_oats):.1f} °C, {max(all_oats):.1f} °C]")

    # 3. Engine-to-Engine Diversity Verification
    print("\n[3/4] Testing Engine-to-Engine Physical Diversity in Steady Cruise...")
    cruise_telemetries = []
    for eng in profiles[:15]:
        tel = simulate_engine_mission(
            profile=eng,
            mission=CanonicalMission.CRUISE,
            dt=0.5,
            duration_scale=0.2,
        )
        # Steady state at final step
        final_rec = tel[-1]
        cruise_telemetries.append(final_rec)

    chts = [r.cht for r in cruise_telemetries]
    egts = [r.egt for r in cruise_telemetries]
    oil_temps = [r.oil_temp for r in cruise_telemetries]
    oil_press = [r.oil_pressure for r in cruise_telemetries]
    fuel_flows = [r.fuel_flow for r in cruise_telemetries]
    vibrations = [r.vibration for r in cruise_telemetries]

    print(f"  Steady CHT spread       : mean={np.mean(chts):.2f}°C, std={np.std(chts):.2f}°C, min={np.min(chts):.2f}°C, max={np.max(chts):.2f}°C")
    print(f"  Steady EGT spread       : mean={np.mean(egts):.2f}°C, std={np.std(egts):.2f}°C, min={np.min(egts):.2f}°C, max={np.max(egts):.2f}°C")
    print(f"  Steady Oil Temp spread  : mean={np.mean(oil_temps):.2f}°C, std={np.std(oil_temps):.2f}°C, min={np.min(oil_temps):.2f}°C, max={np.max(oil_temps):.2f}°C")
    print(f"  Steady Oil Press spread : mean={np.mean(oil_press):.3f} bar, std={np.std(oil_press):.3f} bar, min={np.min(oil_press):.3f} bar, max={np.max(oil_press):.3f} bar")
    print(f"  Steady Fuel Flow spread : mean={np.mean(fuel_flows):.2f} L/h, std={np.std(fuel_flows):.2f} L/h, min={np.min(fuel_flows):.2f} L/h, max={np.max(fuel_flows):.2f} L/h")
    print(f"  Steady Vibration spread : mean={np.mean(vibrations):.3f} g, std={np.std(vibrations):.3f} g, min={np.min(vibrations):.3f} g, max={np.max(vibrations):.3f} g")

    diversity_ok = (np.std(chts) > 0.5 and np.std(oil_press) > 0.01 and np.std(fuel_flows) > 0.05)
    print(f"\n[4/4] Physical Diversity Check: {'PASS (Meaningful physical variation confirmed)' if diversity_ok else 'FAIL'}")

    overall_pass = pop_report.is_fully_valid and diversity_ok
    print(f"\n=== Overall Population Validation Result: {'PASS' if overall_pass else 'FAIL'} ===")
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
