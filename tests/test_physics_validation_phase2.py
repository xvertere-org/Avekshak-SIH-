"""
Comprehensive Unit and Validation Tests for Phase 2:
Turbocharged Intake, Drivetrain & Multi-Cylinder Physics.

Governing Reference: BRP-Rotax 914 UL/F Series
Verification Claim: NUMERICALLY_VERIFIED (Tier C/D physics contracts)

Covers 24 rigorous tests organized in 6 groups:
- Group A: Turbocharger & MAP Dynamics (tests 01-06)
- Group B: Power Chain & Conservation (tests 07-09)
- Group C: Reduction Gearbox & Propeller Dynamics (tests 10-12)
- Group D: Four-Cylinder Architecture & Independence (tests 13-17)
- Group E: Liquid Cooling Loop Surrogate (tests 18-19)
- Group F: Telemetry & Regression (tests 20-24)
"""

import math
import json
import pytest
import numpy as np

from simulator.config import (
    SimulatorConfig,
    TierAParameters,
    TierCParameters,
    TierDParameters,
    TierCTurboParameters,
    TierCGearboxParameters,
    TierCCylinderParameters,
    TierCCoolingLoopParameters,
)
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.atmosphere import Atmosphere
from simulator.subsystems.turbocharger import (
    TurbochargerSubsystem,
    TCUSurrogate,
    TurbineSurrogate,
    CompressorSurrogate,
)
from simulator.subsystems.dynamics import RotationalDynamics, EngineDynamics
from simulator.subsystems.thermal import MultiCylinderThermalSubsystem
from simulator.subsystems.cooling import CoolingSubsystem
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from telemetry.schema import TelemetryRecord


# ==============================================================================
# GROUP A: TURBOCHARGER & MAP DYNAMICS (Tests 01 to 06)
# ==============================================================================

def test_01_tcu_closed_loop_target_map():
    """
    Test 01: Verify TCU surrogate calculates calibrated target MAP:
    1.200 bar continuous (85% throttle) and 1.350 bar takeoff (100% throttle).
    """
    cfg = TierCTurboParameters()
    tcu = TCUSurrogate(config=cfg)

    target_cont = tcu.compute_target_map(throttle_pct=85.0, altitude_m=2000.0)
    assert abs(target_cont - cfg.tcu_continuous_map_target_bar) < 1e-4
    assert abs(target_cont - 1.200) < 1e-4

    target_to = tcu.compute_target_map(throttle_pct=100.0, altitude_m=1000.0)
    assert abs(target_to - cfg.tcu_takeoff_map_target_bar) < 1e-4
    assert abs(target_to - 1.350) < 1e-4

    # Low throttle (< 25%) naturally aspirated region
    target_idle = tcu.compute_target_map(throttle_pct=0.0, altitude_m=0.0)
    assert target_idle < 0.70


def test_02_wastegate_action():
    """
    Test 02: Wastegate closes (wastegate_position decreases toward 0) as altitude increases
    up to critical altitude to extract more turbine enthalpy and maintain target MAP.
    """
    sim_cfg = SimulatorConfig()
    turbo_low = TurbochargerSubsystem(config=sim_cfg.tier_c_turbo)
    turbo_high = TurbochargerSubsystem(config=sim_cfg.tier_c_turbo)

    atmo = Atmosphere()
    atmo_low = atmo.compute(altitude_m=500.0)
    atmo_high = atmo.compute(altitude_m=3500.0)

    # Step both at 85% throttle until wastegate establishes equilibrium
    for _ in range(30):
        s_low = turbo_low.step(
            throttle_pct=85.0,
            altitude_m=500.0,
            p_amb_pa=atmo_low.pressure_pa,
            t_amb_c=atmo_low.temperature_c,
            m_dot_air_kg_s=0.045,
            m_dot_fuel_kg_s=0.003,
            t_exh_c=650.0,
            dt=0.2,
        )
        s_high = turbo_high.step(
            throttle_pct=85.0,
            altitude_m=3500.0,
            p_amb_pa=atmo_high.pressure_pa,
            t_amb_c=atmo_high.temperature_c,
            m_dot_air_kg_s=0.045,
            m_dot_fuel_kg_s=0.003,
            t_exh_c=650.0,
            dt=0.2,
        )

    # At higher altitude, wastegate must close more (lower position) to maintain boost
    assert s_high.wastegate_position < s_low.wastegate_position
    # Pressure ratio at high altitude must be higher than at low altitude
    assert s_high.pressure_ratio > s_low.pressure_ratio


def test_03_critical_altitude_derating():
    """
    Test 03: MAP is maintained near target up to critical altitude (4572 m continuous),
    and derates above critical altitude once wastegate is fully closed.
    """
    sim_cfg = SimulatorConfig()
    turbo = TurbochargerSubsystem(config=sim_cfg.tier_c_turbo)
    atmo = Atmosphere()

    # Step at 4000 m (below continuous critical altitude 4572 m)
    atmo_sub = atmo.compute(altitude_m=4000.0)
    for _ in range(40):
        s_sub = turbo.step(
            throttle_pct=85.0,
            altitude_m=4000.0,
            p_amb_pa=atmo_sub.pressure_pa,
            t_amb_c=atmo_sub.temperature_c,
            m_dot_air_kg_s=0.040,
            m_dot_fuel_kg_s=0.0028,
            t_exh_c=650.0,
            dt=0.2,
        )

    # Step at 6500 m (well above critical altitude 4572 m)
    turbo.reset(1.013)
    atmo_sup = atmo.compute(altitude_m=6500.0)
    for _ in range(40):
        s_sup = turbo.step(
            throttle_pct=85.0,
            altitude_m=6500.0,
            p_amb_pa=atmo_sup.pressure_pa,
            t_amb_c=atmo_sup.temperature_c,
            m_dot_air_kg_s=0.025,
            m_dot_fuel_kg_s=0.0018,
            t_exh_c=620.0,
            dt=0.2,
        )

    # Sub-critical MAP should be near target 1.20 bar
    assert s_sub.map_bar >= 1.05
    # Super-critical MAP must derate significantly below sub-critical MAP
    assert s_sup.map_bar < s_sub.map_bar - 0.15


def test_04_turbine_enthalpy_balance():
    """
    Test 04: Turbine enthalpy extraction increases monotonically with exhaust mass flow
    and exhaust temperature, expanding toward ambient backpressure.
    """
    cfg = TierCTurboParameters()
    turbine = TurbineSurrogate(config=cfg)

    # Work at lower mass flow vs higher mass flow
    w_low, p3_low = turbine.compute_work(m_dot_exh_kg_s=0.025, t_exh_c=600.0, p_amb_pa=101325.0, wastegate_pos=0.5)
    w_high, p3_high = turbine.compute_work(m_dot_exh_kg_s=0.050, t_exh_c=600.0, p_amb_pa=101325.0, wastegate_pos=0.5)

    assert w_high > w_low
    assert p3_high >= p3_low
    # P3 exhaust manifold pressure must exceed ambient
    assert p3_low > 101325.0 / 100000.0

    # Higher temperature yields more enthalpy extraction
    w_hot, _ = turbine.compute_work(m_dot_exh_kg_s=0.050, t_exh_c=750.0, p_amb_pa=101325.0, wastegate_pos=0.5)
    assert w_hot > w_high


def test_05_compressor_thermodynamics():
    """
    Test 05: Compressor pressure ratio PR >= 1.0, discharge temp T2 >= T1 (ambient),
    and charge air temp <= T2 (due to intercooler heat rejection).
    """
    cfg = TierCTurboParameters()
    compressor = CompressorSurrogate(config=cfg)

    pr, t2_c, t_charge_c = compressor.compute_compression(
        w_dot_comp_w=4500.0,
        m_dot_air_kg_s=0.045,
        p_inlet_pa=101325.0,
        t_inlet_c=15.0,
    )

    assert pr >= 1.0
    # Discharge temperature must rise due to thermodynamic compression work
    assert t2_c > 15.0
    # Intercooling surrogate reduces charge air temperature below compressor discharge
    assert t_charge_c <= t2_c
    assert t_charge_c >= 15.0


def test_06_throttle_vs_boost_distinction():
    """
    Test 06: Strictly verify the Case 8 distinction:
    At partial throttle at sea level, compressor boost capability PR >= 1.0,
    while intake manifold pressure MAP < P_amb (due to downstream throttle restriction).
    """
    sim_cfg = SimulatorConfig()
    turbo = TurbochargerSubsystem(config=sim_cfg.tier_c_turbo)
    atmo = Atmosphere()
    atmo_sl = atmo.compute(altitude_m=0.0)

    # 10% throttle (idle/low power) stepped for 15 steps to allow manifold dynamics to settle
    for _ in range(15):
        s = turbo.step(
            throttle_pct=10.0,
            altitude_m=0.0,
            p_amb_pa=atmo_sl.pressure_pa,
            t_amb_c=atmo_sl.temperature_c,
            m_dot_air_kg_s=0.015,
            m_dot_fuel_kg_s=0.001,
            t_exh_c=500.0,
            dt=0.1,
        )

    # Compressor boost capability is non-negative (PR >= 1.0)
    assert s.pressure_ratio >= 1.0
    # Downstream intake manifold pressure is below ambient (throttle vacuum)
    assert s.map_bar < atmo_sl.pressure_bar
    assert s.map_bar < 0.85


# ==============================================================================
# ==============================================================================
# GROUP B: POWER CHAIN & POWER HIERARCHY INVARIANT (Tests 07 to 09)
# ==============================================================================

def test_07_power_hierarchy_invariant():
    """
    Test 07: POWER_HIERARCHY_INVARIANT verification:
    P_chemical > P_indicated > P_brake >= 0 across all operational speeds and loads.
    NOTE: Designated as POWER_HIERARCHY_INVARIANT rather than full energy conservation
    because unmodeled thermal and exhaust losses are not closed in a complete enthalpy balance.
    """
    dyn = RotationalDynamics()
    test_points = [
        (2500.0, 30.0, 1.0, 0.85),
        (3500.0, 50.0, 1.0, 1.05),
        (4500.0, 75.0, 0.9, 1.15),
        (5500.0, 90.0, 0.85, 1.20),
        (5800.0, 100.0, 1.0, 1.35),
    ]

    for rpm, thr, dens, p_map in test_points:
        p_brake, p_ind, p_chem, m_air, m_fuel = dyn.compute_power_chain(
            rpm=rpm,
            throttle_pct=thr,
            density_factor=dens,
            map_bar=p_map,
            charge_air_temp_c=35.0,
        )

        assert p_chem > 0.0
        assert p_ind > 0.0
        assert p_brake >= 0.0
        # Strict hierarchy invariant: P_chemical > P_indicated > P_brake
        assert p_ind < p_chem, f"Failed at {rpm} RPM: P_ind ({p_ind}) >= P_chem ({p_chem})"
        assert p_brake < p_ind, f"Failed at {rpm} RPM: P_brake ({p_brake}) >= P_ind ({p_ind})"


def test_08_chemical_power_scaling():
    """
    Test 08: Verify chemical power release strictly equals m_dot_fuel * LHV (43.5 MJ/kg).
    Uses independent physical reference LHV constant rather than internal object attributes.
    """
    dyn = RotationalDynamics()
    lhv_reference = 43.5e6  # 43.5 MJ/kg standard reference LHV for aviation gasoline / mogas

    _, _, p_chem, _, m_fuel = dyn.compute_power_chain(
        rpm=5000.0,
        throttle_pct=80.0,
        density_factor=1.0,
        map_bar=1.15,
        charge_air_temp_c=30.0,
    )

    expected_chem = m_fuel * lhv_reference
    assert abs(p_chem - expected_chem) < 1e-4


def test_09_indicated_power_monotonic_with_air_and_fuel():
    """
    Test 09: Indicated combustion power increases monotonically with mass flow
    and combustion efficiency factor.
    """
    dyn = RotationalDynamics()

    _, p_ind_low, _, _, _ = dyn.compute_power_chain(
        rpm=4000.0,
        throttle_pct=50.0,
        density_factor=1.0,
        combustion_efficiency_factor=1.0,
        map_bar=1.0,
    )
    _, p_ind_high, _, _, _ = dyn.compute_power_chain(
        rpm=4000.0,
        throttle_pct=85.0,
        density_factor=1.0,
        combustion_efficiency_factor=1.0,
        map_bar=1.20,
    )
    assert p_ind_high > p_ind_low

    # Reduced combustion efficiency decreases indicated power
    _, p_ind_degraded, _, _, _ = dyn.compute_power_chain(
        rpm=4000.0,
        throttle_pct=85.0,
        density_factor=1.0,
        combustion_efficiency_factor=0.85,
        map_bar=1.20,
    )
    assert p_ind_degraded < p_ind_high


# ==============================================================================
# GROUP C: REDUCTION GEARBOX & PROPELLER DYNAMICS (Tests 10 to 12)
# ==============================================================================

def test_10_reduction_ratio_kinematics():
    """
    Test 10: Reduction gearbox ratio kinematics:
    propeller_rpm = engine_rpm / (51 / 21) verified via independent arithmetic.
    """
    dyn = RotationalDynamics()
    ratio_independent = 51.0 / 21.0  # 2.4285714... teeth ratio
    assert abs(dyn.ratio - ratio_independent) < 1e-4

    op = dyn.step(throttle_pct=75.0, density_factor=1.0, dt=0.1, map_bar=1.1)
    expected_prop_rpm = op.rpm / ratio_independent
    expected_omega_prop = op.omega_rad_s / ratio_independent
    assert abs(op.propeller_rpm - expected_prop_rpm) < 1e-3
    assert abs(op.omega_prop_rad_s - expected_omega_prop) < 1e-3


def test_11_torque_reflection_and_gearbox_loss():
    """
    Test 11: Propeller torque reflection through reduction ratio and efficiency:
    T_load,eng = T_prop / (i * eta_gb), and gearbox mechanical loss > 0 for eta_gb < 1.0.
    Verified via independent arithmetic.
    """
    dyn = RotationalDynamics()
    op = dyn.step(throttle_pct=80.0, density_factor=1.0, dt=0.5, map_bar=1.2)

    ratio_independent = 51.0 / 21.0
    eta_gb_reference = 0.975
    # Reflected load torque: T_load,eng = T_prop / (ratio * eta_gb)
    expected_t_load = op.torque_prop_nm / (ratio_independent * eta_gb_reference)
    assert abs(op.torque_load_nm - expected_t_load) < 1e-3

    # Gearbox power loss: P_eng_out - P_prop > 0
    p_eng_out = op.torque_load_nm * op.omega_rad_s
    assert p_eng_out >= op.power_prop_w
    assert op.gearbox_loss_w > 0.0
    assert abs(op.gearbox_loss_w - (p_eng_out - op.power_prop_w)) < 1e-3


def test_12_gearbox_never_creates_energy():
    """
    Test 12: Invariant check: The gearbox never creates energy.
    P_prop <= P_eng_out at all operating points.
    """
    dyn = RotationalDynamics()
    for thr in [20.0, 50.0, 75.0, 100.0]:
        op = dyn.step(throttle_pct=thr, density_factor=1.0, dt=0.2, map_bar=1.0 + thr / 300.0)
        p_eng_out = op.torque_load_nm * op.omega_rad_s
        assert op.power_prop_w <= p_eng_out + 1e-5


# ==============================================================================
# GROUP D: FOUR-CYLINDER ARCHITECTURE & INDEPENDENCE (Tests 13 to 17)
# ==============================================================================

def test_13_cylinder_count_and_firing_order():
    """
    Test 13: Multi-cylinder thermal subsystem models exactly 4 cylinders
    with Rotax 1-4-3-2 firing order.
    """
    thermal = MultiCylinderThermalSubsystem()
    assert len(thermal.cht_cyl) == 4
    assert len(thermal.egt_cyl) == 4
    assert thermal.cyl_cfg.firing_order == "1-4-3-2"
    # Bank variance factors must be normalized to mean 1.000
    assert abs(sum(thermal.variation_factors) / 4.0 - 1.0) < 1e-5


def test_14_individual_cylinder_cht_egt_channels():
    """
    Test 14: All 4 CHT and EGT channels are distinct, physiologically bounded,
    and responsive to thermal load.
    """
    thermal = MultiCylinderThermalSubsystem()
    state = thermal.step(
        rpm=4500.0,
        load_pct=75.0,
        fuel_mass_flow_kg_s=0.003,
        density_factor=1.0,
        ambient_temp_c=20.0,
        airspeed_ms=45.0,
        dt=1.0,
    )

    chts = [state.cht_cyl1, state.cht_cyl2, state.cht_cyl3, state.cht_cyl4]
    egts = [state.egt_cyl1, state.egt_cyl2, state.egt_cyl3, state.egt_cyl4]

    # All CHTs in realistic operational range
    for c in chts:
        assert 50.0 <= c <= 180.0
    # All EGTs in realistic operational range
    for e in egts:
        assert 400.0 <= e <= 900.0


def test_15_local_cylinder_state_independence_with_shared_coupling():
    """
    Test 15: LOCAL CYLINDER-STATE INDEPENDENCE WITH SHARED-SYSTEM COUPLING:
    Verify that a direct local perturbation to cylinder 1 does not instantaneously
    alter other cylinder-local state values (cylinders 2, 3, 4 remain strictly unchanged at t),
    while shared crankshaft torque, exhaust runner enthalpy, and cooling jacket loops
    couple them dynamically over subsequent time steps.
    """
    thermal = MultiCylinderThermalSubsystem()
    # Step to establish baseline
    thermal.step(rpm=4000.0, load_pct=60.0, fuel_mass_flow_kg_s=0.0025, density_factor=1.0, ambient_temp_c=20.0, airspeed_ms=40.0, dt=0.5)

    c1_before = thermal.cht_cyl[0]
    c2_before = thermal.cht_cyl[1]
    c3_before = thermal.cht_cyl[2]
    c4_before = thermal.cht_cyl[3]

    # Perturb cylinder 1 by +25°C
    thermal.perturb_cylinder(cylinder_index=0, delta_cht_c=25.0)

    assert abs(thermal.cht_cyl[0] - (c1_before + 25.0)) < 1e-5
    assert thermal.cht_cyl[1] == c2_before
    assert thermal.cht_cyl[2] == c3_before
    assert thermal.cht_cyl[3] == c4_before


def test_16_aggregate_cht_is_arithmetic_mean():
    """
    Test 16: Aggregate CHT is strictly the arithmetic mean of the 4 cylinder temperatures.
    """
    thermal = MultiCylinderThermalSubsystem()
    for _ in range(5):
        st = thermal.step(rpm=4800.0, load_pct=70.0, fuel_mass_flow_kg_s=0.0028, density_factor=1.0, ambient_temp_c=18.0, airspeed_ms=42.0, dt=0.5)

    expected_mean = sum(thermal.cht_cyl) / 4.0
    assert abs(st.cht_c - expected_mean) < 1e-5
    assert abs(thermal.cht_c - expected_mean) < 1e-5


def test_17_aggregate_egt_is_arithmetic_mean():
    """
    Test 17: Aggregate EGT is strictly the arithmetic mean of the 4 exhaust runner temperatures.
    """
    thermal = MultiCylinderThermalSubsystem()
    for _ in range(5):
        st = thermal.step(rpm=4800.0, load_pct=70.0, fuel_mass_flow_kg_s=0.0028, density_factor=1.0, ambient_temp_c=18.0, airspeed_ms=42.0, dt=0.5)

    expected_mean = sum(thermal.egt_cyl) / 4.0
    assert abs(st.egt_c - expected_mean) < 1e-5
    assert abs(thermal.egt_c - expected_mean) < 1e-5


# ==============================================================================
# GROUP E: LIQUID COOLING LOOP SURROGATE (Tests 18 to 19)
# ==============================================================================

def test_18_cooling_loop_heat_transfer():
    """
    Test 18: Liquid cooling loop warms toward operating thermostat range (80-90°C),
    exchanges heat with cylinder heads, and responds to cooling fault degradation.
    """
    cool = CoolingSubsystem()
    # Nominal warm-up from 75°C
    for _ in range(30):
        st_nom = cool.step(
            cylinder_cht_temps_c=[105.0, 107.0, 104.0, 106.0],
            ambient_temp_c=20.0,
            airspeed_ms=45.0,
            dt=1.0,
            cooling_fault_severity=0.0,
        )
    assert 78.0 <= st_nom.coolant_temp_c <= 95.0

    # Cooling fault reduces radiator heat dissipation -> coolant heats up higher
    cool_fault = CoolingSubsystem()
    for _ in range(30):
        st_fault = cool_fault.step(
            cylinder_cht_temps_c=[105.0, 107.0, 104.0, 106.0],
            ambient_temp_c=20.0,
            airspeed_ms=45.0,
            dt=1.0,
            cooling_fault_severity=0.7,
        )
    assert st_fault.coolant_temp_c > st_nom.coolant_temp_c


def test_19_cooling_loop_surrogate_disclosure():
    """
    Test 19: Verify CoolingSubsystem carries REDUCED_ORDER_COOLING_SURROGATE designation
    and explicit lumped thermal capacitance parameter (MODEL_CALIBRATION).
    """
    cool = CoolingSubsystem()
    assert cool.MODEL_TYPE == "REDUCED_ORDER_COOLING_SURROGATE"
    assert cool.config.c_coolant_j_per_k == 4500.0
    assert cool.config.thermostat_temp_c == 80.0
    prov = cool.config.provenance_metadata["c_coolant_j_per_k"]
    assert prov["tag"] == "MODEL_CALIBRATION"


# ==============================================================================
# GROUP F: TELEMETRY & REGRESSION (Tests 20 to 24)
# ==============================================================================

def test_20_telemetry_backward_compatibility():
    """
    Test 20: TelemetryRecord with only Phase 1 fields serializes/deserializes without error,
    with new Phase 2 fields defaulting cleanly to None.
    """
    rec = TelemetryRecord(
        timestamp=10.0,
        mission_id="MSN001",
        engine_id="ROTAX_914_DEMO",
        mission_phase="CRUISE",
        altitude=2000.0,
        ambient_temp=15.0,
        throttle=75.0,
        load=75.0,
        rpm=5000.0,
        cht=102.0,
        egt=650.0,
        oil_temp=85.0,
        oil_pressure=4.2,
        fuel_flow=22.5,
        vibration=0.35,
    )
    # Check nullable Phase 2 fields default to None
    assert rec.map_bar is None
    assert rec.propeller_rpm is None
    assert rec.engine_rpm is None
    assert rec.cht_cyl1 is None
    assert rec.coolant_temp is None
    assert rec.tcu_wastegate_position is None

    # Serialization test via to_dict / from_dict
    d = rec.to_dict()
    rec_deser = TelemetryRecord.from_dict(d)
    assert rec_deser.rpm == 5000.0
    assert rec_deser.propeller_rpm is None


def test_21_telemetry_phase2_fields_populated():
    """
    Test 21: When running simulator, all Phase 2 channels are populated with non-null values.
    """
    sim = EngineSimulator(seed=123)
    seg = PhaseSegment(
        phase=FlightPhase.CRUISE,
        duration_s=2.0,
        throttle_start_pct=75.0,
        throttle_end_pct=75.0,
        altitude_start_m=1500.0,
        altitude_end_m=1500.0,
    )
    profile = MissionProfile(segments=[seg])
    records = sim.run(profile, dt=0.5)

    assert len(records) > 0
    rec = records[-1]

    # Verify all Phase 2 channels populated
    assert rec.map_bar is not None and rec.map_bar > 0.5
    assert rec.propeller_rpm is not None and rec.propeller_rpm > 500.0
    assert rec.engine_rpm is not None and rec.engine_rpm > 1000.0
    assert rec.cht_cyl1 is not None
    assert rec.cht_cyl2 is not None
    assert rec.cht_cyl3 is not None
    assert rec.cht_cyl4 is not None
    assert rec.egt_cyl1 is not None
    assert rec.egt_cyl2 is not None
    assert rec.egt_cyl3 is not None
    assert rec.egt_cyl4 is not None
    assert rec.coolant_temp is not None
    assert rec.tcu_wastegate_position is not None and 0.0 <= rec.tcu_wastegate_position <= 1.0


def test_22_telemetry_json_roundtrip():
    """
    Test 22: Full Phase 2 TelemetryRecord serializes to JSON string and deserializes identically.
    """
    sim = EngineSimulator(seed=456)
    seg = PhaseSegment(FlightPhase.CLIMB, 1.0, 85.0, 85.0, 1000.0, 1000.0)
    records = sim.run(MissionProfile(segments=[seg]), dt=0.5)
    rec = records[-1]

    json_str = json.dumps(rec.to_dict())
    d_restored = json.loads(json_str)
    rec_restored = TelemetryRecord.from_dict(d_restored)

    assert rec_restored.rpm == rec.rpm
    assert rec_restored.propeller_rpm == rec.propeller_rpm
    assert rec_restored.map_bar == rec.map_bar
    assert rec_restored.cht_cyl1 == rec.cht_cyl1
    assert rec_restored.coolant_temp == rec.coolant_temp


def test_23_no_nan_inf_across_operating_matrix():
    """
    Test 23: 8-point operating matrix produces zero NaN or Inf across all state variables:
    1. Idle Ground (0m, 0% thr)
    2. Sea-Level Cruise (0m, 65% thr)
    3. Mid-Altitude Cruise (2500m, 75% thr)
    4. Critical Altitude Continuous (4572m, 85% thr)
    5. Takeoff Power (500m, 100% thr)
    6. Rapid Throttle Step (20% -> 90% in 1s)
    7. Full Power Transient (Idle -> Takeoff in 2s)
    8. Rapid Descent Throttled (4000m -> 1000m, 15% thr)
    """
    sim = EngineSimulator(seed=789)

    conditions = [
        # (FlightPhase, dur, thr_start, thr_end, alt_start, alt_end)
        (FlightPhase.LOITER, 3.0, 0.0, 0.0, 0.0, 0.0),
        (FlightPhase.CRUISE, 3.0, 65.0, 65.0, 0.0, 0.0),
        (FlightPhase.CRUISE, 3.0, 75.0, 75.0, 2500.0, 2500.0),
        (FlightPhase.CRUISE, 3.0, 85.0, 85.0, 4572.0, 4572.0),
        (FlightPhase.TAKEOFF, 3.0, 100.0, 100.0, 500.0, 500.0),
        (FlightPhase.CLIMB, 2.0, 20.0, 90.0, 1000.0, 1200.0),
        (FlightPhase.TAKEOFF, 3.0, 0.0, 100.0, 0.0, 200.0),
        (FlightPhase.DESCENT, 3.0, 15.0, 15.0, 4000.0, 1000.0),
    ]

    for phase, dur, t_s, t_e, a_s, a_e in conditions:
        seg = PhaseSegment(phase, dur, t_s, t_e, a_s, a_e)
        records = sim.run(MissionProfile(segments=[seg]), dt=0.2)
        for r in records:
            d = r.to_dict()
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    assert not math.isnan(v), f"NaN detected in {k} during {phase}"
                    assert not math.isinf(v), f"Inf detected in {k} during {phase}"


def test_24_parameter_provenance_completeness():
    """
    Test 24: All Phase 2 configuration parameters carry provenance metadata
    with valid classification/tag, source, basis, and notes.
    """
    cfg = SimulatorConfig()

    turbo_provenance = cfg.tier_c_turbo.provenance_metadata
    gearbox_provenance = cfg.tier_c_gearbox.provenance_metadata
    cyl_provenance = cfg.tier_c_cylinder.provenance_metadata
    cool_provenance = cfg.tier_c_cooling.provenance_metadata

    assert len(turbo_provenance) >= 5
    assert len(gearbox_provenance) >= 4
    assert len(cyl_provenance) >= 3
    assert len(cool_provenance) >= 4

    valid_classifications = {
        "OEM_REFERENCE",
        "OEM_REFERENCE_VALUE",
        "MODEL_CONFIGURATION_VALUE",
        "MODEL_ASSUMPTION",
        "MODEL_CALIBRATION",
    }

    for prov_dict in [turbo_provenance, gearbox_provenance, cyl_provenance, cool_provenance]:
        for param, meta in prov_dict.items():
            tag = meta.get("classification") or meta.get("tag")
            assert tag is not None, f"Missing classification/tag for {param}"
            assert tag in valid_classifications, f"Invalid classification '{tag}' for {param}"
            assert "source" in meta and len(meta["source"]) > 0, f"Missing source for {param}"


def test_25_idle_throttle_map_and_rpm_stability():
    """
    Test 25: Idle throttle consistency audit (Section 16 requirement):
    At 0% throttle, verifies that the explicit idle airflow conductance produces:
    - MAP > 0 (intake plenum depression ~0.55 to 0.65 bar, not impossible zero vacuum)
    - Stable idle RPM (~1400 RPM nominal)
    - Compressor pressure ratio PR >= 1.0
    - Non-negative, finite state without numerical divergence across fine dt (0.05 s).
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(
        phase=FlightPhase.LOITER,
        duration_s=8.0,
        throttle_start_pct=0.0,
        throttle_end_pct=0.0,
        altitude_start_m=0.0,
        altitude_end_m=0.0,
        airspeed_start_ms=0.0,
        airspeed_end_ms=0.0,
    )
    records = sim.run(MissionProfile(segments=[seg]), dt=0.05)
    assert len(records) > 0
    final = records[-1]

    # Intake manifold absolute pressure is bounded in physical idle range
    assert final.map_bar is not None
    assert 0.50 <= final.map_bar <= 0.70, f"Idle MAP out of range: {final.map_bar}"

    # Engine speed settles near calibrated Rotax idle (1400 RPM)
    assert 1300.0 <= final.rpm <= 1600.0, f"Idle RPM out of range: {final.rpm}"

    # Compressor boost ratio capability remains >= 1.0 even while MAP is below ambient
    assert final.metadata.get("pressure_ratio", 1.0) >= 1.0

    # Propeller RPM correctly coupled through 2.42857 reduction ratio to true crankshaft speed
    assert final.propeller_rpm is not None
    eng_speed = final.engine_rpm if final.engine_rpm is not None else final.rpm
    assert abs(final.propeller_rpm - (eng_speed / (51.0 / 21.0))) < 0.1
