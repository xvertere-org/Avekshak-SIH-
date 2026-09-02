"""
Simulator Physics Validation and Telemetry Visualization Script.

Generates diagnostic validation plots for:
1. RPM vs Time (Throttle Step Response)
2. Throttle vs Steady-State RPM
3. CHT vs Time under high load
4. EGT vs Time under load step
5. Oil Temperature vs Time
6. Oil Pressure vs RPM & Temperature
7. Fuel Flow vs Engine Power
8. Vibration Time-Domain Signal
9. Vibration FFT Order Spectrum (1x & 2x Orders)
10. Altitude vs Atmospheric Density Factor
11. Full 6-Phase MALE UAV Mission Telemetry Overview
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.config import SimulatorConfig
from simulator.subsystems.atmosphere import Atmosphere
from simulator.subsystems.mission import MissionProfile, FlightPhase, PhaseSegment
from simulator.subsystems.dynamics import RotationalDynamics
from simulator.subsystems.fuel import FuelSystem
from simulator.subsystems.thermal import ThermalSystem
from simulator.subsystems.lubrication import LubricationSystem
from simulator.subsystems.vibration import VibrationSystem
from simulator.engine_simulator import EngineSimulator


OUTPUT_DIR = Path("docs") / "plots"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_throttle_step_rpm():
    """1 & 2: RPM response to throttle step changes."""
    sim = EngineSimulator(seed=42)
    dt = 0.05
    time_pts = []
    rpm_pts = []
    throttle_pts = []

    # Step: 0-10s idle (0%), 10-25s takeoff (100%), 25-45s cruise (75%), 45-60s loiter (55%), 60-75s idle (0%)
    steps_schedule = [
        (10.0, 0.0),
        (15.0, 100.0),
        (20.0, 75.0),
        (15.0, 55.0),
        (15.0, 0.0),
    ]

    t_curr = 0.0
    for duration, thr in steps_schedule:
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            t_curr += dt
            op = sim.dynamics.step(throttle_pct=thr, density_factor=1.0, dt=dt)
            time_pts.append(t_curr)
            rpm_pts.append(op.rpm)
            throttle_pts.append(thr)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_pts, y=rpm_pts, name="Engine RPM", line=dict(color="#00bcd4", width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=time_pts, y=throttle_pts, name="Throttle (%)", line=dict(color="#ff9800", width=1.5, dash="dot")), secondary_y=True)

    fig.update_layout(title="<b>Rotational Dynamics: Throttle-Step Transient Response</b>", template="plotly_dark", height=450)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Engine Speed (RPM)", secondary_y=False)
    fig.update_yaxes(title_text="Throttle (%)", secondary_y=True)
    fig.write_html(OUTPUT_DIR / "1_rpm_throttle_response.html")
    print(" -> Saved 1_rpm_throttle_response.html")


def plot_thermal_dynamics():
    """3, 4 & 5: CHT, EGT, and Oil Temperature thermal dynamics."""
    sim = EngineSimulator(seed=42)
    dt = 0.2
    total_time = 300.0
    time_pts, cht_pts, egt_pts, oil_t_pts = [], [], [], []

    # Run step at 80% throttle continuous climb/cruise
    for step_idx in range(int(total_time / dt)):
        t = step_idx * dt
        thr = 85.0 if t < 180.0 else 40.0
        rec = sim.step(time_step=dt)
        time_pts.append(t)
        cht_pts.append(rec.cht)
        egt_pts.append(rec.egt)
        oil_t_pts.append(rec.oil_temp)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["<b>Cylinder Head Temperature (CHT)</b>", "<b>Exhaust Gas Temperature (EGT)</b>", "<b>Engine Oil Temperature</b>"])

    fig.add_trace(go.Scatter(x=time_pts, y=cht_pts, name="CHT (°C)", line=dict(color="#e91e63", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_pts, y=egt_pts, name="EGT (°C)", line=dict(color="#ff5722", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_pts, y=oil_t_pts, name="Oil Temp (°C)", line=dict(color="#ffeb3b", width=2)), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=700, title="<b>Thermal Subsystem: CHT, EGT, and Oil Temperature Transient Convergence</b>")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.write_html(OUTPUT_DIR / "2_thermal_dynamics.html")
    print(" -> Saved 2_thermal_dynamics.html")


def plot_oil_pressure_and_fuel_flow():
    """6 & 7: Oil Pressure vs RPM & Fuel Flow vs Power."""
    sim = EngineSimulator(seed=42)
    rpms = np.linspace(1400.0, 5800.0, 50)
    powers = np.linspace(0.0, 58000.0, 50)

    lub = LubricationSystem()
    oil_p_cold = [lub.step(rpm=r, cht_c=80.0, fuel_mass_flow_kg_s=0.002, ambient_temp_c=15.0, dt=0.1).oil_pressure_bar for r in rpms]
    lub.set_oil_temp(105.0)
    oil_p_hot = [lub.step(rpm=r, cht_c=110.0, fuel_mass_flow_kg_s=0.002, ambient_temp_c=15.0, dt=0.1).oil_pressure_bar for r in rpms]

    fuel_sys = FuelSystem()
    fuel_flows = [fuel_sys.compute(p).volumetric_flow_l_h for p in powers]
    bsfc_vals = [fuel_sys.compute(p).bsfc_g_kwh for p in powers]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["<b>Oil Pressure vs RPM (Viscosity Effect)</b>", "<b>Fuel Flow vs Engine Power (Willans Line)</b>"])

    fig.add_trace(go.Scatter(x=rpms, y=oil_p_cold, name="Oil Press @ 65°C", line=dict(color="#00e676", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpms, y=oil_p_hot, name="Oil Press @ 105°C", line=dict(color="#ff9100", width=2.5, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=powers / 1000.0, y=fuel_flows, name="Fuel Flow (L/h)", line=dict(color="#00b0ff", width=2.5)), row=1, col=2)

    fig.update_layout(template="plotly_dark", height=450, title="<b>Fluids & Lubrication Physics Validation</b>")
    fig.update_xaxes(title_text="Engine Speed (RPM)", row=1, col=1)
    fig.update_yaxes(title_text="Oil Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Engine Mechanical Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Fuel Flow (L/h)", row=1, col=2)
    fig.write_html(OUTPUT_DIR / "3_oil_and_fuel_physics.html")
    print(" -> Saved 3_oil_and_fuel_physics.html")


def plot_vibration_and_fft():
    """8 & 9: Vibration waveform & FFT order spectrum."""
    vib_sys = VibrationSystem()
    rpm = 3000.0
    duration = 1.0
    fs = 1000.0

    t, sig, f1, f2 = vib_sys.generate_waveform(rpm=rpm, load_pct=75.0, duration_s=duration, sampling_rate_hz=fs)

    n = len(sig)
    fft_vals = np.abs(np.fft.rfft(sig)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["<b>Vibration Time-Domain Signal (100 ms excerpt)</b>", "<b>Vibration FFT Spectrum (Rotational Orders 1x & 2x)</b>"])

    mask_100ms = t <= 0.10
    fig.add_trace(go.Scatter(x=t[mask_100ms] * 1000.0, y=sig[mask_100ms], name="Vibration (g)", line=dict(color="#7c4dff", width=1.8)), row=1, col=1)

    mask_freq = (freqs >= 0) & (freqs <= 200)
    fig.add_trace(go.Scatter(x=freqs[mask_freq], y=fft_vals[mask_freq], name="Spectral Amplitude", line=dict(color="#00e5ff", width=2)), row=1, col=2)

    fig.add_vline(x=50.0, line_width=1.5, line_dash="dash", line_color="#ff5252", annotation_text="1x Order (50 Hz)", row=1, col=2)
    fig.add_vline(x=100.0, line_width=1.5, line_dash="dash", line_color="#ff4081", annotation_text="2x Order (100 Hz)", row=1, col=2)

    fig.update_layout(template="plotly_dark", height=450, title=f"<b>Vibration Synthesis Validation @ {rpm:.0f} RPM (Crankshaft Order Harmonics)</b>")
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (g)", row=1, col=2)
    fig.write_html(OUTPUT_DIR / "4_vibration_fft_spectrum.html")
    print(" -> Saved 4_vibration_fft_spectrum.html")


def plot_full_mission_overview():
    """10 & 11: Full 6-phase MALE UAV mission profile telemetry overview."""
    sim = EngineSimulator(seed=42)
    profile = MissionProfile()
    df = sim.run_to_dataframe(mission_profile=profile, dt=0.5)

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "<b>Flight Altitude & Throttle</b>",
            "<b>Engine Speed (RPM)</b>",
            "<b>Cylinder Head Temp (CHT)</b>",
            "<b>Exhaust Gas Temp (EGT)</b>",
            "<b>Oil Temperature</b>",
            "<b>Oil Pressure</b>",
            "<b>Fuel Flow Rate</b>",
            "<b>Vibration (RMS)</b>",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    t = df["timestamp"]
    # Row 1, Col 1
    fig.add_trace(go.Scatter(x=t, y=df["altitude"], name="Altitude (m)", line=dict(color="#00e5ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["throttle"] * 30.0, name="Throttle scaled", line=dict(color="#ff9100", dash="dot")), row=1, col=1)

    # Row 1, Col 2
    fig.add_trace(go.Scatter(x=t, y=df["rpm"], name="RPM", line=dict(color="#00e676")), row=1, col=2)

    # Row 2, Col 1
    fig.add_trace(go.Scatter(x=t, y=df["cht"], name="CHT (°C)", line=dict(color="#ff1744")), row=2, col=1)

    # Row 2, Col 2
    fig.add_trace(go.Scatter(x=t, y=df["egt"], name="EGT (°C)", line=dict(color="#ff6d00")), row=2, col=2)

    # Row 3, Col 1
    fig.add_trace(go.Scatter(x=t, y=df["oil_temp"], name="Oil Temp (°C)", line=dict(color="#ffd600")), row=3, col=1)

    # Row 3, Col 2
    fig.add_trace(go.Scatter(x=t, y=df["oil_pressure"], name="Oil Press (bar)", line=dict(color="#00b0ff")), row=3, col=2)

    # Row 4, Col 1
    fig.add_trace(go.Scatter(x=t, y=df["fuel_flow"], name="Fuel Flow (L/h)", line=dict(color="#d500f9")), row=4, col=1)

    # Row 4, Col 2
    fig.add_trace(go.Scatter(x=t, y=df["vibration"], name="Vibration (g)", line=dict(color="#651fff")), row=4, col=2)

    fig.update_layout(template="plotly_dark", height=900, title="<b>Representative MALE UAV 6-Phase Flight Mission Synthetic Telemetry Overview</b>")
    fig.write_html(OUTPUT_DIR / "5_full_mission_overview.html")
    print(" -> Saved 5_full_mission_overview.html")


def plot_timestep_comparison():
    """6: Timestep stability comparison across dt = [0.05, 0.1, 0.2, 0.5, 1.0] s."""
    timesteps = [0.05, 0.1, 0.2, 0.5, 1.0]
    colors = ["#00e5ff", "#00e676", "#ffea00", "#ff9100", "#ff1744"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["<b>Engine RPM Trajectory across Integration Timesteps</b>", "<b>Cylinder Head Temperature (CHT) Trajectory</b>"])

    profile = MissionProfile(
        segments=[
            PhaseSegment(FlightPhase.TAKEOFF, 10.0, 100.0, 100.0, 0.0, 200.0),
            PhaseSegment(FlightPhase.CRUISE, 20.0, 75.0, 75.0, 2000.0, 2000.0),
        ]
    )

    for dt, col in zip(timesteps, colors):
        sim = EngineSimulator(seed=42)
        df = sim.run_to_dataframe(mission_profile=profile, dt=dt)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rpm"], name=f"dt = {dt}s", line=dict(color=col, width=1.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cht"], name=f"dt = {dt}s", line=dict(color=col, width=1.8), showlegend=False), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=600, title="<b>Empirical Timestep Stability Sweep Comparison (dt = 0.05s to 1.0s)</b>")
    fig.update_xaxes(title_text="Simulation Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="RPM", row=1, col=1)
    fig.update_yaxes(title_text="CHT (°C)", row=2, col=1)
    fig.write_html(OUTPUT_DIR / "6_timestep_stability_comparison.html")
    print(" -> Saved 6_timestep_stability_comparison.html")


def plot_correlation_matrix():
    """7: Cross-channel correlation matrix heatmap."""
    sim = EngineSimulator(seed=42)
    profile = MissionProfile()
    df = sim.run_to_dataframe(mission_profile=profile, dt=0.5)

    channels = ["throttle", "load", "rpm", "fuel_flow", "cht", "egt", "oil_temp", "oil_pressure", "vibration"]
    corr_matrix = df[channels].corr().round(3)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=channels,
        y=channels,
        colorscale="Viridis",
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 11},
    ))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title="<b>Cross-Channel Correlation Matrix (Physical Telemetry Coherence)</b>",
    )
    fig.write_html(OUTPUT_DIR / "7_cross_channel_correlation_matrix.html")
    print(" -> Saved 7_cross_channel_correlation_matrix.html")


def plot_cooling_degradation_transient():
    """8: CHT, Oil Temperature, and Fault Severity vs Time during cooling fault and recovery."""
    from simulator.fault_interface import FaultState, FaultType
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 180.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.65,
        start_time=30.0,
        end_time=110.0,
        parameters={"ramp_duration": 15.0},
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.2)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.2, fault_schedule=fault)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[
            "<b>Cylinder Head Temperature (CHT): Healthy vs Cooling Degradation</b>",
            "<b>Oil Temperature: Natural Secondary Rise via Conduction Coupling</b>",
            "<b>Active Fault Severity & Activation Window</b>",
        ]
    )

    t = df_f["timestamp"]
    # Row 1: CHT
    fig.add_trace(go.Scatter(x=t, y=df_h["cht"], name="Nominal CHT", line=dict(color="#00e676", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["cht"], name="Degraded CHT", line=dict(color="#ff1744", width=2.5)), row=1, col=1)
    fig.add_hline(y=150.0, line_dash="dash", line_color="#ff5252", annotation_text="CHT Limit (150°C)", row=1, col=1)

    # Row 2: Oil Temp
    fig.add_trace(go.Scatter(x=t, y=df_h["oil_temp"], name="Nominal Oil Temp", line=dict(color="#00e5ff", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["oil_temp"], name="Degraded Oil Temp", line=dict(color="#ff9100", width=2.5)), row=2, col=1)

    # Row 3: Fault Severity
    fig.add_trace(go.Scatter(x=t, y=df_f["fault_severity"], name="Fault Severity", line=dict(color="#d500f9", width=2), fill="tozeroy"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=800,
        title="<b>Phase 4B: Cooling Degradation Transient Response, Secondary Coupling & Natural Recovery</b>",
    )
    fig.update_xaxes(title_text="Simulation Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="CHT (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Oil Temp (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Severity", row=3, col=1)
    fig.write_html(OUTPUT_DIR / "8_cooling_degradation_transient.html")
    print(" -> Saved 8_cooling_degradation_transient.html")


def plot_cooling_severity_sweep():
    """9: Steady-State CHT and Oil Temperature as a function of cooling fault severity."""
    from simulator.fault_interface import FaultState, FaultType
    severities = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    chts, oils = [], []

    seg = PhaseSegment(FlightPhase.CRUISE, 150.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for s in severities:
        sim = EngineSimulator(seed=42)
        f = FaultState(FaultType.COOLING_DEGRADATION, severity=s, start_time=10.0, end_time=150.0)
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=f)
        chts.append(float(df["cht"].iloc[-1]))
        oils.append(float(df["oil_temp"].iloc[-1]))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=["<b>Steady-State CHT vs Cooling Severity</b>", "<b>Steady-State Oil Temp vs Cooling Severity</b>"]
    )

    fig.add_trace(go.Scatter(x=severities, y=chts, mode="lines+markers", name="CHT (°C)", line=dict(color="#ff5252", width=2.5), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=severities, y=oils, mode="lines+markers", name="Oil Temp (°C)", line=dict(color="#ffab00", width=2.5), marker=dict(size=8)), row=1, col=2)
    fig.add_hline(y=150.0, line_dash="dash", line_color="#ff1744", annotation_text="150°C CHT Limit", row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=450,
        title="<b>Phase 4B: Steady-State Thermal Response across Cooling Severity Sweep [0.0 - 1.0]</b>"
    )
    fig.update_xaxes(title_text="Cooling Severity (0 = nominal, 1 = max degradation)", row=1, col=1)
    fig.update_xaxes(title_text="Cooling Severity (0 = nominal, 1 = max degradation)", row=1, col=2)
    fig.update_yaxes(title_text="Steady-State CHT (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Steady-State Oil Temp (°C)", row=1, col=2)
    fig.write_html(OUTPUT_DIR / "9_cooling_severity_sweep.html")
    print(" -> Saved 9_cooling_severity_sweep.html")


def plot_lubrication_degradation_transient():
    """10: Oil Pressure, Oil Temperature, and Severity vs Time during lubrication fault and recovery."""
    from simulator.fault_interface import FaultState, FaultType
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 180.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.65,
        start_time=30.0,
        end_time=110.0,
        parameters={"ramp_duration": 15.0},
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.2)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.2, fault_schedule=fault)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[
            "<b>Oil Pressure (bar): Healthy vs Lubrication Degradation Drop</b>",
            "<b>Oil Temperature (°C): Secondary Frictional & Heat Rejection Rise</b>",
            "<b>Active Lubrication Fault Severity & Activation Window</b>",
        ]
    )

    t = df_f["timestamp"]
    # Row 1: Oil Pressure
    fig.add_trace(go.Scatter(x=t, y=df_h["oil_pressure"], name="Nominal Oil Press", line=dict(color="#00e5ff", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["oil_pressure"], name="Degraded Oil Press", line=dict(color="#ff1744", width=2.5)), row=1, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ff5252", annotation_text="Low Oil Press Warning (2.0 bar)", row=1, col=1)

    # Row 2: Oil Temp
    fig.add_trace(go.Scatter(x=t, y=df_h["oil_temp"], name="Nominal Oil Temp", line=dict(color="#00e676", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["oil_temp"], name="Degraded Oil Temp", line=dict(color="#ff9100", width=2.5)), row=2, col=1)
    fig.add_hline(y=130.0, line_dash="dash", line_color="#ff1744", annotation_text="Max Oil Temp Limit (130°C)", row=2, col=1)

    # Row 3: Fault Severity
    fig.add_trace(go.Scatter(x=t, y=df_f["fault_severity"], name="Lubrication Severity", line=dict(color="#00e5ff", width=2), fill="tozeroy"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=800,
        title="<b>Phase 4C: Lubrication Degradation Transient Dynamics (Pressure Drop, Thermal Rise & Recovery)</b>",
    )
    fig.update_xaxes(title_text="Simulation Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Oil Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Oil Temp (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Severity", row=3, col=1)
    fig.write_html(OUTPUT_DIR / "10_lubrication_degradation_transient.html")
    print(" -> Saved 10_lubrication_degradation_transient.html")


def plot_lubrication_severity_sweep():
    """11: Steady-State Oil Pressure and Oil Temperature vs Lubrication Fault Severity."""
    from simulator.fault_interface import FaultState, FaultType
    severities = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    pressures, oil_temps = [], []

    seg = PhaseSegment(FlightPhase.CRUISE, 150.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for s in severities:
        sim = EngineSimulator(seed=42)
        f = FaultState(FaultType.LUBRICATION_DEGRADATION, severity=s, start_time=10.0, end_time=150.0)
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=f)
        pressures.append(float(df["oil_pressure"].iloc[-1]))
        oil_temps.append(float(df["oil_temp"].iloc[-1]))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=["<b>Steady-State Oil Pressure vs Severity (Monotonic Drop)</b>", "<b>Steady-State Oil Temp vs Severity (Monotonic Rise)</b>"]
    )

    fig.add_trace(go.Scatter(x=severities, y=pressures, mode="lines+markers", name="Oil Press (bar)", line=dict(color="#00e5ff", width=2.5), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=severities, y=oil_temps, mode="lines+markers", name="Oil Temp (°C)", line=dict(color="#ff9100", width=2.5), marker=dict(size=8)), row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ff1744", annotation_text="2.0 bar Warning", row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=450,
        title="<b>Phase 4C: Steady-State Lubrication Response across Severity Sweep [0.0 - 1.0]</b>"
    )
    fig.update_xaxes(title_text="Lubrication Severity (0 = nominal, 1 = max degradation)", row=1, col=1)
    fig.update_xaxes(title_text="Lubrication Severity (0 = nominal, 1 = max degradation)", row=1, col=2)
    fig.update_yaxes(title_text="Steady-State Oil Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Steady-State Oil Temp (°C)", row=1, col=2)
    fig.write_html(OUTPUT_DIR / "11_lubrication_severity_sweep.html")
    print(" -> Saved 11_lubrication_severity_sweep.html")


def plot_fuel_injection_transient():
    """12: Fuel Flow, EGT, RPM, and Fault Severity vs Time during scheduled Lean and Rich abnormality."""
    from simulator.fault_interface import FaultState, FaultSchedule, FaultType
    seg = PhaseSegment(FlightPhase.CRUISE, 220.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    f_lean = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.65,
        start_time=30.0,
        end_time=85.0,
        parameters={"mode": "lean", "ramp_duration": 10.0},
    )
    f_rich = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.65,
        start_time=130.0,
        end_time=185.0,
        parameters={"mode": "rich", "ramp_duration": 10.0},
    )
    sched = FaultSchedule(faults=[f_lean, f_rich])

    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    df_h = sim_h.run_to_dataframe(profile, dt=0.2)
    df_f = sim_f.run_to_dataframe(profile, dt=0.2, fault_schedule=sched)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "<b>Fuel Flow (L/h): Lean Drop vs Rich Increase</b>",
            "<b>Exhaust Gas Temperature (EGT °C): Lean Elevation vs Rich Fuel Quench</b>",
            "<b>Engine Speed (RPM): Natural Combustion Efficiency Droop</b>",
            "<b>Active Fault State & Abnormality Window</b>",
        ]
    )

    t = df_f["timestamp"]
    # Row 1: Fuel Flow
    fig.add_trace(go.Scatter(x=t, y=df_h["fuel_flow"], name="Nominal Fuel Flow", line=dict(color="#00e676", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["fuel_flow"], name="Abnormal Fuel Flow", line=dict(color="#d500f9", width=2.5)), row=1, col=1)

    # Row 2: EGT
    fig.add_trace(go.Scatter(x=t, y=df_h["egt"], name="Nominal EGT", line=dict(color="#ff9100", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["egt"], name="Abnormal EGT", line=dict(color="#ff1744", width=2.5)), row=2, col=1)
    fig.add_hline(y=800.0, line_dash="dash", line_color="#ff1744", annotation_text="EGT Max Reference Limit (800°C)", row=2, col=1)

    # Row 3: RPM
    fig.add_trace(go.Scatter(x=t, y=df_h["rpm"], name="Nominal RPM", line=dict(color="#00e5ff", width=2, dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_f["rpm"], name="Abnormal RPM", line=dict(color="#ffd600", width=2.5)), row=3, col=1)

    # Row 4: Severity
    fig.add_trace(go.Scatter(x=t, y=df_f["fault_severity"], name="Fault Severity", line=dict(color="#e040fb", width=2), fill="tozeroy"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=900,
        title="<b>Phase 4D: Fuel / Injection Abnormality Dynamic Response (Lean Window -> Recovery -> Rich Window)</b>"
    )
    fig.update_xaxes(title_text="Simulation Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Fuel Flow (L/h)", row=1, col=1)
    fig.update_yaxes(title_text="EGT (°C)", row=2, col=1)
    fig.update_yaxes(title_text="RPM", row=3, col=1)
    fig.update_yaxes(title_text="Severity", row=4, col=1)
    fig.write_html(OUTPUT_DIR / "12_fuel_injection_transient.html")
    print(" -> Saved 12_fuel_injection_transient.html")


def plot_fuel_injection_severity_sweep():
    """13: Steady-State Fuel Flow and EGT vs Severity for Lean and Rich Abnormality."""
    from simulator.fault_interface import FaultState, FaultType
    severities = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    fuel_lean, egt_lean = [], []
    fuel_rich, egt_rich = [], []

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for s in severities:
        sim_l = EngineSimulator(seed=42)
        fl = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=s, start_time=5.0, end_time=60.0, parameters={"mode": "lean"})
        df_l = sim_l.run_to_dataframe(profile, dt=0.5, fault_schedule=fl)
        fuel_lean.append(float(df_l["fuel_flow"].iloc[-1]))
        egt_lean.append(float(df_l["egt"].iloc[-1]))

        sim_r = EngineSimulator(seed=42)
        fr = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=s, start_time=5.0, end_time=60.0, parameters={"mode": "rich"})
        df_r = sim_r.run_to_dataframe(profile, dt=0.5, fault_schedule=fr)
        fuel_rich.append(float(df_r["fuel_flow"].iloc[-1]))
        egt_rich.append(float(df_r["egt"].iloc[-1]))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=["<b>Fuel Flow vs Severity: Lean vs Rich</b>", "<b>EGT vs Severity: Lean vs Rich</b>"]
    )

    # Subplot 1: Fuel Flow
    fig.add_trace(go.Scatter(x=severities, y=fuel_lean, mode="lines+markers", name="Lean Fuel Flow (L/h)", line=dict(color="#00e676", width=2.5), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=severities, y=fuel_rich, mode="lines+markers", name="Rich Fuel Flow (L/h)", line=dict(color="#d500f9", width=2.5), marker=dict(size=8)), row=1, col=1)

    # Subplot 2: EGT
    fig.add_trace(go.Scatter(x=severities, y=egt_lean, mode="lines+markers", name="Lean EGT (°C)", line=dict(color="#ff1744", width=2.5), marker=dict(size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=severities, y=egt_rich, mode="lines+markers", name="Rich EGT (°C)", line=dict(color="#2979ff", width=2.5), marker=dict(size=8)), row=1, col=2)
    fig.add_hline(y=800.0, line_dash="dash", line_color="#ff1744", annotation_text="800°C Max Limit", row=1, col=2)

    fig.update_layout(
        template="plotly_dark", height=450,
        title="<b>Phase 4D: Steady-State Lean & Rich Fuel / EGT Response across Severity Sweep [0.0 - 1.0]</b>"
    )
    fig.update_xaxes(title_text="Abnormality Severity", row=1, col=1)
    fig.update_xaxes(title_text="Abnormality Severity", row=1, col=2)
    fig.update_yaxes(title_text="Steady-State Fuel Flow (L/h)", row=1, col=1)
    fig.update_yaxes(title_text="Steady-State EGT (°C)", row=1, col=2)
    fig.write_html(OUTPUT_DIR / "13_fuel_injection_severity_sweep.html")
    print(" -> Saved 13_fuel_injection_severity_sweep.html")


def main():
    ensure_output_dir()
    print("Generating Validation & Calibration Plots (Phases 2B, 3, 4B, 4C, 4D)...")
    plot_throttle_step_rpm()
    plot_thermal_dynamics()
    plot_oil_pressure_and_fuel_flow()
    plot_vibration_and_fft()
    plot_full_mission_overview()
    plot_timestep_comparison()
    plot_correlation_matrix()
    plot_cooling_degradation_transient()
    plot_cooling_severity_sweep()
    plot_lubrication_degradation_transient()
    plot_lubrication_severity_sweep()
    plot_fuel_injection_transient()
    plot_fuel_injection_severity_sweep()
    print(f"[SUCCESS] All validation plots generated in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
