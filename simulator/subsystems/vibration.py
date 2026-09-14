"""
Vibration Synthesis Model for SIH26054.

Implements:
1. Rotational order harmonics (1x, 2x crankshaft orders) scaled with load and mechanical condition.
2. Broadband process vibration noise.
3. Metric calculations (RMS amplitude, dominant order frequencies, and time-domain waveform generator for FFT analysis).
"""

import math
import numpy as np
from typing import NamedTuple, Optional, Tuple
from simulator.config import TierCParameters, TierDParameters


class VibrationState(NamedTuple):
    """Calculated vibration state."""
    rms_g: float
    instantaneous_g: float
    order_1x_freq_hz: float
    order_2x_freq_hz: float
    dominant_freq_hz: float
    amplitude_1x_g: float
    amplitude_2x_g: float


class VibrationSystem:
    """
    Synthesizes engine mechanical vibration signals.
    """

    def __init__(
        self,
        tier_c: TierCParameters = TierCParameters(),
        tier_d: TierDParameters = TierDParameters(),
        rng: Optional[np.random.Generator] = None,
    ):
        self.tier_c = tier_c
        self.tier_d = tier_d
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.phase_1 = 0.0
        self.phase_2 = 0.0

    def step(
        self,
        rpm: float,
        load_pct: float,
        dt: float,
        mechanical_condition: Optional[float] = None,
        mechanical_noise_factor: float = 1.0,
        misfire_imbalance_factor: float = 0.0,
    ) -> VibrationState:
        """
        Advance vibration state and calculate RMS & instantaneous vibration metrics.

        Args:
            mechanical_noise_factor: Multiplier for broadband process noise std dev.
                1.0 = nominal. Values > 1.0 model increased broadband vibration
                from mechanical degradation without affecting deterministic harmonic
                frequencies. (Tier C/D calibration assumption.)
            misfire_imbalance_factor: Combustion misfire torque deficit factor [0.0 to 1.0]
                causing 1X rotational torque ripple pulsation. (Tier C/D calibration assumption.)
        """
        load_norm = max(0.0, min(100.0, load_pct)) / 100.0
        m_cond = mechanical_condition if mechanical_condition is not None else self.tier_d.mechanical_condition

        # Fundamental rotational frequencies (deterministic, strictly tied to crankshaft RPM)
        f_rot = max(0.0, rpm) / 60.0
        f_order1 = f_rot
        f_order2 = 2.0 * f_rot

        # Amplitudes scaled with engine load, mechanical condition, and misfire torque ripple
        amp_1x = (self.tier_c.vib_order1_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond
        if misfire_imbalance_factor > 0.0:
            amp_1x += 0.55 * max(0.0, min(1.0, float(misfire_imbalance_factor)))

        amp_2x = (self.tier_c.vib_order2_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond

        # Advance harmonic phase
        self.phase_1 = (self.phase_1 + 2.0 * math.pi * f_order1 * dt) % (2.0 * math.pi)
        self.phase_2 = (self.phase_2 + 2.0 * math.pi * f_order2 * dt) % (2.0 * math.pi)

        # Process noise sample — broadband component scales with mechanical_noise_factor
        effective_noise_std = self.tier_c.vib_noise_std_g * mechanical_noise_factor
        noise_g = float(self.rng.normal(0.0, effective_noise_std))

        # Instantaneous waveform point
        inst_g = amp_1x * math.sin(self.phase_1) + amp_2x * math.sin(self.phase_2) + noise_g

        # Theoretical RMS value for multi-sine + Gaussian noise: sqrt(0.5*A1^2 + 0.5*A2^2 + sigma^2)
        rms_g = math.sqrt(0.5 * (amp_1x ** 2) + 0.5 * (amp_2x ** 2) + (effective_noise_std ** 2))

        # Dominant frequency based on largest amplitude order
        dominant_f = f_order1 if amp_1x >= amp_2x else f_order2

        return VibrationState(
            rms_g=rms_g,
            instantaneous_g=inst_g,
            order_1x_freq_hz=f_order1,
            order_2x_freq_hz=f_order2,
            dominant_freq_hz=dominant_f,
            amplitude_1x_g=amp_1x,
            amplitude_2x_g=amp_2x,
        )

    def generate_waveform(
        self,
        rpm: float,
        load_pct: float,
        duration_s: float = 1.0,
        sampling_rate_hz: float = 1000.0,
        mechanical_condition: Optional[float] = None,
        mechanical_noise_factor: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Generate a high-rate time-domain vibration waveform for FFT and spectral validation.

        Args:
            mechanical_noise_factor: Multiplier for broadband process noise std dev.

        Returns:
            (time_array, signal_array, order_1x_freq, order_2x_freq)
        """
        num_samples = int(duration_s * sampling_rate_hz)
        t = np.linspace(0.0, duration_s, num_samples, endpoint=False)

        f_order1 = max(0.0, rpm) / 60.0
        f_order2 = 2.0 * f_order1

        load_norm = max(0.0, min(100.0, load_pct)) / 100.0
        m_cond = mechanical_condition if mechanical_condition is not None else self.tier_d.mechanical_condition

        amp_1x = (self.tier_c.vib_order1_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond
        amp_2x = (self.tier_c.vib_order2_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond

        effective_noise_std = self.tier_c.vib_noise_std_g * mechanical_noise_factor
        noise = self.rng.normal(0.0, effective_noise_std, size=num_samples)
        signal = amp_1x * np.sin(2.0 * np.pi * f_order1 * t) + amp_2x * np.sin(2.0 * np.pi * f_order2 * t) + noise

        return t, signal, f_order1, f_order2
