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
    ) -> VibrationState:
        """
        Advance vibration state and calculate RMS & instantaneous vibration metrics.
        """
        load_norm = max(0.0, min(100.0, load_pct)) / 100.0
        m_cond = mechanical_condition if mechanical_condition is not None else self.tier_d.mechanical_condition

        # Fundamental rotational frequencies
        f_rot = max(0.0, rpm) / 60.0
        f_order1 = f_rot
        f_order2 = 2.0 * f_rot

        # Amplitudes scaled with engine load and mechanical condition
        amp_1x = (self.tier_c.vib_order1_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond
        amp_2x = (self.tier_c.vib_order2_base_g + self.tier_c.vib_load_gain * load_norm) * m_cond

        # Advance harmonic phase
        self.phase_1 = (self.phase_1 + 2.0 * math.pi * f_order1 * dt) % (2.0 * math.pi)
        self.phase_2 = (self.phase_2 + 2.0 * math.pi * f_order2 * dt) % (2.0 * math.pi)

        # Process noise sample
        noise_g = float(self.rng.normal(0.0, self.tier_c.vib_noise_std_g))

        # Instantaneous waveform point
        inst_g = amp_1x * math.sin(self.phase_1) + amp_2x * math.sin(self.phase_2) + noise_g

        # Theoretical RMS value for multi-sine + Gaussian noise: sqrt(0.5*A1^2 + 0.5*A2^2 + sigma^2)
        rms_g = math.sqrt(0.5 * (amp_1x ** 2) + 0.5 * (amp_2x ** 2) + (self.tier_c.vib_noise_std_g ** 2))

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
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Generate a high-rate time-domain vibration waveform for FFT and spectral validation.

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

        noise = self.rng.normal(0.0, self.tier_c.vib_noise_std_g, size=num_samples)
        signal = amp_1x * np.sin(2.0 * np.pi * f_order1 * t) + amp_2x * np.sin(2.0 * np.pi * f_order2 * t) + noise

        return t, signal, f_order1, f_order2
