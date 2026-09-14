"""
Weakest-link End-of-Life (EOL) evaluator for Phase 11.
Combines composite Health Index threshold with physical subsystem redlines.
Inherits Phase 9 sensor-isolation to prevent observation-layer faults from tripping physical EOL.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from prognostics.schema import EOLCriteriaConfig, EOLCriterion


class WeakestLinkEOLEvaluator:
    """
    Evaluates multi-criteria EOL functional failure limits.
    Enforces that sensor-isolated channels are excluded from physical redline evaluations.
    """

    def __init__(self, config: Optional[EOLCriteriaConfig] = None):
        self.config = config or EOLCriteriaConfig()

    def check_immediate_eol(
        self,
        current_health_index: float,
        current_telemetry: Dict[str, float],
        excluded_channels: Optional[Set[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Check if current operating condition breaches any active EOL boundary.

        Args:
            current_health_index: Current smoothed Health Index from Phase 9
            current_telemetry: Dict of current physical sensor readings
            excluded_channels: Channels isolated by Phase 9 as sensor faults

        Returns:
            Tuple of (is_breached: bool, limiting_factor: str)
        """
        excluded = excluded_channels or set()

        # 1. Check Global Health Index EOL (HI <= 0.35)
        if self.config.hi_eol.is_breached(current_health_index):
            return True, "GLOBAL_HEALTH_INDEX"

        # 2. Check Physical Redlines (only on unisolated channels)
        redlines = [
            self.config.cht_redline,
            self.config.oil_pressure_redline,
            self.config.oil_temp_redline,
            self.config.vibration_redline,
        ]

        for crit in redlines:
            if crit.channel in excluded:
                continue
            if crit.channel in current_telemetry:
                val = current_telemetry[crit.channel]
                if crit.is_breached(val):
                    return True, f"REDLINE_{crit.channel.upper()}"

        return False, "NONE"

    def find_earliest_trajectory_crossing(
        self,
        future_timestamps: np.ndarray,
        future_health_index: np.ndarray,
        future_telemetry_dict: Optional[Dict[str, np.ndarray]] = None,
        excluded_channels: Optional[Set[str]] = None,
        current_time: float = 0.0,
    ) -> Tuple[Optional[float], str]:
        """
        Find earliest threshold crossing along future trajectories.

        Args:
            future_timestamps: 1D array of future simulation timestamps
            future_health_index: 1D array of projected Health Index values
            future_telemetry_dict: Optional dict of predicted channel arrays
            excluded_channels: Channels isolated as sensor faults
            current_time: Current simulation timestamp

        Returns:
            Tuple of (earliest_rul_seconds or None, limiting_factor: str)
        """
        excluded = excluded_channels or set()
        earliest_time = None
        limiting_factor = "NONE"

        # Check Global Health Index crossing
        hi_threshold = self.config.hi_eol.threshold_value + self.config.hi_eol.tolerance
        hi_breach_indices = np.where(future_health_index <= hi_threshold)[0]
        if len(hi_breach_indices) > 0:
            idx = hi_breach_indices[0]
            earliest_time = float(future_timestamps[idx])
            limiting_factor = "GLOBAL_HEALTH_INDEX"

        # Check Physical Redlines across future telemetry
        if future_telemetry_dict is not None:
            redlines = [
                self.config.cht_redline,
                self.config.oil_pressure_redline,
                self.config.oil_temp_redline,
                self.config.vibration_redline,
            ]
            for crit in redlines:
                if crit.channel in excluded or crit.channel not in future_telemetry_dict:
                    continue
                arr = future_telemetry_dict[crit.channel]
                if len(arr) != len(future_timestamps):
                    continue

                if crit.comparison == ">=":
                    breach_idx = np.where(arr >= crit.threshold_value)[0]
                else:
                    breach_idx = np.where(arr <= crit.threshold_value)[0]

                if len(breach_idx) > 0:
                    t_breach = float(future_timestamps[breach_idx[0]])
                    if earliest_time is None or t_breach < earliest_time:
                        earliest_time = t_breach
                        limiting_factor = f"REDLINE_{crit.channel.upper()}"

        if earliest_time is not None:
            rul_s = max(0.0, earliest_time - current_time)
            return rul_s, limiting_factor

        return None, "NONE"
