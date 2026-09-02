"""
Validation and Calibration package for SIH26054 Aero Piston Engine Simulator.
"""

from validation.metrics import (
    check_bounds,
    check_monotonicity,
    compute_settling_time,
    compute_channel_stability,
    compute_cross_channel_correlations,
    compute_vibration_order_metrics,
)
from validation.calibration import (
    ParameterRecord,
    get_parameter_inventory,
    generate_calibration_markdown_table,
    run_inertia_sensitivity_sweep,
    run_thermal_capacitance_sweep,
)
from validation.validation_runner import ValidationRunner

__all__ = [
    "check_bounds",
    "check_monotonicity",
    "compute_settling_time",
    "compute_channel_stability",
    "compute_cross_channel_correlations",
    "compute_vibration_order_metrics",
    "ParameterRecord",
    "get_parameter_inventory",
    "generate_calibration_markdown_table",
    "run_inertia_sensitivity_sweep",
    "run_thermal_capacitance_sweep",
    "ValidationRunner",
]
