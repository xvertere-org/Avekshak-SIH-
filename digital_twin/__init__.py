"""
Digital Twin module for SIH26054 Aero Piston Engine Digital Twin.

Includes:
- Physics-informed nominal state estimation (twin_model.py)
- Dynamic residual generation and schema (residuals.py)
"""

from digital_twin.twin_model import DigitalTwin, DigitalTwinModel
from digital_twin.residuals import (
    ResidualFrame,
    ResidualGenerator,
    DEFAULT_RESIDUAL_SCALES,
    SUPPORTED_RESIDUAL_CHANNELS,
)

__all__ = [
    "DigitalTwin",
    "DigitalTwinModel",
    "ResidualFrame",
    "ResidualGenerator",
    "DEFAULT_RESIDUAL_SCALES",
    "SUPPORTED_RESIDUAL_CHANNELS",
]
