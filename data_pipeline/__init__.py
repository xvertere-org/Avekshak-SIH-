"""
SIH26054 Data Pipeline — Dataset-specific preprocessing module.

This module is ISOLATED from the physics/simulator/Digital Twin implementation.
It provides dataset loaders, feature extractors, and adapters for each external dataset.

Submodules:
    common   — Shared schemas, validation, provenance, splitting
    cmapss   — NASA C-MAPSS turbofan (benchmark only)
    cwru     — CWRU bearing vibration
    femto    — FEMTO/PRONOSTIA bearing degradation
    nasa_battery — NASA battery aging (benchmark only)
    nust     — NUST IC-engine journal bearing
    paderborn — Paderborn University bearing

RESTRICTIONS:
    - No connection to physics model or production telemetry
    - C-MAPSS columns must use cmapss_s_N naming, never cht/egt/oil_temp/etc.
    - Battery variables must not map to engine channels
    - BASIC dataset is unavailable and must not be used
"""

__version__ = "0.1.0"
