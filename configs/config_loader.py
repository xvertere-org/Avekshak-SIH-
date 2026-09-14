"""
Configuration loader utilities for SIH26054.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
from telemetry.schema import MissionConfig, EngineConfig


CONFIG_DIR = Path(__file__).parent


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON config file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mission_config(filepath: Union[str, Path] = CONFIG_DIR / "default_mission.json") -> MissionConfig:
    """Load and parse MissionConfig from JSON."""
    data = load_json(filepath)
    return MissionConfig.from_dict(data)


def load_engine_config(filepath: Union[str, Path] = CONFIG_DIR / "default_engine.json") -> EngineConfig:
    """Load and parse EngineConfig from JSON."""
    data = load_json(filepath)
    return EngineConfig.from_dict(data)


def load_telemetry_settings(filepath: Union[str, Path] = CONFIG_DIR / "telemetry_settings.json") -> Dict[str, Any]:
    """Load telemetry channel settings and metadata."""
    return load_json(filepath)


def load_engine_reference(filepath: Union[str, Path] = CONFIG_DIR / "engine_reference" / "rotax_914_ul_f.json") -> Dict[str, Any]:
    """Load authoritative Rotax 914 UL/F engine reference specification."""
    return load_json(filepath)


def load_physics_contract(filepath: Union[str, Path] = CONFIG_DIR / "physics_contract.json") -> Dict[str, Any]:
    """Load system-wide physics contract defining implemented vs unmodeled subsystems."""
    return load_json(filepath)

