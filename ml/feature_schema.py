"""
Feature Schemas and Taxonomy for Grey-Box ML Integration.

Defines separate, physically meaningful feature groups:
1. Vibration features (statistical time-domain, spectral, band energy)
2. Degradation features (cycle/time index, SOH, degradation state, operating regime)
3. Physics-derived features (RESERVED for digital twin integration; not fabricated)
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
import pandas as pd


# Canonical vibration feature names across bearing & vibration datasets
VIBRATION_TIME_DOMAIN_FEATURES: Set[str] = frozenset({
    "mean", "std", "variance", "RMS", "rms", "peak", "peak_to_peak",
    "skewness", "kurtosis", "crest_factor", "shape_factor",
    "impulse_factor", "clearance_factor", "absolute_mean", "minimum", "maximum",
    "energy", "h_rms", "h_std", "h_peak", "h_peak_to_peak", "h_crest_factor",
    "h_kurtosis", "h_skewness", "h_energy", "v_rms", "v_std", "v_peak",
    "v_peak_to_peak", "v_crest_factor", "v_kurtosis", "v_skewness", "v_energy",
    "Channel 1 Kurtosis", "Channel 2 Kurtosis", "Channel 3 Kurtosis", "Channel 4 Kurtosis",
})

VIBRATION_SPECTRAL_FEATURES: Set[str] = frozenset({
    "dominant_frequency", "spectral_energy", "spectral_centroid", "frequency_band_energy",
    "band_energy_0_1500hz", "band_energy_1500_3000hz", "band_energy_3000_4500hz", "band_energy_4500_6000hz",
    "band_energy_0_5khz", "band_energy_5_15khz", "band_energy_15_32khz",
})

ALL_VIBRATION_FEATURES: Set[str] = VIBRATION_TIME_DOMAIN_FEATURES | VIBRATION_SPECTRAL_FEATURES

# Degradation features (temporal wear, cycle counts, electrochemical / physical wear markers)
DEGRADATION_FEATURES: Set[str] = frozenset({
    "time_cycles", "cycle", "health_index", "soh", "soh_clipped",
    "capacity_loss", "discharge_capacity", "reference_capacity",
    "re_electrolyte_resistance", "rct_charge_transfer_resistance",
    "operating_condition", "degradation_stage", "unit_number", "battery_id",
    "bearing_id", "rul_label", "duration_s",
})

# Physics-derived features (STRICTLY RESERVED for future Digital Twin integration)
# These represent variables from the physics synchronizer and residual calculator.
# MUST NEVER BE FABRICATED in Phase 1.
RESERVED_PHYSICS_DERIVED_FEATURES: Set[str] = frozenset({
    "physics_residuals",
    "normalized_residuals",
    "subsystem_health_scores",
    "predicted_state",
    "observed_state",
    "operating_regime",
    "map_residual",
    "cht_residual",
    "egt_residual",
    "cooling_residual",
    "fuel_flow_residual",
})


@dataclass
class FeatureClassification:
    """Categorization of a dataset's columns into discrete feature groups."""
    dataset_name: str
    vibration_features: List[str] = field(default_factory=list)
    degradation_features: List[str] = field(default_factory=list)
    reserved_physics_features: List[str] = field(default_factory=list)
    metadata_columns: List[str] = field(default_factory=list)
    unclassified_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "vibration_features": self.vibration_features,
            "degradation_features": self.degradation_features,
            "reserved_physics_features": self.reserved_physics_features,
            "metadata_columns": self.metadata_columns,
            "unclassified_columns": self.unclassified_columns,
        }


class FeatureSchema:
    """
    Validates and classifies features against standard grey-box schema definitions.
    """

    METADATA_COLUMN_PATTERNS: Set[str] = frozenset({
        "dataset_name", "source_file", "source_id", "record_id", "window_id",
        "measurement_id", "source_channel", "source_checksum", "preprocessing_version",
        "feature_version", "split", "subset", "file_index", "total_files",
        "file_sequence", "source_bearing_id", "source_run_id", "dataset_variant",
        "window_start", "window_end", "window_length", "sensor_location",
        "sampling_frequency", "original_sampling_rate", "RPM", "load_condition",
        "speed_rpm", "torque_nm", "radial_force_n", "motor_load_hp", "rpm",
        "Time", "Demand 1", "Control 1", "Output Drive 1", "Channel 1", "Channel 2",
        "Channel 3", "Channel 4", "Rear Input 1", "Rear Input 2", "Rear Input 3",
        "Rear Input 4", "Rear Input 5", "Rear Input 6", "Rear Input 7", "Rear Input 8",
        "voltage_mean", "voltage_min", "voltage_max", "current_mean", "current_min", "current_max",
        "temperature_mean", "temperature_min", "temperature_max", "ambient_temperature",
        "fault_label", "fault_size", "fault_size_mils", "bearing_condition",
        "humidity_pct", "temperature_celsius", "operating_region",
    })

    @classmethod
    def classify_columns(cls, columns: List[str], dataset_name: str = "") -> FeatureClassification:
        """Classify a list of columns into respective feature groups."""
        classification = FeatureClassification(dataset_name=dataset_name)

        for col in columns:
            if col in ALL_VIBRATION_FEATURES:
                classification.vibration_features.append(col)
            elif col in DEGRADATION_FEATURES:
                classification.degradation_features.append(col)
            elif col in RESERVED_PHYSICS_DERIVED_FEATURES:
                classification.reserved_physics_features.append(col)
            elif col in cls.METADATA_COLUMN_PATTERNS or any(p in col.lower() for p in ["id", "source", "time", "setting", "cmapss_s"]):
                classification.metadata_columns.append(col)
            else:
                classification.unclassified_columns.append(col)

        return classification

    @classmethod
    def assert_no_fabricated_physics(cls, columns: List[str]) -> None:
        """
        Verify that Phase 1 datasets do not contain fabricated or mock physics residual columns.
        Physics residuals must only come from the Phase 5/Phase 6 Digital Twin integration.
        """
        present = [c for c in columns if c in RESERVED_PHYSICS_DERIVED_FEATURES]
        if present:
            raise ValueError(
                f"Fabricated physics features detected in Phase 1 data: {present}. "
                f"Physics features must be generated solely by the digital twin synchronizer in later phases."
            )
