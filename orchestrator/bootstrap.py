"""
Deterministic synthetic bootstrap initialization for Phase 13.
Reuses the exact Phase 7 and Phase 8 dataset generation and model fitting routines.
Ensures zero drift, reproducible deterministic seeds, and explicit provenance.

CRITICAL:
This uses ONLY the physics-informed aero-piston engine simulator to generate synthetic calibration data.
It does NOT claim external dataset training, real aero-engine validation, or trained foundation models.
"""

import time
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from digital_twin.twin_model import DigitalTwin
from digital_twin.residuals import ResidualFrame
from telemetry.ingestion import CanonicalTelemetryFrame

from anomaly_detection.pipeline import HybridAnomalyDetector
from fault_diagnosis.schema import CANONICAL_FAULT_LABELS
from fault_diagnosis.dataset import DatasetConfig, generate_fault_diagnosis_dataset
from fault_diagnosis.features import FeatureExtractor, compute_class_weights
from fault_diagnosis.classifier import XGBoostFaultClassifier, FaultClassifierConfig
from fault_diagnosis.pipeline import FaultDiagnosisPipeline


class SyntheticBootstrapManager:
    """
    Manages deterministic synthetic bootstrap training for Phase 7 and Phase 8 models.
    Reuses existing Phase 7 and Phase 8 codebase methods directly.
    """

    @staticmethod
    def bootstrap_models(
        seed: int = 42,
        fast_mode: bool = True,
        enable_phase7_gating: bool = True,
        n_estimators_xgb: int = 30,
        max_depth_xgb: int = 4,
    ) -> Tuple[HybridAnomalyDetector, FaultDiagnosisPipeline, Dict[str, Any]]:
        """
        Execute deterministic synthetic bootstrap calibration.

        Returns:
            (fitted_anomaly_detector, fitted_diagnosis_pipeline, bootstrap_metadata)
        """
        start_time = time.perf_counter()

        # =========================================================================
        # 1. Phase 7 Hybrid Anomaly Detector Calibration (Healthy baseline only)
        # =========================================================================
        # Reuses exact Phase 7 test fixture logic: 40s steady cruise simulation
        sim = EngineSimulator(seed=seed + 58)  # Deterministic healthy seed
        seg = PhaseSegment(
            phase=FlightPhase.CRUISE,
            duration_s=35.0 if fast_mode else 50.0,
            throttle_start_pct=75.0,
            throttle_end_pct=75.0,
            altitude_start_m=2000.0,
            altitude_end_m=2000.0,
        )
        profile = MissionProfile(mission_id="BOOTSTRAP_HEALTHY_CALIB", segments=[seg])
        records = sim.run(profile, dt=1.0)
        telemetry_df = pd.DataFrame([r.to_dict() for r in records])
        canonical_frame = CanonicalTelemetryFrame(telemetry_df)

        twin = DigitalTwin()
        healthy_res_frame = twin.process_frame(canonical_frame)

        # Drop initial startup transient (< 3s)
        h_df = healthy_res_frame.to_dataframe()
        steady_h_df = h_df[h_df["timestamp"] >= 3.0].reset_index(drop=True)
        filtered_residuals = ResidualFrame(steady_h_df, metadata=healthy_res_frame.metadata)

        anomaly_detector = HybridAnomalyDetector(
            iforest_n_estimators=50 if fast_mode else 100,
            iforest_random_state=seed,
        )
        anomaly_detector.fit(filtered_residuals)

        # =========================================================================
        # 2. Phase 8 Supervised XGBoost Multiclass Diagnosis Calibration
        # =========================================================================
        # Reuses exact fault_diagnosis.dataset.generate_fault_diagnosis_dataset
        # AUDIT-FD-001 FIX: Use >=20s per class run (was 2s, giving only ~242 total rows).
        # Altitude changed from 1000m to 2000m to match live simulation conditions
        # and eliminate the domain shift that caused systematic healthy-data misclassification.
        fast_profile = MissionProfile(
            mission_id="BOOTSTRAP_DIAGNOSIS_PROFILE",
            segments=[
                PhaseSegment(
                    phase=FlightPhase.CRUISE,
                    duration_s=20.0 if fast_mode else 40.0,
                    throttle_start_pct=75.0,
                    throttle_end_pct=75.0,
                    altitude_start_m=2000.0,
                    altitude_end_m=2000.0,
                )
            ],
        )
        dataset_cfg = DatasetConfig(
            dt=1.0,
            fault_onset_time=0.0,
            mission_profile=fast_profile,
        )
        # Generate canonical 6-class dataset using Phase 8 simulator matrix
        training_dataset = generate_fault_diagnosis_dataset(config=dataset_cfg, verbose=False)

        feature_extractor = FeatureExtractor()
        feature_extractor.fit(training_dataset)
        X_train = feature_extractor.transform(training_dataset)
        y_train = training_dataset["fault_type"]

        weights = compute_class_weights(y_train)
        clf = XGBoostFaultClassifier(
            FaultClassifierConfig(
                n_estimators=n_estimators_xgb,
                max_depth=max_depth_xgb,
                random_state=seed,
            )
        )
        clf.fit(X_train, y_train, sample_weight=weights)

        diagnosis_pipeline = FaultDiagnosisPipeline(
            classifier=clf,
            feature_extractor=feature_extractor,
            enable_phase7_gating=enable_phase7_gating,
        )

        elapsed_s = time.perf_counter() - start_time

        meta = {
            "bootstrap_elapsed_seconds": round(elapsed_s, 4),
            "fast_mode": fast_mode,
            "seed": seed,
            "phase7_calibration_samples": len(steady_h_df),
            "phase8_training_samples": len(training_dataset),
            "phase8_classes": sorted(list(training_dataset["fault_type"].unique())),
            "provenance": "PHYSICS_INFORMED_SYNTHETIC_BOOTSTRAP",
            "disclaimer": (
                "Calibrated entirely on synthetic physics simulation data. "
                "No external flight-test or operational aero-engine datasets used."
            ),
        }

        return anomaly_detector, diagnosis_pipeline, meta
