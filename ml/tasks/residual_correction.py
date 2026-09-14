"""
Phase 3: Physics–ML Residual Integration for Aero-Piston Grey-Box Digital Twin.

Architectural Objectives:
1. Physics baseline: deterministic physics-informed predictions using DigitalTwinModel.
2. Residual learning:
   residual = observed_target - physics_prediction
   corrected_prediction = physics_prediction + predicted_residual
3. Leakage safety: Preprocessor and ML models fitted STRICTLY on training split/groups.
4. Three-way baseline comparison: Physics-Only vs. Pure ML vs. Grey-Box (Physics + ML).
5. Explainability: Deterministic attribution separating physics drivers, residual drivers,
   and health indicators without claiming causality.
6. Target mapping: Explicitly documented mapping between inputs, physics states,
   observed sensors, and residual targets.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from digital_twin.twin_model import DigitalTwinModel
from ml.tasks.preprocessing import LeakageSafePreprocessor
from digital_twin.quality import PHYSICAL_INSTRUMENT_LIMITS


# =====================================================================
# Canonical Channels, Mappings, and Units
# =====================================================================

PHYSICS_INPUT_CHANNELS: List[str] = [
    "throttle",
    "altitude",
    "ambient_temp",
]

# Monitored engine targets where both observed telemetry and physics baseline exist
CANONICAL_TARGET_CHANNELS: List[str] = [
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "rpm",
    "map_bar",
    "coolant_temp",
    "vibration",
]

# Mapping from observed telemetry channel name to DigitalTwinModel output name
TARGET_TO_PHYSICS_MAP: Dict[str, str] = {
    "cht": "cht_expected",
    "egt": "egt_expected",
    "oil_temp": "oil_temp_expected",
    "oil_pressure": "oil_pressure_expected",
    "fuel_flow": "fuel_flow_expected",
    "rpm": "rpm_expected",
    "map_bar": "map_bar_expected",
    "coolant_temp": "coolant_temp_expected",
    "vibration": "vibration_expected",
}

# Standard engineering units per channel
CHANNEL_UNITS: Dict[str, str] = {
    "throttle": "%",
    "altitude": "m",
    "ambient_temp": "°C",
    "airspeed": "m/s",
    "cht": "°C",
    "egt": "°C",
    "oil_temp": "°C",
    "oil_pressure": "bar",
    "fuel_flow": "L/h",
    "rpm": "RPM",
    "map_bar": "bar",
    "coolant_temp": "°C",
    "vibration": "g",
    "load": "%",
}

# Physical bounds for validation and numerical stability
CHANNEL_PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "throttle": (0.0, 100.0),
    "altitude": (-500.0, 15000.0),
    "ambient_temp": (-60.0, 60.0),
    "airspeed": (0.0, 150.0),
    "cht": (0.0, 300.0),
    "egt": (0.0, 1100.0),
    "oil_temp": (-30.0, 180.0),
    "oil_pressure": (0.0, 15.0),
    "fuel_flow": (0.0, 60.0),
    "rpm": (0.0, 7500.0),
    "map_bar": (0.1, 3.5),
    "coolant_temp": (-30.0, 150.0),
    "vibration": (0.0, 25.0),
}


@dataclass
class ResidualCorrectionConfig:
    """Configuration for Grey-Box Residual Learning."""
    targets: List[str] = field(default_factory=lambda: list(CANONICAL_TARGET_CHANNELS))
    model_type: str = "ridge"  # "ridge", "gradient_boosting", or "random_forest"
    ridge_alpha: float = 1.0
    random_state: int = 42
    clipping_safety: bool = True
    default_dt: float = 0.1
    mission_phase: str = "CRUISE"


@dataclass
class ChannelPrediction:
    """Channel-level grey-box prediction output."""
    channel: str
    unit: str
    physics_estimate: float
    predicted_residual: float
    corrected_prediction: float
    observed_target: Optional[float] = None
    detected_deviation: Optional[float] = None  # observed - physics_estimate
    model_confidence: float = 1.0


@dataclass
class GreyBoxPrediction:
    """Full system state prediction across all monitored channels."""
    timestamp: float
    channels: Dict[str, ChannelPrediction]
    operating_context: Dict[str, float]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelComparisonMetrics:
    """Statistical evaluation metrics comparing models on a held-out test partition."""
    target: str
    unit: str
    n_samples: int
    physics_only_mae: float
    physics_only_rmse: float
    physics_only_r2: Optional[float]
    pure_ml_mae: float
    pure_ml_rmse: float
    pure_ml_r2: Optional[float]
    grey_box_mae: float
    grey_box_rmse: float
    grey_box_r2: Optional[float]
    delta_mae_vs_physics: float            # Positive indicates grey-box MAE is lower (better)
    delta_mae_vs_pure_ml: float            # Positive indicates grey-box MAE is lower (better)
    performance_vs_physics: str            # "IMPROVED", "DEGRADED", or "SIMILAR"
    performance_vs_pure_ml: str            # "IMPROVED", "DEGRADED", or "SIMILAR"
    error_reduction_vs_physics_pct: float  # Percentage reduction; negative indicates error increased
    error_reduction_vs_pure_ml_pct: float  # Percentage reduction; negative indicates error increased
    r2_meaningful: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class ModelComparisonResult:
    """Overall Phase 3 Model Comparison summary across all targets."""
    model_type: str
    train_samples: int
    test_samples: int
    evaluated_targets: List[str]
    channel_metrics: Dict[str, ChannelComparisonMetrics]
    mean_error_reduction_vs_physics_pct: float
    mean_error_reduction_vs_pure_ml_pct: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["channel_metrics"] = {k: v.to_dict() for k, v in self.channel_metrics.items()}
        return d


@dataclass
class ExplanationContribution:
    """Local attribution of feature contributions to physics and residual correction."""
    target: str
    physics_drivers: Dict[str, float]       # Ambient and demand features driving baseline
    residual_drivers: Dict[str, float]      # Operational dynamics driving ML correction
    health_indicator: Dict[str, Any]        # Z-score / deviation indicators
    disclaimer: str = "Feature attributions reflect local model sensitivities and do not establish physical causation."


# =====================================================================
# Grey-Box Residual Pipeline
# =====================================================================

class GreyBoxResidualPipeline:
    """
    Deterministic Grey-Box Residual Learning Pipeline for Aero-Piston Engines.
    
    Couples a deterministic physics model (DigitalTwinModel) with leakage-safe
    machine-learning residual regressors.
    """

    def __init__(self, config: Optional[ResidualCorrectionConfig] = None):
        self.config = config or ResidualCorrectionConfig()
        self.twin_model = DigitalTwinModel()

        # Preprocessors fitted strictly on training data
        self.preprocessor_pure_ml = LeakageSafePreprocessor(with_imputer=True)
        self.preprocessor_residual = LeakageSafePreprocessor(with_imputer=True)

        # Trained models per target
        self.pure_ml_models: Dict[str, Any] = {}
        self.residual_models: Dict[str, Any] = {}

        # Feature column definitions
        self.pure_ml_features_: List[str] = []
        self.residual_features_: List[str] = []
        self.is_fitted_: bool = False

    # -----------------------------------------------------------------
    # Data Validation and Alignment
    # -----------------------------------------------------------------

    @staticmethod
    def validate_telemetry_dataframe(
        df: pd.DataFrame,
        require_monotonic_time: bool = True,
        clamp_limits: bool = True,
    ) -> pd.DataFrame:
        """
        Validate input telemetry dataframe for monotonicity, unit boundaries, and NaN safety.
        Returns a validated and safely cleaned DataFrame copy.
        """
        df_clean = df.copy()

        # 1. Monotonic timestamp check
        if "timestamp" in df_clean.columns and require_monotonic_time:
            t_diff = df_clean["timestamp"].diff().dropna()
            if (t_diff <= 0).any():
                non_mono_idx = (t_diff <= 0).to_numpy().nonzero()[0]
                raise ValueError(
                    f"Telemetry timestamps must be strictly monotonically increasing. "
                    f"Non-monotonic progression found at row {non_mono_idx[0] + 1}."
                )

        # 2. Check and safely handle physical limits
        for col, (min_val, max_val) in CHANNEL_PHYSICAL_LIMITS.items():
            if col in df_clean.columns:
                # Detect invalid values (NaN or Inf)
                series = pd.to_numeric(df_clean[col], errors="coerce")
                if clamp_limits:
                    # Clip out-of-range sensor readings to safe instrument limits
                    df_clean[col] = series.clip(lower=min_val, upper=max_val)
                else:
                    out_of_bounds = (series < min_val) | (series > max_val)
                    if out_of_bounds.any():
                        raise ValueError(
                            f"Channel '{col}' has values outside physical limits [{min_val}, {max_val}]."
                        )

        return df_clean

    # -----------------------------------------------------------------
    # Physics Prediction
    # -----------------------------------------------------------------

    def generate_physics_predictions(
        self,
        df: pd.DataFrame,
        dt: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Execute deterministic physics model across input telemetry records.
        Decoupled: strictly uses observable conditions (throttle, altitude, ambient_temp).
        """
        physics_records = []
        self.twin_model.reset()

        step_dt = dt or self.config.default_dt
        last_time = None

        for _, row in df.iterrows():
            curr_time = row.get("timestamp", None)
            if last_time is not None and curr_time is not None:
                calc_dt = max(1e-4, float(curr_time - last_time))
            else:
                calc_dt = step_dt
            last_time = curr_time

            throttle = float(row.get("throttle", 75.0))
            altitude = float(row.get("altitude", 2000.0))
            amb_temp = float(row.get("ambient_temp", 15.0))
            airspeed = float(row.get("airspeed", 45.0)) if "airspeed" in row else None
            phase = str(row.get("mission_phase", self.config.mission_phase))

            exp_dict = self.twin_model.step_expected(
                throttle_pct=throttle,
                altitude_m=altitude,
                ambient_temp_c=amb_temp,
                dt=calc_dt,
                airspeed_ms=airspeed,
                mission_phase=phase,
            )
            physics_records.append(exp_dict)

        return pd.DataFrame(physics_records, index=df.index)

    # -----------------------------------------------------------------
    # Feature Construction
    # -----------------------------------------------------------------

    def _build_feature_matrices(
        self,
        df: pd.DataFrame,
        physics_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build feature matrices for Pure ML and Residual models.
        
        Pure ML features:
        - Observable flight context: throttle, altitude, ambient_temp
        - Derived kinematics/rates: delta throttle, delta altitude
        
        Residual features:
        - Observable flight context
        - Physics expected states (so the residual model knows what the physics predicted)
        - Derived context interaction terms
        """
        # Base observable features
        feat_base = pd.DataFrame(index=df.index)
        for col in PHYSICS_INPUT_CHANNELS:
            feat_base[col] = df[col].astype(float) if col in df.columns else 0.0

        if "airspeed" in df.columns:
            feat_base["airspeed"] = df["airspeed"].astype(float)

        # Operational derivatives
        feat_base["d_throttle"] = feat_base["throttle"].diff().fillna(0.0)
        feat_base["d_altitude"] = feat_base["altitude"].diff().fillna(0.0)

        # Pure ML features
        X_pure_ml = feat_base.copy()

        # Residual features: includes physics state predictions
        X_residual = feat_base.copy()
        for ch in self.config.targets:
            phys_col = TARGET_TO_PHYSICS_MAP.get(ch)
            if phys_col and phys_col in physics_df.columns:
                X_residual[f"phys_{ch}"] = physics_df[phys_col].astype(float)

        return X_pure_ml, X_residual

    # -----------------------------------------------------------------
    # Model Instantiation
    # -----------------------------------------------------------------

    def _create_regressor(self) -> Any:
        """Instantiate regressor according to configuration."""
        m_type = self.config.model_type.lower()
        if m_type == "ridge":
            return Ridge(alpha=self.config.ridge_alpha, random_state=self.config.random_state)
        elif m_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=self.config.random_state,
            )
        elif m_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unsupported model_type '{self.config.model_type}'. Choose 'ridge', 'gradient_boosting', or 'random_forest'.")

    # -----------------------------------------------------------------
    # Training (Strictly Leakage-Free on Train Partition)
    # -----------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        dt: Optional[float] = None,
    ) -> "GreyBoxResidualPipeline":
        """
        Fit Pure ML and Residual models strictly using the training partition.
        Scalers, imputers, and regressors NEVER see validation or test data.
        """
        train_clean = self.validate_telemetry_dataframe(train_df)
        physics_train = self.generate_physics_predictions(train_clean, dt=dt)

        X_pure_train, X_res_train = self._build_feature_matrices(train_clean, physics_train)

        self.pure_ml_features_ = list(X_pure_train.columns)
        self.residual_features_ = list(X_res_train.columns)

        # Fit preprocessors strictly on training features
        X_pure_scaled = self.preprocessor_pure_ml.fit_transform(X_pure_train, self.pure_ml_features_)
        X_res_scaled = self.preprocessor_residual.fit_transform(X_res_train, self.residual_features_)

        self.pure_ml_models.clear()
        self.residual_models.clear()

        # Train models for each target channel
        for target in self.config.targets:
            if target not in train_clean.columns:
                continue

            phys_col = TARGET_TO_PHYSICS_MAP.get(target)
            if not phys_col or phys_col not in physics_train.columns:
                continue

            y_obs = train_clean[target].to_numpy(dtype=float)
            y_phys = physics_train[phys_col].to_numpy(dtype=float)

            # Target residual: observed - physics
            residual_target = y_obs - y_phys

            # 1. Train Pure ML model: X_pure -> y_obs
            pure_model = self._create_regressor()
            pure_model.fit(X_pure_scaled, y_obs)
            self.pure_ml_models[target] = pure_model

            # 2. Train Residual model: X_res -> residual
            res_model = self._create_regressor()
            res_model.fit(X_res_scaled, residual_target)
            self.residual_models[target] = res_model

        self.is_fitted_ = True
        return self

    # -----------------------------------------------------------------
    # Inference Methods
    # -----------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        dt: Optional[float] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Produce physics-only, pure ML, residual, and corrected grey-box predictions.
        
        Returns:
            Dictionary with keys:
            - 'physics': {target: array}
            - 'pure_ml': {target: array}
            - 'residual_ml': {target: array}
            - 'corrected': {target: array}
        """
        if not self.is_fitted_:
            raise RuntimeError("GreyBoxResidualPipeline must be fitted before calling predict().")

        df_clean = self.validate_telemetry_dataframe(df)
        physics_df = self.generate_physics_predictions(df_clean, dt=dt)

        X_pure, X_res = self._build_feature_matrices(df_clean, physics_df)
        X_pure_scaled = self.preprocessor_pure_ml.transform(X_pure)
        X_res_scaled = self.preprocessor_residual.transform(X_res)

        out_physics: Dict[str, np.ndarray] = {}
        out_pure_ml: Dict[str, np.ndarray] = {}
        out_residual: Dict[str, np.ndarray] = {}
        out_corrected: Dict[str, np.ndarray] = {}

        for target in self.config.targets:
            phys_col = TARGET_TO_PHYSICS_MAP.get(target)
            if not phys_col or phys_col not in physics_df.columns:
                continue

            y_phys = physics_df[phys_col].to_numpy(dtype=float)
            out_physics[target] = y_phys

            # Pure ML prediction
            if target in self.pure_ml_models:
                y_pure = self.pure_ml_models[target].predict(X_pure_scaled)
                out_pure_ml[target] = y_pure

            # Residual prediction and corrected prediction
            if target in self.residual_models:
                y_res = self.residual_models[target].predict(X_res_scaled)
                out_residual[target] = y_res

                # CORRECTED PREDICTION: physics + residual
                y_corrected = y_phys + y_res

                # Optional physical safety clamping
                if self.config.clipping_safety and target in CHANNEL_PHYSICAL_LIMITS:
                    min_lim, max_lim = CHANNEL_PHYSICAL_LIMITS[target]
                    y_corrected = np.clip(y_corrected, min_lim, max_lim)

                out_corrected[target] = y_corrected

        return {
            "physics": out_physics,
            "pure_ml": out_pure_ml,
            "residual_ml": out_residual,
            "corrected": out_corrected,
        }

    # -----------------------------------------------------------------
    # Three-Way Model Comparison
    # -----------------------------------------------------------------

    def evaluate_three_way(
        self,
        test_df: pd.DataFrame,
        dt: Optional[float] = None,
    ) -> ModelComparisonResult:
        """
        Execute reproducible 3-way evaluation on a held-out test partition.
        Compares:
        1. Physics-only
        2. Pure ML
        3. Grey-Box (Physics + ML residual)
        """
        predictions = self.predict(test_df, dt=dt)
        df_clean = self.validate_telemetry_dataframe(test_df)

        channel_metrics: Dict[str, ChannelComparisonMetrics] = {}
        reductions_vs_phys: List[float] = []
        reductions_vs_pure: List[float] = []

        for target in self.config.targets:
            if target not in df_clean.columns or target not in predictions["corrected"]:
                continue

            y_obs = df_clean[target].to_numpy(dtype=float)
            y_phys = predictions["physics"][target]
            y_pure = predictions["pure_ml"].get(target, y_phys)
            y_grey = predictions["corrected"][target]

            n = len(y_obs)
            if n == 0:
                continue

            # Physics-only metrics
            phys_mae = float(np.mean(np.abs(y_obs - y_phys)))
            phys_rmse = float(np.sqrt(np.mean((y_obs - y_phys) ** 2)))

            # Pure ML metrics
            pure_mae = float(np.mean(np.abs(y_obs - y_pure)))
            pure_rmse = float(np.sqrt(np.mean((y_obs - y_pure) ** 2)))

            # Grey-box metrics
            grey_mae = float(np.mean(np.abs(y_obs - y_grey)))
            grey_rmse = float(np.sqrt(np.mean((y_obs - y_grey) ** 2)))

            # R^2 calculation (check if target variance is non-zero)
            y_var = np.var(y_obs)
            r2_valid = bool(y_var > 1e-6)

            if r2_valid:
                phys_r2 = float(1.0 - (np.sum((y_obs - y_phys) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)))
                pure_r2 = float(1.0 - (np.sum((y_obs - y_pure) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)))
                grey_r2 = float(1.0 - (np.sum((y_obs - y_grey) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)))
                notes = None
            else:
                phys_r2 = None
                pure_r2 = None
                grey_r2 = None
                notes = "Target variance is zero; R² is undefined for constant target."

            # Absolute MAE difference: positive means grey-box MAE is smaller (better)
            d_mae_phys = float(phys_mae - grey_mae)
            d_mae_pure = float(pure_mae - grey_mae)

            if d_mae_phys > 1e-3:
                perf_phys = "IMPROVED"
            elif d_mae_phys < -1e-3:
                perf_phys = "DEGRADED"
            else:
                perf_phys = "SIMILAR"

            if d_mae_pure > 1e-3:
                perf_pure = "IMPROVED"
            elif d_mae_pure < -1e-3:
                perf_pure = "DEGRADED"
            else:
                perf_pure = "SIMILAR"

            # Relative percentage error reduction: (baseline_mae - grey_mae) / baseline_mae * 100
            red_phys = float(((phys_mae - grey_mae) / phys_mae * 100.0) if phys_mae > 1e-6 else 0.0)
            red_pure = float(((pure_mae - grey_mae) / pure_mae * 100.0) if pure_mae > 1e-6 else 0.0)

            reductions_vs_phys.append(red_phys)
            reductions_vs_pure.append(red_pure)

            channel_metrics[target] = ChannelComparisonMetrics(
                target=target,
                unit=CHANNEL_UNITS.get(target, ""),
                n_samples=n,
                physics_only_mae=round(phys_mae, 4),
                physics_only_rmse=round(phys_rmse, 4),
                physics_only_r2=round(phys_r2, 4) if phys_r2 is not None else None,
                pure_ml_mae=round(pure_mae, 4),
                pure_ml_rmse=round(pure_rmse, 4),
                pure_ml_r2=round(pure_r2, 4) if pure_r2 is not None else None,
                grey_box_mae=round(grey_mae, 4),
                grey_box_rmse=round(grey_rmse, 4),
                grey_box_r2=round(grey_r2, 4) if grey_r2 is not None else None,
                delta_mae_vs_physics=round(d_mae_phys, 4),
                delta_mae_vs_pure_ml=round(d_mae_pure, 4),
                performance_vs_physics=perf_phys,
                performance_vs_pure_ml=perf_pure,
                error_reduction_vs_physics_pct=round(red_phys, 2),
                error_reduction_vs_pure_ml_pct=round(red_pure, 2),
                r2_meaningful=r2_valid,
                notes=notes,
            )


        mean_red_phys = float(np.mean(reductions_vs_phys)) if reductions_vs_phys else 0.0
        mean_red_pure = float(np.mean(reductions_vs_pure)) if reductions_vs_pure else 0.0

        return ModelComparisonResult(
            model_type=self.config.model_type,
            train_samples=len(self.preprocessor_residual.means_) if self.preprocessor_residual.means_ is not None else 0,
            test_samples=len(test_df),
            evaluated_targets=list(channel_metrics.keys()),
            channel_metrics=channel_metrics,
            mean_error_reduction_vs_physics_pct=round(mean_red_phys, 2),
            mean_error_reduction_vs_pure_ml_pct=round(mean_red_pure, 2),
            timestamp=pd.Timestamp.now().isoformat(),
        )

    # -----------------------------------------------------------------
    # Explainability (Deterministic Feature Attribution)
    # -----------------------------------------------------------------

    def explain_instance(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        dt: Optional[float] = None,
    ) -> Dict[str, ExplanationContribution]:
        """
        Generate deterministic feature attribution for a single telemetry instance.
        
        Separates:
        - Physics drivers: primary operating demands (throttle, altitude, ambient_temp)
        - Residual drivers: operational parameters influencing the ML correction
        - Health indicator: detected deviation between measured and physics estimates
        """
        if isinstance(row, dict):
            row_s = pd.Series(row)
        else:
            row_s = row

        sample_df = pd.DataFrame([row_s])
        preds = self.predict(sample_df, dt=dt)

        explanations = {}
        throttle = float(row_s.get("throttle", 75.0))
        altitude = float(row_s.get("altitude", 2000.0))
        ambient = float(row_s.get("ambient_temp", 15.0))

        for target in self.config.targets:
            if target not in preds["corrected"]:
                continue

            phys_val = float(preds["physics"][target][0])
            res_val = float(preds["residual_ml"][target][0])
            corr_val = float(preds["corrected"][target][0])
            obs_val = float(row_s[target]) if target in row_s and pd.notna(row_s[target]) else None

            # Physics baseline drivers
            phys_drivers = {
                "throttle_pct": throttle,
                "altitude_m": altitude,
                "ambient_temp_c": ambient,
            }

            # Residual drivers: based on model coefficients or feature importances if linear
            res_model = self.residual_models.get(target)
            res_drivers = {}
            if res_model is not None and hasattr(res_model, "coef_"):
                coefs = res_model.coef_
                # Match coefficients with residual feature names
                for f_name, c_val in zip(self.residual_features_, coefs):
                    if abs(c_val) > 1e-4:
                        res_drivers[f_name] = round(float(c_val), 4)
            elif res_model is not None and hasattr(res_model, "feature_importances_"):
                imps = res_model.feature_importances_
                for f_name, imp in zip(self.residual_features_, imps):
                    if imp > 0.01:
                        res_drivers[f_name] = round(float(imp), 4)
            else:
                res_drivers = {"predicted_residual": round(res_val, 4)}

            # Health indicator: deviation magnitude
            detected_dev = (obs_val - phys_val) if obs_val is not None else None
            health_ind = {
                "physics_estimate": round(phys_val, 2),
                "sensor_informed_correction": round(res_val, 3),
                "corrected_prediction": round(corr_val, 2),
                "detected_deviation": round(detected_dev, 3) if detected_dev is not None else None,
                "status": "ELEVATED" if (detected_dev is not None and abs(detected_dev) > 10.0) else "NOMINAL",
            }

            explanations[target] = ExplanationContribution(
                target=target,
                physics_drivers=phys_drivers,
                residual_drivers=res_drivers,
                health_indicator=health_ind,
            )

        return explanations
