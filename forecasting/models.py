"""
Model abstractions and TimesFM-3 adapter for Phase 10 telemetry trajectory forecasting.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Protocol
import numpy as np
import torch

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    ForecastingConfig,
    ModelStatus,
)


class ForecastModel(Protocol):
    """Protocol defining the standard interface for forecasting models."""
    name: str

    def predict(
        self,
        context_2d: np.ndarray,
        horizon: int,
        target_channels: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts from context."""
        ...


class TimesFM3ModelAdapter:
    """
    Adapter for Google TimesFM-3 foundation model.

    Supports:
    - 2D joint multivariate input: activates Sequence Attention AND Variate Attention across channels.
    - 1D univariate input mode: independent channel forecasting for ablation studies.
    - Transparent runtime status: LOADED_PRETRAINED, LOCAL_UNCHECKPOINTED_GRAPH, or BLOCKED_UNAUTHENTICATED_GATED.
    - Offline graph verification without fabricating pretrained weights.
    """

    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        use_multivariate: bool = True,
        force_local_graph: bool = False,
    ):
        self.config = config or ForecastingConfig()
        self.use_multivariate = use_multivariate
        self.name = "timesfm-3.0"
        self.runtime_status: str = ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value
        self.status_detail: str = ""
        self.forecaster: Any = None
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._initialize_model(force_local_graph=force_local_graph)

    def _initialize_model(self, force_local_graph: bool = False) -> None:
        """Initialize TimesFM3 forecaster or establish local graph verification."""
        try:
            import timesfm
            from timesfm3 import ModelConfig, TimesFM3Forecaster, configs
            import timesfm3.timesfm3_forecaster as tfm_mod
        except ImportError as e:
            self.runtime_status = ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value
            self.status_detail = f"timesfm package not importable: {e}"
            return

        # Case 1: Explicit local graph requested (for unit tests / offline graph verification)
        if force_local_graph:
            try:
                res_cfg = configs.ResidualBlockConfig(
                    hidden_dims=32, output_dims=32, use_bias=False, activation="relu"
                )
                tr_cfg = configs.StackedTransformersConfig(
                    num_layers=1,
                    transformer=configs.TransformerConfig(
                        model_dims=32,
                        hidden_dims=32,
                        num_heads=2,
                        attention_norm="rms",
                        feedforward_norm="rms",
                        qk_norm="rms",
                        use_rope_seq=True,
                        use_rope_var=True,
                        use_bias=False,
                        ff_activation="relu",
                        deterministic=True,
                    ),
                )
                m_cfg = ModelConfig(
                    checkpoint_path="local_graph_verification",
                    input_patch_length=8,
                    output_patch_length=16,
                    quantiles=[0.1, 0.5, 0.9],
                    median_quantile_index=1,
                    residual_block_config=res_cfg,
                    transformer_config=tr_cfg,
                    device=self.device,
                )
                forecaster = TimesFM3Forecaster.__new__(TimesFM3Forecaster)
                forecaster.config = m_cfg
                forecaster.device = torch.device(self.device)
                forecaster.model = tfm_mod._make_torch_model(m_cfg)
                forecaster.model.to(forecaster.device)
                forecaster.model.eval()
                self.forecaster = forecaster
                self.runtime_status = ModelStatus.LOCAL_UNCHECKPOINTED_GRAPH.value
                self.status_detail = "Local PyTorch graph initialized for verification (no pretrained weights)."
                return
            except Exception as e:
                self.runtime_status = ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value
                self.status_detail = f"Failed to initialize local graph: {e}"
                return

        # Case 2: Attempt loading from checkpoint or Hugging Face Hub
        has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        is_local_path = os.path.exists(self.config.checkpoint_path)

        if not has_token and not is_local_path:
            self.runtime_status = ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value
            self.status_detail = (
                "Checkpoint 'google/timesfm-3.0-pytorch' is a gated Hugging Face repository. "
                "HF_TOKEN is unset and terms not accepted. Pretrained weights not loaded."
            )
            return

        try:
            self.forecaster = TimesFM3Forecaster.from_pretrained(
                pretrained_model_name_or_path=self.config.checkpoint_path,
                device=self.device,
            )
            self.runtime_status = ModelStatus.LOADED_PRETRAINED.value
            self.status_detail = f"Pretrained TimesFM-3 loaded successfully on {self.device}."
        except Exception as e:
            self.runtime_status = ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value
            self.status_detail = f"Failed to load checkpoint: {e}"

    def is_available(self) -> bool:
        """Returns True if forecaster instance is ready for inference."""
        return self.forecaster is not None

    def predict(
        self,
        context_2d: np.ndarray,
        horizon: int,
        target_channels: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
        return_quantiles: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Generate telemetry forecast via TimesFM-3.

        Args:
            context_2d: Array of shape (num_channels, context_len) float32
            horizon: Forecast steps
            target_channels: Optional list of channel names
            timestamps: Optional array of historical timestamps
            return_quantiles: Whether to return quantile uncertainty bounds

        Returns:
            (point_forecasts, quantile_forecasts):
            - point_forecasts: Dict mapping channel -> np.ndarray of shape (horizon,)
            - quantile_forecasts: Dict mapping channel -> np.ndarray of shape (horizon, num_q) or None
        """
        if not self.is_available():
            raise RuntimeError(
                f"TimesFM-3 forecaster is unavailable. Status: {self.runtime_status}. Detail: {self.status_detail}"
            )

        channels = target_channels or DEFAULT_FORECAST_CHANNELS
        num_channels, context_len = context_2d.shape

        if self.use_multivariate:
            # 2D Joint Multivariate Input: (num_channels, context_len)
            # Passes single 2D array inside contexts list -> activates Variate Attention
            ctx = np.array(context_2d, dtype=np.float32)
            results = list(
                self.forecaster.predict_batch(
                    contexts=[ctx],
                    horizon=horizon,
                    return_quantiles=return_quantiles,
                )
            )
            res = results[0]
            # res.forecast shape is (num_channels, horizon)
            point_fc = res.forecast
            q_fc = res.quantiles if return_quantiles else None

            forecasts: Dict[str, np.ndarray] = {}
            quantiles: Optional[Dict[str, np.ndarray]] = {} if return_quantiles else None

            for i in range(num_channels):
                ch = channels[i] if i < len(channels) else f"channel_{i}"
                forecasts[ch] = point_fc[i]
                if return_quantiles and q_fc is not None:
                    quantiles[ch] = q_fc[i]

            return forecasts, quantiles
        else:
            # 1D Univariate Batch Mode: list of 1D series (ablation mode, no variate attention)
            ctx_list = [np.array(context_2d[i], dtype=np.float32) for i in range(num_channels)]
            results = list(
                self.forecaster.predict_batch(
                    contexts=ctx_list,
                    horizon=horizon,
                    return_quantiles=return_quantiles,
                )
            )
            forecasts = {}
            quantiles = {} if return_quantiles else None

            for i, res in enumerate(results):
                ch = channels[i] if i < len(channels) else f"channel_{i}"
                forecasts[ch] = res.forecast
                if return_quantiles and res.quantiles is not None:
                    quantiles[ch] = res.quantiles

            return forecasts, quantiles
