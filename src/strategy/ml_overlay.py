"""
QuantAI ML Overlay (R1)

ML prediction + quality-gate + A/B logic extracted from
PaperTradingRunner into the strategy/model layer.

The Runner no longer owns ML mechanics; it delegates to this component.
Public behavior is unchanged (same decisions, same reasons).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml_engine import MLEngine, MLConfig


@dataclass
class MLQualityGateConfig:
    """Configuration for ML model quality gate (canonical location)."""
    enabled: bool = True
    min_balanced_accuracy: float = 0.52
    min_f1_score: float = 0.30
    min_precision: float = 0.25
    min_recall: float = 0.25
    max_models_without_retrain: int = 10
    require_walk_forward_validation: bool = True


class MLOverlay:
    """
    Owns: model loading, A/B decisioning, quality gating, prediction.
    """

    def __init__(
        self,
        enable_ml: bool = False,
        ml_config: MLConfig | None = None,
        ml_quality_gate=None,          # MLQualityGateConfig | None
        ab_test_ml: bool = False,
    ) -> None:
        self.enable_ml = bool(enable_ml)
        self.ml_config = ml_config or MLConfig()
        self.ml_quality_gate = ml_quality_gate
        self.ab_test_ml = bool(ab_test_ml)

        self.engine: MLEngine | None = None
        self.models_since_retrain = 0
        self._ab_counter = 0

        if self.enable_ml:
            self.load_model()

    # -----------------------------------------------------

    def load_model(self) -> bool:
        """Load model from disk. Returns success."""
        try:
            self.engine = MLEngine(
                config=self.ml_config,
                load_existing=True,
            )
            if self.engine.model is not None:
                self.models_since_retrain = 0
                return True
            self.engine = None
            return False
        except Exception as e:
            print(f"[ML] Failed to load model: {e}")
            self.engine = None
            return False

    def should_use(self) -> bool:
        """A/B aware decision whether ML participates in this step."""
        if not self.enable_ml:
            return False

        if self.ab_test_ml:
            self._ab_counter += 1
            return self._ab_counter % 2 == 0

        return True

    def check_quality_gate(self) -> tuple[bool, str]:
        gate = self.ml_quality_gate

        if gate is None or not getattr(gate, "enabled", False):
            return True, "ML quality gate disabled."

        if self.engine is None or self.engine.model is None:
            return False, "No ML model loaded."

        limit = getattr(gate, "max_models_without_retrain", 10)
        if self.models_since_retrain >= limit:
            return (
                False,
                f"Model retrain limit exceeded ({self.models_since_retrain} windows)",
            )

        return True, "ML quality gate passed."

    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Predict (signal, confidence) for the current market window.
        Fail-soft: any error -> HOLD / 0.0.
        """
        if self.engine is None:
            return "HOLD", 0.0

        try:
            from src.feature_engine import build_features

            features = build_features(df)
            return self.engine.predict_signal(pd.DataFrame([features]))
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return "HOLD", 0.0


__all__ = ["MLOverlay", "MLQualityGateConfig"]
