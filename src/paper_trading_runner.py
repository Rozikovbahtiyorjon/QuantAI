"""
=========================================================
QuantAI Professional v5
Paper Trading Runner

Connects Strategy Engine with PaperTradingEngine and,
optionally, applies paper-trading risk controls before
opening a new virtual position.

This module does NOT:
    - connect to Binance
    - execute real orders
    - train ML models
    - calculate indicators
    - modify Strategy logic
    - modify PaperTradingEngine
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.ml_engine import MLEngine, MLConfig
from src.model_manager import ModelManager
from src.paper_trading_engine import (
    PaperTrade,
    PaperTradingEngine,
)
from src.risk.risk_context import RiskContext
from src.risk.risk_orchestrator import (
    RiskOrchestrator,
    create_default_orchestrator,
)
from src.strategy import (
    SignalResult,
    generate_signal_result,
)


@dataclass
class PaperTradingStepResult:
    signal: SignalResult
    trade: Optional[PaperTrade]
    position_opened: bool
    position_closed: bool
    risk_approved: bool = True
    risk_reason: str = ""
    # ML diagnostics
    ml_signal: str = "HOLD"
    ml_confidence: float = 0.0
    ml_used: bool = False
    ml_quality_gate_passed: bool = True
    ml_quality_reason: str = ""


@dataclass
class MLQualityGateConfig:
    """Configuration for ML model quality gate."""
    enabled: bool = True
    min_balanced_accuracy: float = 0.52
    min_f1_score: float = 0.30
    min_precision: float = 0.25
    min_recall: float = 0.25
    max_models_without_retrain: int = 10
    require_walk_forward_validation: bool = True


class PaperTradingRunner:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
        enable_risk_controls: bool = True,
        risk_percent: float = 1.0,
        max_drawdown_percent: float = 10.0,
        max_total_exposure_percent: float = 60.0,
        max_position_exposure_percent: float = 5.0,
        leverage: float = 1.0,
        min_leverage: float = 1.0,
        max_leverage: float = 50.0,
        risk_orchestrator: RiskOrchestrator | None = None,
        # ML Integration
        enable_ml: bool = False,
        ml_model_path: str = "models/quantai_v5.pkl",
        ml_config: MLConfig | None = None,
        ml_quality_gate: MLQualityGateConfig | None = None,
        ab_test_ml: bool = False,  # A/B test: alternate ML vs no-ML
    ) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be greater than zero.")

        if risk_percent <= 0:
            raise ValueError(
                "risk_percent must be greater than zero."
            )

        if leverage <= 0:
            raise ValueError(
                "leverage must be greater than zero."
            )

        self.engine = PaperTradingEngine(
            initial_balance=initial_balance,
            commission=commission,
        )

        self.quantity = float(quantity)

        self.enable_risk_controls = bool(
            enable_risk_controls
        )

        self.risk_percent = float(
            risk_percent
        )

        self.leverage = float(
            leverage
        )

        # Use provided orchestrator or create default
        self.risk_orchestrator = (
            risk_orchestrator
            if risk_orchestrator is not None
            else create_default_orchestrator(
                max_drawdown_percent=max_drawdown_percent,
                max_total_exposure_percent=max_total_exposure_percent,
                max_position_exposure_percent=max_position_exposure_percent,
                min_leverage=min_leverage,
                max_leverage=max_leverage,
                default_risk_percent=risk_percent,
                default_leverage=leverage,
            )
        )

        # =====================================================
        # ML INTEGRATION
        # =====================================================
        self.enable_ml = bool(enable_ml)
        self.ml_model_path = ml_model_path
        self.ml_config = ml_config or MLConfig()
        self.ml_quality_gate = ml_quality_gate or MLQualityGateConfig()
        self.ab_test_ml = bool(ab_test_ml)
        self._ml_engine: MLEngine | None = None
        self._ml_models_since_retrain = 0
        self._ab_test_counter = 0

        if self.enable_ml:
            self._load_ml_model()

    # =====================================================
# CURRENT EXPOSURE
# =====================================================

    def _current_exposure(self) -> float:
        """
        Return the current paper position notional exposure.

        Exposure is calculated from the current position entry
        price and quantity.
        """

        if not self.engine.has_position:
            return 0.0

        position = self.engine.position

        if position is None:
            return 0.0

        return float(
            position.entry_price
            * position.quantity
        )

    # =====================================================
    # ML INTEGRATION METHODS
    # =====================================================

    def _load_ml_model(self) -> bool:
        """Load ML model from disk."""
        try:
            self._ml_engine = MLEngine(
                config=self.ml_config,
                load_existing=True,
            )
            if self._ml_engine.model is not None:
                self._ml_models_since_retrain = 0
                return True
            return False
        except Exception as e:
            print(f"[ML] Failed to load model: {e}")
            self._ml_engine = None
            return False

    def _check_ml_quality_gate(self) -> tuple[bool, str]:
        """
        Check if ML model passes quality gate.
        
        Returns:
            (passed, reason)
        """
        if not self.ml_quality_gate.enabled:
            return True, "ML quality gate disabled."

        if self._ml_engine is None or self._ml_engine.model is None:
            return False, "No ML model loaded."

        if self._ml_models_since_retrain >= self.ml_quality_gate.max_models_without_retrain:
            return False, f"Model retrain limit exceeded ({self._ml_models_since_retrain} windows)"

        return True, "ML quality gate passed."

    def _get_ml_prediction(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Get ML prediction for current market state.
        
        Returns:
            (signal, confidence)
        """
        if self._ml_engine is None:
            return "HOLD", 0.0

        try:
            from src.feature_engine import build_features
            features = build_features(df)
            signal, confidence = self._ml_engine.predict_signal(
                pd.DataFrame([features])
            )
            return signal, confidence
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return "HOLD", 0.0

    def _should_use_ml(self) -> bool:
        """Determine if ML should be used (supports A/B testing)."""
        if not self.enable_ml:
            return False
        
        if self.ab_test_ml:
            self._ab_test_counter += 1
            return self._ab_test_counter % 2 == 0
        
        return True

    # =====================================================
    # RISK CHECK (via RiskOrchestrator)
    # =====================================================

    def _risk_check(
        self,
        signal: SignalResult,
        is_flip: bool = False,
        position_side: str | None = None,
    ) -> tuple[bool, str, float]:
        """
        Validate a new position using the RiskOrchestrator.

        R0.1: builds a canonical RiskContext. On a flip the exposure
        limit check runs against the PROJECTED (post-close) baseline,
        so the decision matches the state that will exist if the
        trade commits.

        Returns:
            approved: bool
            reason: str
            approved_quantity: float
        """

        if not self.enable_risk_controls:
            return (
                True,
                "Risk controls disabled.",
                self.quantity,
            )

        if signal.entry <= 0:
            return (
                False,
                "Risk rejected: entry must be greater than zero.",
                0.0,
            )

        if signal.stop_loss <= 0:
            return (
                False,
                "Risk rejected: stop_loss must be greater than zero.",
                0.0,
            )

        if signal.stop_loss == signal.entry:
            return (
                False,
                "Risk rejected: stop_loss cannot equal entry.",
                0.0,
            )

        current_exposure = self._current_exposure()

        # On a flip the old position will be closed before the new
        # one opens; projected exposure after that close:
        projected_exposure = 0.0 if is_flip else current_exposure

        # Check for NaN/Inf in signal
        import math
        if (math.isnan(signal.entry) or math.isinf(signal.entry) or
            math.isnan(signal.stop_loss) or math.isinf(signal.stop_loss) or
            (signal.take_profit is not None and (math.isnan(signal.take_profit) or math.isinf(signal.take_profit)))):
            return (
                False,
                "Risk rejected: NaN or Inf detected in signal",
                0.0,
            )

        # Check for zero or negative balance before risk evaluation
        if self.engine.balance <= 0:
            return (
                False,
                "Risk rejected: account balance must be greater than zero.",
                0.0,
            )

        context = RiskContext(
            equity=float(self.engine.balance),
            balance=float(self.engine.balance),
            current_exposure=current_exposure,
            projected_exposure=projected_exposure,
            position_side=position_side,
            requested_side="LONG" if signal.signal == "BUY" else "SHORT",
            is_flip=is_flip,
            entry=float(signal.entry),
            stop_loss=float(signal.stop_loss),
            take_profit=float(signal.take_profit or 0.0),
            risk_percent=self.risk_percent,
            leverage=self.leverage,
        )

        decision = self.risk_orchestrator.evaluate(
            signal=signal,
            equity=self.engine.balance,
            current_exposure=current_exposure,
            risk_percent=self.risk_percent,
            leverage=self.leverage,
            context=context,
        )

        return (
            decision.allowed,
            decision.reason,
            decision.quantity,
        )

    # =====================================================
    # PROCESS SIGNAL
    # =====================================================

    def process_signal(
        self,
        signal: SignalResult,
        df: pd.DataFrame | None = None,
    ) -> PaperTradingStepResult:
        """
        Process one Strategy signal.

        BUY:
            open LONG or flip SHORT → LONG

        SELL:
            open SHORT or flip LONG → SHORT

        HOLD:
            do nothing

        R0.1 risk ordering (atomic flip decision):

            1. Risk check runs BEFORE any state change.
               On a flip, exposure limits are evaluated against the
               PROJECTED post-close baseline via RiskContext.
            2. Only on approval: close old position -> open new one.
            3. On rejection: engine state remains UNCHANGED
               (old position stays open).
        """

        if not isinstance(
            signal,
            SignalResult,
        ):
            raise TypeError(
                "signal must be SignalResult."
            )

        # -------------------------------------------------
        # HOLD
        # -------------------------------------------------

        if signal.signal == "HOLD":
            return PaperTradingStepResult(
                signal=signal,
                trade=None,
                position_opened=False,
                position_closed=False,
                risk_approved=True,
                risk_reason="HOLD signal.",
                ml_signal="HOLD",
                ml_confidence=0.0,
                ml_used=False,
                ml_quality_gate_passed=True,
                ml_quality_reason="",
            )

        # -------------------------------------------------
        # VALIDATE SIGNAL
        # -------------------------------------------------

        if signal.signal not in {
            "BUY",
            "SELL",
        }:
            raise ValueError(
                f"Unsupported signal: {signal.signal}"
            )

        requested_side = (
            "LONG"
            if signal.signal == "BUY"
            else "SHORT"
        )

        current_side = None

        if (
            self.engine.has_position
            and self.engine.position is not None
        ):
            current_side = (
                self.engine.position.side
            )

        is_flip = (
            current_side is not None
            and current_side != requested_side
        )

        # -------------------------------------------------
        # DEFAULT RISK STATE
        # -------------------------------------------------

        risk_approved = True

        risk_reason = (
            "Risk approved."
            if self.enable_risk_controls
            else "Risk controls disabled."
        )

        risk_quantity = self.quantity

        # ML diagnostics
        ml_signal = "HOLD"
        ml_confidence = 0.0
        ml_used = False
        ml_quality_gate_passed = True
        ml_quality_reason = ""

        # -------------------------------------------------
        # ML PREDICTION & QUALITY GATE
        # -------------------------------------------------

        if self._should_use_ml():
            ml_used = True
            ml_quality_gate_passed, ml_quality_reason = self._check_ml_quality_gate()
            
            if ml_quality_gate_passed:
                ml_signal, ml_confidence = self._get_ml_prediction(df)
            else:
                ml_signal, ml_confidence = "HOLD", 0.0

            # ML can block trades if it disagrees with strategy
            if ml_quality_gate_passed and ml_signal != "HOLD":
                strategy_signal = signal.signal
                if (strategy_signal == "BUY" and ml_signal == "SELL") or \
                   (strategy_signal == "SELL" and ml_signal == "BUY"):
                    risk_approved = False
                    risk_reason = f"ML disagrees: strategy={strategy_signal}, ML={ml_signal}"
                    return PaperTradingStepResult(
                        signal=signal,
                        trade=None,
                        position_opened=False,
                        position_closed=False,
                        risk_approved=False,
                        risk_reason=risk_reason,
                        ml_signal=ml_signal,
                        ml_confidence=ml_confidence,
                        ml_used=ml_used,
                        ml_quality_gate_passed=ml_quality_gate_passed,
                        ml_quality_reason=ml_quality_reason,
                    )

        # -------------------------------------------------
        # RISK GATE
        # -------------------------------------------------

        needs_new_position = (
            not self.engine.has_position
            or is_flip
        )

        if (
            self.enable_risk_controls
            and needs_new_position
        ):
            (
                risk_approved,
                risk_reason,
                risk_quantity,
            ) = self._risk_check(
                signal,
                is_flip=is_flip,
                position_side=current_side,
            )

            if not risk_approved:
                # R0.1 invariant: rejected risk leaves the engine
                # state UNCHANGED. On a flip the old position is NOT
                # closed (decision is atomic: close+open together).
                return PaperTradingStepResult(
                    signal=signal,
                    trade=None,
                    position_opened=False,
                    position_closed=False,
                    risk_approved=False,
                    risk_reason=risk_reason,
                    ml_signal=ml_signal,
                    ml_confidence=ml_confidence,
                    ml_used=ml_used,
                    ml_quality_gate_passed=ml_quality_gate_passed,
                    ml_quality_reason=ml_quality_reason,
                )

        # -------------------------------------------------
        # SAME-SIDE SIGNAL - Reject duplicate same-side signals
        # -------------------------------------------------

        if (
            self.engine.has_position
            and current_side == requested_side
        ):
            return PaperTradingStepResult(
                signal=signal,
                trade=None,
                position_opened=False,
                position_closed=False,
                risk_approved=False,
                risk_reason="Already have position on same side",
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                ml_used=ml_used,
                ml_quality_gate_passed=ml_quality_gate_passed,
                ml_quality_reason=ml_quality_reason,
            )

        # -------------------------------------------------
        # FLIP / CLOSE EXISTING POSITION
        # -------------------------------------------------

        position_closed = False
        trade = None

        if self.engine.has_position:
            trade = self.engine.close_position(
                price=signal.entry
            )

            position_closed = True

        # -------------------------------------------------
        # OPEN NEW POSITION
        # -------------------------------------------------

        self.engine.open_position(
            side=requested_side,
            price=signal.entry,
            quantity=(
                risk_quantity
                if self.enable_risk_controls
                else self.quantity
            ),
        )

        return PaperTradingStepResult(
            signal=signal,
            trade=trade,
            position_opened=True,
            position_closed=position_closed,
            risk_approved=risk_approved,
            risk_reason=risk_reason,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            ml_used=ml_used,
            ml_quality_gate_passed=ml_quality_gate_passed,
            ml_quality_reason=ml_quality_reason,
        )

    # =====================================================
    # CLOSE POSITION
    # =====================================================

    def close_position(
        self,
        price: float,
        signal: SignalResult | None = None,
    ) -> PaperTradingStepResult:
        """
        Manually close the current paper position.
        """

        if price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if signal is None:
            signal = SignalResult(
                signal="HOLD",
                entry=price,
            )

        trade = self.engine.close_position(
            price=price
        )

        return PaperTradingStepResult(
            signal=signal,
            trade=trade,
            position_opened=False,
            position_closed=True,
            risk_approved=True,
            risk_reason="Manual position close.",
        )

    # =====================================================
    # PROCESS DATAFRAME
    # =====================================================

    def process_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[PaperTradingStepResult]:
        """
        Generate Strategy signals from a DataFrame
        and process them sequentially.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        results: list[
            PaperTradingStepResult
        ] = []

        ml_model = self._ml_engine.model if self.enable_ml and self._ml_engine else None

        for end in range(
            1,
            len(df) + 1,
        ):
            window = df.iloc[
                :end
            ].copy()

            signal = generate_signal_result(
                window,
                model=ml_model,
            )

            result = self.process_signal(
                signal,
                df=window,
            )

            results.append(
                result
            )

        return results

    # =====================================================
    # ACCOUNT STATE
    # =====================================================

    @property
    def balance(self) -> float:
        return self.engine.balance

    @property
    def has_position(self) -> bool:
        return self.engine.has_position

    @property
    def realized_profit(self) -> float:
        return self.engine.realized_profit

    @property
    def current_exposure(self) -> float:
        return self._current_exposure()

    # =====================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # =====================================================

    @property
    def drawdown_guard(self):
        """Backward compatibility: access DrawdownGuard from orchestrator."""
        return self.risk_orchestrator.drawdown_guard

    @property
    def exposure_manager(self):
        """Backward compatibility: access ExposureManager from orchestrator."""
        return self.risk_orchestrator.exposure_manager

    @property
    def position_sizer(self):
        """Backward compatibility: access PositionSizer from orchestrator."""
        return self.risk_orchestrator.position_sizer

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset paper-trading engine and risk state.
        """

        self.engine.reset()
        self.risk_orchestrator.reset()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingStepResult",
    "PaperTradingRunner",
]