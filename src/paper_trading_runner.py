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
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd

# Canonical risk policy (Audit: single source of truth)
try:
    from src.risk.policy import get_policy as _get_canonical_policy  # canonical src/risk/policy.py
    _CANONICAL_PAPER = _get_canonical_policy("paper")
except Exception:
    _CANONICAL_PAPER = None  # type: ignore

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
    RiskDecision,
)
from src.strategy.ml_overlay import MLOverlay, MLQualityGateConfig  # noqa: F401 (re-export)
from src.strategy import (
    SignalResult,
    generate_signal_result,
)


@dataclass
class DecisionRecord:
    """
    Complete audit trail for every trading decision.
    
    Captures the full context of why a trade was made (or not made).
    """
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Market context
    market_state: Dict[str, Any]
    
    # Features used
    features: Dict[str, float]
    
    # ML predictions
    ml_prediction: str
    ml_probability: float
    
    # Confidence & strategy
    confidence: float
    strategy_signal: str
    
    # Risk decision
    risk_decision: Dict[str, Any]
    
    # Position sizing
    position_size: float
    entry: float
    stop_loss: float
    take_profit: float
    
    # Final outcome
    final_decision: str  # "OPEN", "REJECT", "HOLD", "FLIP", "CLOSE"
    reason_codes: List[str]
    
    # Metadata
    step_index: int
    balance_before: float
    equity_before: float


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
    # Decision record
    decision_record: Optional[DecisionRecord] = None


class PaperTradingRunner:
    # Canonical PaperPolicy: 30% total / 3% per position / 10% DD / 10x leverage
    # Previously diverged 60%/5%/50x — now unified to src/risk/policies.py (Audit single canonical)
    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
        enable_risk_controls: bool = True,
        risk_percent: float = 1.0,
        max_drawdown_percent: float = 10.0,  # canonical PaperPolicy 10.0
        max_total_exposure_percent: float = 30.0,  # was 60.0 → now 30.0 PaperPolicy
        max_position_exposure_percent: float = 3.0,  # was 5.0 → now 3.0 PaperPolicy
        leverage: float = 1.0,
        min_leverage: float = 1.0,
        max_leverage: float = 10.0,  # was 50.0 → now 10.0 canonical
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

        # R1: ML mechanics live in strategy layer (MLOverlay).
        self.ml_overlay = MLOverlay(
            enable_ml=self.enable_ml,
            ml_config=ml_config,
            ml_quality_gate=ml_quality_gate,
            ab_test_ml=ab_test_ml,
        )
        self.ml_config = self.ml_overlay.ml_config
        self.ml_quality_gate = ml_quality_gate
        self.ab_test_ml = bool(ab_test_ml)

        if self.enable_ml:
            self._load_ml_model()

        # Decision records for audit trail
        self.decision_records: List[DecisionRecord] = []

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
        """Delegate: model loading lives in MLOverlay (R1)."""
        return self.ml_overlay.load_model()

    def _check_ml_quality_gate(self) -> tuple[bool, str]:
        """Delegate: quality gate lives in MLOverlay (R1)."""
        return self.ml_overlay.check_quality_gate()

    def _get_ml_prediction(self, df: pd.DataFrame) -> tuple[str, float]:
        """Delegate: prediction lives in MLOverlay (R1)."""
        return self.ml_overlay.predict(df)

    def _should_use_ml(self) -> bool:
        """Delegate: A/B decisioning lives in MLOverlay (R1)."""
        return self.ml_overlay.should_use()

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

    def _create_decision_record(
        self,
        signal: SignalResult,
        df: pd.DataFrame | None,
        final_decision: str,
        reason_codes: list[str],
        risk_approved: bool,
        risk_reason: str,
        ml_signal: str,
        ml_confidence: float,
        ml_used: bool,
        position_size: float,
        position_opened: bool,
        position_closed: bool,
    ) -> DecisionRecord:
        """
        Create a comprehensive DecisionRecord for audit trail.
        
        Captures full context of the trading decision including
        market state, features, ML predictions, risk assessment,
        and final outcome.
        """
        # Extract features from df if available
        features = {}
        market_state = {}
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            # Extract numeric features (exclude target, future_return, etc.)
            for col in df.columns:
                if col not in ["target", "future_return", "index"]:
                    val = last_row.get(col)
                    if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float)):
                        features[col] = float(val) if not pd.isna(val) else 0.0
            
            # Market state summary
            market_state = {
                "close": float(last_row.get("close", 0)),
                "volume": float(last_row.get("volume", 0)),
                "atr": float(last_row.get("atr", 0)) if "atr" in last_row else 0,
                "regime": str(last_row.get("regime", "UNKNOWN")) if "regime" in last_row else "UNKNOWN",
            }
        
        # Get confidence from signal if available
        confidence = getattr(signal, 'confidence', 0.0) or 0.0
        
        # Build risk decision dict
        risk_decision = {
            "allowed": risk_approved,
            "reason": risk_reason,
            "position_size": position_size,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        }
        
        return DecisionRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=signal.intent.symbol if hasattr(signal, 'intent') and signal.intent else "UNKNOWN",
            timeframe="15m",  # Default, could be configurable
            market_state=market_state,
            features=features,
            ml_prediction=ml_signal,
            ml_probability=ml_confidence / 100.0 if ml_confidence > 1 else ml_confidence,
            confidence=confidence,
            strategy_signal=signal.signal,
            risk_decision=risk_decision,
            position_size=position_size,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit or 0.0,
            final_decision=final_decision,
            reason_codes=reason_codes,
            step_index=len(self.decision_records),
            balance_before=self.engine.balance,
            equity_before=self.engine.equity if hasattr(self.engine, 'equity') else self.engine.balance,
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
            decision_record = self._create_decision_record(
                signal=signal,
                df=df,
                final_decision="HOLD",
                reason_codes=["HOLD_SIGNAL"],
                risk_approved=True,
                risk_reason="HOLD signal.",
                ml_signal="HOLD",
                ml_confidence=0.0,
                ml_used=False,
                position_size=0.0,
                position_opened=False,
                position_closed=False,
            )
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
                decision_record=decision_record,
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
                    decision_record = self._create_decision_record(
                        signal=signal,
                        df=df,
                        final_decision="REJECT",
                        reason_codes=["ML_DISAGREES"],
                        risk_approved=False,
                        risk_reason=risk_reason,
                        ml_signal=ml_signal,
                        ml_confidence=ml_confidence,
                        ml_used=ml_used,
                        position_size=0.0,
                        position_opened=False,
                        position_closed=False,
                    )
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
                        decision_record=decision_record,
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
                decision_record = self._create_decision_record(
                    signal=signal,
                    df=df,
                    final_decision="REJECT",
                    reason_codes=["RISK_REJECTED"],
                    risk_approved=False,
                    risk_reason=risk_reason,
                    ml_signal=ml_signal,
                    ml_confidence=ml_confidence,
                    ml_used=ml_used,
                    position_size=0.0,
                    position_opened=False,
                    position_closed=False,
                )
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
                    decision_record=decision_record,
                )

        # -------------------------------------------------
        # SAME-SIDE SIGNAL - Reject duplicate same-side signals
        # -------------------------------------------------

        if (
            self.engine.has_position
            and current_side == requested_side
        ):
            decision_record = self._create_decision_record(
                signal=signal,
                df=df,
                final_decision="REJECT",
                reason_codes=["SAME_SIDE_POSITION_EXISTS"],
                risk_approved=False,
                risk_reason="Already have position on same side",
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                ml_used=ml_used,
                position_size=0.0,
                position_opened=False,
                position_closed=False,
            )
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
                decision_record=decision_record,
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

        final_decision = "FLIP" if position_closed else "OPEN"
        reason_codes = ["POSITION_OPENED"]
        if position_closed:
            reason_codes.append("FLIP")

        decision_record = self._create_decision_record(
            signal=signal,
            df=df,
            final_decision=final_decision,
            reason_codes=reason_codes,
            risk_approved=risk_approved,
            risk_reason=risk_reason,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            ml_used=ml_used,
            position_size=risk_quantity if self.enable_risk_controls else self.quantity,
            position_opened=True,
            position_closed=position_closed,
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
            decision_record=decision_record,
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

        decision_record = self._create_decision_record(
            signal=signal,
            df=None,
            final_decision="CLOSE",
            reason_codes=["MANUAL_CLOSE"],
            risk_approved=True,
            risk_reason="Manual position close.",
            ml_signal="HOLD",
            ml_confidence=0.0,
            ml_used=False,
            position_size=0.0,
            position_opened=False,
            position_closed=True,
        )
        return PaperTradingStepResult(
            signal=signal,
            trade=trade,
            position_opened=False,
            position_closed=True,
            risk_approved=True,
            risk_reason="Manual position close.",
            decision_record=decision_record,
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

        # Need at least warmup_bars for indicators to be valid
        warmup_bars = 250  # EMA 200 needs 200, plus buffer
        if len(df) < warmup_bars + 1:
            raise ValueError(
                f"DataFrame must have at least {warmup_bars + 1} rows for indicators to be valid"
            )

        results: list[
            PaperTradingStepResult
        ] = []

        ml_model = self._ml_engine.model if self.enable_ml and self._ml_engine else None

        # Start from warmup_bars to ensure indicators are valid
        for end in range(
            warmup_bars,
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
        self.decision_records.clear()


    def get_decision_records(self) -> List[DecisionRecord]:
        """Get all recorded decisions."""
        return list(self.decision_records)


    def export_decision_records(self) -> list[dict]:
        """Export decision records as list of dicts for serialization."""
        return [
            {
                "timestamp": dr.timestamp.isoformat(),
                "symbol": dr.symbol,
                "timeframe": dr.timeframe,
                "market_state": dr.market_state,
                "features": dr.features,
                "ml_prediction": dr.ml_prediction,
                "ml_probability": dr.ml_probability,
                "confidence": dr.confidence,
                "strategy_signal": dr.strategy_signal,
                "risk_decision": dr.risk_decision,
                "position_size": dr.position_size,
                "entry": dr.entry,
                "stop_loss": dr.stop_loss,
                "take_profit": dr.take_profit,
                "final_decision": dr.final_decision,
                "reason_codes": dr.reason_codes,
                "step_index": dr.step_index,
                "balance_before": dr.balance_before,
                "equity_before": dr.equity_before,
            }
            for dr in self.decision_records
        ]


# =========================================================
# PAPER TRADING RUNNER FUNCTION
# =========================================================

async def run_paper_trading(
    state: Any,
    duration_minutes: Optional[int] = None,
    df: Optional[Any] = None,
) -> Any:
    """
    Run paper trading simulation with the given state.
    
    Args:
        state: Application state from lifecycle startup
        duration_minutes: Optional duration in minutes to run
        df: Optional pre-loaded DataFrame with prepared data
    
    Returns:
        PaperTradingRunner instance with completed simulation
    """
    from src.paper_trading_runner import PaperTradingRunner
    from src.strategy.signal_generator import SignalGenerator, SignalConfig
    import pandas as pd
    
    # Load data if not provided
    if df is None:
        df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
    
    # Create paper trading runner with canonical PaperPolicy (30%/3%/10%)
    # Use get_policy('paper') if available, else explicit canonical values
    try:
        _paper = _get_canonical_policy("paper")  # type: ignore
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=_paper.commission,
            enable_risk_controls=True,
            risk_percent=_paper.risk_per_trade * 100,
            max_drawdown_percent=_paper.max_drawdown_pct,
            max_total_exposure_percent=_paper.max_total_exposure_pct,
            max_position_exposure_percent=_paper.max_position_exposure_pct,
            leverage=1.0,
            max_leverage=_paper.max_leverage,
        )
    except Exception:
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=0.0004,
            enable_risk_controls=True,
            risk_percent=1.0,
            max_drawdown_percent=10.0,
            max_total_exposure_percent=30.0,
            max_position_exposure_percent=3.0,
            leverage=1.0,
            max_leverage=10.0,
        )
    
    # Process the data
    results = runner.process_dataframe(df)
    
    # Print summary
    trades = [r for r in results if r.trade is not None]
    print("Paper Trading Complete:")
    print(f"  Trades: {len(trades)}")
    if trades:
        # Get final balance from the engine
        final_balance = 1000.0  # default
        if hasattr(runner, 'engine') and runner.engine:
            final_balance = runner.engine.balance
        else:
            # Calculate from trades
            final_balance = 1000.0 + sum(t.net_profit for t in trades)
        print(f"  Final Balance: {final_balance:.2f}")
    else:
        print("  Final Balance: 1000.00 (no trades)")

    return results


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "DecisionRecord",
    "PaperTradingStepResult",
    "PaperTradingRunner",
    "run_paper_trading",
]