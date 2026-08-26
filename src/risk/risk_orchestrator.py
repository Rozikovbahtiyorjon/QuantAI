"""
====================================================
QuantAI Professional
Risk Orchestrator
====================================================

Unified risk facade coordinating:
- DrawdownGuard: equity peak tracking + drawdown limits
- ExposureManager: position/notional exposure limits
- PositionSizer: risk-based position sizing

Provides single RiskDecision interface for Strategy/Runner.
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.drawdown_guard import DrawdownGuard, DrawdownGuardResult
from src.exposure_manager import ExposureManager, ExposureResult
from src.position_sizer import PositionSizer, PositionSizeResult
from src.risk.risk_context import RiskContext
from src.strategy import SignalResult


@dataclass(frozen=True)
class RiskDecision:
    """
    Unified risk decision from the orchestrator.
    
    Attributes:
        allowed: Whether the position is allowed to be opened
        quantity: Approved position size (0 if not allowed)
        stop_loss: Stop loss price (from signal)
        take_profit: Take profit price (from signal)
        reason: Human-readable reason for decision
        drawdown_result: Detailed drawdown analysis
        exposure_result: Detailed exposure analysis
        position_size_result: Detailed position sizing analysis
        metadata: Additional context (drawdown_pct, exposure_pct, etc.)
    """
    allowed: bool
    quantity: float
    stop_loss: float
    take_profit: float
    reason: str
    drawdown_result: DrawdownGuardResult
    exposure_result: ExposureResult
    position_size_result: PositionSizeResult
    metadata: dict = field(default_factory=dict)


class RiskOrchestrator:
    """
    Orchestrates multiple risk components into a single decision.
    
    Flow:
        SignalResult + Equity + Current Exposure
                    ↓
            DrawdownGuard (check equity health)
                    ↓
            PositionSizer (calculate risk-based size)
                    ↓
            ExposureManager (cap to exposure limits)
                    ↓
            RiskDecision (unified result)
    
    Usage:
        orchestrator = RiskOrchestrator(
            drawdown_guard=DrawdownGuard(max_drawdown_percent=10.0),
            exposure_manager=ExposureManager(max_total_exposure_percent=60.0, max_position_exposure_percent=5.0),
            position_sizer=PositionSizer(min_leverage=1.0, max_leverage=50.0),
            default_risk_percent=1.0,
            default_leverage=1.0,
        )
        
        decision = orchestrator.evaluate(
            signal=signal_result,
            equity=1000.0,
            current_exposure=0.0,
        )
        
        if decision.allowed:
            open_position(decision.quantity, decision.stop_loss, decision.take_profit)
    """
    
    def __init__(
        self,
        drawdown_guard: DrawdownGuard,
        exposure_manager: ExposureManager,
        position_sizer: PositionSizer,
        default_risk_percent: float = 1.0,
        default_leverage: float = 1.0,
    ) -> None:
        self.drawdown_guard = drawdown_guard
        self.exposure_manager = exposure_manager
        self.position_sizer = position_sizer
        self.default_risk_percent = float(default_risk_percent)
        self.default_leverage = float(default_leverage)
    
    def evaluate(
        self,
        signal: SignalResult,
        equity: float,
        current_exposure: float = 0.0,
        risk_percent: Optional[float] = None,
        leverage: Optional[float] = None,
        context: Optional["RiskContext"] = None,
    ) -> RiskDecision:
        """
        Evaluate a trading signal against all risk controls.

        Args:
            signal: Strategy signal with entry, stop_loss, take_profit
            equity: Current account equity/balance
            current_exposure: Current notional exposure from open positions
            risk_percent: Override default risk % (e.g., 1.0 = 1%)
            leverage: Override default leverage
            context: Optional RiskContext (R0.1). When provided, exposure
                limits are evaluated against context.projected_exposure
                (post-close baseline on flips) instead of current_exposure.

        Returns:
            RiskDecision with allowed flag, approved quantity, and diagnostics
        """
        # Use defaults if not provided
        risk_pct = risk_percent if risk_percent is not None else self.default_risk_percent
        lev = leverage if leverage is not None else self.default_leverage

        # R0.1: projected (post-action) exposure baseline for limit checks.
        limit_baseline_exposure = (
            context.effective_exposure
            if context is not None
            else current_exposure
        )
        
        # -------------------------------------------------
        # 1. DRAWDOWN CHECK
        # -------------------------------------------------
        drawdown_result = self.drawdown_guard.evaluate(equity)
        
        if not drawdown_result.allowed:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=(
                    f"maximum drawdown exceeded "
                    f"({drawdown_result.drawdown_percent:.8f}%)"
                ),
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity,
                    total_exposure=current_exposure,
                    total_exposure_percent=0.0,
                    available_exposure=0.0,
                    available_exposure_percent=0.0,
                    position_exposure=0.0,
                    position_exposure_percent=0.0,
                    within_limit=False,
                ),
                position_size_result=PositionSizeResult(
                    balance=equity,
                    risk_percent=risk_pct,
                    risk_amount=0.0,
                    entry_price=signal.entry,
                    stop_price=signal.stop_loss,
                    stop_distance=0.0,
                    stop_distance_percent=0.0,
                    position_size=0.0,
                    position_notional=0.0,
                    leverage=lev,
                    margin_required=0.0,
                ),
                metadata={
                    "drawdown_pct": drawdown_result.drawdown_percent,
                    "max_drawdown_pct": self.drawdown_guard.max_drawdown_percent,
                    "stage": "drawdown_blocked",
                },
            )
        
        # -------------------------------------------------
        # 2. POSITION SIZING
        # -------------------------------------------------
        try:
            position_size_result = self.position_sizer.calculate(
                balance=equity,
                risk_percent=risk_pct,
                entry_price=signal.entry,
                stop_price=signal.stop_loss,
                leverage=lev,
            )
        except ValueError as e:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=f"Position sizing failed: {e}",
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity,
                    total_exposure=current_exposure,
                    total_exposure_percent=0.0,
                    available_exposure=0.0,
                    available_exposure_percent=0.0,
                    position_exposure=0.0,
                    position_exposure_percent=0.0,
                    within_limit=False,
                ),
                position_size_result=PositionSizeResult(
                    balance=equity,
                    risk_percent=risk_pct,
                    risk_amount=0.0,
                    entry_price=signal.entry,
                    stop_price=signal.stop_loss,
                    stop_distance=0.0,
                    stop_distance_percent=0.0,
                    position_size=0.0,
                    position_notional=0.0,
                    leverage=lev,
                    margin_required=0.0,
                ),
                metadata={"stage": "sizing_failed", "error": str(e)},
            )
        
        risk_quantity = position_size_result.position_size
        
        # -------------------------------------------------
        # 3. EXPOSURE CAP (per-position limit)
        # -------------------------------------------------
        max_position_capital = self.exposure_manager.max_position_capital(equity)
        exposure_limited_quantity = max_position_capital / signal.entry if signal.entry > 0 else 0.0
        
        approved_quantity = min(risk_quantity, exposure_limited_quantity)
        
        if approved_quantity <= 0:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason="Calculated position size is zero after exposure cap",
                drawdown_result=drawdown_result,
                exposure_result=self.exposure_manager.calculate(
                    equity=equity,
                    current_exposure=current_exposure,
                ),
                position_size_result=position_size_result,
                metadata={
                    "risk_quantity": risk_quantity,
                    "exposure_limited_quantity": exposure_limited_quantity,
                    "stage": "exposure_cap_zero",
                },
            )
        
        approved_position_exposure = approved_quantity * signal.entry
        
        # -------------------------------------------------
        # 4. TOTAL EXPOSURE CHECK
        # -------------------------------------------------
        exposure_result = self.exposure_manager.calculate(
            equity=equity,
            current_exposure=limit_baseline_exposure,
            position_exposure=approved_position_exposure,
        )

        can_open = self.exposure_manager.can_open_position(
            equity=equity,
            current_exposure=limit_baseline_exposure,
            new_position_exposure=approved_position_exposure,
        )
        
        if not can_open:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason="exposure limit exceeded",
                drawdown_result=drawdown_result,
                exposure_result=exposure_result,
                position_size_result=position_size_result,
                metadata={
                    "approved_quantity": approved_quantity,
                    "approved_exposure": approved_position_exposure,
                    "current_exposure": current_exposure,
                    "limit_baseline_exposure": limit_baseline_exposure,
                    "is_flip": context.is_flip if context is not None else False,
                    "max_total_exposure_pct": self.exposure_manager.max_total_exposure_percent,
                    "stage": "total_exposure_blocked",
                },
            )
        
        # -------------------------------------------------
        # 5. ALL CHECKS PASSED
        # -------------------------------------------------
        return RiskDecision(
            allowed=True,
            quantity=round(approved_quantity, 8),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reason="Risk approved.",
            drawdown_result=drawdown_result,
            exposure_result=exposure_result,
            position_size_result=position_size_result,
                metadata={
                    "drawdown_pct": drawdown_result.drawdown_percent,
                    "total_exposure_pct": exposure_result.total_exposure_percent,
                    "position_exposure_pct": exposure_result.position_exposure_percent,
                    "risk_quantity": risk_quantity,
                    "approved_quantity": approved_quantity,
                    "leverage": lev,
                    "is_flip": context.is_flip if context is not None else False,
                    "stage": "approved",
                },
        )
    
    def reset(self) -> None:
        """Reset all stateful components."""
        self.drawdown_guard.reset()
        # ExposureManager and PositionSizer are stateless


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_default_orchestrator(
    max_drawdown_percent: float = 10.0,
    max_total_exposure_percent: float = 60.0,
    max_position_exposure_percent: float = 5.0,
    min_leverage: float = 1.0,
    max_leverage: float = 50.0,
    default_risk_percent: float = 1.0,
    default_leverage: float = 1.0,
) -> RiskOrchestrator:
    """
    Create a RiskOrchestrator with default settings from config.
    """
    return RiskOrchestrator(
        drawdown_guard=DrawdownGuard(max_drawdown_percent=max_drawdown_percent),
        exposure_manager=ExposureManager(
            max_total_exposure_percent=max_total_exposure_percent,
            max_position_exposure_percent=max_position_exposure_percent,
        ),
        position_sizer=PositionSizer(
            min_leverage=min_leverage,
            max_leverage=max_leverage,
        ),
        default_risk_percent=default_risk_percent,
        default_leverage=default_leverage,
    )


def create_orchestrator_from_settings(settings) -> RiskOrchestrator:
    """
    Create RiskOrchestrator from Pydantic Settings object.
    
    Expected settings structure:
        settings.risk.drawdown_limit_pct
        settings.risk.max_correlation (unused here)
        settings.account.max_risk_percent
        settings.account.max_open_positions (unused here)
    """
    # Default exposure limits from config or sensible defaults
    max_total_exposure = getattr(settings.risk, 'max_total_exposure_percent', 60.0)
    max_position_exposure = getattr(settings.risk, 'max_position_exposure_percent', 5.0)
    
    return RiskOrchestrator(
        drawdown_guard=DrawdownGuard(
            max_drawdown_percent=settings.risk.drawdown_limit_pct
        ),
        exposure_manager=ExposureManager(
            max_total_exposure_percent=max_total_exposure,
            max_position_exposure_percent=max_position_exposure,
        ),
        position_sizer=PositionSizer(
            min_leverage=1.0,
            max_leverage=50.0,
        ),
        default_risk_percent=settings.account.risk_per_trade * 100,  # config is 0.01 = 1%
        default_leverage=1.0,
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RiskContext",
    "RiskDecision",
    "RiskOrchestrator",
    "create_default_orchestrator",
    "create_orchestrator_from_settings",
]