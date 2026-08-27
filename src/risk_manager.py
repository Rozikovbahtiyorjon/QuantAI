"""
====================================================
QuantAI Professional Risk Manager (Facade)
====================================================

DEPRECATED: This module is a compatibility facade.
New code should use src.risk.risk_orchestrator.RiskOrchestrator directly.

This facade delegates to the new RiskOrchestrator for position sizing,
SL/TP calculation, and risk evaluation.
====================================================
"""

import warnings
from config.settings import (
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_MULTIPLIER,
    MIN_POSITION_SIZE,
    MAX_POSITION_SIZE,
)

from src.risk.risk_orchestrator import (
    RiskOrchestrator,
    create_default_orchestrator,
    RiskDecision,
)
from src.risk.position_sizer import PositionSizer
from src.risk.drawdown_guard import DrawdownGuard
from src.risk.exposure_manager import ExposureManager
from src.strategy import SignalResult


# Global default orchestrator for backward compatibility
_default_orchestrator: RiskOrchestrator | None = None


def _get_orchestrator() -> RiskOrchestrator:
    """Get or create the default risk orchestrator."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = create_default_orchestrator()
    return _default_orchestrator


def set_default_orchestrator(orchestrator: RiskOrchestrator) -> None:
    """Set the default orchestrator for facade functions."""
    global _default_orchestrator
    _default_orchestrator = orchestrator


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Расчет размера позиции по фиксированному риску.
    
    DEPRECATED: Use RiskOrchestrator.evaluate() for full risk evaluation,
    or PositionSizer.calculate() for position sizing only.
    
    Parameters
    ----------
    balance : float
        Баланс счета.

    risk_percent : float
        Риск на сделку в процентах.

    entry_price : float
        Цена входа.

    stop_loss : float
        Цена Stop Loss.

    Returns
    -------
    float
        Размер позиции.
    """
    warnings.warn(
        "calculate_position_size is deprecated. "
        "Use RiskOrchestrator or PositionSizer directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    stop_distance = abs(entry_price - stop_loss)

    if stop_distance <= 0:
        return 0.0

    risk_amount = balance * (risk_percent / 100)

    position_size = risk_amount / stop_distance

    # Ограничиваем размер позиции
    position_size = max(position_size, MIN_POSITION_SIZE)
    position_size = min(position_size, MAX_POSITION_SIZE)

    return round(position_size, 6)


def calculate_sl_tp(
    entry_price: float,
    atr: float,
    rr: float | None = None,
):
    """
    Расчет Stop Loss и Take Profit.
    
    DEPRECATED: Use SLTPCalculator from src.strategy.sl_tp_calculator
    for regime-adaptive SL/TP calculation.
    
    Если rr не передан,
    используется отношение ATR_TAKE_MULTIPLIER /
    ATR_STOP_MULTIPLIER.
    """
    warnings.warn(
        "calculate_sl_tp is deprecated. "
        "Use SLTPCalculator from src.strategy.sl_tp_calculator for regime-adaptive SL/TP.",
        DeprecationWarning,
        stacklevel=2,
    )

    stop_multiplier = ATR_STOP_MULTIPLIER

    if rr is None:
        rr = ATR_TAKE_MULTIPLIER / ATR_STOP_MULTIPLIER

    stop_loss = entry_price - atr * stop_multiplier

    take_profit = entry_price + atr * stop_multiplier * rr

    return (
        round(stop_loss, 2),
        round(take_profit, 2),
    )


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """
    Возвращает Risk/Reward.
    
    DEPRECATED: This utility function will be removed.
    """
    warnings.warn(
        "calculate_risk_reward is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )

    risk = abs(entry_price - stop_loss)

    reward = abs(take_profit - entry_price)

    if risk == 0:
        return 0.0

    return round(reward / risk, 2)


def calculate_trade_risk(
    balance: float,
    risk_percent: float,
) -> float:
    """
    Сколько долларов допускается потерять.
    
    DEPRECATED: Use PositionSizer.calculate() via RiskOrchestrator.
    """
    warnings.warn(
        "calculate_trade_risk is deprecated. "
        "Use PositionSizer.calculate() via RiskOrchestrator.",
        DeprecationWarning,
        stacklevel=2,
    )

    return round(balance * risk_percent / 100, 2)


def break_even_price(
    entry_price: float,
    commission: float,
) -> float:
    """
    Цена безубытка с учетом комиссии.
    
    DEPRECATED: This utility function will be removed.
    """
    warnings.warn(
        "break_even_price is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )

    return round(entry_price * (1 + commission * 2), 2)


# =========================================================
# NEW FACADE FUNCTIONS (delegate to RiskOrchestrator)
# =========================================================

def evaluate_risk(
    signal: SignalResult,
    equity: float,
    current_exposure: float = 0.0,
    risk_percent: float = 1.0,
    leverage: float = 1.0,
) -> RiskDecision:
    """
    Evaluate a trading signal against all risk controls.
    
    This is the NEW recommended entry point for risk evaluation.
    Delegates to RiskOrchestrator.
    
    Args:
        signal: Strategy signal with entry, stop_loss, take_profit
        equity: Current account equity/balance
        current_exposure: Current notional exposure from open positions
        risk_percent: Risk per trade (e.g., 1.0 = 1%)
        leverage: Leverage to use
        
    Returns:
        RiskDecision with allowed flag, approved quantity, and diagnostics
    """
    orchestrator = _get_orchestrator()
    return orchestrator.evaluate(
        signal=signal,
        equity=equity,
        current_exposure=current_exposure,
        risk_percent=risk_percent,
        leverage=leverage,
    )


def check_drawdown(equity: float) -> bool:
    """
    Check if drawdown limits allow new positions.
    
    Delegates to DrawdownGuard via RiskOrchestrator.
    """
    orchestrator = _get_orchestrator()
    result = orchestrator.drawdown_guard.evaluate(equity)
    return result.allowed


def check_exposure(
    equity: float,
    current_exposure: float,
    new_position_exposure: float,
) -> bool:
    """
    Check if new position would exceed exposure limits.
    
    Delegates to ExposureManager via RiskOrchestrator.
    """
    orchestrator = _get_orchestrator()
    return orchestrator.exposure_manager.can_open_position(
        equity=equity,
        current_exposure=current_exposure,
        new_position_exposure=new_position_exposure,
    )


def get_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    leverage: float = 1.0,
) -> float:
    """
    Calculate position size using PositionSizer.
    
    Delegates to PositionSizer via RiskOrchestrator.
    """
    orchestrator = _get_orchestrator()
    result = orchestrator.position_sizer.calculate(
        balance=balance,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_price=stop_price,
        leverage=leverage,
    )
    return result.position_size


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    # Legacy functions (deprecated)
    "calculate_position_size",
    "calculate_sl_tp",
    "calculate_risk_reward",
    "calculate_trade_risk",
    "break_even_price",
    
    # New facade functions
    "evaluate_risk",
    "check_drawdown",
    "check_exposure",
    "get_position_size",
    "set_default_orchestrator",
    
    # Re-export core types
    "RiskDecision",
    "RiskOrchestrator",
    "create_default_orchestrator",
    "PositionSizer",
    "DrawdownGuard",
    "ExposureManager",
]