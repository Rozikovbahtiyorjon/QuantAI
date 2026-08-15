"""
====================================================
QuantAI Professional Risk Manager
====================================================
"""

from config.settings import (
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_MULTIPLIER,
    MIN_POSITION_SIZE,
    MAX_POSITION_SIZE,
)


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Расчет размера позиции по фиксированному риску.

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

    Если rr не передан,
    используется отношение ATR_TAKE_MULTIPLIER /
    ATR_STOP_MULTIPLIER.
    """

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
    """

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
    """

    return round(balance * risk_percent / 100, 2)


def break_even_price(
    entry_price: float,
    commission: float,
) -> float:
    """
    Цена безубытка с учетом комиссии.
    """

    return round(entry_price * (1 + commission * 2), 2)