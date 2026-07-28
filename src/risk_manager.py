def calculate_position_size(
    balance,
    risk_percent,
    entry_price,
    stop_loss
):
    """
    Расчет размера позиции.
    """

    risk_amount = balance * (risk_percent / 100)

    stop_distance = abs(entry_price - stop_loss)

    if stop_distance == 0:
        return 0

    position_size = risk_amount / stop_distance

    return round(position_size, 6)


def calculate_sl_tp(
    entry_price,
    atr,
    rr=2
):
    """
    Расчет Stop Loss и Take Profit.
    """

    stop_loss = entry_price - atr * 1.5
    take_profit = entry_price + atr * 1.5 * rr

    return (
        round(stop_loss, 2),
        round(take_profit, 2)
    )