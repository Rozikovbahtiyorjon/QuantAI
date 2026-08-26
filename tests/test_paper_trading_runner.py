import pytest
import pandas as pd
from unittest.mock import patch

from src.paper_trading_runner import PaperTradingRunner
from src.strategy import SignalResult

@pytest.fixture
def runner():
    return PaperTradingRunner(
        enable_risk_controls=False,
        initial_balance=1000.0, 
        commission=0.001, 
        quantity=1.0
    )

def test_runner_initialization(runner):
    assert runner.balance == 1000.0
    assert not runner.has_position
    assert runner.realized_profit == 0.0

def test_process_signal_buy_then_sell_flips_position(runner):
    # 1. Открываем LONG
    signal_buy = SignalResult(signal="BUY", entry=100.0)
    res_buy = runner.process_signal(signal_buy)
    
    assert res_buy.position_opened is True
    assert res_buy.position_closed is False
    assert runner.engine.position.side == "LONG"
    
    # 2. Получаем SELL -> Закрываем LONG, открываем SHORT
    signal_sell = SignalResult(signal="SELL", entry=110.0)
    res_sell = runner.process_signal(signal_sell)
    
    assert res_sell.position_closed is True
    assert res_sell.position_opened is True
    assert runner.engine.position.side == "SHORT"
    
    # 3. Проверяем фиксацию прибыли (trade)
    assert res_sell.trade is not None
    assert res_sell.trade.gross_profit == 10.0  # (110 - 100) * 1.0

@patch("src.paper_trading_runner.generate_signal_result")
def test_process_dataframe_executes_pipeline(mock_generate, runner):
    # Мокаем стратегию, чтобы не зависеть от реальной логики индикаторов
    mock_generate.side_effect = [
        SignalResult(signal="BUY", entry=100.0),
        SignalResult(signal="HOLD", entry=105.0),
        SignalResult(signal="SELL", entry=110.0)
    ]
    
    df = pd.DataFrame({"close": [100.0, 105.0, 110.0]})
    results = runner.process_dataframe(df)
    
    assert len(results) == 3
    
    # Шаг 1: BUY
    assert results[0].position_opened is True
    assert results[0].position_closed is False
    
    # Шаг 2: HOLD
    assert results[1].position_opened is False
    assert results[1].position_closed is False
    
    # Шаг 3: SELL (Flip)
    assert results[2].position_opened is True
    assert results[2].position_closed is True
    
    # Итоговая позиция должна быть SHORT
    assert runner.engine.position.side == "SHORT"