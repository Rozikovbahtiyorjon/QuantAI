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
    # Mock returns HOLD for most steps, BUY at step 250, SELL at step 255
    call_count = {"count": 0}
    
    def mock_generate_func(df, model=None):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return SignalResult(signal="BUY", entry=float(df.iloc[-1]["close"]))
        elif call_count["count"] == 6:
            return SignalResult(signal="SELL", entry=float(df.iloc[-1]["close"]))
        return SignalResult(signal="HOLD", entry=float(df.iloc[-1]["close"]))
    
    mock_generate.side_effect = mock_generate_func
    
    # Need at least 251 rows for warmup (250) + 1
    closes = [100.0 + i * 0.1 for i in range(260)]
    df = pd.DataFrame({"close": closes})
    results = runner.process_dataframe(df)
    
    # With warmup_bars=250 and 260 rows, we get 11 results (indices 250-260)
    assert len(results) == 11
    
    # Step 250 (index 0 in results): BUY
    assert results[0].position_opened is True
    assert results[0].position_closed is False
    assert results[0].signal.signal == "BUY"
    
    # Steps 1-4 (indices 1-4): HOLD
    for i in range(1, 5):
        assert results[i].signal.signal == "HOLD"
        assert results[i].position_opened is False
        assert results[i].position_closed is False
    
    # Step 5 (index 5, corresponds to df index 255): SELL
    assert results[5].position_opened is True
    assert results[5].position_closed is True
    assert results[5].signal.signal == "SELL"
    
    # Final position should be SHORT
    assert runner.engine.position.side == "SHORT"