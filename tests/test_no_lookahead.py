"""
No look-ahead bias protection tests (Phase 1).

1. Indicators must be causal: values computed on a prefix of data
   must equal the corresponding rows computed on the full series.
2. System-level: trades opened before bar N must be identical
   whether the future after N is normal or shocked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_indicators


def make_ohlcv(
    rows: int = 600,
    seed: int = 42,
    drift: float = 0.0005,
    shock_after: int | None = None,
    shock_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV series (15m bars).
    Optionally applies a violent price shock after `shock_after` bars
    to simulate a changed future.
    """

    rng = np.random.default_rng(seed)

    rets = rng.normal(drift, 0.004, size=rows)

    close = 100.0 * np.cumprod(1.0 + rets)

    if shock_after is not None:
        close[shock_after:] *= shock_factor

    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    spread = np.abs(rng.normal(0, 0.001, size=rows)) * close

    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    volume = rng.uniform(50, 500, size=rows)

    ts = pd.date_range("2024-01-01", periods=rows, freq="15min")

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


INDICATOR_COLS = [
    "ema_fast",
    "ema_slow",
    "ema_trend",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "plus_di",
    "minus_di",
    "adx",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "vwap",
    "obv",
    "volume_sma20",
    "supertrend",
]


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(float)


class TestIndicatorCausality:
    """Indicators computed on a prefix == rows of full-series computation."""

    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_prefix_rows_identical(self, seed: int) -> None:
        full = add_indicators(make_ohlcv(rows=600, seed=seed))
        n = 450

        # IMPORTANT: prefix must be a SLICE of the same series,
        # not a fresh shorter generation (RNG samples differ).
        head_src = make_ohlcv(rows=600, seed=seed).iloc[:n].copy()
        head = add_indicators(head_src)

        left = _numeric_frame(head, INDICATOR_COLS).to_numpy()
        right = _numeric_frame(full.iloc[:n], INDICATOR_COLS).to_numpy()

        np.testing.assert_allclose(
            left,
            right,
            rtol=1e-9,
            atol=1e-9,
            err_msg=(
                "Indicator values differ between prefix-computed and "
                "full-series-computed rows -> future information leak."
            ),
        )

    def test_shocked_future_does_not_change_past(self) -> None:
        base = make_ohlcv(rows=600, seed=99)
        shocked = make_ohlcv(rows=600, seed=99, shock_after=500, shock_factor=3.0)

        full_a = add_indicators(base)
        full_b = add_indicators(shocked)

        n = 500
        left = _numeric_frame(full_a.iloc[:n], INDICATOR_COLS).to_numpy()
        right = _numeric_frame(full_b.iloc[:n], INDICATOR_COLS).to_numpy()

        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-9)


class TestSystemLevelNoLookahead:
    """
    Trades opened strictly before the shock must be identical
    regardless of what happens in the future.
    """

    def test_trades_before_shock_are_invariant(self) -> None:
        from src.trade_engine import TradeEngine

        rows = 600
        cutoff = 500

        base = add_indicators(make_ohlcv(rows=rows, seed=5))
        shocked = add_indicators(
            make_ohlcv(rows=rows, seed=5, shock_after=cutoff, shock_factor=3.0)
        )

        eng_a = TradeEngine()
        eng_a.initial_balance = 1000.0
        eng_a.balance = 1000.0
        eng_a.equity = 1000.0
        eng_a.run(base)

        eng_b = TradeEngine()
        eng_b.initial_balance = 1000.0
        eng_b.balance = 1000.0
        eng_b.equity = 1000.0
        eng_b.run(shocked)

        def entries_before(engine, ts_cutoff):
            out = []
            for p in engine.closed_positions:
                t = pd.Timestamp(p.entry_time)
                if t < ts_cutoff:
                    out.append((p.side.value, round(p.entry_price, 6), str(p.entry_time)))
            return out

        ts_cut = base["timestamp"].iloc[cutoff]

        a = entries_before(eng_a, ts_cut)
        b = entries_before(eng_b, ts_cut)

        assert len(a) > 0, "test data produced no pre-cutoff trades"

        assert a == b, (
            "Trades before the shock differ when the future changes "
            "-> system has look-ahead bias."
        )
