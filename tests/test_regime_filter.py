"""
RegimeFilter (Stage A) unit tests.

Covers: trend detection, range detection, hysteresis,
counter-trend gate, causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_filter import (
    RANGE,
    TREND_DOWN,
    TREND_UP,
    RegimeConfig,
    RegimeFilter,
)


def make_prepared(
    rows: int = 300,
    seed: int = 1,
    drift: float = 0.0,
    vol: float = 0.002,
) -> pd.DataFrame:
    """
    Synthetic PREPARED series with ema_trend / adx columns
    produced by the real indicator pipeline (guarantees column parity).
    """

    from src.indicators import add_indicators

    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=rows)
    close = 100.0 * np.cumprod(1 + rets)

    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 0.0008, rows)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 200, rows)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return add_indicators(df)


class TestClassification:
    def test_uptrend_detected(self) -> None:
        df = make_prepared(drift=0.003, vol=0.001, seed=3)

        rf = RegimeFilter()
        last = None
        for i in range(80, len(df)):
            last = rf.classify(df.iloc[: i + 1])

        assert last == TREND_UP

    def test_downtrend_detected(self) -> None:
        df = make_prepared(drift=-0.003, vol=0.001, seed=4)

        rf = RegimeFilter()
        last = None
        for i in range(80, len(df)):
            last = rf.classify(df.iloc[: i + 1])

        assert last == TREND_DOWN

    def test_deterministic_flat_is_range(self) -> None:
        # Hand-built prepared frame: flat ema_trend, weak ADX.
        rows = 120
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min"),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 100.0,
                "ema_trend": 100.0,
                "adx": 12.0,
            }
        )

        rf = RegimeFilter()
        states = [rf.classify(df.iloc[: i + 1]) for i in range(60, rows)]

        assert set(states) == {RANGE}

    def test_stationary_ar1_stays_range_mostly(self) -> None:
        """
        Mean-reverting AR(1) has no persistent direction;
        trend states must be episodic, not dominant.
        """

        from src.indicators import add_indicators

        rng = np.random.default_rng(77)
        n = 500

        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = 0.80 * x[t - 1] + rng.normal(0, 0.003)

        close = 100.0 * (1.0 + x)
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        spread = np.abs(rng.normal(0, 0.0005, n)) * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.uniform(50, 200, n)

        df = add_indicators(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

        rf = RegimeFilter()
        states = [rf.classify(df.iloc[: i + 1]) for i in range(80, len(df))]

        range_share = states.count(RANGE) / len(states)
        assert range_share >= 0.30, (
            f"RANGE share {range_share:.0%} too low for a "
            f"mean-reverting series (hysteresis too sticky?)"
        )


class TestHysteresis:
    def test_trend_persists_while_adx_above_exit(self) -> None:
        cfg = RegimeConfig(adx_enter=22.0, adx_exit=18.0)
        rf = RegimeFilter(cfg)

        # Force state into trend.
        rf._state = TREND_UP

        # ADX between exit and enter: must STAY in trend.
        df = make_prepared(rows=120, seed=11)
        df = df.copy()
        df.loc[df.index[-1], "adx"] = 20.0  # exit < adx < enter

        assert rf.classify(df) == TREND_UP

    def test_trend_exits_when_adx_below_exit(self) -> None:
        cfg = RegimeConfig(adx_enter=22.0, adx_exit=18.0)
        rf = RegimeFilter(cfg)
        rf._state = TREND_UP

        df = make_prepared(rows=120, seed=12)
        df = df.copy()
        df.loc[df.index[-1], "adx"] = 10.0

        assert rf.classify(df) == RANGE

    def test_config_validates_thresholds(self) -> None:
        with pytest.raises(ValueError):
            RegimeConfig(adx_enter=18.0, adx_exit=22.0)


class TestGate:
    def test_counter_trend_blocked(self) -> None:
        rf = RegimeFilter()
        rf._state = TREND_UP

        assert rf.allows("BUY") is True
        assert rf.allows("SELL") is False

        rf._state = TREND_DOWN
        assert rf.allows("BUY") is False
        assert rf.allows("SELL") is True

        rf._state = RANGE
        assert rf.allows("BUY") is True
        assert rf.allows("SELL") is True


class TestCausality:
    def test_prefix_classification_stable_under_future_change(self) -> None:
        base = make_prepared(rows=300, drift=0.002, vol=0.001, seed=21)
        shocked = base.copy()
        shocked.loc[shocked.index[250:], "close"] *= 2.0
        shocked.loc[shocked.index[250:], "high"] *= 2.0

        n = 250
        rf_a = RegimeFilter()
        rf_b = RegimeFilter()

        a = None
        b = None
        for i in range(80, n):
            a = rf_a.classify(base.iloc[: i + 1])
            b = rf_b.classify(shocked.iloc[: i + 1])

        # classify() consumes only ema_trend/adx of the prefix window;
        # both filters see identical prefixes here.
        assert a == b
