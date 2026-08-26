"""
Tests for Phase A/B modules: MultiTFConfirm, labels, FilteredGenerator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.meta_label import (
    BarrierConfig,
    FEATURE_NAMES,
    FilteredGenerator,
    MetaLabelModel,
    MultiTFConfirm,
    entry_features,
    label_entry,
)
from src.strategy.signal_generator import SignalResult


def make_window(rows: int = 400, close: float = 100.0, slope: float = -0.5) -> pd.DataFrame:
    """
    Synthetic prepared window. Default slope makes the HTF trend DOWN
    (closes decline), which MultiTFConfirm must read correctly.
    """
    ts = pd.date_range("2024-01-01", periods=rows, freq="1h")
    closes = close + np.arange(rows) * slope

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": 100.0,
            "ema_fast": closes * 0.999,
            "ema_slow": closes * 0.998,
            "ema_trend": closes * 0.997,
            "rsi": 50.0,
            "adx": 30.0,
            "atr": 1.0,
        }
    )


class StubBase:
    """Base generator always returning BUY."""

    def generate(self, df):
        r = SignalResult()
        r.signal = "BUY"
        r.trade_approved = True
        r.entry = float(df.iloc[-1]["close"])
        r.stop_loss = r.entry - 3.0
        r.take_profit = r.entry + 6.0
        r.confidence = 70.0
        return r


class TestMultiTFConfirm:
    def test_blocks_buy_in_htf_downtrend(self) -> None:
        df = make_window(slope=-0.5)  # declining -> HTF bearish

        gen = MultiTFConfirm(StubBase())
        res = gen.generate(df)

        assert res.signal == "HOLD"
        assert any("MTF" in s for s in res.reasons)

    def test_allows_buy_in_htf_uptrend(self) -> None:
        df = make_window(slope=+0.5)

        gen = MultiTFConfirm(StubBase())
        res = gen.generate(df)

        assert res.signal == "BUY"

    def test_causal_prefix_stability(self) -> None:
        """Appending future rows must not change past-window decision."""
        df_full = make_window(rows=600, slope=-0.5)

        gen_a = MultiTFConfirm(StubBase())
        gen_b = MultiTFConfirm(StubBase())

        n = 500
        a = gen_a.generate(df_full.iloc[:n])
        b = gen_b.generate(df_full)  # same prefix content, longer tail

        assert a.signal == b.signal


class TestLabels:
    def _frame(self, highs, lows):
        rows = len(highs)
        ts = pd.date_range("2024-01-01", periods=rows, freq="1h")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": 100.0,
                "high": highs,
                "low": lows,
                "close": 100.0,
                "volume": 1.0,
                "atr": 1.0,
            }
        )

    def test_tp_first_is_win(self) -> None:
        # entry 100, R=3 -> SL 97, TP 106 (2R)
        df = self._frame([101, 107], [99, 100])
        assert label_entry(df, 0, "BUY", 100.0, BarrierConfig()) == 1

    def test_sl_first_is_loss(self) -> None:
        df = self._frame([101, 102], [96.9, 97])
        assert label_entry(df, 0, "BUY", 100.0, BarrierConfig()) == 0

    def test_tie_goes_to_sl(self) -> None:
        # same bar touches both -> conservative loss
        df = self._frame([107.0], [96.9])
        assert label_entry(df, 0, "BUY", 100.0, BarrierConfig()) == 0

    def test_timeout_is_loss(self) -> None:
        df = self._frame([100.5] * 80, [99.5] * 80)
        assert label_entry(df, 0, "BUY", 100.0, BarrierConfig()) == 0

    def test_short_side_mirrored(self) -> None:
        # short entry 100: TP 94, SL 103; price falls -> win
        df = self._frame([101.0, 95.0], [99.5, 93.5])
        assert label_entry(df, 0, "SELL", 100.0, BarrierConfig()) == 1


class TestFeaturesAndFilter:
    def test_feature_vector_complete_and_finite(self) -> None:
        df = make_window()
        feats = entry_features(df, "BUY")

        assert set(feats.keys()) == set(FEATURE_NAMES)
        assert all(np.isfinite(v) for v in feats.values())

    def test_filtered_generator_drops_low_proba(self) -> None:
        class LowModel:
            threshold = 0.55

            def approve(self, feats):
                return False

        gen = FilteredGenerator(StubBase(), LowModel(), history_window=300)
        res = gen.generate(make_window())

        assert res.signal == "HOLD"

    def test_meta_model_fit_predict_smoke(self) -> None:
        rng = np.random.default_rng(0)
        n = 200

        X = pd.DataFrame(
            {
                name: rng.normal(size=n)
                for name in FEATURE_NAMES
            }
        )
        y = pd.Series((X["adx"] > 0).astype(int))  # learnable rule

        model = MetaLabelModel(threshold=0.5)
        model.fit(X, y)

        good = {name: 1.0 for name in FEATURE_NAMES}
        bad = {name: -1.0 for name in FEATURE_NAMES}

        assert model.approve(good) is True
        assert model.approve(bad) is False
