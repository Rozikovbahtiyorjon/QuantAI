"""
Data quality validator tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_quality import drop_duplicates_and_sort, validate_ohlcv


def make_ok_frame(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min")
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 0.5, rows).cumsum()
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = np.abs(rng.normal(0, 0.1, rows))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(10, 100, rows),
        }
    )


class TestCleanSeries:
    def test_clean_series_passes(self) -> None:
        rep = validate_ohlcv(make_ok_frame())
        assert rep.passed, rep.summary()
        assert rep.duplicate_timestamps == 0
        assert rep.missing_bars == 0
        assert rep.ohlc_violations == 0


class TestDuplicates:
    def test_duplicates_fail(self) -> None:
        df = make_ok_frame()
        df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
        rep = validate_ohlcv(df)
        assert not rep.passed
        assert rep.duplicate_timestamps == 1

    def test_cleanup_removes_duplicates_and_sorts(self) -> None:
        df = make_ok_frame()
        shuffled = df.iloc[::-1]
        dirty = pd.concat([shuffled, df.iloc[[5]]], ignore_index=True)

        out = drop_duplicates_and_sort(dirty)

        assert len(out) == len(df)
        assert out["timestamp"].is_monotonic_increasing
        assert validate_ohlcv(out).passed


class TestGaps:
    def test_missing_bars_detected(self) -> None:
        df = make_ok_frame(400)
        with_gaps = df.drop(index=range(100, 160)).reset_index(drop=True)

        rep = validate_ohlcv(with_gaps, max_gap_pct=0.5)

        assert rep.missing_bars == 60
        assert not rep.passed

    def test_small_gap_within_tolerance_warns_not_fails(self) -> None:
        df = make_ok_frame(400)
        with_gap = df.drop(index=[50]).reset_index(drop=True)

        rep = validate_ohlcv(with_gap, max_gap_pct=1.0)

        # 1 of ~400 bars = 0.25% < 1% tolerance
        assert rep.passed
        assert rep.missing_bars == 1


class TestOHLCIntegrity:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda df: df.assign(high=df["low"]),          # high < low swapped
            lambda df: df.assign(high=df["close"] - 5),    # high below close
            lambda df: df.assign(low=df["open"] + 5),      # low above open
        ],
    )
    def test_violations_fail(self, mutate) -> None:
        df = mutate(make_ok_frame())
        rep = validate_ohlcv(df)
        assert not rep.passed
        assert rep.ohlc_violations > 0

    def test_non_positive_price_fails(self) -> None:
        df = make_ok_frame()
        df.loc[df.index[10], "close"] = 0.0
        rep = validate_ohlcv(df)
        assert not rep.passed
        assert rep.non_positive_prices >= 1


class TestOutliersAndVolume:
    def test_outlier_warns_but_passes(self) -> None:
        df = make_ok_frame(600)
        df.loc[df.index[300], "close"] *= 2.0
        # keep OHLC consistent after shock
        row = df.index[300]
        df.loc[row, "high"] = max(df.loc[row, "high"], df.loc[row, "close"])
        df.loc[row, "open"] = df.loc[row - 1, "close"]

        rep = validate_ohlcv(df)
        assert rep.outliers_6sigma >= 1
        assert rep.passed  # outlier is a warning, not a gate

    def test_zero_volume_reported(self) -> None:
        df = make_ok_frame()
        df.loc[df.index[:30], "volume"] = 0.0
        rep = validate_ohlcv(df)
        assert rep.zero_volume_bars == 30
        assert rep.warnings  # 6% share -> warning path
