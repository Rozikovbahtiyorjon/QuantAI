"""
QuantAI Data Quality Validator

Structural and statistical checks for OHLCV series.
Every check returns findings; nothing raises for normal issues -
quality gates are decided by the caller (download pipeline / WF prep).

Checks:
    1. monotonic timestamps, no duplicates
    2. gaps vs expected interval (median diff)
    3. OHLC integrity: high >= max(o,c), low <= min(o,c), positive prices
    4. non-positive / zero volume bars
    5. outlier returns (beyond N sigma of rolling vol)
    6. coverage summary (rows, span, expected vs actual)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class QualityReport:
    """Aggregated result of all checks."""

    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    duplicate_timestamps: int = 0
    non_monotonic: int = 0

    missing_bars: int = 0
    missing_pct: float = 0.0
    gap_runs: list = field(default_factory=list)

    ohlc_violations: int = 0
    non_positive_prices: int = 0
    zero_volume_bars: int = 0

    outliers_6sigma: int = 0

    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rows={self.rows}",
            f"span={self.start} .. {self.end}",
            f"duplicates={self.duplicate_timestamps}, non_monotonic={self.non_monotonic}",
            f"missing_bars={self.missing_bars} ({self.missing_pct:.3f}%)",
            f"ohlc_violations={self.ohlc_violations}",
            f"zero_volume={self.zero_volume_bars}",
            f"outliers(>6s)={self.outliers_6sigma}",
            f"PASSED={self.passed}",
        ]
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARN: {w}")
        return "\n".join(lines)


def validate_ohlcv(
    df: pd.DataFrame,
    max_gap_pct: float = 1.0,
    max_ohlc_violations: int = 0,
) -> QualityReport:
    """
    Run full quality pipeline.

    Gates:
        - duplicates / non-monotonic -> FAIL
        - missing bars > max_gap_pct (%)   -> FAIL
        - ohlc violations > allowed  -> FAIL
        - zero volume / outliers     -> WARN only (crypto reality)
    """

    rep = QualityReport()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.passed = False
        rep.errors.append(f"missing columns: {missing_cols}")
        return rep

    if df.empty:
        rep.passed = False
        rep.errors.append("empty frame")
        return rep

    ts = pd.to_datetime(df["timestamp"], utc=True)
    rep.rows = len(df)
    rep.start = ts.iloc[0]
    rep.end = ts.iloc[-1]

    # ---- 1. duplicates & order ----
    dup = int(ts.duplicated().sum())
    rep.duplicate_timestamps = dup
    non_mono = int((ts.diff().dropna() <= pd.Timedelta(0)).sum())
    rep.non_monotonic = non_mono

    if dup:
        rep.passed = False
        rep.errors.append(f"{dup} duplicate timestamps")
    if non_mono:
        rep.passed = False
        rep.errors.append(f"{non_mono} non-monotonic steps")

    # ---- 2. gaps ----
    diffs = ts.diff().dropna()
    if len(diffs):
        interval = diffs.mode().iloc[0]
        expected = (rep.end - rep.start) / interval + 1
        missing = int(expected - len(df)) if expected > len(df) else 0
        rep.missing_bars = missing
        rep.missing_pct = round(missing / float(expected) * 100.0, 4)

        big_gaps = diffs[diffs > interval * 3]
        rep.gap_runs = [
            (str(ts.iloc[i]), str(v))
            for i, v in zip(big_gaps.index, big_gaps)
        ][:20]

        if rep.missing_pct > max_gap_pct:
            rep.passed = False
            rep.errors.append(
                f"missing {rep.missing_bars} bars ({rep.missing_pct:.3f}% > {max_gap_pct}%)"
            )

    # ---- 3. OHLC integrity ----
    o, h, l, c = (
        df["open"].astype(float),
        df["high"].astype(float),
        df["low"].astype(float),
        df["close"].astype(float),
    )
    v = df["volume"].astype(float)

    bad = (
        (h < l)
        | (h < np.maximum(o, c) - 1e-9)
        | (l > np.minimum(o, c) + 1e-9)
    )
    rep.ohlc_violations = int(bad.sum())

    neg = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
    rep.non_positive_prices = neg

    if rep.ohlc_violations > max_ohlc_violations:
        rep.passed = False
        rep.errors.append(f"{rep.ohlc_violations} OHLC violations")
    if neg:
        rep.passed = False
        rep.errors.append(f"{neg} non-positive prices")

    # ---- 4. volume ----
    rep.zero_volume_bars = int((v <= 0).sum())
    if rep.zero_volume_bars:
        share = rep.zero_volume_bars / rep.rows
        if share > 0.05:
            rep.warnings.append(f"{share:.1%} zero-volume bars")

    # ---- 5. outliers ----
    rets = np.log(c).diff().dropna()
    if len(rets) > 30:
        sigma = rets.rolling(200, min_periods=30).std()
        z = (rets - rets.rolling(200, min_periods=30).mean()) / sigma
        rep.outliers_6sigma = int((z.abs() > 6).sum())
        if rep.outliers_6sigma:
            rep.warnings.append(
                f"{rep.outliers_6sigma} return outliers beyond 6 sigma"
            )

    return rep


def drop_duplicates_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleanup used by the downloader before persisting."""

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(subset="timestamp", keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out
