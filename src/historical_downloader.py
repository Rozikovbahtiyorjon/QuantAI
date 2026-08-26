"""
QuantAI Historical Data Downloader

Primary source:  data.binance.vision monthly kline archives
                 (bulk zips, no API rate limits, spot market).
Fallback:        ccxt paginated fetch for the current (partial) month.

Output: one parquet per symbol/timeframe under data/.

Usage (module CLI):
    python -m src.historical_downloader \
        --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h --years 3
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error

from src.data_quality import drop_duplicates_and_sort, validate_ohlcv

VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


# =====================================================
# binance.vision monthly archives
# =====================================================

def _month_iter(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def _download_zip(url: str, timeout: int = 60) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quantai/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Parse epoch timestamps auto-detecting unit.

    binance.vision switched open_time to microseconds in 2025;
    naive unit='ms' produces dates like year 58551.
    """

    num = pd.to_numeric(series, errors="coerce")
    vmax = float(num.max()) if len(num) else 0.0

    # Epoch magnitudes: s ~1.8e9 | ms ~1.8e12 | us ~1.8e15
    if vmax >= 1e15:         # microseconds
        return pd.to_datetime(num, unit="us", utc=True)
    if vmax >= 1e11:         # milliseconds
        return pd.to_datetime(num, unit="ms", utc=True)
    # seconds
    return pd.to_datetime(num, unit="s", utc=True)


def _parse_klines_csv(raw: bytes) -> pd.DataFrame:
    """
    Parse a binance.vision klines CSV from a zip archive.
    Handles both header-less and header-ed files; some rows carry
    a trailing 'ignore' column in newer exports.
    """

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.readline().decode("utf-8", errors="ignore")
            fh.seek(0)

            has_header = not head.split(",")[0].strip().isdigit()

            df = pd.read_csv(
                fh,
                header=0 if has_header else None,
                skipinitialspace=True,
            )

    # Normalize columns: keep first 6 relevant ones.
    if has_header:
        df.columns = [str(c).strip().lower() for c in df.columns]
        ts_col = (
            "open_time" if "open_time" in df.columns else df.columns[0]
        )
        o, h, l, c, v = (
            "open" if "open" in df.columns else df.columns[1],
            "high" if "high" in df.columns else df.columns[2],
            "low" if "low" in df.columns else df.columns[3],
            "close" if "close" in df.columns else df.columns[4],
            "volume" if "volume" in df.columns else df.columns[5],
        )
    else:
        ts_col, o, h, l, c, v = df.columns[:6]

    out = pd.DataFrame(
        {
            "timestamp": _to_utc_datetime(df[ts_col]),
            "open": pd.to_numeric(df[o], errors="coerce"),
            "high": pd.to_numeric(df[h], errors="coerce"),
            "low": pd.to_numeric(df[l], errors="coerce"),
            "close": pd.to_numeric(df[c], errors="coerce"),
            "volume": pd.to_numeric(df[v], errors="coerce"),
        }
    ).dropna()

    return out


def fetch_month(symbol: str, timeframe: str, year: int, month: int) -> pd.DataFrame | None:
    """Fetch one monthly archive; None when it does not exist yet."""

    url = f"{VISION_BASE}/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month:02d}.zip"
    raw = _download_zip(url)
    if raw is None:
        return None
    try:
        return _parse_klines_csv(raw)
    except Exception:
        return None


# =====================================================
# ccxt fallback for the recent partial month
# =====================================================

def fetch_recent_ccxt(
    symbol: str,
    timeframe: str,
    since_ms: int,
    max_requests: int = 2000,
) -> pd.DataFrame | None:
    """
    Paginated OHLCV fetch via ccxt (used to top-up after the last
    complete monthly archive). Silent-None on connectivity failure.
    """

    try:
        import ccxt
    except ImportError:
        return None

    ex = ccxt.binance({"enableRateLimit": True})
    step_ms = TIMEFRAME_MS[timeframe] * 1000  # fetch_ohlcv limit=1000 bars

    frames = []
    cursor = since_ms
    for _ in range(max_requests):
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        except Exception:
            break
        if not batch:
            break
        frames.append(pd.DataFrame(batch, columns=COLUMNS))
        cursor = batch[-1][0] + TIMEFRAME_MS[timeframe]
        if len(batch) < 1000:
            break

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# =====================================================
# Public pipeline
# =====================================================

def download_series(
    symbol: str,
    timeframe: str,
    years: float = 3.0,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Download `years` of history for one symbol/timeframe.

    Returns (df, report). Raises RuntimeError on total failure or
    failed quality gates.
    """

    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    now_utc = datetime.now(timezone.utc)

    # Start point: first day of the month `years` ago.
    start_dt = now_utc - pd.DateOffset(months=int(years * 12))
    start_year, start_month = start_dt.year, start_dt.month

    # Monthly archives exist only up to LAST full month.
    last_full = now_utc.replace(day=1) - pd.Timedelta(days=1)
    last_year, last_month = last_full.year, last_full.month

    frames = []
    fetched_months = 0

    for y, m in _month_iter(start_year, start_month, last_year, last_month):
        part = fetch_month(symbol, timeframe, y, m)
        if part is not None and len(part):
            frames.append(part)
            fetched_months += 1

    if not frames:
        raise RuntimeError(
            f"No monthly archives fetched for {symbol} {timeframe} "
            f"({start_year}-{start_month:02d} .. {last_year}-{last_month:02d})"
        )

    df = pd.concat(frames, ignore_index=True)

    # Top-up the current partial month via REST.
    tail = fetch_recent_ccxt(
        symbol,
        timeframe,
        since_ms=int(pd.Timestamp(last_full).timestamp() * 1000),
    )
    topped = False
    if tail is not None and len(tail):
        df = pd.concat([df, tail], ignore_index=True)
        topped = True

    df = drop_duplicates_and_sort(df)

    rep = validate_ohlcv(df)
    report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "months_fetched": fetched_months,
        "topped_with_rest": topped,
        "quality": rep,
    }

    if not rep.passed:
        raise RuntimeError(
            f"Quality gates FAILED for {symbol} {timeframe}:\n{rep.summary()}"
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol.lower()}_{timeframe}.parquet"
        df.to_parquet(path, index=False)
        report["path"] = str(path)

    return df, report


__all__ = [
    "download_series",
    "fetch_month",
    "fetch_recent_ccxt",
    "TIMEFRAME_MS",
]
