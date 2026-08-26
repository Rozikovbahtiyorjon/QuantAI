"""P-C1 data: download extra 1h series + build daily wide price frame."""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src.historical_downloader import download_series

DATA = Path("data")

EXTRA = ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]

for sym in EXTRA:
    path = DATA / f"{sym.lower()}_1h.parquet"
    prep = DATA / f"{sym.lower()}_1h_prepared.parquet"
    if path.exists():
        print(f"skip {sym} (exists)")
        continue
    t0 = time.time()
    try:
        df, rep = download_series(sym, "1h", years=3.0, out_dir=DATA)
        q = rep["quality"]
        print(f"OK {sym}: {len(df)} rows {time.time()-t0:.0f}s | "
              f"{q.start.date()}..{q.end.date()} miss={q.missing_pct}%")
    except Exception as e:
        print(f"FAIL {sym}: {type(e).__name__}: {str(e)[:120]}")


# ---- daily wide closes ----
SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
           "adausdt", "dogeusdt", "linkusdt"]

closes = {}
for sym in SYMBOLS:
    p = DATA / f"{sym}_1h.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = df.set_index("timestamp")["close"].resample("1D").last().dropna()
    closes[sym.upper()] = s

wide = pd.DataFrame(closes)
# drop days where majority missing (early history of some alts)
wide = wide.dropna(how="all")
first_valid = wide.apply(lambda col: col.first_valid_index())
start = max(v for v in first_valid if v is not None)
wide = wide[wide.index >= start]

out = DATA / "portfolio_daily_closes.parquet"
wide.to_parquet(out, index=True)
print(f"\nwide frame: {wide.shape[0]} days x {wide.shape[1]} symbols "
      f"({wide.index[0].date()} .. {wide.index[-1].date()})")
print("coverage per symbol (% non-null):")
print((wide.notna().mean() * 100).round(1).to_string())
