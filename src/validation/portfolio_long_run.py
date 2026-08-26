"""
QuantAI Long-Run PORTFOLIO session (R3 extension for P-C1 champion)

Runs the cross-sectional champion (top-K weekly rebalance) over LIVE
daily candles with crash-safe checkpointing, producing Gate-compatible
artifacts:

    <dir>/state.json     started_at/balance/last_date/counters + meta
    <dir>/journal.csv    SAME schema as single-symbol long-run
                         (one row per closed SLOT: symbol entry/exit)

Broker: multi-name cash ledger (strict: cash = initial + realized_gross
- fees at every step), equal or inv_vol weights, per-side commission.

Gate compatibility: evaluate_long_run() works unchanged; portfolio
class declares expected trade pace via meta so min_trades adapts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.strategies.cross_sectional import CrossSectionParams


# ============================================================
# STATE / CONFIG
# ============================================================

@dataclass
class PortfolioState:
    symbols: list[str]
    params: dict
    started_at: str
    balance: float
    last_completed_day: str | None = None   # ISO date of last processed day
    steps_done: int = 0                     # days processed
    rebalances: int = 0
    trades_closed: int = 0                  # closed slots journaled
    last_rebalance_day: str | None = None
    incidents: int = 0
    updated_at: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PortfolioState":
        d = json.loads(text)
        d["symbols"] = list(d.get("symbols", []))
        return cls(**d)


@dataclass
class PortfolioLongRunConfig:
    params: CrossSectionParams
    duration_days: int = 30
    initial_balance: float = 10_000.0
    # broker commission per side on notional (defaults to params fee)
    commission_per_side: float | None = None
    meta_extra: dict = field(default_factory=dict)


# ============================================================
# BROKER (multi-name, strict ledger)
# ============================================================

class PortfolioPaperBroker:
    """
    Multi-name cash broker with average-entry positions.

    Invariant at ALL times:
        cash + open_notional + open_entry_fees
            == initial_cash + realized_gross - fees_closed
    """

    def __init__(self, initial_cash: float, fee_per_side: float) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(self.initial_cash)
        self.fee = float(fee_per_side)
        self.positions: dict[str, dict] = {}
        self.realized_gross = 0.0
        self.fees_closed = 0.0
        self.open_notional = 0.0
        self.open_entry_fees = 0.0

    # -------------------------------------------------- views

    def equity(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None:
                px = pos["entry_price"]
            eq += pos["qty"] * px
        return eq

    def identity_gap(self) -> float:
        lhs = self.cash + self.open_notional + self.open_entry_fees
        rhs = self.initial_cash + self.realized_gross - self.fees_closed
        return abs(lhs - rhs)

    # -------------------------------------------------- internals

    def _add(self, sym: str, price: float, qty: float) -> None:
        fee = price * qty * self.fee
        pos = self.positions.get(sym)
        if pos is None:
            self.positions[sym] = {
                "qty": qty,
                "entry_price": price,
                "entry_fee": fee,
            }
        else:
            tot = pos["qty"] + qty
            pos["entry_price"] = (
                (pos["entry_price"] * pos["qty"] + price * qty) / tot
            )
            pos["entry_fee"] += fee
            pos["qty"] = tot
        self.cash -= price * qty + fee
        self.open_notional += price * qty
        self.open_entry_fees += fee

    def _reduce(self, sym: str, price: float, dq: float) -> dict:
        """Realize dq units at `price` using average entry."""
        pos = self.positions[sym]
        dq = min(dq, pos["qty"])
        gross = (price - pos["entry_price"]) * dq
        exit_fee = price * dq * self.fee
        fee_share = pos["entry_fee"] * (dq / pos["qty"]) if pos["qty"] else 0.0
        net = gross - fee_share - exit_fee

        self.cash += dq * price - exit_fee
        self.realized_gross += gross
        self.fees_closed += fee_share + exit_fee
        self.open_notional -= pos["entry_price"] * dq
        self.open_entry_fees -= fee_share

        pos["qty"] -= dq
        pos["entry_fee"] -= fee_share
        row = {
            "side": "LONG",
            "entry": round(pos["entry_price"], 8),
            "exit": round(float(price), 8),
            "qty": round(dq, 10),
            "gross": round(gross, 8),
            "fees": round(fee_share + exit_fee, 8),
            "net": round(net, 8),
        }
        if pos["qty"] <= 1e-12:
            del self.positions[sym]
        return row

    # -------------------------------------------------- public ops

    def open_slot(self, sym: str, price: float, qty: float) -> None:
        if qty <= 0 or price <= 0:
            return
        self._add(sym, float(price), float(qty))

    def close_slot(self, sym: str, price: float) -> list[dict]:
        if sym not in self.positions:
            return []
        pos = self.positions[sym]
        row = self._reduce(sym, float(price), pos["qty"])
        return [row]

    def rebalance_slot(
        self, sym: str, price: float, target_qty: float
    ) -> list[dict]:
        """Move holding toward target_qty. Sells produce journal rows."""
        rows: list[dict] = []
        price = float(price)
        target_qty = max(float(target_qty), 0.0)

        pos = self.positions.get(sym)
        current = pos["qty"] if pos else 0.0
        diff = target_qty - current

        if diff < 0 and pos:
            rows.append(self._reduce(sym, price, -diff))
        elif diff > 0:
            # cash guard on the BUY leg
            budget = min(diff * price * (1 + self.fee), self.cash)
            buy_qty = budget / (price * (1 + self.fee)) if price > 0 else 0.0
            if buy_qty > 0:
                self._add(sym, price, buy_qty)
        return rows


# ============================================================
# SESSION
# ============================================================

class PortfolioLongRunSession:
    def __init__(
        self,
        out_dir: Path,
        cfg: PortfolioLongRunConfig,
        symbols: list[str],
        candle_provider=None,          # (symbol, since_ms|None) -> DataFrame 1d raw
    ) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.symbols = list(symbols)
        self.provider = candle_provider or self._ccxt_provider

        fee = cfg.commission_per_side or cfg.params.fee_per_side
        self.broker = PortfolioPaperBroker(cfg.initial_balance, fee)

        self.state_path = self.out / "state.json"
        self.journal_path = self.out / "journal.csv"

        self.state = self._load_or_init()
        self.closes: dict[str, pd.Series] = {}

        if not self.journal_path.exists():
            self.journal_path.write_text(
                "close_time,side,entry,exit,qty,gross,fees,net,balance\n",
                encoding="utf-8",
            )

    # -------------------------------------------------- providers

    @staticmethod
    def _ccxt_provider(symbol: str, since_ms: int | None):
        """Daily klines from Binance spot (public endpoint)."""
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True})
        rows = ex.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=1000)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "close"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D")
        return df[["timestamp", "close"]]

    # -------------------------------------------------- persistence

    def _load_or_init(self) -> PortfolioState:
        if self.state_path.exists():
            return PortfolioState.from_json(self.state_path.read_text(encoding="utf-8"))

        return PortfolioState(
            symbols=list(self.symbols),
            params={
                "lookback_days": self.cfg.params.lookback_days,
                "top_k": self.cfg.params.top_k,
                "rebalance_days": self.cfg.params.rebalance_days,
                "weighting": self.cfg.params.weighting,
                "fee_per_side": self.cfg.params.fee_per_side,
                "target_ann_vol": self.cfg.params.target_ann_vol,
                "dd_soft_stop_pct": self.cfg.params.dd_soft_stop_pct,
                "dd_reentry_pct": self.cfg.params.dd_reentry_pct,
            },
            started_at=datetime.now(timezone.utc).isoformat(),
            balance=self.cfg.initial_balance,
            updated_at=datetime.now(timezone.utc).isoformat(),
            meta={
                "asset_class": "portfolio_xs_momentum",
                # weekly top-K => ~2*K slots/week; gate default 30 trades
                # would need ~4 months; scale expectation to class:
                "min_trades_per_30d": max(4, 2 * self.cfg.params.top_k * 4),
                **self.cfg.meta_extra,
            },
        )

    def _save(self) -> None:
        self.state.balance = round(self._last_prices_equity(), 6)
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(self.state.to_json(), encoding="utf-8")
        tmp.replace(self.state_path)

    def _last_prices_equity(self) -> float:
        prices = {
            s: float(self.closes[s].iloc[-1])
            for s in self.broker.positions
            if s in self.closes and len(self.closes[s])
        }
        return self.broker.equity(prices)

    def _journal_append(self, rows: list[dict], when: str) -> None:
        if not rows:
            return
        with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            eq = round(self._last_prices_equity(), 6)
            for r in rows:
                w.writerow([when, r["side"], r["entry"], r["exit"],
                            r["qty"], r["gross"], r["fees"], r["net"], eq])
        self.state.trades_closed += len(rows)

    # -------------------------------------------------- history maintenance

    def _ingest(self, sym: str) -> None:
        since_ms = None
        if sym in self.closes and len(self.closes[sym]):
            last = self.closes[sym].index[-1]
            since_ms = int(last.timestamp() * 1000) + 1

        df = self.provider(sym, since_ms)
        if df is None or df.empty:
            return

        s = df.set_index("timestamp")["close"].astype(float)
        s = s[~s.index.duplicated(keep="last")]

        if sym in self.closes:
            combined = pd.concat([self.closes[sym], s])
            self.closes[sym] = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            self.closes[sym] = s.sort_index()

    def _drop_forming_day(self, series: pd.Series, today: pd.Timestamp) -> pd.Series:
        """Keep only COMPLETED UTC days."""
        return series[series.index < today.normalize()]

    # -------------------------------------------------- daily step

    def _process_new_day(self, day: pd.Timestamp) -> int:
        """
        Process one completed day: append closes, rebalance when due.
        Returns number of journal rows produced.
        """
        try:
            # Per-day VIEW (no mutation): stored history must survive,
            # otherwise the lookback window dies with the first day.
            views = {}
            for sym in self.symbols:
                ser = self.closes.get(sym)
                if ser is None:
                    return 0
                views[sym] = ser[ser.index <= day.normalize()]

            closes_df = pd.DataFrame(views).dropna(how="any")
            closes_df = closes_df.iloc[-400:]   # trailing window cap

            # universe completeness guard: never trade on partial data
            if any(s not in closes_df.columns or closes_df[s].isna().any()
                   for s in self.symbols):
                return 0

            # Backtest alignment: first period starts when exactly
            # lookback+1 completed bars exist (entry at close(t)).
            need = self.cfg.params.lookback_days + 1
            if len(closes_df) < need:
                return 0

            # ---- pick targets using data up to `day` (causal) ----
            now_row = closes_df.iloc[-1]
            past_row = closes_df.iloc[-1 - self.cfg.params.lookback_days]
            mom = (now_row / past_row - 1.0).dropna()

            picked = list(mom.nlargest(min(self.cfg.params.top_k, len(mom))).index)

            # ---- weekly cadence gate ----
            last_rb = (
                pd.Timestamp(self.state.last_rebalance_day, tz="UTC")
                if self.state.last_rebalance_day else None
            )
            due = last_rb is None or (day - last_rb).days >= self.cfg.params.rebalance_days
            if not due:
                return 0

            # ---- rebalance to equal-weight targets ----
            journal_rows: list[dict] = []
            today_iso = day.date().isoformat()

            last_prices = {
                s: float(closes_df.iloc[-1][s]) for s in closes_df.columns
            }

            # full exits for dropped names
            for held in list(self.broker.positions.keys()):
                if held not in picked:
                    journal_rows.extend(
                        self.broker.close_slot(held, last_prices[held])
                    )

            equity_now = self.broker.equity(last_prices)
            per_name = equity_now / len(picked)

            for sym in picked:
                target_qty = per_name / last_prices[sym]
                journal_rows.extend(
                    self.broker.rebalance_slot(sym, last_prices[sym], target_qty)
                )

            self.state.last_rebalance_day = today_iso
            self.state.rebalances += 1
            self._journal_append(journal_rows, today_iso)
            return len(journal_rows)

        except Exception as e:  # noqa: BLE001
            self.state.incidents += 1
            self.state.meta["last_error"] = f"{type(e).__name__}: {e}"
            return 0

    @staticmethod
    def _keep_upto(series: pd.Series, day: pd.Timestamp) -> pd.Series:
        return series[series.index <= day.normalize()].iloc[-400:]

    # -------------------------------------------------- main loop

    def run_until_target(self, max_wall_seconds: int | None = None) -> PortfolioState:
        deadline = time.time() + max_wall_seconds if max_wall_seconds else None

        today_utc = pd.Timestamp.now(tz="UTC").normalize()

        # ingest all symbols once per invocation
        for sym in self.symbols:
            self._ingest(sym)

        # determine unprocessed completed days
        common_idx = None
        for s in self.closes.values():
            idx = self._drop_forming_day(s, today_utc).index
            common_idx = idx if common_idx is None else common_idx.union(idx)

        if common_idx is None:
            self._save()
            return self.state

        last_done = (
            pd.Timestamp(self.state.last_completed_day, tz="UTC")
            if self.state.last_completed_day else None
        )
        new_days = [d for d in common_idx if last_done is None or d > last_done]

        for day in new_days:
            made = self._process_new_day(day)
            self.state.steps_done += 1
            self.state.last_completed_day = day.date().isoformat()
            if made or self.state.steps_done % 5 == 0:
                self._save()
            if deadline and time.time() > deadline:
                break

        # mark-to-market balance & persist
        self._save()

        gap = self.broker.identity_gap()
        if gap > 1e-6:
            self.state.incidents += 1
        return self.state



# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/long_run_portfolio")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,"
                                         "ADAUSDT,DOGEUSDT,LINKUSDT,LTCUSDT,AVAXUSDT,"
                                         "DOTUSDT,TRXUSDT,UNIUSDT,ATOMUSDT,NEARUSDT,"
                                         "APTUSDT,ARBUSDT,SUIUSDT")
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--rebalance", type=int, default=7)
    ap.add_argument("--duration-days", type=int, default=30)
    ap.add_argument("--max-minutes", type=int, default=None)
    a = ap.parse_args()

    cfg = PortfolioLongRunConfig(
        params=CrossSectionParams(lookback_days=a.lookback, top_k=a.top_k,
                                  rebalance_days=a.rebalance),
        duration_days=a.duration_days,
    )
    sess = PortfolioLongRunSession(Path(a.out), cfg, a.symbols.split(","))
    st = sess.run_until_target(
        max_wall_seconds=a.max_minutes * 60 if a.max_minutes else None
    )
    print(json.dumps(asdict(st), indent=2, default=str))
