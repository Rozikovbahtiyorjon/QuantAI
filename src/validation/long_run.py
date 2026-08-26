"""
QuantAI Long-Run Paper Validation harness (R3)

Runs a paper-trading session over live candles for DAYS/WEEKS with
crash-safe checkpointing, producing the evidence consumed by
Gate check `long_run_paper`:

    <dir>/state.json     last timestamp, balance, counters
    <dir>/journal.csv    one row per closed trade

Design:
    - candle provider is INJECTED (default: ccxt binance fetch_ohlcv)
      so tests can feed synthetic candles without network.
    - resume: continues from state.last_open_time; dedupes by ts.
    - risk controls always ON (safe default), execution via
      ExecutionBridge PAPER mode.

Usage (live):
    python -m src.validation.long_run --hours 720 --symbol BTCUSDT --timeframe 1h
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

from src.execution.intent_bridge import ExecutionBridge
from src.paper_trading_runner import PaperTradingRunner


# ============================================================
# STATE / CONFIG
# ============================================================

@dataclass
class LongRunConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    duration_hours: int = 24 * 30          # 30 days target
    initial_balance: float = 1000.0
    commission: float = 0.0004
    history_window: int = 300              # bars passed to strategy per step
    warmup_bars: int = 200                 # initial history fetched at start
    checkpoint_every_steps: int = 1


@dataclass
class LongRunState:
    symbol: str
    timeframe: str
    started_at: str
    balance: float
    last_open_time_ms: int | None = None
    steps_done: int = 0
    signals_processed: int = 0
    trades_closed: int = 0
    incidents: int = 0
    updated_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LongRunState":
        return cls(**json.loads(text))


# ============================================================
# CANDLE PROVIDERS
# ============================================================

def ccxt_provider(symbol: str, timeframe: str, since_ms: int | None, limit: int = 1000):
    """Default provider: Binance spot via ccxt."""
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"])


# ============================================================
# SESSION
# ============================================================

class LongRunSession:
    def __init__(
        self,
        out_dir: Path,
        config: LongRunConfig | None = None,
        candle_provider=None,
        signal_generator_factory=None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or LongRunConfig()
        self.provider = candle_provider or ccxt_provider
        self.signal_generator_factory = signal_generator_factory  # None -> runner default

        self.state_path = self.out_dir / "state.json"
        self.journal_path = self.out_dir / "journal.csv"

        self.runner = PaperTradingRunner(
            initial_balance=self.cfg.initial_balance,
            commission=self.cfg.commission,
            enable_risk_controls=True,     # R3: risk ALWAYS on in long-run
        )
        self.bridge = ExecutionBridge.paper(self.runner)

        self.state = self._load_or_init_state()

        if not self.journal_path.exists():
            self.journal_path.write_text(
                "close_time,side,entry,exit,qty,gross,fees,net,balance\n",
                encoding="utf-8",
            )

    # -------------------------------------------------- persistence

    def _load_or_init_state(self) -> LongRunState:
        if self.state_path.exists():
            return LongRunState.from_json(self.state_path.read_text(encoding="utf-8"))

        return LongRunState(
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            started_at=datetime.now(timezone.utc).isoformat(),
            balance=self.cfg.initial_balance,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _save_state(self) -> None:
        self.state.balance = self.runner.engine.balance
        self.state.trades_closed = len(self.runner.engine.trade_history)
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(self.state.to_json(), encoding="utf-8")
        tmp.replace(self.state_path)

    def _append_journal(self) -> int:
        """Append trades not yet journaled; returns appended count."""
        total = len(self.runner.engine.trade_history)
        appended = 0

        if total > self.state.trades_closed:
            with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for trade in self.runner.engine.trade_history[self.state.trades_closed:]:
                    writer.writerow([
                        datetime.now(timezone.utc).isoformat(),
                        trade.side, trade.entry_price, trade.exit_price,
                        trade.quantity, trade.gross_profit, trade.fees,
                        trade.net_profit, self.runner.engine.balance,
                    ])
                    appended += 1
            self.state.trades_closed += appended
        return appended

    # -------------------------------------------------- main loop

    def run_until_target(self, max_wall_seconds: int | None = None) -> LongRunState:
        """
        Fetch candles step-by-step until duration target reached
        (or wall-clock budget exhausted). Crash-safe via checkpoints.
        """
        tf_ms = _timeframe_ms(self.cfg.timeframe)
        deadline = time.time() + max_wall_seconds if max_wall_seconds else None

        since_ms = self.state.last_open_time_ms
        if since_ms is None:
            # seed history for indicator warm-up
            seed_since = int(
                (
                    datetime.now(timezone.utc) - timedelta(
                        hours=_tf_hours(self.cfg.timeframe) * self.cfg.warmup_bars
                    )
                ).timestamp() * 1000
            )
            since_ms = seed_since

        window: pd.DataFrame = getattr(self, "_window_cache", None)

        while True:
            batch = self.provider(self.cfg.symbol, self.cfg.timeframe, since_ms)

            if batch is None or batch.empty:
                break  # no new data (caught up)

            for row in batch.itertuples(index=False):
                open_ms = int(pd.Timestamp(row.timestamp).timestamp() * 1000)

                if self.state.last_open_time_ms is not None and \
                        open_ms <= self.state.last_open_time_ms:
                    continue

                bar = pd.DataFrame([{
                    "timestamp": row.timestamp,
                    "open": row.open, "high": row.high, "low": row.low,
                    "close": row.close, "volume": row.volume,
                }])
                window = bar.copy() if window is None or window.empty else \
                    pd.concat([window, bar], ignore_index=True).tail(
                        self.cfg.history_window
                    )

                self._process_window(window)
                self.state.last_open_time_ms = open_ms
                self.state.steps_done += 1

                if self.state.steps_done % self.cfg.checkpoint_every_steps == 0:
                    self._append_journal()
                    self._save_state()

                if deadline and time.time() > deadline:
                    self._append_journal()
                    self._save_state()
                    return self.state

            since_ms = self.state.last_open_time_ms or since_ms

            target_end = _started_plus_duration(self.state, self.cfg.duration_hours)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            last_bar_ms = self.state.last_open_time_ms or 0
            if last_bar_ms >= target_end or now_ms >= target_end:
                break

        self._append_journal()
        self._save_state()
        self._window_cache = window
        return self.state

    # -------------------------------------------------- per-bar processing

    def _process_window(self, window: pd.DataFrame) -> None:
        try:
            # Live candles arrive raw; strategy requires prepared columns.
            # Indicators on a trailing window are causal (prefix-stable).
            if "ema_fast" not in window.columns:
                from src.indicators import add_indicators
                prepared = add_indicators(window.copy())
            else:
                prepared = window

            if self.signal_generator_factory is not None:
                signal = self.signal_generator_factory().generate(prepared)
            else:
                from src.strategy import generate_signal_result
                signal = generate_signal_result(prepared)

            # Long-run keeps risk inside the runner (orchestrator path).
            self.runner.process_signal(signal, df=prepared)
            self.state.signals_processed += 1

        except Exception:
            self.state.incidents += 1


# ============================================================
# CRITERIA (consumed by Gate)
# ============================================================

def evaluate_long_run(
    directory: Path,
    min_days: int = 30,
    min_trades: int = 30,
) -> dict:
    d = Path(directory)
    state_path = d / "state.json"
    journal = d / "journal.csv"

    if not state_path.exists():
        raise FileNotFoundError(f"no state.json in {d}")

    st = LongRunState.from_json(state_path.read_text(encoding="utf-8"))

    started = datetime.fromisoformat(st.started_at)
    updated = (
        datetime.fromisoformat(st.updated_at)
        if st.updated_at else datetime.now(timezone.utc)
    )
    days_covered = (updated - started).total_seconds() / 86400.0

    n_trades = 0
    net = 0.0
    if journal.exists():
        with open(journal, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            n_trades = len(rows)
            net = sum(float(r["net"]) for r in rows)

    days_ok = days_covered >= min_days
    trades_ok = n_trades >= min_trades
    incidents_ok = st.incidents == 0
    alive_ok = st.balance > 0

    passed = all([days_ok, trades_ok, incidents_ok, alive_ok])

    summary = (
        f"days={days_covered:.1f}/{min_days} trades={n_trades}/{min_trades} "
        f"incidents={st.incidents} net={net:.2f} balance={st.balance:.2f}"
    )

    return {
        "passed": passed,
        "summary": summary,
        "days_covered": round(days_covered, 2),
        "trades": n_trades,
        "net_pnl": round(net, 4),
        "balance": round(st.balance, 4),
        "incidents": st.incidents,
        "signals_processed": st.signals_processed,
        "checks": {
            "days_ok": days_ok, "trades_ok": trades_ok,
            "incidents_ok": incidents_ok, "alive_ok": alive_ok,
        },
    }


# ============================================================
# helpers / CLI
# ============================================================

def _timeframe_ms(tf: str) -> int:
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(tf[:-1]) * mult[tf[-1]]


def _tf_hours(tf: str) -> float:
    return _timeframe_ms(tf) / 3_600_000.0


def _started_plus_duration(state: LongRunState, hours: int) -> int:
    end = datetime.fromisoformat(state.started_at) + timedelta(hours=hours)
    return int(end.timestamp() * 1000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/long_run")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--hours", type=int, default=24 * 30)
    ap.add_argument("--max-minutes", type=int, default=None,
                    help="wall-clock budget for this invocation")
    a = ap.parse_args()

    cfg = LongRunConfig(symbol=a.symbol, timeframe=a.timeframe, duration_hours=a.hours)
    sess = LongRunSession(Path(a.out), cfg)
    st = sess.run_until_target(
        max_wall_seconds=a.max_minutes * 60 if a.max_minutes else None
    )
    print(json.dumps(asdict(st), indent=2))
