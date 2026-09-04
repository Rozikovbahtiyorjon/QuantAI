"""
QuantAI Quant Researcher Agent
Real alpha research with tournament-based strategy evaluation
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from src.indicators import add_indicators
from src.feature_engine import build_features
from src.dataset_builder import DatasetBuilder, DatasetConfig
from src.ml_engine import MLEngine, MLConfig
from src.labeling import TripleBarrierConfig, label_dataset
from src.backtest_engine import BacktestEngine
from src.walk.walk_forward_engine import WalkForwardEngine
from src.ml_engine import MLEngine, MLConfig
from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
from src.risk.policy import get_policy, ResearchPolicy
from src.control_plane.task_manager import Task, TaskPriority
from src.control_plane.evidence_manager import EvidenceManager, Evidence, EvidenceType
from src.ml_engine import MLEngine, MLConfig
from src.dataset_builder import DatasetBuilder, DatasetConfig
from src.labeling import TripleBarrierConfig

# ============================================================
# Data Structures
# ============================================================

@dataclass
class StrategyResult:
    """Result of a single strategy evaluation."""
    name: str
    backtest_pf: float
    backtest_return_pct: float
    backtest_maxdd_pct: float
    backtest_trades: int
    backtest_win_rate: float
    wf_pf_median: float
    wf_profit: float
    wf_sharpe: float
    wf_maxdd_pct: float
    wf_windows: int
    wf_profitable_share: float
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TournamentResult:
    """Complete tournament results."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    best_strategy: Optional[str] = None
    best_pf: float = 0.0
    recommendation: str = "reject"  # "deploy" | "reject" | "iterate"
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================
# Quant Researcher Agent
# ============================================================

class QuantResearcher:
    """
    Real Quant Researcher Agent.
    Runs tournament of strategies on multi-asset 4h data with Walk-Forward validation.
    """

    def __init__(self, 
                 data_path: str = "data",
                 symbols: Optional[List[str]] = None,
                 timeframe: str = "4h",
                 risk_policy: Any = None,
                 evidence_manager: Any = None):
        self.data_path = Path(data_path)
        self.symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "UNIUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
            "NEARUSDT", "FILUSDT", "ICPUSDT", "VETUSDT", "TRXUSDT"
        ]
        self.timeframe = timeframe
        self.risk_policy = risk_policy or ResearchPolicy
        self.evidence_manager = evidence_manager
        
        # Strategy configurations (frozen after tuning)
        self.strategy_configs = {
            "baseline": {
                "class": "SignalGenerator",
                "config": {"use_regime_adaptive": True, "use_ml": False},
                "weight": 0.3
            },
            "breakout": {
                "class": "BreakoutSignalGenerator",
                "config": {
                    "channel_bars": 96,
                    "min_adx": 20.0,
                    "sl_atr_mult": 3.0,
                    "cooldown_bars": 12
                },
                "weight": 0.5
            },
            "mean_reversion": {
                "class": "MeanReversionSignalGenerator",
                "config": {
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "rsi_period": 14,
                    "rsi_oversold": 30.0,
                    "rsi_overbought": 70.0,
                    "max_adx": 60.0,
                    "sl_atr_mult": 1.5,
                    "tp_atr_mult": 3.0,
                    "cooldown_bars": 8
                },
                "weight": 0.2
            }
        }

    async def run_tournament(self, 
                            data_path: Optional[str] = None,
                            symbols: Optional[List[str]] = None,
                            timeframe: str = "4h",
                            train_size: int = 3000,
                            test_size: int = 600,
                            step_size: int = 600,
                            n_windows: int = 6) -> Dict[str, Any]:
        """
        Run full tournament: load data → build features → evaluate 3 strategies → WF → recommend.
        Fail-closed: any exception returns success=False.
        """
        try:
            # 1. Load & prepare data
            data_path = Path(data_path) if data_path else Path("data")
            symbols = symbols or self.symbols[:10]  # top-10 for speed
            
            # Load multi-asset data
            prices = {}
            for symbol in symbols:
                fp = Path("data") / f"{symbol.lower()}_4h_prepared.parquet"
                if not fp.exists():
                    # Try raw
                    fp_raw = Path("data") / f"{symbol.lower()}_4h.parquet"
                    if fp_raw.exists():
                        df = pd.read_parquet(fp_raw)
                        from src.indicators import add_indicators
                        df = add_indicators(df)
                        df.to_parquet(Path("data") / f"{symbol.lower()}_4h_prepared.parquet")
                        prices[symbol] = df["close"]
                    continue
                df = pd.read_parquet(fp)
                prices[symbol] = df["close"]
            
            if not prices:
                return {"success": False, "error": "No data loaded"}
            
            prices_df = pd.DataFrame(prices).sort_index().ffill().dropna()
            
            # Run tournament — guard that string/exception never propagates as dict
            results = await self._run_tournament(prices_df, n_windows=6)
            if isinstance(results, str):
                return {"success": False, "error": results, "tournament": results, "recommendation": "reject"}
            if not isinstance(results, dict):
                return {"success": False, "error": f"Unexpected results type {type(results)}", "tournament": str(results), "recommendation": "reject"}
            
            # Store evidence (fail-closed, never crash tournament)
            try:
                evidence_id = await self._store_evidence(results)
            except Exception as e:
                evidence_id = ""
            
            recommendation = self._make_recommendation(results)
            
            return {
                "success": True,
                "tournament": results,
                "recommendation": recommendation,
                "evidence_id": evidence_id
            }
        except Exception as e:
            return {"success": False, "error": str(e), "tournament": {}, "recommendation": "reject"}

    async def _run_tournament(self, prices_df: pd.DataFrame, n_windows: int = 6) -> Dict[str, Any]:
        """Run full tournament on single-asset (BTC) or multi-asset."""
        from src.backtest_engine import BacktestEngine
        from src.walk.walk_forward_engine import WalkForwardEngine
        from src.strategy.signal_generator import SignalGenerator, SignalConfig
        from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
        from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
        from src.walk.walk_forward_engine import WalkForwardEngine
        import src.trade_engine as te_mod

        strategies = {
            "baseline": self._create_baseline_signal,
            "breakout": self._create_breakout_signal,
            "mean_reversion": self._create_mean_reversion_signal
        }
        
        results = {}
        for name, signal_fn in strategies.items():
            print(f"\n=== Tournament: {name} ===")
            try:
                # Backtest - directly call signal function
                be = __import__('src.backtest_engine', fromlist=['BacktestEngine']).BacktestEngine(initial_balance=1000.0)
                
                # Inject our signal generator into trade_engine
                import src.trade_engine as te_mod
                original_generate = te_mod.generate_signal_result
                te_mod.generate_signal_result = lambda df_hist: signal_fn(df_hist)
                
                try:
                    res = be.run(self._prepare_data_for_backtest())
                finally:
                    te_mod.generate_signal_result = original_generate
                
                # Walk-forward
                wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
                wf_res = wf.run(self._prepare_data_for_backtest())

                # Compute WF metrics before storing
                wf_pf_median = float(np.median([
                    w.backtest_result.profit_factor 
                    for w in wf_res.windows 
                    if w.backtest_result.profit_factor != float('inf')
                ])) if wf_res.windows else 0

                results[name] = {
                    "backtest_pf": res.profit_factor if res.profit_factor != float('inf') else 99,
                    "backtest_ret": res.total_return_pct,
                    "backtest_dd": res.max_drawdown_pct,
                    "backtest_trades": res.total_trades,
                    "backtest_win": res.win_rate,
                    "wf_profit": wf_res.net_profit,
                    "wf_pf_median": wf_pf_median,
                    "wf_sharpe": float(np.median([w.backtest_result.sharpe for w in wf_res.windows])) if wf_res.windows else 0,
                    "wf_maxdd": float(np.median([w.backtest_result.max_drawdown_pct for w in wf_res.windows])) if wf_res.windows else 0,
                    "wf_windows": wf_res.total_windows,
                    "wf_win_rate": float(np.mean([w.backtest_result.win_rate for w in wf_res.windows])) if wf_res.windows else 0,
                    "wf_profitable_share": sum(1 for w in wf_res.windows if w.backtest_result.net_profit > 0) / len(wf_res.windows) if wf_res.windows else 0
                }

                print(f"  {name}: BT PF={res.profit_factor:.3f} ret={res.total_return_pct:.1f}% DD={res.max_drawdown_pct:.1f}% | WF PF={wf_pf_median:.3f} profit={wf_res.net_profit:.1f}$")
            except Exception as e:
                print(f"  {name}: ERROR {e}")
                results[name] = {
                    "backtest_pf": 0.0,
                    "backtest_ret": 0.0,
                    "backtest_dd": 0.0,
                    "backtest_trades": 0,
                    "backtest_win": 0.0,
                    "wf_profit": 0.0,
                    "wf_pf_median": 0.0,
                    "wf_sharpe": 0.0,
                    "wf_maxdd": 0.0,
                    "wf_windows": 0,
                    "wf_win_rate": 0.0,
                    "wf_profitable_share": 0.0,
                    "error": str(e)
                }
        
        if not results:
            return {"strategies": {}, "best": None, "best_wf": None, "error": "No strategies evaluated"}
        try:
            best = max(results.items(), key=lambda x: x[1].get("backtest_pf", 0) if isinstance(x[1], dict) else -1)[0]
        except Exception:
            best = next(iter(results))
        try:
            best_wf = max(results.items(), key=lambda x: x[1].get("wf_pf_median", 0) if isinstance(x[1], dict) else -1)[0]
        except Exception:
            best_wf = best
        return {
            "strategies": results,
            "best": best,
            "best_wf": best_wf
        }

    def _prepare_data_for_backtest(self):
        """Load and prepare BTC 4h data with indicators."""
        import pandas as pd
        from src.indicators import add_indicators
        df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
        if 'bb_position' not in df.columns:
            df = add_indicators(df)
        return df

    def _create_baseline_signal(self, df: pd.DataFrame) -> dict:
        from src.strategy.signal_generator import SignalGenerator, SignalConfig
        sg = SignalGenerator(SignalConfig(use_regime_adaptive=True, use_ml=False))
        return sg.generate(df)

    def _create_breakout_signal(self, df: pd.DataFrame) -> dict:
        from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
        gen = BreakoutSignalGenerator(BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12))
        return gen.generate(df)

    def _create_mean_reversion_signal(self, df: pd.DataFrame) -> dict:
        from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
        gen = MeanReversionSignalGenerator(MeanReversionConfig(max_adx=60.0))
        return gen.generate(df)

    async def _store_evidence(self, results: dict) -> str:
        """Store tournament results as evidence."""
        if hasattr(self, 'evidence_manager') and self.evidence_manager:
            try:
                from src.control_plane.evidence_manager import Evidence, EvidenceType
                # Only persist dict-type strategy results; skip string metadata (best/best_wf)
                safe_tournament = {}
                for k, v in results.items():
                    if isinstance(v, dict):
                        safe_tournament[k] = {kk: vv for kk, vv in v.items() if kk != 'metadata'}
                    else:
                        # preserve scalar metadata as-is under metadata key
                        safe_tournament[k] = v
                evidence = Evidence(
                    type=EvidenceType.BACKTEST_RESULT,
                    data={"tournament": safe_tournament},
                    source="quant_researcher",
                    tags=["tournament", "4h", "btcusdt"]
                )
                return await self.evidence_manager.store(evidence)
            except Exception:
                return ""
        return ""

    def _make_recommendation(self, results: Any) -> str:
        """Derive deploy/iterate/reject from tournament results — fail-closed."""
        try:
            if isinstance(results, str) or not isinstance(results, dict):
                return "reject"
            # Tournament dict has shape {"strategies": {name: metrics}, "best": str, ...}
            # Legacy fallback: results itself may be strategies dict
            if "strategies" in results and isinstance(results["strategies"], dict):
                strategies = results["strategies"]
            else:
                strategies = {k: v for k, v in results.items() if isinstance(v, dict) and "backtest_pf" in v}
            if not strategies:
                return "reject"
            # filter non-dict entries safely
            best = max(
                strategies.items(),
                key=lambda x: x[1].get("backtest_pf", 0) if isinstance(x[1], dict) else -1
            )
            best_name, best_metrics = best
            if not isinstance(best_metrics, dict):
                return "reject"
            wf_pf = best_metrics.get("wf_pf_median", 0)
            wf_profit = best_metrics.get("wf_profit", 0)
            if wf_pf >= 1.15 and wf_profit > 0:
                return "deploy"
            elif wf_pf >= 1.05:
                return "iterate"
            else:
                return "reject"
        except Exception:
            return "reject"

# ============================================================
# Factory for Control Plane Integration
# ============================================================

async def create_quant_researcher(evidence_manager=None) -> "QuantResearcher":
    """Factory for Control Plane integration."""
    return QuantResearcher(evidence_manager=evidence_manager)


# Quick test
if __name__ == "__main__":
    import asyncio
    async def test():
        researcher = QuantResearcher()
        result = await researcher.run_tournament()
        print(f"Recommendation: {result['recommendation']}")
        print(f"Best: {result['tournament']['best_strategy'] if 'best_strategy' in result['tournament'] else 'N/A'}")
    
    asyncio.run(test())