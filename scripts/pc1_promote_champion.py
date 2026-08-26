"""P-C1: honest OOS evaluation + first champion promotion."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src.champion.evaluation_pipeline import PromotionRules
from src.champion.feedback import MutationBounds
from src.champion.pipeline import ChampionPipeline
from src.champion.portfolio_adapter import evaluate_portfolio_candidate
from src.strategies.cross_sectional import CrossSectionParams

DATA = Path("data")
wide = pd.read_parquet(DATA / "portfolio_daily_closes.parquet")

# Portfolio-class rules (crypto alt basket): relaxed std/DD vs defaults,
# PF/window/net gates kept meaningful.
RULES = PromotionRules(
    min_pf_median=1.05,
    min_profitable_window_share=0.5,
    max_drawdown_median_pct=-60.0,
    min_net_median_pct=0.0,
    min_trades_total=100,
    max_net_std_pct=80.0,
)


def make_genome(sid, params):
    from src.strategy_genome import StrategyGenome

    return StrategyGenome(
        strategy_id=sid,
        version="1.0",
        market="crypto",
        timeframes=("1D",),
        features=("momentum",),
        indicators=("pct_change",),
        ml_model="none",
        regime_filters=("none",),
        entry_logic={"type": "cross_sectional_topk"},
        exit_logic={"rebalance_days": params.get("rebalance_days", 7)},
        risk_profile="portfolio",
        position_sizing={"equal_weight": True},
        portfolio_constraints={"top_k": params.get("top_k")},
        parameters=dict(params),
    )


VARIANTS = {
    "xs_lb14_k2": dict(lookback_days=14, top_k=2),
    "xs_lb7_k2": dict(lookback_days=7, top_k=2),
    "xs_lb28_k2": dict(lookback_days=28, top_k=2),
    "xs_lb14_k1": dict(lookback_days=14, top_k=1),
}

pipe = ChampionPipeline(
    registry=__import__("src.strategy_bank", fromlist=["StrategyRegistry"]).StrategyRegistry(),
    rules=RULES,
    store_path=DATA / "champions" / "state.json",
)

for sid, params in VARIANTS.items():
    pipe.submit_candidate(
        __import__("src.champion.evaluation_pipeline", fromlist=["CandidateSpec"]).CandidateSpec(
            sid,
            lambda p=params: CrossSectionParams(**p),
            params=dict(params),
        ),
        make_genome(sid, params),
    )

evals = pipe.evaluate_all(
    wide,
    evaluate_fn=evaluate_portfolio_candidate,
    test_days=180,
)

print(f"{'candidate':<12} {'net_med%':>9} {'pf_med':>7} {'win_share':>10} "
      f"{'dd_med%':>8} {'trades':>7} {'rules':>6}")
print("-" * 66)
for sid, r in evals.items():
    m = r["metrics"]
    print(f"{sid:<12} {m['net_median_pct']:>9.2f} {m['pf_median']:>7.3f} "
          f"{m['profitable_window_share']:>10.2f} {m['maxdd_median_pct']:>8.1f} "
          f"{m['trades']:>7} {str(r['rules_passed']):>6}")

dec = pipe.decide_promotion(evals)
pipe.save()

print("\nPROMOTION:", dec)
print("CHAMPION:", pipe.current_champion_id())
print("history events:", [h.event for h in pipe.history])

# feedback-style mutation seeds for next generation
bounds = MutationBounds(bounds={
    "lookback_days": (5, 35, 3),
})
champ_params = pipe.specs[pipe.current_champion_id()].params if pipe.current_champion_id() else {}
print("\nnext-gen mutation candidates:", suggest := __import__("src.champion.feedback", fromlist=["suggest_mutations"]).suggest_mutations(champ_params, bounds))
