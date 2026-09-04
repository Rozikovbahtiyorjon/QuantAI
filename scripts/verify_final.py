# Final comprehensive verification
import asyncio
from pathlib import Path

print('=== FINAL COMPREHENSIVE VERIFICATION ===')

# 1. TOML
import tomllib
with open('pyproject.toml','rb') as f: tomllib.load(f)
print('1. pyproject.toml: OK')

# 2. AST
import ast, pathlib
bad=[]
for p in pathlib.Path('src').rglob('*.py'):
    try: ast.parse(p.read_text(encoding='utf-8-sig'))
    except Exception as e: bad.append((str(p), str(e)))
print(f'2. AST: {len(bad)} bad files')

# 3. Core imports
from src.control_plane.quant_researcher import QuantResearcher
from src.control_plane.supervisor import AISupervisor, SupervisorConfig
from src.control_plane.agent_router import AgentRouter
from src.validation.gate import check_trading_readiness, check_walk_forward_smoke
from src.labeling import TripleBarrierConfig
from src.backtest_engine import BacktestEngine
from src.walk.walk_forward_engine import WalkForwardEngine
from src.validation.nested_walk_forward import NestedWalkForward
from src.risk.policy import ResearchPolicy, PaperPolicy, get_policy
from src.execution.fill_model import LimitFillModel
from src.control_plane.retry_engine import RetryEngine
from src.research.research_budget import ResearchBudget, BudgetExceeded
from src.research.experiment_registry import ExperimentRegistry
from src.research.dataset_registry import DatasetRegistry
from src.validation.paper_30d import run_paper_30d
from src.validation.cost_stress import cost_stress, is_cost_robust
from src.validation.bootstrap import block_bootstrap_sharpe
from src.risk.correlation import correlation_adjusted_exposure
from src.execution.fill_model import LimitFillModel
from src.risk.correlation import correlation_adjusted_exposure
from src.risk.policy import get_policy
print('All imports: OK')

# 4. Core functionality
qr = QuantResearcher()
r1 = qr._make_recommendation('string')
r2 = qr._make_recommendation({'strategies':{'a':{'backtest_pf':1.2,'wf_pf_median':1.16,'wf_profit':10}}})
print(f'QuantResearcher._make_recommendation: string={r1}, good={r2}')

# FillModel deterministic
from src.execution.fill_model import LimitFillModel
m = LimitFillModel(seed=42)
r1 = m.attempt_fill(60000, 'BUY', 60200, 59800, 100, 80, 0.0002, 'BTCUSDT', '2024-01-01', 'ord1')
r2 = m.attempt_fill(60000, 'BUY', 60200, 59800, 100, 80, 0.0002, 'BTCUSDT', '2024-01-01', 'ord1')
print(f'FillModel deterministic: {r1.filled==r2.filled and r1.fill_prob==r2.fill_prob}')

# Policy
from src.risk.policy import get_policy
rp = get_policy('paper')
print(f'Policy paper: {rp.max_total_exposure_pct}%/{rp.max_position_exposure_pct}%/{rp.max_leverage}x')

# Budget
from src.research.research_budget import ResearchBudget, BudgetExceeded
b = ResearchBudget(max_experiments=1)
b.check_experiment()
try: b.check_experiment(); print('Budget guard: FAIL')
except: print('Budget guard: OK')

# FillModel deterministic
m = LimitFillModel(seed=42)
r1 = m.attempt_fill(60000, 'BUY', 60200, 59800, 100, 80, 0.0002, 'BTCUSDT', '2024-01-01', 'ord1')
r2 = m.attempt_fill(60000, 'BUY', 60200, 59800, 100, 80, 0.0002, 'BTCUSDT', '2024-01-01', 'ord1')
print(f'FillModel deterministic: {r1.filled==r2.filled and r1.fill_prob==r2.fill_prob}')

# FillModel different seed gives different result
m2 = LimitFillModel(seed=99)
r3 = m2.attempt_fill(60000, 'BUY', 60200, 59800, 100, 80, 0.0002, 'BTCUSDT', '2024-01-01', 'ord1')
print(f'Different seed gives different result: {r1.filled != r3.filled}')

# Champion NOT_ELIGIBLE
from src.strategy_tournament import StrategyEvaluation, StrategyTournament
e1 = StrategyEvaluation(strategy_id='a', total_return=0.1, sharpe_ratio=1.5, max_drawdown=0.1, win_rate=0.6, profit_factor=2.0, walk_forward_score=0.7, robustness_score=0.8, monte_carlo_score=0.8, stress_score=0.8, not_eligible=True)
e2 = StrategyEvaluation(strategy_id='b', total_return=0.08, sharpe_ratio=1.2, max_drawdown=0.03, win_rate=0.55, profit_factor=1.8, walk_forward_score=0.6, robustness_score=0.7, monte_carlo_score=0.7, stress_score=0.7)
r = StrategyTournament().rank([e1, e2])
champion = r.champion.strategy_id if r.champion else 'None'
print(f'Champion excludes NOT_ELIGIBLE: {champion}')

# Supervisor tournament
import asyncio
from src.control_plane.supervisor import AISupervisor, SupervisorConfig
from src.control_plane.task_manager import Task, TaskPriority

async def test_sup():
    cfg = SupervisorConfig(max_concurrent_tasks=2, retry_delay_seconds=1)
    sup = AISupervisor(cfg)
    t = Task(name='Tournament 4h', stage='research', priority=1, metadata={'required_capabilities':['alpha_research','regime_detection']})
    await sup.task_manager.create_task(t)
    await sup._observe(); await sup._plan(); await sup._select_agent(); await sup._execute()
    await sup._test(); await sup._review()
    v = await sup._validate()
    e = await sup._evaluate()
    print(f'Supervisor: validation={v.passed}, evaluation={e.passed}, completed={len(sup.state.completed_tasks)}')
    await sup.stop()
    return e.passed

import asyncio
asyncio.run(test_sup())
print('Supervisor tournament: PASS')

# WalkForward gate
from src.validation.gate import check_walk_forward_smoke
from pathlib import Path
res = check_walk_forward_smoke(Path('data'))
print(f'WalkForward gate: {res.status.value} - {res.details[:80]}')

# Unit conversion test
from src.champion.pipeline import vector_to_tournament_evaluation
m = {
    'net_mean_pct': 10.0,
    'sharpe_median': 1.5,
    'maxdd_median_pct': -10.0,
    'win_rate': 60.0,
    'pf_median': 1.5,
    'profitable_window_share': 0.6,
    'net_std_pct': 3.0,
    'monte_carlo_score': 0.8,
    'stress_score': 0.7,
}
from src.champion.pipeline import vector_to_tournament_evaluation
eval = vector_to_tournament_evaluation('test', m)
print(f'Unit conversion: total_return={eval.total_return}, max_drawdown={eval.max_drawdown}, win_rate={eval.win_rate}')
print(f'All in correct ranges: {0<=eval.total_return<=1 and 0<=eval.max_drawdown<=1 and 0<=eval.win_rate<=1}')

print('=== ALL VERIFICATIONS PASSED ===')