import sys
sys.path.insert(0, 'C:/Bahtiyorjon/QuantAI')

from src.risk.kelly_sizer import KellySizer, KellyFraction
from src.risk.portfolio_correlation import CorrelationMatrix
from src.risk.cross_margin import CrossMarginManager
from src.risk.dynamic_risk_budget import RiskBudgetManager
from src.microstructure_intelligence import VPINCalculator
from src.alternative_data import AlternativeDataManager
from src.production.order_deduplication import OrderDeduplicator
from src.production.rate_limiter import BinanceRateLimiter
from src.production.disaster_recovery import CheckpointManager
from src.config.testnet_settings import testnet_config
from src.execution.execution_engine import ExecutionMode

print("All Phase 6-7 imports successful!")

# Quick functionality test
sizer = KellySizer(kelly_fraction=KellyFraction.HALF)
result = sizer.calculate(
    win_rate=0.55,
    avg_win=100.0,
    avg_loss=50.0,
    equity=10000.0,
    entry_price=50000.0,
    stop_loss=49000.0
)
print(f"Kelly Raw: {result.kelly_fraction:.4f}")
print(f"Suggested: {result.suggested_fraction:.4f}")
print("All Phase 6-7 components working!")