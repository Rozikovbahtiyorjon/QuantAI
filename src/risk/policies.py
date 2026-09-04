"""
DEPRECATED — canonical is src/risk/policy.py (Audit: duplicate canonical)

This file kept for backward compat only. Use:
  from src.risk.policy import ResearchPolicy, PaperPolicy, TestnetPolicy, ProductionPolicy, get_policy
Will be removed in 6.0.
"""

from src.risk.policy import (  # noqa: F401
    BasePolicy,
    ResearchPolicy,
    PaperPolicy,
    TestnetPolicy,
    ProductionPolicy,
    POLICIES,
    get_policy,
)
