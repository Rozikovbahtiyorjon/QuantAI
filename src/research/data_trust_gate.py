"""
Data/Result Trust Gate

Ensures no function works on data/results that system overly trusts
without independent verification.

- Data: must pass DataGates (7 checks) before any research
- Results: must be from verified evidence (L2+), not just self-reported
"""

from __future__ import annotations

import pandas as pd
from typing import Any


def enforce_data_trust(df: pd.DataFrame, timeframe: str = "1h", source: str = "unknown") -> None:
    """
    Ensure DataFrame has passed DataGates before use.
    Raises ValueError if data is not trusted (would be working on unvalidated data).
    """
    try:
        from src.data.data_gates import DataGates
        gates = DataGates()
        gates.validate(df, timeframe=timeframe)
    except Exception as e:
        raise ValueError(f"Data trust failed for {source}: {e} — cannot work on unvalidated data (would overly trust)") from e


def enforce_result_trust(result: Any, required_trust: int = 2, source: str = "unknown") -> None:
    """
    Ensure result is from verified evidence, not just self-reported.
    required_trust: 2 = EXECUTION_VERIFIED, 3 = INDEPENDENTLY_VALIDATED
    """
    # Check if result has trust_level
    trust = 0
    if isinstance(result, dict):
        trust = int(result.get("trust_level", result.get("trust", 0)) or 0)
        verified = result.get("verification_status") == "VERIFIED" or result.get("verified") is True
    elif hasattr(result, "trust_level"):
        trust = int(getattr(result, "trust_level", 0) or 0)
        verified = getattr(result, "verification_status", "") == "VERIFIED"
    else:
        trust = 0
        verified = False

    if trust < required_trust or not verified:
        raise ValueError(
            f"Result trust insufficient for {source}: trust {trust} < required {required_trust} or not verified "
            f"— cannot work on results that system overly trusts (need independent verification)"
        )
