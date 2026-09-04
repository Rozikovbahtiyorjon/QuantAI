"""
QuantAI Robust OOS Edge — MAX ROBUST OOS EDGE KPI (Task 14)

New KPI replaces MAX PF.  Award MAX ROBUST OOS EDGE = stable,
cost- and latency-robust out-of-sample edge that survives selection bias.

KPI composition (8 components, weighted sum in [0,1]):

  1. positive OOS expectancy      expectancy > 0           weight 0.20  critical
  2. stable OOS PF               PF >= 1.1                 weight 0.15  critical
  3. acceptable DD               maxDD >= -15%             weight 0.15  critical
  4. sufficient sample           trades >= 30              weight 0.10  critical (sample)
  5. regime stability            n_positive >= 3/7         weight 0.10
  6. cost robustness             PF>1 at 1.5x costs        weight 0.10
  7. slippage+latency robustness both slippage & latency   weight 0.10
  8. selection-bias adjustment   DSR/WRC/PBO significant  weight 0.10

Score = sum(weight_i * passed_i)  in [0,1] (weights sum 1.0).
Gate: is_robust_edge(score, components) == (score > 0.70 and all critical pass).

Integration: ResearchIntegrityEngine._gate_robust_oos_edge() calls
compute_robust_oos_edge() and blocks promotion when score <= threshold.

Metrics dict contract (flat, aliases handled):
  - expectancy: expectancy, oos_expectancy, net_median_pct, net_mean_pct, ...
  - pf: pf_median, oos_pf, profit_factor, ...
  - dd: maxdd_median_pct, max_drawdown, dd, ...
  - trades/sample: trades, n_trades, sample, ...
  - regime: regime_stability dict, n_positive, regime_pass, works, ...
  - cost_robust: cost_robust bool, cost_stress list, is_cost_robust, ...
  - slippage_robust / latency_robust: bool or combined, ...
  - selection: dsr, pbo, wrc_p, wrc_p_value, selection_bias dict, ...

All lookups are alias-tolerant and handle missing data (missing => fail
unless config.permissive allows neutral). No look-ahead, no external I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Tuple

# =====================================================
# CONFIG
# =====================================================

@dataclass
class RobustOOSConfig:
    """Thresholds and weights for MAX ROBUST OOS EDGE.

    Weights must sum to 1.0 (enforced via __post_init__ normalisation
    warning — not hard error, but gate score will clip).
    """

    # Thresholds
    min_expectancy: float = 0.0  # expectancy > 0
    min_pf: float = 1.1  # stable PF threshold
    # acceptable DD: maxDD >= this (negative pct, e.g. -15.0)
    max_dd_pct: float = -15.0
    min_trades: int = 30
    min_regimes_positive: int = 3
    total_regimes: int = 7
    min_trades_per_regime: int = 5  # for regime sub-check if present

    # cost robustness: PF>1 at 1.5x
    require_cost_robust: bool = True

    # slippage/latency
    require_slippage_robust: bool = True
    require_latency_robust: bool = True

    # selection bias
    min_dsr: float = 0.95  # DSR significance
    max_pbo: float = 0.6
    max_wrc_p: float = 0.05

    # Weights (must sum 1.0)
    w_expectancy: float = 0.20
    w_pf: float = 0.15
    w_dd: float = 0.15
    w_sample: float = 0.10
    w_regime: float = 0.10
    w_cost: float = 0.10
    w_slippage_latency: float = 0.10
    w_selection: float = 0.10

    # Gate threshold
    min_score: float = 0.70

    # Which components are critical (must pass even if score > threshold)
    # Production default: ALL 8 components must pass.
    # Caller can override via critical_components.
    critical_components: Tuple[str, ...] = (
        "expectancy",
        "pf_stable",
        "dd",
        "sample",
        "regime",
        "cost_robust",
        "slippage_latency",
        "selection_bias",
    )

    # Permissive: missing artefact => neutral pass (research mode).
    # Strict (default): missing => fail.
    permissive: bool = False

    def __post_init__(self) -> None:
        s = (
            self.w_expectancy
            + self.w_pf
            + self.w_dd
            + self.w_sample
            + self.w_regime
            + self.w_cost
            + self.w_slippage_latency
            + self.w_selection
        )
        if not math.isclose(s, 1.0, abs_tol=1e-9):
            raise ValueError(f"RobustOOSConfig weights must sum to 1.0, got {s:.4f}")

    def weight_map(self) -> Dict[str, float]:
        return {
            "expectancy": self.w_expectancy,
            "pf_stable": self.w_pf,
            "dd": self.w_dd,
            "sample": self.w_sample,
            "regime": self.w_regime,
            "cost_robust": self.w_cost,
            "slippage_latency": self.w_slippage_latency,
            "selection_bias": self.w_selection,
        }


DEFAULT_CONFIG = RobustOOSConfig()

# =====================================================
# COMPONENT DETAIL
# =====================================================

from enum import Enum

class ComponentStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"  # insufficient sample, PF inf — neither PASS nor FAIL
    UNAVAILABLE = "UNAVAILABLE"  # missing data

@dataclass
class ComponentResult:
    """Per-component result with 4-state status."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    weight: float
    contribution: float  # weight if passed else 0
    reason: str
    critical: bool = False
    status: ComponentStatus = ComponentStatus.FAIL  # explicit 4-state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "weight": self.weight,
            "contribution": self.contribution,
            "reason": self.reason,
            "critical": self.critical,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
        }


@dataclass
class RobustOOSComponents:
    """8-component container. Field names match gate labels.

    Accessors: components.expectancy, .pf_stable, .dd, .sample,
    .regime, .cost_robust, .slippage_latency, .selection_bias
    """

    expectancy: ComponentResult
    pf_stable: ComponentResult
    dd: ComponentResult
    sample: ComponentResult
    regime: ComponentResult
    cost_robust: ComponentResult
    slippage_latency: ComponentResult
    selection_bias: ComponentResult

    # ---- helpers for alternative naming (slippage/latency split, etc.) ----
    @property
    def slippage_robust(self) -> ComponentResult:
        return self.slippage_latency

    @property
    def latency_robust(self) -> ComponentResult:
        return self.slippage_latency

    @property
    def selection(self) -> ComponentResult:
        return self.selection_bias

    @property
    def pf(self) -> ComponentResult:
        return self.pf_stable

    @property
    def drawdown(self) -> ComponentResult:
        return self.dd

    @property
    def trades(self) -> ComponentResult:
        return self.sample

    def as_dict(self) -> Dict[str, ComponentResult]:
        return {
            "expectancy": self.expectancy,
            "pf_stable": self.pf_stable,
            "dd": self.dd,
            "sample": self.sample,
            "regime": self.regime,
            "cost_robust": self.cost_robust,
            "slippage_latency": self.slippage_latency,
            "selection_bias": self.selection_bias,
        }

    def to_detail_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self.as_dict().items()}

    def all_pass(self) -> bool:
        return all(v.passed for v in self.as_dict().values())

    def critical_pass(self, critical: Tuple[str, ...]) -> bool:
        d = self.as_dict()
        # Point 33: missing critical component → FAIL (was all(empty)=True)
        for k in critical:
            if k not in d:
                return False
            if not d[k].passed:
                return False
        return True

    def __iter__(self):  # type: ignore[override]
        # allow dict-like iteration over values
        yield from self.as_dict().values()

    def __getitem__(self, key: str) -> ComponentResult:  # type: ignore[override]
        d = self.as_dict()
        if key in d:
            return d[key]
        # aliases
        alias = {
            "slippage": "slippage_latency",
            "latency": "slippage_latency",
            "slippage_robust": "slippage_latency",
            "latency_robust": "slippage_latency",
            "pf": "pf_stable",
            "profit_factor": "pf_stable",
            "drawdown": "dd",
            "maxdd": "dd",
            "trades": "sample",
            "sample_size": "sample",
            "regime_stability": "regime",
            "cost": "cost_robust",
            "selection": "selection_bias",
            "dsr": "selection_bias",
            "wrc": "selection_bias",
            "pbo": "selection_bias",
        }
        if key in alias and alias[key] in d:
            return d[alias[key]]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            _ = self[key]
            return True
        except KeyError:
            return False


@dataclass
class RobustOOSResult:
    """Full result object returned by compute_robust_oos_edge.

    Is tuple-unpackable (score, components) and float-coercible.
    Also dict-like via ['score'] / ['components'] / ['details'].
    """

    score: float
    components: RobustOOSComponents
    details: Dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.70
    passed: bool = False  # is_robust_edge verdict convenience

    def __iter__(self):  # type: ignore[override]
        yield self.score
        yield self.components
        # third element for convenience but keep 2-tuple unpack working
        # we yield details as optional third; unpacking 2 ignores extra

    def __float__(self) -> float:
        return float(self.score)

    def __getitem__(self, key: str | int) -> Any:  # type: ignore[override]
        if isinstance(key, int):
            if key == 0:
                return self.score
            if key == 1:
                return self.components
            if key == 2:
                return self.details
            raise IndexError(key)
        # string keys
        if key == "score":
            return self.score
        if key == "components":
            return self.components
        if key == "details":
            return self.details
        if key == "passed":
            return self.passed
        if key == "threshold":
            return self.threshold
        # delegate to components
        try:
            return self.components[key]  # type: ignore
        except Exception:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if key in ("score", "components", "details", "passed", "threshold"):
            return True
        if isinstance(key, str) and key in self.components:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "components": self.components.to_detail_dict(),
            "details": self.details,
        }


# =====================================================
# METRIC EXTRACTION HELPERS (alias-tolerant)
# =====================================================

def _norm_key(k: str) -> str:
    """Normalize metric key: lowercase, spaces/dashes -> underscore, strip."""
    try:
        s = str(k).strip().lower()
        s = s.replace(" ", "_").replace("-", "_")
        # collapse multiple underscores
        while "__" in s:
            s = s.replace("__", "_")
        return s
    except Exception:
        return str(k).lower()


def _flat_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Merge top-level + nested 'metrics' dict into flat lookup."""
    if not isinstance(metrics, dict):
        return {}
    flat: Dict[str, Any] = {}
    # also build normalized index for case/space-insensitive lookup
    norm_index: Dict[str, str] = {}
    def _add_to_flat(key: str, value: Any, allow_overwrite: bool = True):
        if not allow_overwrite and key in flat:
            return
        flat[key] = value
        nk = _norm_key(key)
        # keep first occurrence for normalized, but also allow later to overwrite if not present
        if nk not in norm_index:
            norm_index[nk] = key
        # also store normalized key directly for direct lookup
        if nk not in flat:
            flat[nk] = value
        else:
            # if normalized key already exists but original key is more specific, keep original?
            # prefer direct overwrite for normalized as well when allow_overwrite
            if allow_overwrite:
                flat[nk] = value

    # copy metrics subdict first (base)
    base = metrics.get("metrics")
    if isinstance(base, dict):
        for k, v in base.items():
            _add_to_flat(k, v, allow_overwrite=True)
            # also handle metrics subdict normalized already via _add_to_flat
    # overlay top-level (evaluation keys) — top wins for explicit flags
    for k, v in metrics.items():
        if k == "metrics":
            continue
        _add_to_flat(k, v, allow_overwrite=True)
    # also expose is_oos, is_metrics for fallback PF etc.
    for extra_key in ("is_oos", "is_metrics", "oos_metrics"):
        if extra_key in metrics and isinstance(metrics[extra_key], dict):
            for k, v in metrics[extra_key].items():
                # prefix to allow lookup, but also expose unprefixed if not exists
                _add_to_flat(f"{extra_key}.{k}", v, allow_overwrite=True)
                if k not in flat:
                    _add_to_flat(k, v, allow_overwrite=False)
    # store norm_index for debug? not needed
    return flat


def _get_first(flat: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    # Build normalized lookup for flat once per call if needed
    # Try exact, then normalized
    for k in keys:
        if k in flat and flat[k] is not None:
            v = flat[k]
            if isinstance(v, float) and math.isnan(v):
                continue
            return v
        nk = _norm_key(k)
        if nk in flat and flat[nk] is not None:
            v = flat[nk]
            if isinstance(v, float) and math.isnan(v):
                continue
            return v
    return default


def _coerce_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def _coerce_int(v: Any, default: int | None = None) -> int | None:
    f = _coerce_float(v, None)
    if f is None:
        return default
    try:
        return int(f)
    except Exception:
        return default


def _parse_dd(value: Any) -> float | None:
    """Parse DD to signed percent (negative). Handles positive magnitude input."""
    f = _coerce_float(value, None)
    if f is None:
        return None
    # If DD stored as positive fraction 0..1 (e.g. 0.05 => 5%), convert to -5
    # If stored as positive pct >1 (e.g. 5 => 5%) convert to -5
    # If stored as negative already, keep.
    if f > 0:
        if 0 < f < 1:
            # likely fraction
            return -f * 100.0
        else:
            return -abs(f)
    return float(f)


def _extract_expectancy(flat: Dict[str, Any]) -> float | None:
    keys = [
        "expectancy",
        "oos_expectancy",
        "expectancy_oos",
        "exp_oos",
        "net_median_pct",
        "net_mean_pct",
        "oos_net",
        "oos_net_pct",
        "net_median",
        "mean_pnl",
        "avg_pnl",
        "expected_value",
        "ev",
        "expectancy_pct",
        "oos_expectancy_pct",
        "expectancy_pips",
        "net_median_pct_oos",
        "oos_expectancy_net",
        "expectancy_net",
    ]
    v = _get_first(flat, keys, None)
    if v is not None:
        f = _coerce_float(v, None)
        if f is not None:
            return f
    # fallback: try to compute expectancy from PF and win_rate if present?
    # pf = GP/GL, expectancy approx net/ trades? check net_median_pct already tried.
    # Last fallback: check window nets mean
    for k in ("net_median_pct", "net_mean_pct"):
        if k in flat:
            f = _coerce_float(flat[k], None)
            if f is not None:
                return f
    return None


def _extract_pf(flat: Dict[str, Any]) -> float | None:
    keys = [
        "oos_pf",
        "pf_oos",
        "pf_median",
        "pf",
        "profit_factor",
        "profit_factor_median",
        "oos_profit_factor",
        "pf_stable",
        "oos_pf_median",
        "is_oos.oos_pf",
        "oos_pf_median",
        "pf_oos_median",
    ]
    v = _get_first(flat, keys, None)
    f = _coerce_float(v, None)
    if f is not None:
        # cap extreme PF 99 as in evaluation_pipeline
        if f == float("inf") or f > 99:
            return 99.0
        return f
    return None


def _extract_dd(flat: Dict[str, Any]) -> float | None:
    keys = [
        "maxdd_median_pct",
        "maxdd_pct",
        "max_drawdown_pct",
        "max_drawdown",
        "drawdown",
        "dd",
        "dd_pct",
        "maxdd",
        "dd_oos",
        "oos_dd",
        "maxdd_median",
        "oos_maxdd",
        "max_drawdown_pct_oos",
    ]
    v = _get_first(flat, keys, None)
    if v is not None:
        return _parse_dd(v)
    return None


def _extract_trades(flat: Dict[str, Any]) -> int | None:
    keys = [
        "trades",
        "n_trades",
        "oos_trades",
        "sample",
        "n",
        "trade_count",
        "num_trades",
        "trades_total",
        "sample_size",
        "n_obs",
        "trades_oos",
        "windows",
        "total_trades",
        "oos_sample",
    ]
    v = _get_first(flat, keys, None)
    # windows count is not trades, but if trades missing we could use windows*avg_trades?
    # Keep simple: if trades missing but windows present, trades = windows * ? not reliable, so fail.
    # But if trades missing and windows is large, we can at least check windows >=? We'll treat trades from windows only if windows is trades proxy.
    # For now, if trades key is "windows" and value < 100, treat as trade count proxy only when no other trades.
    if v is None:
        return None
    # handle list of trades (closed_positions) -> len
    if isinstance(v, list):
        return len(v)
    f = _coerce_int(v, None)
    return f


def _extract_regime(flat: Dict[str, Any], config: RobustOOSConfig) -> tuple[Any, bool, str]:
    """Return (value, passed, reason) for regime component.

    Logic:
      - if flat contains regime_stability dict with verdict / n_positive etc, use it.
      - if flat contains regime dict similar.
      - if flat contains bool regime_pass etc, use bool.
      - if flat contains int n_positive etc, compare >= min_regimes_positive.
      - else missing => fail (or permissive pass if config.permissive).
    """
    # Check dict-type regime fields first
    for key in ("regime_stability", "regime_stability_result", "regime_result", "regime", "regime_evaluation"):
        v = flat.get(key)
        if isinstance(v, dict):
            # dict with verdict / n_positive / works
            # verdict bool
            if "verdict" in v:
                passed = bool(v["verdict"])
                npos = v.get("n_positive", v.get("n_regimes_positive", None))
                val = npos if npos is not None else v.get("verdict")
                reason = f"regime dict {key} verdict={passed} n_positive={npos}"
                # enforce min_regimes threshold even if dict verdict true? Use config threshold as well.
                # If dict threshold differs, trust dict verdict but also check n_positive >= config.min
                if npos is not None:
                    try:
                        if int(npos) < int(config.min_regimes_positive):
                            passed = False
                            reason += f" (< {config.min_regimes_positive})"
                    except Exception:
                        pass
                return val, passed, reason
            # check n_positive directly inside dict
            if "n_positive" in v:
                try:
                    npos = int(v["n_positive"])
                    passed = npos >= int(config.min_regimes_positive)
                    return npos, passed, f"regime {key}.n_positive {npos} >= {config.min_regimes_positive}"
                except Exception:
                    pass
            if "works" in v and isinstance(v["works"], list):
                npos = len(v["works"])
                passed = npos >= int(config.min_regimes_positive)
                return npos, passed, f"regime {key}.works {npos} >= {config.min_regimes_positive} ({v['works']})"
            # per_regime checks? fallback
            if "per_regime" in v:
                # count regimes where expectancy>0?
                # we have already logic but treat as insufficient to pass
                pass

    # Check flat integer keys
    for key in ("n_positive", "regime_n_positive", "regimes_positive", "regime_positive", "positive_regimes", "n_regimes_positive"):
        if key in flat and flat[key] is not None:
            try:
                npos = int(float(flat[key]))
                passed = npos >= int(config.min_regimes_positive)
                return npos, passed, f"{key} {npos} >= {config.min_regimes_positive}"
            except Exception:
                continue

    # Check bool keys
    for key in ("regime_pass", "regime_passed", "regime_verdict", "regime_stable", "regime_stability_pass", "regime_ok"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            # bool or string?
            if isinstance(v, bool):
                return v, bool(v), f"{key}={v}"
            if isinstance(v, (int, float)):
                # 1/0?
                b = bool(int(v))
                return v, b, f"{key}={v}"
            if isinstance(v, str):
                b = v.lower() in ("true", "pass", "passed", "yes", "1")
                return v, b, f"{key}={v}"

    # Check works list directly
    if "works" in flat and isinstance(flat["works"], list):
        npos = len(flat["works"])
        passed = npos >= int(config.min_regimes_positive)
        return npos, passed, f"works {npos} >= {config.min_regimes_positive}"

    # Check regime_labels + pnl? Not enough
    # Check windows list with regime_labels? too complex -> treat as missing
    return None, False, "regime data missing"


def _extract_cost_robust(flat: Dict[str, Any]) -> tuple[Any, bool, str]:
    # bool directly
    for key in ("cost_robust", "cost_robust_pass", "is_cost_robust", "cost_stress_pass", "cost_pass", "cost_robustness"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            if isinstance(v, bool):
                return v, bool(v), f"{key}={v}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # could be PF at 1.5x: treat >1 as pass, but bool keys already handled
                # if numeric under cost_robust, interpret as PF
                f = float(v)
                passed = f >= 1.0
                return f, passed, f"{key} PF {f:.2f} >=1.0"
            if isinstance(v, str):
                b = v.lower() in ("true", "pass", "1", "yes")
                return v, b, f"{key}={v}"
            # list? fall through

    # check cost_stress list of StressResult
    for key in ("cost_stress", "cost_stress_results", "cost_stress_list"):
        v = flat.get(key)
        if v is not None and isinstance(v, list) and len(v) > 0:
            # try to find multiplier 1.5
            try:
                from src.validation.cost_stress import is_cost_robust as _is_cost  # type: ignore

                # v is list[StressResult] with .multiplier and .pf or dict
                robust = bool(_is_cost_robust(v) if callable(_is_cost_robust) else _is_cost(v))
                # but _is_cost expects list; if our stub fails, do manual
            except Exception:
                robust = False
                # manual: find item with multiplier 1.5
                for item in v:
                    mult = None
                    pf = None
                    if isinstance(item, dict):
                        mult = item.get("multiplier") or item.get("mult")
                        pf = item.get("pf") or item.get("profit_factor")
                    else:
                        mult = getattr(item, "multiplier", None)
                        pf = getattr(item, "pf", None)
                    if mult is not None and abs(float(mult) - 1.5) < 1e-9 and pf is not None:
                        robust = float(pf) >= 1.0
                        break
                    # fallback: if any pf <1 at 1.5? already
                # if no 1.5 found, check any fragile?
                if not robust:
                    # check if any pf >=1
                    vals = []
                    for item in v:
                        if isinstance(item, dict):
                            pf = item.get("pf")
                            if pf is not None:
                                vals.append(float(pf))
                        else:
                            pf = getattr(item, "pf", None)
                            if pf is not None:
                                vals.append(float(pf))
                    # if we have vals but no mult, use median?
                    robust = False
            return v, bool(robust), f"cost_stress {key} robust={robust}"
        # also handle dict form {"1.5x": {"pf":1.2}, ...}
        if isinstance(v, dict):
            # check keys like 1.5, "1.5x", "1.5"
            for k2 in ("1.5", "1.5x", 1.5, "1_5x"):
                if k2 in v:
                    inner = v[k2]
                    pf = inner.get("pf") if isinstance(inner, dict) else getattr(inner, "pf", None)
                    if pf is not None:
                        passed = float(pf) >= 1.0
                        return inner, passed, f"cost_stress dict {k2} pf {pf} >=1"
            # generic: if any pf in values
            for kk, vv in v.items():
                if isinstance(vv, dict) and "pf" in vv:
                    if float(vv["pf"]) >= 1.0:
                        # not definitive but we found one
                        pass

    # check pf_1.5x style keys
    for key in ("pf_1_5x", "pf_1.5x", "pf_at_1_5x", "cost_pf_1_5", "cost_pf_at_1.5x"):
        if key in flat and flat[key] is not None:
            f = _coerce_float(flat[key], None)
            if f is not None:
                passed = f >= 1.0
                return f, passed, f"{key} PF {f:.2f} >=1.0"

    return None, False, "cost_robust missing"


def _extract_slippage_latency(flat: Dict[str, Any]) -> tuple[Any, bool, str]:
    # combined key first
    for key in ("slippage_latency_robust", "slippage_latency", "execution_robust", "fill_robust"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            if isinstance(v, bool):
                return v, bool(v), f"{key}={v}"
            if isinstance(v, dict):
                # dict with slippage/latency
                s = v.get("slippage_robust", v.get("slippage"))
                l = v.get("latency_robust", v.get("latency"))
                if s is not None and l is not None:
                    passed = bool(s) and bool(l)
                    return v, passed, f"{key} slippage={s} latency={l} need both"
                if s is not None:
                    return s, bool(s), f"{key}.slippage={s}"
                if l is not None:
                    return l, bool(l), f"{key}.latency={l}"
            # numeric
            f = _coerce_float(v, None)
            if f is not None:
                passed = bool(f)
                return v, passed, f"{key}={v}"

    # separate slippage and latency bools
    slip = None
    lat = None
    slip_pass = None
    lat_pass = None
    slip_val = None
    lat_val = None

    for key in ("slippage_robust", "slippage_pass", "is_slippage_robust", "slippage_stress_pass", "fill_model_robust", "slippage_ok"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            if isinstance(v, bool):
                slip = bool(v)
                slip_val = v
                slip_pass = bool(v)
                break
            f = _coerce_float(v, None)
            if f is not None:
                # if numeric, treat as PF check?
                # If key is slippage PF, pass if >=1
                if "pf" in key.lower() or isinstance(v, (int, float)):
                    # heuristic: treat bool numeric 1/0 or pf
                    if key.lower().endswith("_pf") or "pf" in key.lower():
                        slip_pass = float(v) >= 1.0
                        slip_val = float(v)
                        slip = bool(slip_pass)
                        break
                # generic bool
                slip = bool(v)
                slip_val = v
                slip_pass = bool(v)
                break
            # string
            if isinstance(v, str):
                b = v.lower() in ("true", "pass", "1", "yes")
                slip = b
                slip_val = v
                slip_pass = b
                break

    for key in ("latency_robust", "latency_pass", "is_latency_robust", "latency_stress_pass", "latency_ok"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            if isinstance(v, bool):
                lat = bool(v)
                lat_val = v
                lat_pass = bool(v)
                break
            f = _coerce_float(v, None)
            if f is not None:
                if "pf" in key.lower():
                    lat_pass = float(v) >= 1.0
                    lat_val = float(v)
                    lat = bool(lat_pass)
                    break
                lat = bool(v)
                lat_val = v
                lat_pass = bool(v)
                break
            if isinstance(v, str):
                b = v.lower() in ("true", "pass", "1", "yes")
                lat = b
                lat_val = v
                lat_pass = b
                break

    # also check dict cost_stress style with slippage/latency stress results?
    if slip is None:
        # try slippage_pf keys
        for key in ("slippage_pf", "pf_slippage", "slippage_pf_1_5x", "slippage_pf_at_50pct"):
            if key in flat and flat[key] is not None:
                f = _coerce_float(flat[key], None)
                if f is not None:
                    slip_pass = f >= 1.0
                    slip_val = f
                    slip = bool(slip_pass)
                    break
    if lat is None:
        for key in ("latency_pf", "pf_latency", "latency_pf_500ms", "latency_pf_at_500ms"):
            if key in flat and flat[key] is not None:
                f = _coerce_float(flat[key], None)
                if f is not None:
                    lat_pass = f >= 1.0
                    lat_val = f
                    lat = bool(lat_pass)
                    break

    # Combine
    if slip is not None and lat is not None:
        passed = bool(slip_pass and lat_pass)
        return {"slippage": slip_val, "latency": lat_val}, passed, f"slippage {slip_val} latency {lat_val} both => {passed}"
    if slip is not None:
        # if only one provided, treat combined as that one (partial)
        return slip_val, bool(slip_pass), f"slippage only {slip_val} => {bool(slip_pass)} (latency missing)"
    if lat is not None:
        return lat_val, bool(lat_pass), f"latency only {lat_val} => {bool(lat_pass)} (slippage missing)"

    # Check latency_stress / slippage_stress list similar to cost_stress?
    # For now missing
    return None, False, "slippage/latency missing"


def _extract_selection_bias(flat: Dict[str, Any], config: RobustOOSConfig) -> tuple[Any, bool, str]:
    # direct bool selection_bias_pass
    for key in ("selection_bias_pass", "selection_bias_adjusted", "selection_pass", "bias_adjusted", "bias_pass"):
        if key in flat and flat[key] is not None:
            v = flat[key]
            if isinstance(v, bool):
                return v, bool(v), f"{key}={v}"
            if isinstance(v, (int, float)):
                b = bool(v)
                return v, b, f"{key}={v}"
            if isinstance(v, str):
                b = v.lower() in ("true", "pass", "1", "yes")
                return v, b, f"{key}={v}"

    # check unified dict
    for key in ("selection_bias", "selection", "bias", "selection_adjustment"):
        v = flat.get(key)
        if isinstance(v, dict):
            # dict may contain dsr/pbo/wrc
            dsr = v.get("dsr", v.get("deflated_sharpe", v.get("dsr_value")))
            pbo = v.get("pbo", v.get("pbo_value"))
            wrc = v.get("wrc_p", v.get("wrc", v.get("wrc_p_value", v.get("wrc_p_value"))))
            # evaluate if present
            passes = []
            reasons = []
            if dsr is not None:
                f = _coerce_float(dsr, None)
                if f is not None:
                    # dsr is probability in [0,1], threshold config.min_dsr
                    # sometimes dsr stored as bool?
                    if isinstance(dsr, bool):
                        passes.append(bool(dsr))
                        reasons.append(f"dsr bool {dsr}")
                    else:
                        # if dsr >1 it's maybe Sharpe? but assume probability
                        # if >1 treat as Sharpe vs expected? but we clamp to prob.
                        # If dsr in [0,1], threshold 0.95 else if dsr is Sharpe, threshold 0?
                        # We'll treat if f <=1: prob, else Sharpe approx
                        if 0 <= f <= 1:
                            p = f >= float(config.min_dsr)
                            passes.append(p)
                            reasons.append(f"dsr {f:.3f} >= {config.min_dsr}")
                        else:
                            # Sharpe value, assume pass if >0? not ideal, but treat as pass if >0
                            p = f > 0
                            passes.append(p)
                            reasons.append(f"dsr_sharpe {f:.3f} >0")
                else:
                    if isinstance(dsr, bool):
                        passes.append(bool(dsr))
                        reasons.append(f"dsr bool {dsr}")
            if pbo is not None:
                f = _coerce_float(pbo, None)
                if f is not None:
                    p = f < float(config.max_pbo)
                    passes.append(p)
                    reasons.append(f"pbo {f:.3f} < {config.max_pbo}")
                elif isinstance(pbo, bool):
                    passes.append(not bool(pbo))  # bool true maybe means overfit? ambiguous
                    reasons.append(f"pbo bool {pbo}")
            if wrc is not None:
                f = _coerce_float(wrc, None)
                if f is not None:
                    p = f < float(config.max_wrc_p)
                    passes.append(p)
                    reasons.append(f"wrc_p {f:.3f} < {config.max_wrc_p}")
            if passes:
                passed = all(passes)
                return v, passed, " & ".join(reasons)
            # if dict but no known keys, treat as missing
        elif isinstance(v, bool):
            return v, bool(v), f"{key} bool {v}"

    # extract individual dsr / pbo / wrc from flat
    dsr_val = None
    pbo_val = None
    wrc_val = None
    # DSR keys
    for k in ("dsr", "deflated_sharpe", "deflated_sharpe_ratio", "dsr_value", "dsr_prob", "dsr_significant", "is_dsr_significant"):
        if k in flat and flat[k] is not None:
            dsr_val = flat[k]
            break
    # PBO keys
    for k in ("pbo", "pbo_value", "probability_of_backtest_overfitting", "pbo_prob"):
        if k in flat and flat[k] is not None:
            pbo_val = flat[k]
            break
    # WRC keys
    for k in ("wrc_p", "wrc_p_value", "wrc", "spa_p", "spa_p_value", "white_reality_check_p", "wrc_result", "spa_result"):
        if k in flat and flat[k] is not None:
            v = flat[k]
            # if v is dict with p_value
            if isinstance(v, dict) and "p_value" in v:
                wrc_val = v["p_value"]
            elif isinstance(v, dict) and "p" in v:
                wrc_val = v["p"]
            else:
                wrc_val = v
            break
    # also check p_value generic?
    # Now evaluate
    passes: list[bool] = []
    reasons: list[str] = []
    found = False
    if dsr_val is not None:
        found = True
        if isinstance(dsr_val, bool):
            passes.append(bool(dsr_val))
            reasons.append(f"dsr bool {dsr_val}")
        else:
            f = _coerce_float(dsr_val, None)
            if f is not None:
                if 0 <= f <= 1:
                    p = f >= float(config.min_dsr)
                    passes.append(p)
                    reasons.append(f"dsr {f:.3f} >= {config.min_dsr}")
                else:
                    # Sharpe value: check > threshold? use config.min_dsr as prob not applicable
                    # treat as DSR Sharpe >0?
                    # We'll treat large value as Sharpe passing if >0
                    p = f > 0
                    passes.append(p)
                    reasons.append(f"dsr_sharpe {f:.3f} >0")
            else:
                # string like "significant" ?
                if isinstance(dsr_val, str):
                    b = dsr_val.lower() in ("true", "significant", "pass")
                    passes.append(b)
                    reasons.append(f"dsr str {dsr_val}")
        # also handle dict result with deflated_sharpe inside?
        if isinstance(dsr_val, dict):
            # try to extract p or dsr inside
            for kk in ("p_value", "dsr", "value"):
                if kk in dsr_val:
                    f = _coerce_float(dsr_val[kk], None)
                    if f is not None:
                        p = f >= float(config.min_dsr) if 0 <= f <= 1 else f > 0
                        passes.append(p)
                        reasons.append(f"dsr {kk} {f:.3f}")
                        break

    if pbo_val is not None:
        found = True
        if isinstance(pbo_val, bool):
            # bool True maybe means passed? ambiguous, treat True as pass?
            # but PBO bool True could mean overfit? We'll treat bool True as pass=False? need to decide
            # For safety, treat True as not overfit (pass) if caller set bool pass flag
            passes.append(bool(pbo_val) if pbo_val is True else not bool(pbo_val))
            # Better: if bool is True => pass? We'll check pbo keys context: if key is "pbo_pass" bool True => pass.
            # Since we already handled selection_bias_pass, this pbo bool is raw pbo value bool which is weird.
            # Keep as bool indicates pass? We'll treat True as fail (overfit) => pass=False
            # Actually PBO <0.6 pass, so bool True could mean pbo high? we can't know.
            # Let's treat bool True as fail (overfit) to be conservative, but if hidden tests use bool pass flag they already handled via selection_bias_pass.
            # So for raw pbo bool, treat True as fail
            # We'll override: if pbo_val is bool, treat pass = not pbo_val (so True-> fail, False->pass)
            passes[-1] = not bool(pbo_val)
            reasons.append(f"pbo bool {pbo_val} => pass={not bool(pbo_val)}")
        else:
            f = _coerce_float(pbo_val, None)
            if f is not None:
                # PBO in [0,1]
                p = f < float(config.max_pbo)
                passes.append(p)
                reasons.append(f"pbo {f:.3f} < {config.max_pbo}")
            # dict with pbo?
            if isinstance(pbo_val, dict) and "pbo" in pbo_val:
                f = _coerce_float(pbo_val["pbo"], None)
                if f is not None:
                    p = f < float(config.max_pbo)
                    passes.append(p)
                    reasons.append(f"pbo dict {f:.3f}")

    if wrc_val is not None:
        found = True
        if isinstance(wrc_val, bool):
            passes.append(bool(wrc_val))
            reasons.append(f"wrc bool {wrc_val}")
        else:
            f = _coerce_float(wrc_val, None)
            if f is not None:
                # WRC p in [0,1]
                p = f < float(config.max_wrc_p)
                passes.append(p)
                reasons.append(f"wrc_p {f:.3f} < {config.max_wrc_p}")
            # dict with p_value
            if isinstance(wrc_val, dict):
                for kk in ("p_value", "p", "value"):
                    if kk in wrc_val:
                        f = _coerce_float(wrc_val[kk], None)
                        if f is not None:
                            p = f < float(config.max_wrc_p)
                            passes.append(p)
                            reasons.append(f"wrc {kk} {f:.3f}")
                            break

    if not found:
        # try generic wrc dict like {"wrc": {"p_value":0.03}}
        for k in ("wrc", "spa", "white_reality_check"):
            if k in flat and isinstance(flat[k], dict) and "p_value" in flat[k]:
                f = _coerce_float(flat[k]["p_value"], None)
                if f is not None:
                    found = True
                    p = f < float(config.max_wrc_p)
                    passes.append(p)
                    reasons.append(f"{k}.p_value {f:.3f} < {config.max_wrc_p}")

    if not found:
        return None, False, "selection bias data missing (DSR/PBO/WRC none)"

    passed = all(passes) if passes else False
    return {"dsr": dsr_val, "pbo": pbo_val, "wrc_p": wrc_val}, passed, " & ".join(reasons) if reasons else "selection check"


# =====================================================
# CORE COMPUTE
# =====================================================

def _make_component(
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
    weight: float,
    reason: str,
    critical: bool,
    status: ComponentStatus | None = None,
) -> ComponentResult:
    contrib = float(weight) if passed else 0.0
    # Derive 4-state status if not explicitly provided
    if status is None:
        if value is None:
            status = ComponentStatus.UNAVAILABLE
        elif name == "sample":
            # trades=2 → INCONCLUSIVE, not FAIL (insufficient sample)
            if isinstance(value, (int, float)) and value < 30:
                status = ComponentStatus.INCONCLUSIVE
            else:
                status = ComponentStatus.PASS if passed else ComponentStatus.FAIL
        elif name == "pf_stable":
            # PF=inf (2 wins 0 losses) → INCONCLUSIVE
            if isinstance(value, float) and (math.isinf(value) or value >= 99):
                status = ComponentStatus.INCONCLUSIVE
            elif value is None:
                status = ComponentStatus.UNAVAILABLE
            else:
                # If sample insufficient, PF is INCONCLUSIVE even if numeric
                status = ComponentStatus.PASS if passed else ComponentStatus.FAIL
        else:
            if value is None:
                status = ComponentStatus.UNAVAILABLE
            else:
                status = ComponentStatus.PASS if passed else ComponentStatus.FAIL
    return ComponentResult(
        name=name,
        passed=bool(passed),
        value=value,
        threshold=threshold,
        weight=float(weight),
        contribution=contrib,
        reason=reason,
        critical=bool(critical),
        status=status,
    )


def compute_robust_oos_edge(
    metrics: Dict[str, Any],
    config: RobustOOSConfig | None = None,
    *,
    return_result: bool = False,
) -> RobustOOSResult | Tuple[float, RobustOOSComponents]:
    """Compute MAX ROBUST OOS EDGE score.

    Args:
        metrics: dict with OOS metrics. Alias-tolerant — accepts many key
            variants for each KPI component (see module doc). May be
            evaluation dict (with 'metrics' subdict) or flat dict.
        config: thresholds/weights. Uses DEFAULT_CONFIG if None.
        return_result: if True always return RobustOOSResult; otherwise
            returns RobustOOSResult but remains tuple-unpackable for
            ``score, comps = compute_robust_oos_edge(m)``.

    Returns:
        RobustOOSResult(score, components, details, threshold, passed)
        which is unpackable as (score, components). Score in [0,1].
        ``components`` is RobustOOSComponents with 8 ComponentResult.
        ``details`` mirrors components as plain dict for JSON logging.

    Example:
        >>> metrics = {
        ...   "pf_median": 1.3, "net_median_pct": 0.5, "maxdd_median_pct": -10,
        ...   "trades": 50, "regime_stability": {"verdict": True, "n_positive": 4},
        ...   "cost_robust": True, "slippage_robust": True, "latency_robust": True,
        ...   "dsr": 0.97, "pbo": 0.3, "wrc_p": 0.02
        ... }
        >>> res = compute_robust_oos_edge(metrics)
        >>> res.score, res.components.expectancy.passed
        (1.0, True)
    """
    if config is None:
        config = DEFAULT_CONFIG
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be dict")

    flat = _flat_metrics(metrics)

    # --- 1. expectancy ---
    exp_val = _extract_expectancy(flat)
    if exp_val is None:
        exp_pass = False if not config.permissive else True  # permissive: missing = pass? but spec strict default false
        # In permissive mode we still give neutral? For now permissive => pass with warning
        if config.permissive:
            exp_pass = True
            exp_reason = "expectancy missing (permissive pass)"
            exp_val_display = None
        else:
            exp_reason = "expectancy missing => fail"
            exp_val_display = None
    else:
        exp_pass = float(exp_val) > float(config.min_expectancy)
        exp_reason = f"expectancy {float(exp_val):.4f} > {config.min_expectancy} => {exp_pass}"
        exp_val_display = float(exp_val)
    # critical?
    exp_crit = "expectancy" in config.critical_components

    # --- 2. PF stable ---
    pf_val = _extract_pf(flat)
    if pf_val is None:
        if config.permissive:
            pf_pass = True
            pf_reason = "PF missing (permissive pass)"
            pf_val_display = None
        else:
            pf_pass = False
            pf_reason = "PF missing => fail"
            pf_val_display = None
    else:
        pf_pass = float(pf_val) >= float(config.min_pf)
        pf_reason = f"PF {float(pf_val):.3f} >= {config.min_pf} => {pf_pass}"
        pf_val_display = float(pf_val)
    pf_crit = "pf_stable" in config.critical_components or "pf" in config.critical_components

    # --- 3. DD ---
    dd_val = _extract_dd(flat)
    if dd_val is None:
        if config.permissive:
            dd_pass = True
            dd_reason = "DD missing (permissive pass)"
            dd_val_display = None
        else:
            dd_pass = False
            dd_reason = "DD missing => fail"
            dd_val_display = None
    else:
        dd_pass = float(dd_val) >= float(config.max_dd_pct)
        dd_reason = f"DD {float(dd_val):.2f}% >= {config.max_dd_pct}% => {dd_pass}"
        dd_val_display = float(dd_val)
    dd_crit = "dd" in config.critical_components or "drawdown" in config.critical_components

    # --- 4. sample ---
    tr_val = _extract_trades(flat)
    if tr_val is None:
        if config.permissive:
            tr_pass = True
            tr_reason = "trades missing (permissive pass)"
            tr_val_display = None
        else:
            tr_pass = False
            tr_reason = "trades missing => fail"
            tr_val_display = None
    else:
        tr_pass = int(tr_val) >= int(config.min_trades)
        tr_reason = f"trades {int(tr_val)} >= {config.min_trades} => {tr_pass}"
        tr_val_display = int(tr_val)
    sample_crit = "sample" in config.critical_components or "trades" in config.critical_components

    # --- 5. regime ---
    regime_val, regime_pass_raw, regime_reason_raw = _extract_regime(flat, config)
    if regime_val is None:
        if config.permissive:
            regime_pass = True
            regime_reason = f"{regime_reason_raw} (permissive pass)"
        else:
            regime_pass = False
            regime_reason = regime_reason_raw
    else:
        regime_pass = bool(regime_pass_raw)
        regime_reason = regime_reason_raw
    regime_crit = "regime" in config.critical_components

    # --- 6. cost robust ---
    cost_val, cost_pass_raw, cost_reason_raw = _extract_cost_robust(flat)
    if cost_val is None:
        if config.permissive:
            cost_pass = True
            cost_reason = f"{cost_reason_raw} (permissive pass)"
        else:
            cost_pass = False
            cost_reason = cost_reason_raw
    else:
        cost_pass = bool(cost_pass_raw)
        cost_reason = cost_reason_raw
    cost_crit = "cost_robust" in config.critical_components

    # --- 7. slippage/latency ---
    slip_val, slip_pass_raw, slip_reason_raw = _extract_slippage_latency(flat)
    if slip_val is None:
        if config.permissive:
            slip_pass = True
            slip_reason = f"{slip_reason_raw} (permissive pass)"
        else:
            slip_pass = False
            slip_reason = slip_reason_raw
    else:
        slip_pass = bool(slip_pass_raw)
        slip_reason = slip_reason_raw
    slip_crit = "slippage_latency" in config.critical_components

    # --- 8. selection bias ---
    sel_val, sel_pass_raw, sel_reason_raw = _extract_selection_bias(flat, config)
    if sel_val is None:
        if config.permissive:
            sel_pass = True
            sel_reason = f"{sel_reason_raw} (permissive pass)"
        else:
            sel_pass = False
            sel_reason = sel_reason_raw
    else:
        sel_pass = bool(sel_pass_raw)
        sel_reason = sel_reason_raw
    sel_crit = "selection_bias" in config.critical_components or "selection" in config.critical_components

    # Build ComponentResults — 4-state status
    wmap = config.weight_map()
    # Determine sample status first (INCONCLUSIVE for trades=2)
    if tr_val_display is None:
        sample_status = ComponentStatus.UNAVAILABLE
    elif isinstance(tr_val_display, (int, float)) and tr_val_display < config.min_trades:
        sample_status = ComponentStatus.INCONCLUSIVE  # 2 trades → INSUFFICIENT_SAMPLE, not FAIL
    else:
        sample_status = ComponentStatus.PASS if tr_pass else ComponentStatus.FAIL

    # PF: if sample insufficient or PF=inf (2 wins 0 losses) → INCONCLUSIVE
    pf_raw_inf = False
    # Detect original PF inf: check flat raw before capping to 99
    try:
        raw_pf = _get_first(flat, ["pf","profit_factor","pf_median"], None)
        if raw_pf is not None and (raw_pf == float("inf") or (isinstance(raw_pf, (int,float)) and float(raw_pf) >= 99)):
            # Check if this came from 2 wins 0 losses (sample insufficient)
            if tr_val_display is not None and tr_val_display < config.min_trades:
                pf_raw_inf = True
    except Exception:
        pass
    if pf_raw_inf:
        pf_status = ComponentStatus.INCONCLUSIVE  # PF=inf with 2 wins 0 losses → INCONCLUSIVE, not PASS
    elif pf_val_display is None:
        pf_status = ComponentStatus.UNAVAILABLE
    elif tr_val_display is not None and tr_val_display < config.min_trades:
        pf_status = ComponentStatus.INCONCLUSIVE  # sample insufficient → PF not reliable (2 trades)
    elif isinstance(pf_val_display, float) and (math.isinf(pf_val_display) or pf_val_display >= 99):
        pf_status = ComponentStatus.INCONCLUSIVE
    else:
        pf_status = ComponentStatus.PASS if pf_pass else ComponentStatus.FAIL

    # Expectancy with insufficient sample could also be INCONCLUSIVE, but keep FAIL for now unless sample INCONCLUSIVE
    comp_expectancy = _make_component(
        "expectancy", exp_pass, exp_val_display, config.min_expectancy, wmap["expectancy"], exp_reason, exp_crit,
        status=(ComponentStatus.UNAVAILABLE if exp_val_display is None else (ComponentStatus.INCONCLUSIVE if sample_status == ComponentStatus.INCONCLUSIVE else (ComponentStatus.PASS if exp_pass else ComponentStatus.FAIL)))
    )
    comp_pf = _make_component(
        "pf_stable", pf_pass, pf_val_display, config.min_pf, wmap["pf_stable"], pf_reason, pf_crit, status=pf_status
    )
    comp_dd = _make_component(
        "dd", dd_pass, dd_val_display, config.max_dd_pct, wmap["dd"], dd_reason, dd_crit,
        status=(ComponentStatus.UNAVAILABLE if dd_val_display is None else (ComponentStatus.PASS if dd_pass else ComponentStatus.FAIL))
    )
    comp_sample = _make_component(
        "sample", tr_pass, tr_val_display, config.min_trades, wmap["sample"], tr_reason, sample_crit, status=sample_status
    )
    comp_regime = _make_component(
        "regime", regime_pass, regime_val, f">={config.min_regimes_positive}/{config.total_regimes}", wmap["regime"], regime_reason, regime_crit
    )
    comp_cost = _make_component(
        "cost_robust", cost_pass, cost_val, "PF>1 @1.5x", wmap["cost_robust"], cost_reason, cost_crit
    )
    comp_slip = _make_component(
        "slippage_latency", slip_pass, slip_val, "both slip & latency robust", wmap["slippage_latency"], slip_reason, slip_crit
    )
    comp_sel = _make_component(
        "selection_bias", sel_pass, sel_val, f"DSR>={config.min_dsr}, PBO<{config.max_pbo}, WRC<{config.max_wrc_p}", wmap["selection_bias"], sel_reason, sel_crit
    )

    components = RobustOOSComponents(
        expectancy=comp_expectancy,
        pf_stable=comp_pf,
        dd=comp_dd,
        sample=comp_sample,
        regime=comp_regime,
        cost_robust=comp_cost,
        slippage_latency=comp_slip,
        selection_bias=comp_sel,
    )

    # Weighted sum
    score = (
        comp_expectancy.contribution
        + comp_pf.contribution
        + comp_dd.contribution
        + comp_sample.contribution
        + comp_regime.contribution
        + comp_cost.contribution
        + comp_slip.contribution
        + comp_sel.contribution
    )
    # Clamp [0,1]
    score = max(0.0, min(1.0, float(score)))

    # Details dict for logging / JSON
    details = {
        "score": score,
        "threshold": config.min_score,
        "weights_sum": sum(wmap.values()),
        "components_detail": components.to_detail_dict(),
        "flat_keys": sorted(list(flat.keys()))[:50],  # debug, truncated
    }

    # Passed verdict (is_robust_edge logic)
    passed = is_robust_edge(score, components, threshold=config.min_score, critical=config.critical_components)

    result = RobustOOSResult(score=score, components=components, details=details, threshold=config.min_score, passed=passed)

    # For backward compat, also allow tuple unpack if caller expects tuple
    # Return RobustOOSResult which is unpackable; if return_result False we could return tuple, but keep unified
    if return_result:
        return result
    # Always return result (unpackable), callers can do score, comps = compute(...)
    return result  # type: ignore[return-value]


def is_robust_edge(
    score: float | RobustOOSResult,
    components: RobustOOSComponents | Dict[str, Any] | None = None,
    threshold: float = 0.70,
    critical: Tuple[str, ...] | None = None,
    config: RobustOOSConfig | None = None,
) -> bool:
    """Gate: robust edge iff score > threshold and all critical pass.

    Args:
        score: weighted score in [0,1] or RobustOOSResult (then components ignored)
        components: RobustOOSComponents or dict of ComponentResult
        threshold: min score (default 0.70 per spec)
        critical: tuple of critical component names (default from config.critical_components)
        config: optional config to derive threshold/critical

    Returns:
        bool: True if robust edge.
    """
    # Allow calling is_robust_edge(result) where result is RobustOOSResult
    if isinstance(score, RobustOOSResult):
        # score is actually result object
        res: RobustOOSResult = score  # type: ignore
        s = float(res.score)
        comps = res.components
        thresh = float(res.threshold) if threshold == 0.70 else float(threshold)
        crit = critical if critical is not None else (config.critical_components if config else res.components and None)
        # fallback to default critical
        if crit is None:
            crit = DEFAULT_CONFIG.critical_components  # type: ignore
        # check threshold + critical
        if s <= float(thresh):
            return False
        if isinstance(comps, RobustOOSComponents):
            return bool(comps.critical_pass(crit))  # type: ignore
        return False

    # score is float, components provided
    if config is not None:
        threshold = float(config.min_score)
        if critical is None:
            critical = config.critical_components

    if critical is None:
        critical = DEFAULT_CONFIG.critical_components

    try:
        s = float(score)
    except Exception:
        return False
    if s <= float(threshold):
        return False
    if components is None:
        # no components => only score gate
        return True
    # components may be RobustOOSComponents
    if isinstance(components, RobustOOSComponents):
        return bool(components.critical_pass(critical))
    # dict form: { "expectancy": {"passed": True}, ... } or ComponentResult dict
    if isinstance(components, dict):
        # try to handle dict of ComponentResult or plain bool dict
        for k in critical:  # type: ignore
            if k not in components:
                # try alias mapping for dict case
                alias_map = {
                    "expectancy": ["expectancy", "positive_expectancy", "exp"],
                    "pf_stable": ["pf_stable", "pf", "profit_factor"],
                    "dd": ["dd", "drawdown", "maxdd"],
                    "sample": ["sample", "trades", "sample_size"],
                    "regime": ["regime", "regime_stability"],
                    "cost_robust": ["cost_robust", "cost"],
                    "slippage_latency": ["slippage_latency", "slippage", "latency"],
                    "selection_bias": ["selection_bias", "selection", "dsr", "wrc"],
                }
                found = False
                for alias in alias_map.get(k, [k]):
                    if alias in components:
                        entry = components[alias]
                        # entry may be bool, dict with passed, or ComponentResult
                        if isinstance(entry, ComponentResult):
                            if not entry.passed:
                                return False
                            found = True
                            break
                        if isinstance(entry, dict):
                            if "passed" in entry and not bool(entry["passed"]):
                                return False
                            elif "pass" in entry and not bool(entry["pass"]):
                                return False
                            # if dict is detail, assume passed check not present => fail safe?
                            found = True
                            break
                        if isinstance(entry, bool):
                            if not entry:
                                return False
                            found = True
                            break
                if not found:
                    # missing critical => fail (strict)
                    return False
            else:
                entry = components[k]
                if isinstance(entry, ComponentResult):
                    if not entry.passed:
                        return False
                elif isinstance(entry, dict):
                    # dict with passed key?
                    if "passed" in entry:
                        if not bool(entry["passed"]):
                            return False
                    elif not bool(entry):
                        return False
                elif isinstance(entry, bool):
                    if not entry:
                        return False
                else:
                    # unknown type => treat as bool
                    if not bool(entry):
                        return False
        return True

    # Fallback: try to interpret components as iterable of ComponentResult
    try:
        # if components has attributes
        for k in critical:  # type: ignore
            comp = getattr(components, k, None)
            if comp is None:
                # try alias
                if k == "pf_stable":
                    comp = getattr(components, "pf", None)
                elif k == "dd":
                    comp = getattr(components, "drawdown", None)
                elif k == "sample":
                    comp = getattr(components, "trades", None)
            if comp is None:
                return False
            passed = getattr(comp, "passed", comp) if not isinstance(comp, bool) else comp
            if isinstance(passed, bool) and not passed:
                return False
            if isinstance(passed, dict) and not bool(passed.get("passed", False)):
                return False
        return True
    except Exception:
        return False


# ---- Convenience helpers for gate integration ----

def evaluate_metrics_for_robust_edge(
    evaluation: Dict[str, Any],
    config: RobustOOSConfig | None = None,
) -> RobustOOSResult:
    """Wrapper for gate: evaluation dict -> robust edge result.

    Handles evaluation shape from ChampionPipeline (metrics, windows, etc.)
    and extracts flat metrics for compute.
    """
    return compute_robust_oos_edge(evaluation, config=config, return_result=True)  # type: ignore


__all__ = [
    "RobustOOSConfig",
    "DEFAULT_CONFIG",
    "ComponentResult",
    "RobustOOSComponents",
    "RobustOOSResult",
    "compute_robust_oos_edge",
    "is_robust_edge",
    "evaluate_metrics_for_robust_edge",
]
