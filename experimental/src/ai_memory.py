"""
QuantAI Three-Tier Memory System

Hierarchical memory for AI-driven strategy research:

1. **Observations (7 days)** - Short-term logs of recent runs/experiments
   - Raw experimental outcomes, performance metrics, failures
   - Auto-expires after 7 days unless promoted

2. **Insights (90 days)** - Aggregated patterns per asset/regime
   - Statistical patterns distilled from observations
   - Asset-specific, regime-specific learnings
   - Auto-expires after 90 days unless promoted

3. **Rules (Permanent)** - Global rules auto-generated from patterns
   - High-confidence generalized rules
   - Used directly by strategy generation
   - Never expires, only updated with higher confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from enum import Enum
import json


class MemoryTier(Enum):
    """Three-tier memory hierarchy."""
    OBSERVATION = "observation"      # 7 days TTL
    INSIGHT = "insight"              # 90 days TTL
    RULE = "rule"                    # Permanent


class PromotionReason(Enum):
    """Reasons for promotion between tiers."""
    MANUAL = "manual"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    REPEATED_OBSERVATION = "repeated_observation"
    HIGH_CONFIDENCE = "high_confidence"


# ============================================================
# TTL CONFIGURATION
# ============================================================

TIER_TTL = {
    "observation": timedelta(days=7),
    "insight": timedelta(days=90),
    "rule": None,  # Permanent
}

PROMOTION_THRESHOLDS = {
    "observation_to_insight": {
        "min_occurrences": 3,
        "min_confidence": 0.7,
        "min_days": 2,
    },
    "insight_to_rule": {
        "min_occurrences": 5,
        "min_confidence": 0.8,
        "min_days": 30,
    },
}


# ============================================================
# BASE MEMORY RECORD
# ============================================================

@dataclass
class BaseMemoryRecord:
    """Base class for all memory records."""
    
    id: str
    content: str
    tier: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expires_at is None and self.tier != "rule":
            ttl = TIER_TTL.get(self.tier)
            if ttl:
                self.expires_at = datetime.now(timezone.utc) + ttl
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def score(self) -> float:
        return round(0.6 * self.confidence + 0.4 * self.importance, 4)
    
    def touch(self):
        self.updated_at = datetime.now(timezone.utc)


# ============================================================
# TIER-SPECIFIC RECORDS
# ============================================================

@dataclass
class Observation(BaseMemoryRecord):
    """
    Tier 1: Raw experimental observations (7 days TTL).
    
    Examples:
    - "Run #42: PnL -2.3%, max DD 4.1%, strategy=mean_reversion_v3"
    - "ML model v5.2 balanced_accuracy=0.54 on ETH/USDT 15m"
    - "Regime detector: false positive on BTC flat period"
    """
    tier: str = "observation"
    
    # Experiment context
    run_id: Optional[str] = None
    strategy_name: Optional[str] = None
    asset: Optional[str] = None
    regime: Optional[str] = None
    metrics: dict = field(default_factory=dict)  # PnL, DD, Sharpe, etc.
    outcome: str = "neutral"  # "positive", "negative", "neutral"
    
    def promote_to_insight(self, insight: "Insight") -> None:
        """Link this observation to a promoted insight."""
        self.metadata["promoted_to"] = insight.id
        self.metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()


@dataclass
class Insight(BaseMemoryRecord):
    """
    Tier 2: Aggregated patterns per asset/regime (90 days TTL).
    
    Examples:
    - "Breakout on ETH works best on 1m with ADX filter (win rate 62%)"
    - "Mean reversion fails in high vol regime (vol > 80th pctl)"
    - "BTC lead-lag: BTC 15m move predicts ETH 15m move by 2-3 bars"
    """
    tier: str = "insight"
    
    # Pattern context
    asset: Optional[str] = None
    regime: Optional[str] = None
    pattern_type: str = "general"  # "breakout", "mean_reversion", "trend_follow", "correlation", etc.
    
    # Statistical evidence
    supporting_observations: list[str] = field(default_factory=list)  # observation IDs
    sample_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe: float = 0.0
    
    # Conditions
    conditions: dict = field(default_factory=dict)  # e.g., {"timeframe": "15m", "adx_min": 25}
    
    def add_supporting_observation(self, obs_id: str) -> None:
        if obs_id not in self.supporting_observations:
            self.supporting_observations.append(obs_id)
            self.sample_count = len(self.supporting_observations)
            self.touch()
    
    def can_promote_to_rule(self) -> bool:
        """Check if this insight meets criteria for rule promotion."""
        thresholds = PROMOTION_THRESHOLDS["insight_to_rule"]
        return (
            self.sample_count >= thresholds["min_occurrences"] and
            self.confidence >= thresholds["min_confidence"] and
            (datetime.now(timezone.utc) - self.created_at) >= timedelta(days=thresholds["min_days"])
        )


@dataclass
class Rule(BaseMemoryRecord):
    """
    Tier 3: Global rules auto-generated from insights (Permanent).
    
    Examples:
    - "IF regime=high_vol AND signal=breakout THEN reduce_position_by_50%"
    - "IF asset_correlation > 0.8 THEN max_position_size = 50%"
    - "IF ADX < 20 THEN disable_trend_strategies"
    """
    tier: str = "rule"
    
    # Rule definition
    condition: str = ""  # Human-readable condition
    action: str = ""     # Human-readable action
    dsl: str = ""        # Machine-executable DSL (e.g., "regime == 'high_vol' and signal == 'breakout'")
    
    # Scope
    applies_to: list[str] = field(default_factory=list)  # ["all"] or ["BTC", "ETH", ...]
    regime_filter: Optional[str] = None
    strategy_types: list[str] = field(default_factory=list)  # ["all"] or specific types
    
    # Performance tracking
    applications: int = 0
    successful_applications: int = 0
    last_applied: Optional[datetime] = None
    
    def apply(self, context: dict) -> bool:
        """Evaluate if rule applies to given context."""
        # In production, this would evaluate the DSL
        # For now, simple condition matching
        if self.regime_filter and context.get("regime") != self.regime_filter:
            return False
        if "all" not in self.applies_to and context.get("asset") not in self.applies_to:
            return False
        if self.strategy_types and "all" not in self.strategy_types:
            if context.get("strategy_type") not in self.strategy_types:
                return False
        return True
    
    def record_application(self, success: bool) -> None:
        self.applications += 1
        if success:
            self.successful_applications += 1
        self.last_applied = datetime.now(timezone.utc)
        self.touch()
    
    @property
    def success_rate(self) -> float:
        if self.applications == 0:
            return 0.0
        return self.successful_applications / self.applications


# ============================================================
# THREE-TIER MEMORY STORE
# ============================================================

class ThreeTierMemory:
    """
    Three-tier memory system for QuantAI.
    
    Manages observations, insights, and rules with automatic
    expiration, promotion, and querying capabilities.
    """
    
    def __init__(self):
        self._observations: dict[str, Observation] = {}
        self._insights: dict[str, Insight] = {}
        self._rules: dict[str, Rule] = {}
        
        # Auto-cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired memories."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self._running:
                    self._cleanup_expired()
                    self._check_promotions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ThreeTierMemory] Cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired observations and insights."""
        now = datetime.now(timezone.utc)
        
        # Clean observations
        expired_obs = [oid for oid, obs in self._observations.items() if obs.is_expired]
        for oid in expired_obs:
            del self._observations[oid]
        
        # Clean insights
        expired_ins = [iid for iid, ins in self._insights.items() if ins.is_expired]
        for iid in expired_ins:
            del self._insights[iid]
        
        # Rules never expire
        
        if expired_obs or expired_ins:
            print(f"[ThreeTierMemory] Cleaned {len(expired_obs)} observations, {len(expired_ins)} insights")
    
    def _check_promotions(self):
        """Check for observations/insights ready for promotion."""
        # Observation -> Insight
        for obs in list(self._observations.values()):
            if obs.metadata.get("promoted_to"):
                continue
            
            # Check if we have enough similar observations
            similar = self._find_similar_observations(obs)
            if len(similar) >= PROMOTION_THRESHOLDS["observation_to_insight"]["min_occurrences"]:
                avg_conf = sum(o.confidence for o in similar) / len(similar)
                if avg_conf >= PROMOTION_THRESHOLDS["observation_to_insight"]["min_confidence"]:
                    self._promote_observations_to_insight(similar)
        
        # Insight -> Rule
        for insight in list(self._insights.values()):
            if insight.can_promote_to_rule():
                self._promote_insight_to_rule(insight)
    
    def _find_similar_observations(self, obs: Observation) -> list[Observation]:
        """Find observations with similar context."""
        similar = []
        for other in self._observations.values():
            if other.id == obs.id:
                continue
            if other.asset == obs.asset and other.regime == obs.regime:
                if other.strategy_name == obs.strategy_name:
                    similar.append(other)
        return similar
    
    def _promote_observations_to_insight(self, observations: list[Observation]):
        """Create insight from similar observations."""
        if not observations:
            return
        
        # Aggregate metrics
        avg_pnl = sum(o.metrics.get("pnl", 0) for o in observations) / len(observations)
        avg_dd = sum(o.metrics.get("max_dd", 0) for o in observations) / len(observations)
        win_rate = sum(1 for o in observations if o.outcome == "positive") / len(observations)
        
        # Generate insight content
        asset = observations[0].asset or "unknown"
        regime = observations[0].regime or "unknown"
        strategy = observations[0].strategy_name or "unknown"
        
        content = (
            f"Pattern for {asset} in {regime} regime with {strategy}: "
            f"avg PnL {avg_pnl:.2f}%, avg DD {avg_dd:.2f}%, win rate {win_rate:.1%} "
            f"over {len(observations)} occurrences"
        )
        
        insight = Insight(
            id=f"insight_{len(self._insights) + 1}_{int(datetime.now(timezone.utc).timestamp())}",
            tier="insight",
            content=content,
            tags=("pattern", asset, regime, strategy),
            confidence=sum(o.confidence for o in observations) / len(observations),
            importance=max(o.importance for o in observations),
            asset=asset,
            regime=regime,
            pattern_type="auto",
            supporting_observations=[o.id for o in observations],
            sample_count=len(observations),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            conditions={"strategy": strategy, "regime": regime},
        )
        
        self._insights[insight.id] = insight
        
        # Link back to observations
        for obs in observations:
            obs.promote_to_insight(insight)
        
        print(f"[ThreeTierMemory] Promoted {len(observations)} observations to insight: {insight.id}")
    
    def _promote_insight_to_rule(self, insight: Insight):
        """Promote insight to permanent rule."""
        # Generate DSL from insight
        dsl_parts = []
        if insight.asset:
            dsl_parts.append(f"asset == '{insight.asset}'")
        if insight.regime:
            dsl_parts.append(f"regime == '{insight.regime}'")
        if insight.conditions:
            for k, v in insight.conditions.items():
                dsl_parts.append(f"{k} == '{v}'")
        
        dsl = " and ".join(dsl_parts) if dsl_parts else "true"
        
        # Determine action from insight content
        action = "increase_position" if insight.avg_pnl > 0 else "reduce_position"
        
        rule = Rule(
            id=f"rule_{len(self._rules) + 1}_{int(datetime.now(timezone.utc).timestamp())}",
            tier="rule",
            content=f"Rule from insight {insight.id}: {insight.content}",
            tags=("rule", "auto", insight.asset or "global"),
            confidence=insight.confidence,
            importance=insight.importance,
            condition=insight.content,
            action=action,
            dsl=dsl,
            applies_to=[insight.asset] if insight.asset else ["all"],
            regime_filter=insight.regime,
            strategy_types=["all"],
        )
        
        self._rules[rule.id] = rule
        
        # Link back
        insight.metadata["promoted_to"] = rule.id
        insight.metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
        
        print(f"[ThreeTierMemory] Promoted insight to rule: {rule.id}")
    
    # ============================================================
    # PUBLIC API
    # ============================================================
    
    def add_observation(
        self,
        *,
        id: str,
        content: str,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        strategy_name: Optional[str] = None,
        metrics: Optional[dict] = None,
        outcome: str = "neutral",
        confidence: float = 0.5,
        importance: float = 0.5,
        tags: Iterable[str] = (),
    ) -> Observation:
        """Add a new observation."""
        if id in self._observations:
            raise ValueError(f"Observation already exists: {id}")
        
        obs = Observation(
            id=id,
            tier="observation",
            content=content,
            tags=tuple(dict.fromkeys(tags)),
            confidence=confidence,
            importance=importance,
            asset=asset,
            regime=regime,
            strategy_name=strategy_name,
            metrics=metrics or {},
            outcome=outcome,
        )
        
        self._observations[id] = obs
        return obs
    
    def add_insight(self, insight: Insight) -> Insight:
        """Add an insight manually."""
        if insight.id in self._insights:
            raise ValueError(f"Insight already exists: {insight.id}")
        self._insights[insight.id] = insight
        return insight
    
    def add_rule(self, rule: Rule) -> Rule:
        """Add a rule manually."""
        if rule.id in self._rules:
            raise ValueError(f"Rule already exists: {rule.id}")
        self._rules[rule.id] = rule
        return rule
    
    def get_observation(self, id: str) -> Optional[Observation]:
        return self._observations.get(id)
    
    def get_insight(self, id: str) -> Optional[Insight]:
        return self._insights.get(id)
    
    def get_rule(self, id: str) -> Optional[Rule]:
        return self._rules.get(id)
    
    def search_observations(
        self,
        query: Optional[str] = None,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 50,
    ) -> list[Observation]:
        """Search observations with filters."""
        results = []
        for obs in self._observations.values():
            if obs.is_expired:
                continue
            if asset and obs.asset != asset:
                continue
            if regime and obs.regime != regime:
                continue
            if strategy and obs.strategy_name != strategy:
                continue
            if query and query.lower() not in obs.content.lower():
                continue
            results.append(obs)
        
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]
    
    def search_insights(
        self,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[Insight]:
        """Search insights with filters."""
        results = []
        for ins in self._insights.values():
            if ins.is_expired:
                continue
            if asset and ins.asset != asset:
                continue
            if regime and ins.regime != regime:
                continue
            if pattern_type and ins.pattern_type != pattern_type:
                continue
            if ins.confidence < min_confidence:
                continue
            results.append(ins)
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]
    
    def get_applicable_rules(self, context: dict) -> list[Rule]:
        """Get rules applicable to a given context."""
        applicable = []
        for rule in self._rules.values():
            if rule.apply(context):
                applicable.append(rule)
        applicable.sort(key=lambda r: r.confidence, reverse=True)
        return applicable
    
    def record_rule_application(self, rule_id: str, success: bool) -> bool:
        """Record rule application outcome."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.record_application(success)
            return True
        return False
    
    def export_all(self) -> dict:
        """Export all memory tiers for backup/analysis."""
        return {
            "observations": [vars(o) for o in self._observations.values()],
            "insights": [vars(i) for i in self._insights.values()],
            "rules": [vars(r) for r in self._rules.values()],
        }
    
    def stats(self) -> dict:
        """Get memory statistics."""
        return {
            "observations": len(self._observations),
            "active_observations": sum(1 for o in self._observations.values() if not o.is_expired),
            "insights": len(self._insights),
            "active_insights": sum(1 for i in self._insights.values() if not i.is_expired),
            "rules": len(self._rules),
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

class AIMemory:
    """Legacy interface for backward compatibility."""
    
    def __init__(self):
        self._three_tier = ThreeTierMemory()
        self._records: dict[str, Any] = {}
    
    def add(self, *, memory_id: str, layer: str, content: str, tags: Iterable[str] = (), 
            strategy_id: str | None = None, asset: str | None = None,
            regime: str | None = None, confidence: float = 0.5, importance: float = 0.5):
        """Legacy add method - stores as observation."""
        if memory_id in self._records:
            raise ValueError(f"Memory already exists: {memory_id}")
        
        obs = self._three_tier.add_observation(
            id=memory_id,
            content=content,
            asset=asset,
            regime=regime,
            strategy_name=strategy_id,
            confidence=confidence,
            importance=importance,
            tags=tags,
        )
        
        # Also keep legacy record
        from dataclasses import dataclass
        @dataclass
        class LegacyRecord:
            memory_id: str
            layer: str
            content: str
            tags: tuple
            strategy_id: str | None
            asset: str | None
            regime: str | None
            confidence: float
            importance: float
            created_at: datetime
            updated_at: datetime
            status: str
        
        record = LegacyRecord(
            memory_id=memory_id,
            layer=layer,
            content=content,
            tags=tuple(dict.fromkeys(tags)),
            strategy_id=strategy_id,
            asset=asset,
            regime=regime,
            confidence=confidence,
            importance=importance,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
        )
        self._records[memory_id] = record
        return obs
    
    def get(self, memory_id: str):
        if memory_id in self._records:
            return self._records[memory_id]
        return self._three_tier.get_observation(memory_id) or \
               self._three_tier.get_insight(memory_id) or \
               self._three_tier.get_rule(memory_id)
    
    # Delegate other methods to three-tier memory
    def search(self, *args, **kwargs):
        return self._three_tier.search_observations(*args, **kwargs)
    
    def best(self, *args, **kwargs):
        obs = self._three_tier.search_observations(*args, **kwargs)
        return obs[0] if obs else None
    
    def export(self):
        return self._three_tier.export_all()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MemoryTier",
    "PromotionReason",
    "BaseMemoryRecord",
    "Observation",
    "Insight",
    "Rule",
    "ThreeTierMemory",
    "AIMemory",  # Legacy compatibility
    "TIER_TTL",
    "PROMOTION_THRESHOLDS",
]