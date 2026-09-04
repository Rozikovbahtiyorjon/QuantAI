# QuantAI Architecture 6.0 — Layered Architecture

**Version:** 6.0  
**Date:** 2026-09-02  
**Status:** Canonical — replaces all prior layer diagrams  
**Rule:** Governance must never bypass Risk/Execution. No path reaches Execution without passing PASS/FAIL.

---

## 1. Overview — Canonical Stack

```text
                         QUANTAI
                            |
                +-----------▼-----------+
                |   DATA GOVERNANCE     |
                | Dataset Registry      |
                | Data Quality          |
                | Immutable Dataset     |
                +-----------+-----------+
                            |
                +-----------▼-----------+
                |   RESEARCH ENGINE     |
                | Features              |
                | Labels                |
                | ML                    |
                | Strategies            |
                +-----------+-----------+
                            |
                +-----------▼-----------+
                | RESEARCH INTEGRITY    |
                | Nested WF             |
                | Purging               |
                | PBO                   |
                | DSR                   |
                | Reality Check         |
                | OOS Budget            |
                +-----------+-----------+
                            |
                +-----------▼-----------+
                | ROBUSTNESS ENGINE     |
                | Cost Stress           |
                | Slippage              |
                | Latency               |
                | Regimes               |
                | Bootstrap             |
                +-----------+-----------+
                            |
                       PASS / FAIL
                            |
                +-----------▼-----------+
                |  CHAMPION GOVERNANCE  |
                | Candidate             |
                | Paper                 |
                | Production Candidate  |
                | NO_CHAMPION           |
                +-----------+-----------+
                            |
                +-----------▼-----------+
                |       RISK            |
                | single canonical      |
                | policy + orchestrator |
                +-----------+-----------+
                            |
                +-----------▼-----------+
                |      EXECUTION        |
                | Exchange              |
                | Fill Model            |
                | Reconciliation        |
                | Emergency Flatten     |
                +-----------+-----------+
                            |
                    PAPER → TESTNET
                            |
                       PRODUCTION
```

**Invariant:** Every strategy flows top-down. Lateral bypass is forbidden:
`Data Governance → Research Engine → Research Integrity → Robustness Engine → PASS/FAIL → Champion Governance → Risk → Execution → PAPER → TESTNET → PRODUCTION`

No module in `src/champion/`, `src/risk/`, or `src/execution/` may import from a downstream layer to skip a gate. Verified by `importlinter` / `scripts/verify_final.py`.

---

## 2. Layer Definitions

### Layer 1 — DATA GOVERNANCE

**Purpose:** Make data a trusted, versioned, immutable asset. No research runs on ungoverned data.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Dataset Registry | `src/research/dataset_registry.py:57` | `DatasetRegistry` — `dataset_id`, SHA256 hash, symbol/timeframe, feature/label version, `*.json` index. `canonical_path()` is single source of truth. Legacy `*.bak.parquet` requires registry lookup. |
| Data Quality / Data Gates | `src/data/data_gates.py:120` | `DataGates.validate(df, timeframe)` — 9 automated gates: duplicate timestamps, missing bars, monotonicity, OHLC consistency, zero-volume, bad prices, timezone UTC, exchange outages, gaps. Raises `DataGateError` on failure. Also `src/data_quality.py:1` legacy validator. |
| Immutable Dataset | `src/research/dataset_registry.py:84` | `register()` hashes parquet, writes `*.json`, then `_make_readonly()` (`chmod 0o444` + Windows `FILE_ATTRIBUTE_READONLY`). `load()`/`verify()` recompute SHA256 on every read — mismatch raises `ValueError`. `load()` returns `df.copy()` — no in-place mutation. |

**Contracts:**

- `register(parquet_path, dataset_id, symbol, timeframe)` → runs `DataGates` before hashing (fail-fast).
- `hash_file(path)` — SHA256 of raw parquet bytes (`src/research/dataset_registry.py:74`).
- `verify(dataset_id)` — recomputed hash == stored hash, else `Immutable dataset violation`.
- `load(dataset_id)` — verified + copy.

| PASS | FAIL |
|------|------|
| All `DataGates` pass for declared `timeframe`; hash stored and file is read-only; `load()` returns verified copy. | Any gate raises `DataGateError` (missing bars, naive timestamps, OHLC violation, gap > `outage_multiplier`); hash mismatch → `ValueError`; empty/dup-timestamp → reject. Registration blocked. |

**Canonical rule (Audit #2):** One canonical `DatasetRegistry` per domain; `src/data/data_gates.py:120` is canonical gates implementation. No direct `pd.read_parquet` in research without registry.

---

### Layer 2 — RESEARCH ENGINE

**Purpose:** Generate features, labels, models and strategies deterministically from governed data only.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Features | `src/feature_engine.py:1`, `src/feature_store/store.py:1` | `FeatureVector` (25 ACTIVE + 5 PLANNED per `FEATURE_SCHEMA.json`). `add_indicators()` is single entry (`src/indicators.py:1`). Last-row only, normalized. `FeatureStore`/`drift.py`/`live_logger.py` for drift detection. |
| Labels | `src/labeling.py:1` | Triple Barrier v2 — `both barriers same bar → ambiguous → HOLD` (`discard`), `ret` uses `upper/lower` barrier price not `close`, emits `entry/exit/gross/net`. Versioned via `label_version`. |
| ML | `src/ml_engine.py:1`, `src/ml_config.py:1`, `src/model_manager.py:1`, `src/ml_ensemble.py:1`, `src/ml_regime.py:1` | XGBoost + PurgedKFold/Combinatorial CV, `predict_proba()` with `classes_=[0,1,2]→[SELL,HOLD,BUY]`. `model_manager` loads with pickle guarded by registry hash. |
| Strategies | `src/strategy/signal_generator.py:1`, `src/strategy/mean_reversion_signal.py:1`, `src/strategy/breakout_signal.py:1`, `src/strategy/meta_label.py:1`, `src/strategies/cross_sectional.py:1`, `src/strategy_genome.py:1`, `src/strategy_bank.py:1` | `SignalResult(signal, confidence, entry, sl, tp, reasons, diagnostics)` pipeline: `evaluate_market() → predict_ml() → fuse_ai_ml() → apply_order_flow_gate()`. `MIN_CONFIDENCE=60%`, `AI_WEIGHT=0.6/ML_WEIGHT=0.4`, `CONFLICT_PENALTY=0.7`. |

**Contracts:**

- Feature generation is pure: `DataFrame[OHLCV] → FeatureVector` — no future leak; monotonic timestamp enforced upstream.
- Labels never use `close` as proxy for barrier.
- ML models injected via dependency injection into strategy.

| PASS | FAIL |
|------|------|
| Features validated against schema; Triple Barrier ambiguous→HOLD; ML CV uses `PurgedKFold` (`src/validation/purged_kfold.py:1`) with embargo; strategy outputs `SignalResult` with `confidence≥60` gating. | Feature count mismatch, `ast.parse` fails on indicators, label uses wick proxy, CV without purge → leakage; any `DataGateError` propagates upward. |

---

### Layer 3 — RESEARCH INTEGRITY

**Purpose:** Prevent overfitting, leakage, and selection bias. This is the **hardest gate** — Tournament never sees candidates that fail here.

Hierarchy enforced by `src/champion/research_integrity.py:157`:

`Research → Integrity Checks → Statistical Validation → IS-OOS Consistency → ML Calibration → Robustness → Selection Adjustment → Regime Stability → Tournament → Champion`

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Nested Walk-Forward | `src/validation/nested_walk_forward.py:33`, `src/research/nested_research_pipeline.py:163` | Outer `TRAIN → INNER WF+Optuna → OUTER OOS` isolation. Inner WF runs on `OUTER TRAIN` only; `HoldoutLock` enforces `FINAL HOLDOUT` never touched during development (`src/research/nested_research_pipeline.py:69`). `NestedWalkForward.run(df, param_search_fn, best_param_apply_fn)` enforces fail-fast on optimizer errors. |
| Purging | `src/validation/purged_kfold.py:1` | `PurgedKFold` + `CombinatorialPurgedCV` (`config.settings.Settings.ml.cv_type:143`). Embargo `1%`, purge `0%` with regime-aware option. Monotonic time enforcement. |
| PBO (Prob. of Backtest Overfitting) | `src/research/pbo.py:1`, `src/validation/bootstrap.py:1` | CPCV via `compute_pbo(returns_df)` / `compute_pbo_from_sharpes(is_sharpes, oos_sharpes)` (Bailey et al.). Proxy fallback only when strict data missing. Gate: `PBO < 0.60` (`src/champion/research_integrity.py:59`). |
| DSR (Deflated Sharpe Ratio) | `src/research/dsr.py:1` | Bailey & Prado 2014 DSR with skew/kurtosis/sample_len: `deflated_sharpe_ratio(sharpe, n_trials, T, skew, kurt)` + `expected_max_sharpe`. Strict threshold `0.95` (permissive `0.5`). Window `net_pct` used to infer skew/kurt when not supplied. |
| White Reality Check / Hansen SPA | `src/research/white_reality_check.py:1` | Family-wise data-snooping test via stationary bootstrap (`q=0.1`, `B=1000`). `returns_df_from_evaluations()` builds `T×K` returns. `spa_test` (Hansen studentized) preferred; `white_reality_check` fallback. Global `p < 0.05` (`wrc_enabled`, `wrc_min_trials=100`). Computed once per `assess()` for batch. |
| OOS Budget | `src/research/research_budget.py:1`, `src/research/experiment_registry.py:1` | `ResearchBudget` limits max experiments/OOS touches (`BudgetExceeded` at 101st). `ExperimentRegistry` tracks every trial with `oos_touched` flag (`src/research/experiment_registry.py:1`). `HoldoutLock.mark_touched()` audit trail proves holdout isolation. |

**Integrity Gates (all HARD before Tournament):**

| Gate | File | PASS Threshold | FAIL Action |
|------|------|----------------|-------------|
| Integrity Checks | `src/champion/research_integrity.py:168` | `pf_median ≥1.05`, `profitable_window_share ≥0.45`, `maxdd_median ≥-15%`, `net_median ≥0%`, `trades ≥30`, `net_std ≤10%` (mirrors `src/champion/evaluation_pipeline.py:1` `PromotionRules`) + `trades<30 → INSUFFICIENT_SAMPLE` | Candidate removed before Tournament |
| Statistical Validation | `src/champion/research_integrity.py:281` | `sharpe_median ≥0`, bootstrap `p_sharpe ≤0.05`, `PBO <0.6`, `DSR ≥ threshold`, `WRC/SPA p <0.05` (when `n_trials ≥100`) | Hard FAIL; skipped if prior gate failed |
| IS-OOS Consistency | `src/champion/research_integrity.py:589` | `pf_ratio OOS/IS ≥0.60`, `pf_deterioration ≤50%`, `sharpe_deterioration ≤70%`, `PBO<0.6`, `IS PF≥1.3 ⇒ OOS PF≥1.0` | Blocks breakout-style IS-good/OOS-bad alpha |
| ML Calibration | `src/champion/research_integrity.py:672` | Bucket monotonic `spearman >0.5`, `pearson >0.3`, `cal_error <0.5`, plus `Brier<0.25` / `ECE<0.10` when probabilistic | Permissive → pass with warning; Strict → fail |
| Robustness (pre) | `src/champion/research_integrity.py:535` | `cost_robust=True`, `monte_carlo_score ≥0.3`, `stress_score ≥0.3` | Flagged fragile |
| Selection Adjustment | `src/champion/research_integrity.py:778` | `corrected_sharpe = sharpe/sqrt(1+log N)` > threshold; Bonferroni `p_adj ≤0.05` (strict) | Blocks best-of-many luck |
| Regime Stability | `src/champion/research_integrity.py:811` (`src/research/regime_stability.py:1`) | 7 regimes (Bull/Bear/Sideways/HighVol/LowVol/Crash/Recovery) with `n_positive ≥3` regimes and `min_trades_per_regime=5` | Reports `works/fails`; strict fails if insufficient |

| PASS (Layer) | FAIL (Layer) |
|--------------|--------------|
| All gates above pass (or permissive warnings) → candidate enters `eligible` pool for Champion Governance. | Any hard gate fails → candidate in `rejected` with `failed_stage` and `reasons`; never reaches `_rank_ids()` / `StrategyTournament`. `integrity_report.eligible` is empty → `NO_CHAMPION` path. |

---

### Layer 4 — ROBUSTNESS ENGINE

**Purpose:** Stress-test the edge that passed statistical integrity under real-world market frictions.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Cost Stress | `src/validation/cost_stress.py:31` | `cost_stress(df, base_commission=0.0004, base_slippage=0.0002)` runs BacktestEngine at `1.0x/1.25x/1.5x/2x/3x` costs via `config.settings:44`. `StressResult(multiplier, pf, net_profit, max_dd_pct, fragile)`. `is_cost_robust()` → `PF>1 at 1.5x` else fragile. |
| Slippage | `src/execution/fill_model.py:24` | `LimitFillModel(queue_ahead_pct=0.3, min_volume_ratio=1.5)` — queue-aware heuristic + `+25/+50/+100%` slippage stress. `attempt_fill(limit_price, side, bar_high/low/volume, avg_volume)` → `fill_prob <1.0` (not `touched→filled`). Deterministic via `hash(symbol\|ts\|price\|side\|seed)` (`src/execution/fill_model.py:38`). |
| Latency | `src/validation/cost_stress.py:10` (comment), `src/execution/execution_engine.py:49` | Latency stress `50/100/250/500ms/1s/3s` (audit #39-40). `ExecutionEngine` models queue time-priority; DRY_RUN validates against mark price. |
| Regimes | `src/research/regime_stability.py:1`, `src/research/regime_stability.py:classify_regimes` | 7-regime split + per-regime expectancy. Informs `when strategy WORKS / when FAILS` reporting. Integrated via `_gate_regime_stability` in Integrity as Gate 6. |
| Bootstrap | `src/validation/bootstrap.py:1` | `block_bootstrap_sharpe(returns, block=6, n_iter=200)` with `p_value` and `ci_lower`, `pbo_combinatorial`, `deflated_sharpe` helpers. Used across Statistical and Robustness. |

| PASS | FAIL |
|------|------|
| `is_cost_robust()==True` (PF≥1.0 at 1.5× costs), fill `prob<1` correctly lowers PF but retains edge, latency OOS within tolerance, ≥3 regimes positive, bootstrap `p_sharpe≤0.05`. | `PF<1 at 1.5x` → fragile (promotion blocked in `src/champion/research_integrity.py:557`), `LimitFillModel` previously optimistic `touched→filled` now fixed; any stress score `<0.3` → fragile. |

---

### Central Gate — PASS / FAIL

```
              ROBUSTNESS ENGINE
                     |
                PASS / FAIL        ←  Single chokepoint
                     |
          CHAMPION GOVERNANCE
```

`src/champion/research_integrity.py:956` `IntegrityReport` + `src/champion/pipeline.py:503` decision:

- **PASS** → candidate in `eligible`; enters `StrategyTournament` ranking (`src/strategy_tournament.py:1`).
- **FAIL** → candidate in `rejected`; `TournamentRanking.results` empty for that `strategy_id`. If `eligible` empty after all gates, pipeline emits `NO_CHAMPION` (valid success, not error).

`EdgeProvenGate` (`src/research/edge_validation.py:1`) variant: when `defer_champion_until_edge_proven=True`, pipeline stays in `RESEARCH` state until at least one strict-integrity candidate exists.

```python
# src/champion/pipeline.py:472
if not edge_res.edge_proven:
    return {"champion": "NO_CHAMPION", "state": "RESEARCH", "reason": "NO_EDGE_PROVEN"}
```

---

### Layer 5 — CHAMPION GOVERNANCE

**Purpose:** Own the candidate lifecycle from research to production with full auditability. Valid terminal state `NO_CHAMPION` when 100/100 strategies fail integrity.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Candidate | `src/champion/pipeline.py:159` (`ChampionPipeline`), `src/strategy_genome.py:1`, `src/strategy_bank.py:1`, `src/champion/evaluation_pipeline.py:1`, `src/strategy_tournament.py:1` | `StrategyRegistry` stores genome; `CandidateSpec(factory, params)` + `evaluate_candidate(spec, df, train_size/test_size/step_size)` → `{metrics, windows}`. `PromotionRules.evaluate_flags()` computes per-candidate PASS set. |
| Integrity-Filtered Tournament | `src/champion/research_integrity.py:157`, `src/champion/pipeline.py:504` | `ResearchIntegrityEngine.assess(evaluations)` → `eligible/rejected`. Only `eligible` reaches `StrategyTournament.rank()`. `vector_to_tournament_evaluation()` maps `pf_median→profit_factor`, `net_median_pct→total_return` etc. `monte_carlo/stress NOT_EVALUATED→0.0` (audit fix, not 0.5). |
| Paper | `src/paper_trading_runner.py:1`, `src/paper_trading_engine.py:1`, `src/paper_trading_session.py:1`, `src/validation/paper_30d.py:49`, `src/validation/long_run.py:1`, `src/paper_trading_quality_gate.py:1` | Shadow execution via `PaperTradingRunner`. Long-Run gates: ≥7d paper / ≥30d `Long-Run` validation; `Sharpe>0.5/1.0`, `DD<8%`, `win_rate 45-55%` with `2:1 RR`, `PF≥1.3`. |
| Production Candidate | `src/champion/pipeline.py:425` (`ChampionPipeline.STATES`), `src/champion_governance_engine.py:1` | States: `RESEARCH → CANDIDATE → ROBUST_CANDIDATE → PAPER_CANDIDATE → PAPER_VALIDATED → PRODUCTION_CANDIDATE → PRODUCTION → NO_CHAMPION`. Promotion guarded by `ChampionEvaluator.compare()` (must beat incumbent; `decide_promotion()` at `src/champion/pipeline.py:456`). Demotion via `review_champion()` / `rollback_if_degraded()` with `HoldoutLock` audit. |
| NO_CHAMPION | `src/champion/pipeline.py:538` | First-class success when zero `eligible`. `decide_promotion()` returns `{"champion":"NO_CHAMPION","state":"NO_CHAMPION","fragile":[...],"integrity":{...}}`. Production invariant: `production_champion_id()` returns `None` when champion is `under_review` (`src/champion/pipeline.py:376`). |

**Lifecycle state machine:**

```text
RESEARCH ──integrity pass──► CANDIDATE ──robustness pass──► ROBUST_CANDIDATE
    │                               │
    │ no edge proven                └─cost_robust?─┐
    ▼                                              ▼
NO_CHAMPION                              PAPER_CANDIDATE ──7d clean──► PAPER_VALIDATED
    ▲                                              │
    │ all `                    ──────────────────────┘
    └──── eligible empty (valid terminal) ──beats champion?──► PRODUCTION_CANDIDATE ──readiness gate──► PRODUCTION
                                                                   │
                                                          degraded → rollback / under_review → NO_CHAMPION
```

| PASS | FAIL |
|------|------|
| At least one `eligible` + `Tournament.rank()` non-empty + `ChampionEvaluator.compare` `qualified=True` → `promoted=True` with `from→to`. `PRODUCTION_CANDIDATE` after Paper 7d+ health. | `eligible` empty → `NO_CHAMPION` (not an error). `compare.qualified==False` → not promoted. Flags: `champion_under_review` (`review_champion` at `src/champion/pipeline.py:660`), `rolled_back` path. |

---

### Layer 6 — RISK

**Purpose:** Single canonical risk authority. No order reaches Execution without a `RiskDecision`.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Policy (single canonical) | `src/risk/policy.py:1`, `src/risk/risk_policy.py:1`, `src/risk/policies.py:1` | Declarative risk policy: `max_drawdown`, `max_total_exposure`, `max_position_exposure`, `risk_per_trade=1% (max 3% per trade)`, `max 5% total exposure`, `rule 3-5-7` (profit ≥ loss*1.07), `40% reserve`. Cross-margin vs isolated margin semantics. |
| Orchestrator | `src/risk/risk_orchestrator.py:70` | `RiskOrchestrator(DrawdownGuard, ExposureManager, PositionSizer)` — single `evaluate(signal, equity, current_exposure, context)` → `RiskDecision(allowed, quantity, sl, tp, reason, metadata)`. Flow: `DrawdownGuard → PositionSizer → ExposureManager → FactorRiskGate → TotalExposure`. Fail-closed: `REJECT` on any violation. |
| Context | `src/risk/risk_context.py:1`, `src/risk/correlation.py:1`, `src/risk/factor_risk.py:1` | `RiskContext(effective_exposure, open_positions, correlation_matrix, factor_map, betas)`. `CORR_ADJUSTED_LIMIT=15%`, `max_factor_concentration 70%`, `Herfindahl<0.60`. Handles flip `projected_exposure` correctly. |

**Invariants:**

- `1% risk per trade (max 3%)`, `max 5% total exposure across open trades`, `max 5% per asset`, `40% absolute reserve` (Core Project Rules).
- `RiskOrchestrator` is the **only** entry to position sizing — strategy must not compute `quantity` itself.
- `fail-closed`: any guard raises → `RiskDecision(allowed=False, quantity=0)` (`src/risk/risk_orchestrator.py:159`).

| PASS | FAIL |
|------|------|
| `drawdown_result.allowed && exposure_result.within_limit && position_sizer.ok && factor_gate.allowed && can_open_position(equity, current_exposure, notional)` → `allowed=True, quantity=approved_quantity`. | `max_drawdown` exceeded, `exposure_limit_exceeded`, `sizing_failed`, `Factor risk blocked` (corr-adjusted >15% or Herfindahl breach), `total_exposure_blocked` → `allowed=False`, `reason` surfaced to Execution kill switch logs. |

---

### Layer 7 — EXECUTION

**Purpose:** Exchange-faithful execution with reconciliation and capital-safe emergency halt.

| Canonical Module | Path | Responsibility |
|----------------|------|----------------|
| Exchange / Adapters | `src/execution/binance_adapter.py:1` (`BinanceRestAdapter`, `BinanceWebSocketAdapter`), `src/market_data/fanout.py:6` | REST (HMAC-SHA256, `X-MBX-USED-WEIGHT` rate limit, `timeout=10s`) + WS user data stream with auto-reconnect, keepalive `listenKey`. `withdraw` permission guard (`verify_no_withdraw_permission`). |
| Order Lifecycle | `src/execution/orders.py:1` (`OrderIntent`), `src/execution/order_manager.py:1` (`OrderManager` + `OrderManagerConfig`) | `OrderIntent(symbol, side, order_type, quantity, price, reduceOnly, tif, clientOrderId=UUID, metadata)` with `NEW→SUBMITTED→PARTIAL/FILLED/CANCELED/REJECTED/EXPIRED`. Callbacks `_submit_callback/_cancel_callback`, retry `3×` exponential backoff, expiry `1h`. |
| Fill Model | `src/execution/fill_model.py:23` | `LimitFillModel.attempt_fill()` as above — queue+volume+depth; spread as half-spread cost. |
| Execution Engine | `src/execution/execution_engine.py:98` (`ExecutionEngine`, `ExecutionMode`, `ExecutionConfig`) | Modes `PAPER|DRY_RUN|LIVE`. Flow `submit_intent → safety_check → OrderManager → Binance/Paper → WS callbacks → reconciliation_loop`. `ExecutionStats` Prometheus + structured JSON logs. `_safety_check` enforces kill switch, `max_drawdown=10%`, `max_daily_loss=5%` (`src/execution/execution_engine.py:283`). |
| Reconciliation | `src/execution/reconciliation_engine.py:1`, `src/execution/execution_engine.py:663` | Every `30s`: compare `position_qty ±0.001`, `balance ±$1.0`; ghost positions auto-added; `never auto-flatten` (manual only); `on_reconciliation_fix` callback. |
| Emergency Flatten | `src/execution/execution_engine.py:788` (`emergency_stop`) | 7-step futures-safe HALT: `block new orders (HALTED flag) → cancel all (local+exchange) → query positions → flatten market reduceOnly → verify flat (retries 5×, interval 0.8s) → verify balances → enter HALTED (manual `resume_from_halt`)`. Verified `remaining_positions=={}` else critical error. |

| PASS | FAIL |
|------|------|
| `safety_check(True) && OrderManager.submit_intent → Order(NEW) → filled via exchange/WS → reconciliation 0 drift`; `ExecutionMode` appropriate for deployment stage (see §3). | `HALTED after emergency_stop → submit_intent raises ExecutionError`; `Safety check failed` → `ExecutionError(reject)`; `Rate limit 429→ backoff`; `Auth401→ kill switch+stop`; `Reconciliation drift > tolerance → fix local + alert + metric`; `Flatten verification failed after 5 attempts → CRITICAL`. |

---

## 3. Deployment Track

```text
           Execution (ENGINE)
                   |
          PAPER ───┼───► TESTNET ──► PRODUCTION
           (Phase 1)   (Phase 2)     (Phase 3)
```

| Stage | Mode (`src/execution/execution_engine.py:43`) | Preconditions (gates) | Observability |
|-------|-----------------------------------------------|-----------------------|---------------|
| **PAPER** | `ExecutionMode.PAPER` | Data Governance ✅ → Research ✅ → Integrity ✅ → Robustness ✅ → `ChampionGovernance` promoted (or `NO_CHAMPION` acknowledged) → `RiskOrchestrator` approval on shadow intents. `PaperTradingRunner` (`src/paper_trading_runner.py:1`) with `RiskOrchestrator` gate. | `ExecutionStats{fills, intents, reconciliations}`, `PaperTradingPerformance`, `PaperTradingMonitor`. Prometheus `execution_*`, JSON logs. |
| **TESTNET** | `ExecutionMode.DRY_RUN` | PAPER 7d+ clean + reconciliation `0 drift` + `binance_adapter.validate_order` ok + `Safe Startup Controller` green (`src/quantai_production_safe_startup_controller.py:1`). `BINANCE_TESTNET=true` (`config/settings.py:21`). | Same as PAPER + Binance `markPrice` live, `keepalive_loop`, WS reconnect metrics. |
| **PRODUCTION** | `ExecutionMode.LIVE` | TESTNET parity validated + `Production Readiness Gate` (`src/quantai_production_readiness_gate.py:1`) + `withdraw permission` guard negative (`src/execution/binance_adapter.py:verify_no_withdraw_permission`) + `ReconciliationEngine` green + ops runbook. | `quantai_production_observability.py:1`, `quantai_production_runtime_supervisor.py:1`, incident `P0→ auto-stop`, `P1/P2` alerts, `resume_from_halt` requires operator. |

**Rule:** No skip — `PAPER → TESTNET → PRODUCTION` is linear. `LIVE` never reached without `DRY_RUN` success.

---

## 4. Cross-Cutting Invariants

| Invariant | Enforcement |
|-----------|-------------|
| **Single canonical per domain** | `DatasetRegistry` only, `src/walk/walk_forward_engine.py:28` canonical — facades `src/walk_forward_engine.py:1` are `DEPRECATED` re-exports (`from src.walk.walk_forward_engine import *`) remove after `2026-12`. `RiskOrchestrator` single facade — no direct `DrawdownGuard` calls from Strategy. Lint via `lint_spine.py` / `importlinter`. |
| **Immutability** | Hash is identity (`src/research/dataset_registry.py:74`), file `chmod 0o444`, `verify()` on every `load()`. |
| **Determinism** | `LimitFillModel._deterministic_random()` hash-based, `random_state=42` in `config/settings.py:152`, `py_compile` + `ast.parse` gate. |
| **Audit trail** | `ExperimentRegistry` (`oos_touched`), `DatasetRegistry SHA256`, `HoldoutLock.touch_history`, `ChampionPipeline.history` (`HistoryEvent`), `control_plane/audit_logger.py:1`. |
| **OOS Budget** | `ResearchBudget(101 → BudgetExceeded)` + `oos_touched` flag; holdout `Lock.assert_not_touched_during_development()` (`src/research/nested_research_pipeline.py:88`). |
| **Fail-closed conventions** | Risk `REJECT` on violation, Integrity hard gates before Tournament, Execution `HALTED` blocks new orders. |

---

## 5. Data Flow — End-to-End

```text
CCXT / Binance
    ↓  src/data_loader.py:1 + src/historical_downloader.py:1
OHLCV DataFrame[ts, o,h,l,c,v]  (UTC, monotonic, no NaN)
    ↓  src/data/data_gates.py:120  +  src/research/dataset_registry.py:218
GOVERNED PARQUET (hash + read-only)
    ↓  src/indicators.py::add_indicators  →  src/feature_engine.py + src/labeling.py (TB2)
FeatureVector + Triple-Barrier label
    ↓  src/ml_engine.py (PurgedKFold)  →  src/model_manager.py  →  src/strategy/signal_generator.py
SignalResult (HOLD/BUY/SELL, confidence, entry/sl/tp)
    ↓  src/walk/ + src/validation/nested_walk_forward.py  (Nested WF, inner Optuna on OUTER TRAIN only)
OOS Windows (outer)
    ↓  src/validation/bootstrap.py + src/research/pbo.py + src/research/dsr.py + src/research/white_reality_check.py
Integrity metrics (Sharpe, PBO, DSR, WRC p, IS-OOS consistency)
    ↓  src/champion/research_integrity.py::assess  (7 hard gates)
eligible / rejected
    ↓  src/validation/cost_stress.py + src/execution/fill_model.py + src/research/regime_stability.py + src/validation/bootstrap.py
Robustness (cost_robust?, fill realism, regime works/fails)
    ↓  PASS / FAIL
Only eligible → src/strategy_tournament.py + src/champion/pipeline.py::decide_promotion
Champion or NO_CHAMPION
    ↓  (if champion)  src/risk/risk_orchestrator.py::evaluate  → RiskDecision
OrderIntent (quantity, sl, tp, client_order_id UUID)
    ↓  src/execution/execution_engine.py + src/execution/order_manager.py + src/execution/binance_adapter.py + reconciliation
Order / Fill / Position @ Exchange
    ↓  PAPER ⇒ TESTNET ⇒ PRODUCTION
Live PnL / Metrics / Observability
```

---

## 6. PASS/FAIL Summary Table

| Layer | PASS Condition (all must hold) | FAIL Consequence |
|-------|-------------------------------|------------------|
| Data Governance | 9 gates clean + SHA256 stored + read-only + verified copy | `DataGateError` / `ValueError` → registration rejected; research blocked |
| Research Engine | 25 ACTIVE features + TB2 HOLD on ambiguous + ML CV purged + signal schema valid | `ast.parse` / schema / CV leakage failure → pipeline abort |
| Research Integrity | 7 hard gates pass (incl. WRC p<0.05 when N≥100, DSR≥0.95, PBO<0.6, IS-OOS pf_ratio≥0.6) | Hard `rejected[failed_stage]` — never reaches Tournament |
| Robustness Engine | `cost_robust` true (PF≥1@1.5×), fill_prob<1 modeled, latency/regime/bootstrap within tolerance | `fragile_ids` flagged; promotion blocked; champion stays `NO_CHAMPION` or incumbent |
| PASS/FAIL gate | Central chokepoint: `eligible` non-empty entered ranking | `eligible` empty → `{"champion":"NO_CHAMPION","state":"NO_CHAMPION"}` valid success |
| Champion Governance | Tournament ranked + `ChampionEvaluator.compare` qualified + Paper ≥7d (>=30d for Long-Run) | `not promoted`, `under_review`, `rolled_back`, or `NO_EDGE_PROVEN` deferral |
| Risk | All sub-guards allowed + corr-adjusted ≤15% + `can_open_position` true | `RiskDecision(allowed=False)` → `ExecutionError(Safety check failed)` |
| Execution | `submit→filled` + reconciliation 0 drift within `±0.001 / ±$1` + `verify_flat` after halt | `ExecutionError` / `HALTED` / alert + Prometheus `kill_switch_activations_total` |
| Deployment | PAPER green → TESTNET (`DRY_RUN` validate) → `ReadinessGate` green → LIVE | Promotion halted; `SafeStartupController` fails startup; `DIST` never deployed |

---

## 7. Enforcement & Verification

- **Import boundaries:** `docs/architecture/` + `.importlinter` — `core-trading` → `validation` → `execution-boundary` → `governance`. CI forbids Research → Execution bypass.
- **AST / hygiene gate:** Every `*.py` must satisfy `ast.parse` (`pyproject.toml` / `scripts/verify_final.py`).
- **Contracts CI:** `pytest tests/test_backtest_*.py tests/test_trade_engine_*.py -v` + `validation.gate` (`docs/architecture/validation.md:106`).
- **Observability:** Prometheus `execution_intents_total{mode,result}`, `execution_fills_total`, `execution_latency_seconds`, `reconciliation_fixes_total`; health `src/monitoring/health.py:1`; structured logs `src/monitoring/logging_config.py:1`.
- **Recovery:** `Emergency Flatten` HALT → manual `resume_from_halt()` (`src/execution/execution_engine.py:1117`).

---

## 8. References

- Prior layers doc: `docs/ARCHITECTURE_LAYERS.md:1` (updated to mirror this spine).
- Contracts: `docs/architecture/core-trading.md:1`, `docs/architecture/validation.md:1`, `docs/architecture/execution-boundary.md:1`, `docs/architecture/governance.md:1`.
- Registry status: `docs/architecture/MODULE_STATUS.md:1`.
- Settings single truth: `config/settings.py:285` (`Settings` — `version="5.1.0"` → `6.0` on next bump; `RiskSettings` canonical).

---
*QuantAI 6.0 — Data-governed, integrity-first, robustness-proven, champion-gated, risk-closed, execution-reconciled. No edge → NO_CHAMPION.*
