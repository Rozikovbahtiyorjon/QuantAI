# QuantAI Architecture Layers — v6.0 (Canonical)

**Rule: Governance must never bypass Risk/Execution. No path reaches Execution without passing PASS/FAIL.**

Mirrors `docs/ARCHITECTURE_V6.md:1` — this file is the compact spine; V6 is the full spec.

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

---

## Layer 1 — DATA GOVERNANCE
- **Scope:** `data/` · exchange normalization · quality · immutability
- **Canonical:** `src/research/dataset_registry.py:57` (`DatasetRegistry`) + `src/data/data_gates.py:120` (`DataGates`)
- **Modules:** `DatasetRecord(dataset_id, hash SHA256, symbol, timeframe, schema/feature/label_version, start/end, row_count)`, `_make_readonly()` (`chmod 0o444` + Windows `FILE_ATTRIBUTE_READONLY`), `hash_file()`, `verify()`/`load()` (recompute hash, return copy), `DataGates.validate()` (9 gates: duplicate timestamps, missing bars, monotonicity, OHLC, zero-volume, bad prices, timezone UTC, outages, gaps)
- **PASS/FAIL:**

| PASS | FAIL |
|------|------|
| All `DataGates` clean for declared `timeframe`; hash stored; file read-only; `verify()` recomputed == stored on every `load()` | `DataGateError` (any gate) or `ValueError` hash mismatch / `Immutable violation` → registration rejected, research blocked |
- **Invariant:** `register()` runs gates before hashing; hash is identity, not path.

## Layer 2 — RESEARCH ENGINE
- **Scope:** `features · labels · ML · strategies · backtest`
- **Canonical:** `src/feature_engine.py:1` (25 ACTIVE), `src/labeling.py:1` (Triple Barrier v2), `src/ml_engine.py:1` + `src/model_manager.py:1`, `src/strategy/signal_generator.py:1`
- **Modules:** Features (`FeatureVector` last-row, normalized; drift via `src/feature_store/drift.py:1`), Labels (ambiguous `both barriers same bar → HOLD` discard; `ret` uses `upper/lower` price not `close`), ML (XGBoost, `PurgedKFold`/`CombinatorialPurgedCV` via `src/validation/purged_kfold.py:1`, `predict_proba` `[SELL,HOLD,BUY]`), Strategies (`SignalResult(signal, confidence, entry, sl, tp)` via `evaluate_market→predict_ml→fuse_ai_ml→order_flow_gate`, `MIN_CONFIDENCE=60%`, `AI_WEIGHT=0.6/ML=0.4`)
- **Integrity link:** `ExperimentRegistry` (`src/research/experiment_registry.py:1`) tracks every trial with `oos_touched`
- **PASS/FAIL:** Feature schema == 25; label no wick proxy; CV with purge/embargo; strategy emits valid `SignalResult` — else `ast.parse`/schema fail → pipeline abort.

## Layer 3 — RESEARCH INTEGRITY
- **Scope:** `Nested WF · Purging · PBO · DSR · Reality Check · OOS Budget`
- **Canonical:** `src/champion/research_integrity.py:157` (`ResearchIntegrityEngine`) — enforces `Research → Integrity Checks → Statistical Validation → IS-OOS → ML Calibration → Robustness → Selection Adjustment → Regime Stability → Tournament → Champion` (Tournament cannot ignore integrity)
- **Modules:**
  - **Nested WF** `src/validation/nested_walk_forward.py:33` + `src/research/nested_research_pipeline.py:163` — `OUTER TRAIN → INNER WF+Optuna → OUTER OOS` (inner runs on `OUTER TRAIN` only; `HoldoutLock` at `src/research/nested_research_pipeline.py:69` ensures `FINAL HOLDOUT` never touched in `DEVELOPMENT`)
  - **Purging** `src/validation/purged_kfold.py:1` — `PurgedKFold` / `CombinatorialPurgedCV` (`config/settings.py:142` `cv_type`, `n_splits=5`, `embargo_pct=1%`)
  - **PBO** `src/research/pbo.py:1` (+ `src/validation/bootstrap.py:1` fallback) — CPCV `compute_pbo(returns_df)` / `compute_pbo_from_sharpes()`; gate `PBO<0.6`
  - **DSR** `src/research/dsr.py:1` — `deflated_sharpe_ratio(sharpe, n_trials, T, skew, kurt)` Bailey&Prado 2014 + `expected_max_sharpe`; strict threshold `0.95` (permissive `0.5`); window `net_pct` used to infer `skew/kurt` when not supplied
  - **Reality Check** `src/research/white_reality_check.py:1` — `returns_df_from_evaluations()` → `T×K` DataFrame → `spa_test` (Hansen studentized) / `white_reality_check`; `B=1000`, `q=0.1`, gate `p<0.05` when `n_trials≥100` (`wrc_enabled`, `wrc_min_trials=100` at `src/champion/research_integrity.py:64`)
  - **OOS Budget** `src/research/research_budget.py:1` (`ResearchBudget` → `BudgetExceeded` at 101st) + `src/research/experiment_registry.py:1` (`oos_touched` per trial) + `HoldoutLock.assert_not_touched_during_development()` (`src/research/nested_research_pipeline.py:88`)
- **Gate thresholds (from `src/champion/research_integrity.py:44` `IntegrityConfig`):**

| Gate | PASS | FAIL (hard before Tournament) |
|------|------|-------------------------------|
| Integrity Checks | `pf_median≥1.05`, `profitable_window_share≥0.45`, `maxdd≥-15%`, `net_median≥0%`, `trades≥30`, `net_std≤10%`; `trades<30→INSUFFICIENT_SAMPLE` (via `src/champion/evaluation_pipeline.py:1` `PromotionRules`) | Removed before Tournament |
| Statistical | `sharpe_median≥0`, bootstrap `p_sharpe≤0.05`, `PBO<0.6`, `DSR≥threshold`, `WRC/SPA p<0.05` (when `K≥2`, `T≥10`, `N≥100`) | Hard FAIL |
| IS-OOS | `pf_ratio OOS/IS≥0.60`, `pf_deterioration≤50%`, `sharpe_deterioration≤70%`, `IS PF≥1.3⇒OOS PF≥1.0` | Blocks breakout IS-good→OOS-bad |
| ML Calibration | bucket monotonic `spearman>0.5`, `pearson>0.3`, `cal_error<0.5`, plus `Brier<0.25`/`ECE<0.10` when probabilistic | Permissive warn / Strict fail |
| Robustness (selection) | `cost_robust=True`, `mc_score≥0.3`, `stress≥0.3` — see Layer 4 | Flagged fragile |
| Selection Adjust. | `sharpe_corrected=sharpe/sqrt(1+log N)` > threshold; Bonferroni `p_adj≤0.05` (strict) | Blocks best-of-many luck |
| Regime Stability | 7 regimes (Bull/Bear/Sideways/HighVol/LowVol/Crash/Recovery via `src/research/regime_stability.py:1`); `n_positive≥3` with `min_trades_per_regime=5` | Reports `works/fails`; strict fails if `<3` |

- **Pipeline hook:** `src/champion/pipeline.py:503` `integrity_report = integrity_engine.assess(evaluations)` → `eligible` only reaches `StrategyTournament.rank()`; empty → `NO_CHAMPION`. `src/champion/pipeline.py:472` `EdgeProvenGate` defers `Tournament` until strict edge proven when `defer_champion_until_edge_proven=True`.

## Layer 4 — ROBUSTNESS ENGINE
- **Scope:** `Cost Stress · Slippage · Latency · Regimes · Bootstrap`
- **Canonical:** `src/validation/cost_stress.py:31` + `src/execution/fill_model.py:23` + `src/research/regime_stability.py:1` + `src/validation/bootstrap.py:1`
- **Modules:** Cost (`cost_stress(df)` at `1.0/1.25/1.5/2/3x` via `config/settings.py:51` commission/slippage; `is_cost_robust()==PF>1@1.5x`), Slippage (`LimitFillModel(queue_ahead_pct=0.3, min_volume_ratio=1.5)` handle queue+volume+depth, `fill_prob<1`, deterministic hash at `src/execution/fill_model.py:38`), Latency (`50/100/250/500ms/1s/3s` stress), Regimes (7-class, per-regime expectancy, `when WORKS/when FAILS`), Bootstrap (`block_bootstrap_sharpe(block=6, n_iter=200)` + `ci_lower`, `pbo_combinatorial`)
- **PASS/FAIL:** `is_cost_robust True`, fill realism retained edge, latency OOS within tolerance, ≥3 regimes positive, `p_sharpe≤0.05` — else `fragile_ids` flagged and promotion blocked.

## PASS / FAIL — Central Gate

```text
         ROBUSTNESS ENGINE
                |
           PASS / FAIL      ← single chokepoint
                |
     CHAMPION GOVERNANCE
```

- Implementation: `src/champion/research_integrity.py:956` `IntegrityReport(eligible, rejected)` + `src/champion/pipeline.py:504` `decide_promotion()`.
- **PASS** → `eligible` non-empty → ranked by `src/strategy_tournament.py:1`.
- **FAIL** → `rejected` with `failed_stage/reasons`; if `eligible` empty → `{"champion":"NO_CHAMPION","state":"NO_CHAMPION"}` (valid success, not error). WRC computed batch-wise at `src/champion/pipeline.py:498`.
- **Deferral variant:** `EdgeProvenGate` (`src/research/edge_validation.py:1`) — when `defer_champion_until_edge_proven=True`, state stays `RESEARCH` with `NO_EDGE_PROVEN` until strict edge exists.

## Layer 5 — CHAMPION GOVERNANCE
- **Scope:** `Candidate · Paper · Production Candidate · NO_CHAMPION`
- **Canonical:** `src/champion/pipeline.py:159` (`ChampionPipeline`), `src/strategy_tournament.py:1`, `src/champion/evaluation_pipeline.py:1`, `src/champion_governance_engine.py:1`
- **Modules:**
  - **Candidate** — `StrategyRegistry` (`src/strategy_genome.py:1`, `src/strategy_bank.py:1`) + `CandidateSpec(factory, params)` + `evaluate_candidate(spec, df, train_size/test_size/step_size)` → `{metrics, windows}` (`src/champion/evaluation_pipeline.py:1`); `PromotionRules.evaluate_flags()` per-candidate PASS set.
  - **Tournament** — integrity-filtered `eligible` only → `StrategyTournament.rank()` via `vector_to_tournament_evaluation()` (pf→`profit_factor`, net→`total_return`, etc.; `NOT_EVALUATED` MC/stress → `0.0` not `0.5` audit fix at `src/champion/pipeline.py:86`).
  - **Paper** — `src/paper_trading_runner.py:1` / `src/paper_trading_engine.py:1` / `src/paper_trading_session.py:1` + `src/validation/paper_30d.py:49` / `src/validation/long_run.py:1` + `src/paper_trading_quality_gate.py:1`; gates ≥7d paper, ≥30d Long-Run (`Sharpe>0.5/1.0`, `DD<8%`, `win 45-55%` @ `2:1 RR`, `PF≥1.3`).
  - **Production Candidate** — `ChampionPipeline.STATES` (`src/champion/pipeline.py:425`): `RESEARCH → CANDIDATE → ROBUST_CANDIDATE → PAPER_CANDIDATE → PAPER_VALIDATED → PRODUCTION_CANDIDATE → PRODUCTION → NO_CHAMPION`; promotion guarded by `ChampionEvaluator.compare()` (`src/champion_evaluator.py:1`) — must beat incumbent; `review_champion()`/`rollback_if_degraded()` with `HoldoutLock` audit.
  - **NO_CHAMPION** — `src/champion/pipeline.py:538` first-class terminal: `decide_promotion()` returns `NO_CHAMPION` when zero `eligible`; `production_champion_id()` returns `None` when champion `under_review` (`src/champion/pipeline.py:376`); research view `research_champion_id()` retains last promoted for analysis.
- **Lifecycle:**

```text
RESEARCH ──integrity pass──► CANDIDATE ──robustness pass──► ROBUST_CANDIDATE
    │                               │
    │ no edge proven                └─cost_robust?─┐
    ▼                                              ▼
NO_CHAMPION                              PAPER_CANDIDATE ──7d clean──► PAPER_VALIDATED
    ▲                                              │
    │                          ──────────────────────┘
    └──── eligible empty (valid terminal) ──beats champion?──► PRODUCTION_CANDIDATE ──readiness gate──► PRODUCTION
```

- **PASS/FAIL:** At least one `eligible` + non-empty `rank()` + `compare.qualified=True` → `promoted=True`; else `not promoted` / `under_review` / `rolled_back` / `NO_EDGE_PROVEN`.

## Layer 6 — RISK
- **Scope:** `portfolio · position sizing · drawdown · exposure · correlation`
- **Canonical:** `src/risk/risk_orchestrator.py:70` (`RiskOrchestrator`) + `src/risk/policy.py:1` / `src/risk/risk_policy.py:1` / `src/risk/correlation.py:1` + `src/risk/risk_context.py:1`
- **Policy (single canonical):** `RiskSettings` (`config/settings.py:251`): `drawdown_limit=10%`, `max_total_exposure=60%`, `max_position_exposure=5%`, `risk_per_trade=1% (max 3% per trade)`, `max 5% total exposure across open trades`, `max 5% per asset`, `40% reserve`, `rule 3-5-7` (profit ≥ loss×1.07), `max_open_positions=1`. `DEPRECATED` `AccountSettings` (`config/settings.py:34`) kept only for compat — canonical is `RiskSettings` (sync warns drift at `config/settings.py:307`).
- **Orchestrator flow:** `DrawdownGuard.evaluate(equity)` → `PositionSizer.calculate(balance, risk%, entry, stop, leverage)` → `ExposureManager` (cap per-position) → `FactorRiskGate` (`src/risk/factor_risk.py:1`, corr-adjusted `15%`, concentration `70%`, Herfindahl `<0.60`, `RiskContext`) → `can_open_position(equity, current_exposure, notional)` → `RiskDecision(allowed, quantity, sl, tp, reason, metadata)` (`src/risk/risk_orchestrator.py:43`). Fail-closed: `allow=False` on any violation (`src/risk/risk_orchestrator.py:159`).
- **PASS/FAIL:** `allowed=True` → `quantity=approved_quantity`; else `allowed=False` with `exposure_limit_exceeded` / `Factor risk blocked` / `drawdown exceeded` / `sizing_failed` → Execution kill switch logs.

## Layer 7 — EXECUTION
- **Scope:** `orders · exchange · reconciliation · latency · fills`
- **Canonical:** `src/execution/execution_engine.py:98` (`ExecutionEngine`, `ExecutionMode`, `ExecutionConfig`) + `src/execution/orders.py:1` (`OrderIntent`) + `src/execution/order_manager.py:1` + `src/execution/binance_adapter.py:1` + `src/execution/reconciliation_engine.py:1` + `src/trade_engine.py:1` (backtest) + `src/execution/fill_model.py:23` (queue-aware)
- **Modules:**
  - **Orders** `src/execution/orders.py:1` — `OrderIntent(symbol, side, order_type, quantity, price, reduceOnly, tif, clientOrderId=UUID, metadata)` with `NEW→SUBMITTED→PARTIAL/FILLED/CANCELED/REJECTED/EXPIRED`; idempotent via `metadata`.
  - **OrderManager** `src/execution/order_manager.py:1` — `submit_intent→Order`, `cancel_order`, `get_active_orders`; callbacks `_submit_callback/_cancel_callback`; retry `3×` exponential backoff; expiry `1h`; state persistence SQLite/Redis.
  - **Exchange** `src/execution/binance_adapter.py:1` — `BinanceRestAdapter(place_order, cancel_order, get_balance/positions/markPrice, validate_order, keepalive_loop)` (HMAC-SHA256, `X-MBX-USED-WEIGHT`, `timeout=10s`) + `BinanceWebSocketAdapter` (auto-reconnect, `on_order_update/account/position/balance/error`); guard `verify_no_withdraw_permission()`.
  - **Fill Model** `src/execution/fill_model.py:24` — deterministic `attempt_fill(limit_price, side, bar_high/low/volume, avg_volume, spread, symbol, ts, order_id)` as in Layer 4.
  - **ExecutionEngine** `src/execution/execution_engine.py:43` — modes `PAPER|DRY_RUN|LIVE`; `submit_intent→safety_check(killSwitch, dailyPnL, drawdown)→OrderManager→Binance/Paper→WS callbacks→reconciliation_loop`; `ExecutionStats` + Prometheus `execution_*` + JSON logs; safety `max_drawdown=10%`, `max_daily_loss=5%` (`src/execution/execution_engine.py:283`).
  - **Reconciliation** `src/execution/reconciliation_engine.py:1` / `src/execution/execution_engine.py:663` — every `30s`: `position_qty ±0.001`, `balance ±$1.0`; ghost positions auto-added; `never auto-flattens`; `on_reconciliation_fix(symbol, {type,old,new})`; report `{timestamp, fixes_applied, issues, positions, balance}`.
  - **Emergency Flatten** `src/execution/execution_engine.py:788` `emergency_stop()` — 7-step futures-safe HALT: `1:block new orders (HALTED flag) → 2:cancel all (local+exchange) → 3:query positions → 4:flatten market reduceOnly → 5:verify flat (5× retries @0.8s) → 6:verify balances → 7:HALTED` requiring manual `resume_from_halt()` (`src/execution/execution_engine.py:1117`); remaining `position!=0` after 5 attempts → `CRITICAL`.
- **PASS/FAIL:** `safety_check True` + `filled via exchange/WS` + reconciliation `0 drift` + `verify_flat` after halt — else `ExecutionError(Safety check failed)` / `HALTED` blocks new orders / `401→kill switch+stop` / `429→backoff` / `drift>tol→fix local+alert`.

---

## Deployment Track — PAPER → TESTNET → PRODUCTION

| Stage | Mode (`src/execution/execution_engine.py:43`) | Preconditions | Observability |
|-------|-----------------------------------------------|---------------|---------------|
| **PAPER** | `PAPER` | Data→Research→Integrity→Robustness PASS; Champion promoted or `NO_CHAMPION` acknowledged; `RiskOrchestrator` shadow approval via `PaperTradingRunner` | `src/paper_trading_monitor.py:1`, `src/monitoring/metrics.py:1`, JSON logs |
| **TESTNET** | `DRY_RUN` | PAPER 7d+ clean + reconciliation `0 drift` + `BinanceRestAdapter.validate_order` ok + `quantai_production_safe_startup_controller.py:1` green; `BINANCE_TESTNET=true` (`config/settings.py:21`) | Same + live `markPrice` + WS metrics |
| **PRODUCTION** | `LIVE` | TESTNET parity + `quantai_production_readiness_gate.py:1` green + `verify_no_withdraw_permission` negative + `quantai_production_observability.py:1` + `resume_from_halt` operator-gated | `quantai_production_runtime_supervisor.py:1`, incident `P0→auto-stop`, `P1/P2→alert`, `reconciliation_fixes_total` |

**Rule:** Linear `PAPER → TESTNET → PRODUCTION` — `LIVE` never reached without `DRY_RUN` success.

---

## Canonical rule (Audit #2)

For each domain exactly ONE canonical implementation (e.g., `src/walk/` only for Walk-Forward, `src/risk/risk_orchestrator.py:70` for risk). Legacy facades (`src/walk_forward_*.py` at repo root, `src/risk_manager.py:1` where applicable) are thin re-exports marked:

```python
# DEPRECATED: canonical is src/walk/walk_forward_engine.py — remove after 2026-12
from src.walk.walk_forward_engine import *  # noqa
```

Import boundaries enforced by `.importlinter` / `scripts/verify_final.py:1`. Full spec: `docs/ARCHITECTURE_V6.md:1`; contracts: `docs/architecture/core-trading.md:1`, `docs/architecture/validation.md:1`, `docs/architecture/execution-boundary.md:1`, `docs/architecture/governance.md:1`; status: `docs/architecture/MODULE_STATUS.md:1`.
