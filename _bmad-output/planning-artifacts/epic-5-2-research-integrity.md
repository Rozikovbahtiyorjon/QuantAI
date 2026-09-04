# Epic 5.2 — Research Integrity & Alpha Validation Engine

## Epic 1: Engineering Hygiene (P0)
Fix `pyproject.toml`, `paper_trading_runner.py`, BOM files, `analyze.py` syntax, ensure `every *.py AST MUST PASS` CI gate.

### Story 1.1: Fix pyproject.toml TOML syntax
**As developer** I want valid `pyproject.toml` so that `pytest -q` starts.
**Acceptance:** `tomllib.load(pyproject.toml)` OK, `pytest -q` collects 208 tests.

### Story 1.2: Fix BOM and indentation
**As developer** I want no BOM and no `unexpected indent` so AST passes.
**Acceptance:** `ast.parse` 0 bad files.

### Story 1.3: Fix audit scripts
**As developer** I want `analyze.py` and `analyze_project.py` valid so audit tool self-validates.
**Acceptance:** `ast.parse` OK.

## Epic 2: Configuration Integrity (P1)
Unify `AccountSettings`/`RiskSettings`, enforce fail-fast `settings.strategy.*`.

### Story 2.1: Deduplicate Account vs Risk
**As risk manager** I want single canonical `RiskSettings` so no drift.
**Acceptance:** `Settings.model_post_init` warns and syncs, `AccountSettings` marked deprecated.

### Story 2.2: Fix SignalConfig.from_settings
**As quant** I want `SignalConfig.from_settings` reads `settings.ml.ml_enabled` and `settings.strategy.*` directly.
**Acceptance:** Changing `strategy.ai_weight=0.8` propagates, `getattr(...,fallback)` removed for critical params.

### Story 2.3: Fix MLFusion.from_settings
**As ML engineer** I want `FusionConfig` reads `settings.strategy.ai_weight` correctly.
**Acceptance:** `FusionConfig.from_settings().ai_weight == settings.strategy.ai_weight`.

## Epic 3: Label & Feature Integrity (P1)
Fix Triple Barrier ambiguous handling and barrier price, generate `FEATURE_SCHEMA.json`.

### Story 3.1: Triple Barrier ambiguous
**As researcher** I want `both barriers same bar → ambiguous → HOLD` not wick proxy.
**Acceptance:** `ambiguous_mode discard` yields `barrier=ambiguous target 0`.

### Story 3.2: Triple Barrier barrier price
**As researcher** I want `ret` uses `upper/lower` price not `close`, with `entry/exit/gross/net`.
**Acceptance:** TP 1% candle close 0.2% → ret 1%.

### Story 3.3: Feature schema
**As researcher** I want `FEATURE_SCHEMA.json` with 25 ACTIVE + 5 PLANNED.
**Acceptance:** File exists, `total_features 25`.

## Epic 4: Walk-Forward & Experiment Integrity (P1)
Add sample guard, Nested WF, Experiment/Dataset Registry, Budget.

### Story 4.1: Insufficient sample guard
**As quant** I want `trades<30 → INSUFFICIENT_SAMPLE` not `PF=inf`.
**Acceptance:** `BacktestResult.pf_or_insufficient == "INSUFFICIENT_SAMPLE"` when 2 trades.

### Story 4.2: Nested Walk-Forward
**As researcher** I want `OUTER TRAIN → INNER WF+Optuna → OUTER OOS` no leakage.
**Acceptance:** `NestedWalkForward.run` exists.

### Story 4.3: Experiment/Dataset Registry + Budget
**As governance** I want `ExperimentRegistry` with `oos_touched`, `DatasetRegistry SHA256`, `ResearchBudget` limits.
**Acceptance:** `src/research/*.py` exists, `BudgetExceeded` raised at 101st experiment.

## Epic 5: Risk & Portfolio (P2)
Correlation-adjusted exposure, leverage metrics.

### Story 5.1: Correlation exposure
**As risk manager** I want `BTC+ETH+SOL LONG` as one `CRYPTO_BETA` factor.
**Acceptance:** `correlation_adjusted_exposure` `3*5% corr 0.9 → 13.5%` + `liquidation_distance`.

## Epic 6: Execution & Cost Robustness (P2)
Fill model and cost stress.

### Story 6.1: Fill model
**As execution engineer** I want limit fill with queue/volume probability not `touched→filled`.
**Acceptance:** `LimitFillModel.attempt_fill` returns `fill_prob <1`.

### Story 6.2: Cost stress
**As researcher** I want `1.0/1.25/1.5/2/3x` costs + slippage/latency stress and `is_cost_robust`.
**Acceptance:** `cost_stress(df)` 5 results, fragile flag at 1.5x.

## Epic 7: Champion & Governance (P2)
States + NO_CHAMPION, 5 layers, Docker/data hygiene.

### Story 7.1: Champion NO_CHAMPION
**As PM** I want `NO_CHAMPION` as valid success when 100 failed.
**Acceptance:** `decide_promotion` returns `champion=NO_CHAMPION` when no eligible.

### Story 7.2: Docker & Data hygiene
**As DevOps** I want `GRAFANA_PASSWORD:?` fail-startup, `redis` internal only, `.env.example` canonical, legacy parquet archived.
**Acceptance:** `docker-compose.testnet.yml` no `6379:6379` ports, `data/archive/legacy_bak` 3 files.

### Story 7.3: 5 Layers
**As architect** I want `MarketData→Research→Risk→Execution→Governance` documented and `Governance ↛ Risk` enforced.
**Acceptance:** `docs/ARCHITECTURE_LAYERS.md` exists.

