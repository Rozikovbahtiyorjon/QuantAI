# QuantAI 5.2 — Research Integrity & OOS Validation

**Status:** Planned Milestone — Next  
**Predecessor:** QuantAI 5.1 (Professional)  
**Date Defined:** 2026-09-02  
**Author:** Bahtiyorjon / QuantAI Research Team  
**Governance:** Requires all 10 Phased Gates = PASS before Champion promotion & AI Supervisor autonomy

---

## 0. Verdict Context

### QuantAI is not weak

QuantAI 5.1 has a **serious architecture**. Foundation is solid across:

- **Data** — CCXT loader, historical downloader, parquet prepared datasets, schema validation
- **Feature** — `feature_engine.py`, `indicators.py` (EMA/RSI/MACD/ATR/ADX/BB/VWAP/OBV/SuperTrend), microstructure hooks
- **ML** — `ml_engine.py` (XGBoost + Heterogeneous Ensemble LightGBM/CatBoost), `ml_ensemble.py`, `dataset_builder.py` with triple-barrier
- **Triple Barrier** — `labeling.py` (path-dependent barriers, purged events, `tb_t1`)
- **Backtest** — `backtest_engine.py` + `trade_engine.py` (next-bar-open, slippage, commission, BE/trailing, equity curve)
- **Walk-Forward** — `walk/walk_forward_engine.py` (rolling balance, `train_callback` for ML retrain), `WalkForwardValidator`
- **Risk** — `risk/risk_orchestrator.py` (DrawdownGuard + ExposureManager + PositionSizer, 3-5-7 rules)
- **Paper** — `paper_trading_runner.py`, `paper_trading_session.py`, `paper_trading_engine.py`, quality gate
- **Execution** — `execution_engine.py` (DRY_RUN/LIVE), `binance_adapter.py`, `reconciliation_engine.py`
- **Governance** — `champion_registry.py`, `champion_governance_engine.py`, `champion_admission_controller.py`, `champion_promotion_engine.py`, replacement/rollback guards
- **Champion** — `strategy_champion.py`, `strategy_tournament.py`, `strategy_genome.py`, `strategy_bank.py`
- **Supervisor** — `quantai_production_runtime_supervisor.py`, `ai_strategy_research_lab.py`, `advanced_strategy_architecture.py`
- **Research Budget** — `src/research/research_budget.py` (max 50 trials/run, max 5 params, max 10 indicators, max 10 per OOS)
- **Experiment Registry** — `src/research/experiment_registry.py` (hashes, lineage, OOS reuse counting, DSR/PBO placeholders)

**Score: Architecture / Engineering Maturity = 7–8 / 10**

### Dangerous Gap

```
ENGINEERING MATURITY          TRADING EVIDENCE
 ████████████████░░░░  7-8/10   █████░░░░░░░░░░░░░░  2-3/10

 Data ✅  Feature ✅  ML ✅  TripleBarrier ✅  Backtest ✅  WF ✅  Risk ✅
 Paper ✅  Execution ✅  Governance ✅  Champion ✅  Supervisor ✅  Budget ✅  Registry ✅
                                          ▲
                                          │  GAP: PBO/DSR/WRC not enforced,
                                          │       Walk-Forward is not Nested,
                                          │       OOS is re-touched, samples <30,
                                          │       Cost/Slippage/Latency not stressed,
                                          │       PF 1.1→1.3 is not an edge.
                                          │
 Evidence ❌  Fresh OOS ❌  Bootstrap ❌  PBO ❌  DSR ❌  Cost-robust ❌  Sample gate ❌
```

**Score: Trading Evidence / Statistical Rigor = 2–3 / 10**

**Core problem if Supervisor thinks:**

> `backtest → optimization (Optuna x10) → best PF (1.3) → champion = real edge`

This is **research overfitting**. Without nested isolation, budget, and statistical gates, any PF can be manufactured by multiple testing. QuantAI 5.1 can *generate* champions but cannot *prove* them.

### Main Recommendation

**Next milestone must be QuantAI 5.2 — Research Integrity & OOS Validation**

No new indicators, no 20-param strategies, no 200-trial Optuna, no live scaling until the 10-gate pipeline is PASS. After that, AI Supervisor gets autonomous rights under a hard constraint.

---

## 1. Milestone Goal

Transform QuantAI from *capable of searching* to *capable of proving*.

**Definition of Done:** An independently validated **Robust OOS Edge** (`src/research/robust_oos_edge.py` score > 0.70 and all critical components PASS) survives the full gate pipeline on a **fresh, never-touched OOS**. Only then does a strategy become Champion.

**Principle:** *Less Optimization + More Validation.*

---

## 2. Phased Gates — 10 + Champion (Sequential, No Skip)

Each gate is binary **PASS / FAIL / BLOCKED**. `FAIL` in any gate blocks all downstream gates. No soft PASS.

| # | Gate | Owner | Input → Output | PASS Criteria | Enforcement Module | Failure = |
|---|------|-------|----------------|---------------|--------------------|-----------|
| **G1** | **P0 Code/Config Fixes** | Risk Eng + Quant Dev | `config/settings.py` + `trade_engine.py` + `sl_tp_calculator.py` | `max_total_exposure 60%→5%`, `RISK_PERCENT` decimal unified, duplicate VPIN removed, funding `8h → dynamic Timedelta`, `min_risk_reward 1.5→7.0` enforced, `reserve 40%` field added, `correlation silent pass` removed | `src/validation/gate.py::risk_gates`, `tests/test_risk_*.py` | BLOCKED — cannot trade |
| **G2** | **True Nested Walk-Forward** | ML Eng | Outer Train → Inner WF+Optuna → frozen params → Outer OOS | Inner WF *executed* (not scaffold), outer OOS never seen by optimizer, `inner_test.max < outer_test.min` asserted, Optuna error = EXPERIMENT FAILED (not `{}`) | `src/validation/nested_walk_forward.py` (`NestedWalkForward`) | FAIL — leakage |
| **G3** | **OOS Isolation & Freshness** | Quant Researcher | `oos_period` string | OOS period has `oos_touch_count == 0` at start; any pre-registration touch → BLOCKED; reuse tracked per period | `src/research/experiment_registry.py` (`oos_reuse_count`, `check_oos_valid_for_selection`) | BLOCKED — OOS burned |
| **G4** | **Minimum Sample Gates** | QA | `WalkForwardResult.windows` + `BacktestResult` | `MIN_TRADES_FOR_VALID_PF ≥30` per window & aggregate `≥30`, `MIN_OOS_DAYS ≥90`, `windows ≥10`, `profitable_windows ≥60%` enforced before metrics are trusted | `src/backtest_engine.py`, `src/walk/walk_forward_engine.py`, `src/research/optimization_guard.py` | FAIL — insufficient sample |
| **G5** | **Experiment Registry** | ML Eng / Architect | Every experiment | Mandatory: `experiment_id, dataset_id, dataset_hash, feature_schema_hash, label_schema_hash, code_commit, model_hash, train/valid/oos_period, params, seed, cost/slippage/latency model, gross/net/PF/Sharpe/Sortino/MaxDD/Expectancy/Trades, selection_status, oos_touched, oos_touch_count, parent_id, used_for_selection` | `src/research/experiment_registry.py` (`ExperimentRecord`, `ExperimentRegistry.register`) | BLOCKED — no audit trail |
| **G6** | **Research Budget Enforcement** | Supervisor | `ResearchBudget` counters | `max_optuna_trials=50` (per run 50), `max_experiments=50`, `max_params_per_strategy=5`, `max_indicators=10`, `max_experiments_per_oos=10`, `max_optimizations_per_strategy=3` — exceed → `BudgetExceeded` with *LESS OPTIMIZATION* message; `check_params/check_indicators/check_optimization_attempt` | `src/research/research_budget.py` (`ResearchBudget`) + `src/research/optimization_guard.py` | FAIL — over-optimization |
| **G7** | **Cost / Slippage / Latency Stress** | Quant Dev + Risk Mgr | OOS equity curve | `PF >1.0` at `1.5× costs`, slippage stress (e.g. 0.02%→0.05%) and latency stress (0ms→500ms) both robust; fragile → not robust | `src/validation/cost_stress.py` (`is_cost_robust`) + `src/research/robust_oos_edge.py` (`cost_robust`, `slippage_latency`) | FAIL — not execution-robust |
| **G8** | **Bootstrap** | Quant Researcher | OOS returns series | Block bootstrap (`block=20`, `n_iter=500`, seed 42) on 4h bars; Sharpe 95% CI lower >0 and `p_value <0.05` | `src/validation/bootstrap.py` (`block_bootstrap_sharpe`) | FAIL — CI includes 0 |
| **G9** | **PBO / DSR / WRC** | Quant Researcher | CPCV splits + trials | `PBO <0.6` (Bailey CPCV via `CombinatorialPurgedKFold`), `DSR ≥0.95` (deflated Sharpe, multiple-testing corrected), `WRC p <0.05` (White's Reality Check / SPA) — ALL THREE if provided, else configured subset | `src/research/pbo.py` (`compute_pbo`), `src/research/dsr.py`, `src/research/white_reality_check.py` + `src/research/experiment_registry.py::pbo_placeholder` (real when data supplied) | FAIL — selection bias |
| **G10** | **Robustness Score** | Research Integrity Engine | Gate G1–G9 outputs | `MAX ROBUST OOS EDGE` score `>0.70` and all critical components PASS: `expectancy>0 (0.20) + PF≥1.1 (0.15) + DD≥-15% (0.15) + trades≥30 (0.10) + regime≥3/7 (0.10) + cost-robust (0.10) + slippage&latency (0.10) + DSR/PBO/WRC (0.10)` | `src/research/robust_oos_edge.py` (`compute_robust_oos_edge`, `is_robust_edge`) invoked in `src/champion/research_integrity.py::_gate_robust_oos_edge` | FAIL — not champion-eligible |
| **🏆** | **Champion Admission** | Governance Engine | `RobustOOSResult` | Only if G1–G10 all PASS on **fresh OOS**; registers via `champion_admission_controller.py` → `champion_registry.py`; OOS reuse >50 → PF loses power (hard block) | `src/champion/pipeline.py` + `src/champion/admission` + `docs/validation/acceptance-criteria.md Gate 3-5` | BLOCKED — no champion |

**Pipeline shape:**

```
G1 P0 Fixes ─► G2 Nested WF ─► G3 OOS Freshness ─► G4 Sample Gates ─► G5 Registry
     ─► G6 Budget ─► G7 Cost/Slip/Lat stress ─► G8 Bootstrap ─► G9 PBO/DSR/WRC
     ─► G10 Robust Score (>0.70) ─► 🏆 Champion
                                              │
                                              └─► only then: Supervisor autonomy
```

---

## 3. Success Criteria — Champion Only on Robust OOS Edge

**Hard rule:** Champion promotion requires **ALL**:

- **Each gate G1–G10 = PASS** (not WARN, not CONDITIONAL). Any FAIL or BLOCKED stops the pipeline; no averaging.
- **OOS is independent:** OOS period never used for selection before this experiment (`oos_touch_count == 1` at registration, which is the current touch itself; `is_oos_hard_overused(≥50)` → auto BLOCKED).
- **Robust OOS Edge = PASS:** `compute_robust_oos_edge(metrics).score > 0.70` AND `critical_pass == True` for (`expectancy`, `pf_stable`, `dd`, `sample`). Missing artefact → FAIL (unless `RobustOOSConfig.permissive=True` for research-only runs, never for champion).
- **Statistical significance:** Bootstrap CI excludes 0, and `PBO<0.6 + DSR≥0.95 + WRC p<0.05` (or configured subset for early validation).
- **Execution robustness:** `cost_robust == True` and `slippage_robust == True and latency_robust == True`.
- **Sample:** `Trades ≥30` total OOS and `≥30` per evaluation window where PF is quoted; `OOS days ≥90` (enforced via `acceptance-criteria.md`).

**Negative tests (must be verified to FAIL as expected):**

- WF that skips inner execution → G2 FAIL
- OOS touched 10+ times → G6 / G3 WARN, ≥50 → hard BLOCKED (PF 1.32 example loses power)
- Strategy with 20 params or 15 indicators → G6 `BudgetExceeded`
- PF 1.4 with 12 trades → G4 FAIL despite PF
- PF 1.3 at base cost but PF 0.95 at 1.5× cost → G7 FAIL

---

## 4. AI Supervisor Constraint — Hard

> **AI can infinitely search but cannot declare success if the independent OOS/statistical system is not PASS.**

Formal constraint enforced in code, not in prompt:

```python
# In ResearchIntegrityEngine / ChampionGovernanceEngine / Supervisor:

result = compute_robust_oos_edge(oos_metrics, config=RobustOOSConfig(min_score=0.70))
is_robust = is_robust_edge(result)  # score>0.70 and critical_pass

if not is_robust:
    raise ResearchIntegrityError(
        "SUPERVISOR BLOCKED: Robust OOS Edge FAIL "
        f"(score {result.score:.2f} ≤ 0.70 or critical FAIL: "
        f"{[k for k,v in result.components.to_detail_dict().items() if v['critical'] and not v['passed']]}) — "
        "infinite search allowed, success declaration forbidden."
    )

registry.check_oos_valid_for_selection(oos_period)  # ≥50 touches → RuntimeError
budget.check_optimization_attempt(strategy_id)       # >3 per strategy → BudgetExceeded
# Only after both + G1-G9 PASS does champion_admission_controller admit.
```

**Supervisor rights:**

- ✅ Unlimited hypothesis generation, feature ideas, param proposals, tournament runs, Optuna within `ResearchBudget` caps (50 trials/run).
- ✅ Can run validation infinitely (bootstrap, PBO, WRC, stress) and request fresh OOS slices.
- ❌ **Cannot** set `selection_status=PROMOTED`, call `champion_promotion_engine.promote()`, or report `TRADING READY` unless the independent statistical pipeline above returns PASS.
- ❌ **Cannot** bypass `ExperimentRegistry`, `ResearchBudget`, or `NestedWalkForward` isolation by direct `SignalGenerator` calls — all must route through gated pipeline.

**This is the load-bearing wall of milestone 5.2.**

---

## 5. Gap Diagram — Engineering vs Evidence (Detailed)

```
QuantAI 5.1  Maturity Profile (1-10)

Engineering (what we HAVE)          Evidence (what we can PROVE)
─────────────────────────           ────────────────────────────
Data            ████████░░ 8        Fresh OOS       ██░░░░░░░░ 2
Feature         ████████░░ 8        Nested WF       ███░░░░░░░ 3
ML/Ensemble     ███████░░░ 7        Sample gates    ███░░░░░░░ 3
TripleBarrier   █████████░ 9        Cost stress     ███░░░░░░░ 3
Backtest        ████████░░ 8        Slippage/Lat    ██░░░░░░░░ 2
Walk-Forward    ██████░░░░ 6 *      Bootstrap       █░░░░░░░░░ 1
Risk            █████░░░░░ 5 **     PBO             █░░░░░░░░░ 1
Paper           ██████░░░░ 6        DSR             █░░░░░░░░░ 1
Execution       █████░░░░░ 5        WRC             █░░░░░░░░░ 1
Governance      ████████░░ 8        Robust Score    ██░░░░░░░░ 2
Champion        ███████░░░ 7        Live 30d        ░░░░░░░░░░ 0
Supervisor      ███████░░░ 7        Multi-asset     ░░░░░░░░░░ 0
Budget          ███████░░░ 7        Regime proof    ██░░░░░░░░ 2
Registry        ███████░░░ 7        ──────────────────────────
                ─────────           Evidence AVG    ██░░░░░░░░ 2.3
                AVG 7.2             Engineering AVG ████████░░ 7.2
                                    GAP             ██████░░░░ 4.9

 * WF exists but is NOT nested (leakage risk)
 ** Risk exists but misconfigured: 60% vs 5% rule, missing 40% reserve, RISK_PERCENT fragile

Risk of trusting PF alone:
  Trials=  1 → needed PF for p<0.05 ≈ 1.05
  Trials= 10 → needed PF ≈ 1.18  (your 1.32 is borderline)
  Trials= 50 → needed PF ≈ 1.45  (your 1.32 is NOISE)  — DSR corrects this
```

**Reading:** High engineering maturity makes it *easy* to over-optimize; without evidence gates, more horsepower = more overfit. Milestone 5.2 closes the gap by **raising evidence to 7+**.

Target after 5.2:

```
Engineering  ██████████ 8-9   Evidence  ██████████ 7-8   GAP █░  <1.0
```

---

## 6. Next Steps — Implementation Order (4 Weeks)

### Week 0 — P0 Unblock (1–2 days, must be first)
- [ ] `config/settings.py:258` `max_total_exposure_percent 60→5`, add `reserve_percent=40`, `min_risk_reward_ratio=7.0`
- [ ] `src/trade_engine.py:504` unify `RISK_PERCENT` decimal path, remove `*100` branch
- [ ] `src/strategy/sl_tp_calculator.py:203-205` delete duplicate VPIN multiplier
- [ ] `src/trade_engine.py:1392` `funding = 8h if i%2==0` → `Timedelta(hours=8)/bar_interval`
- [ ] `src/risk/risk_orchestrator.py:247-282` remove silent `try/except`, expose `fail_open` flag + log
- [ ] Verify `pytest tests/test_risk_*.py tests/test_trade_engine_risk.py` green + `python -m src.validation.gate --fast` PASS on risk gates

### Week 1 — Nested WF + OOS Isolation + Sample Gates
- [ ] Harden `src/validation/nested_walk_forward.py`: inner WF execution asserted, overlap assert, optimization error = fail-fast, evidence `win.model_result = inner_wfr`
- [ ] Enforce `ExperimentRegistry.check_oos_valid_for_selection` in `ResearchIntegrityEngine` pre-promotion
- [ ] Enforce `optimization_guard.py` `MIN_TRADES_FOR_VALID_PF=30`, `MIN_OOS_DAYS=90` as gates (not warnings)
- [ ] Tests: `tests/test_nested_walk_forward_leakage.py`, `tests/test_experiment_registry_oos_reuse.py`, `tests/test_sample_gates.py`

### Week 2 — Registry + Budget + Cost/Slippage/Latency
- [ ] Registry mandatory fields audit — every `register()` call must fill all 15 mandatory fields; add `scripts/audit_registry_completeness.py`
- [ ] Wire `ResearchBudget` checks into Supervisor loop: `check_optuna`, `check_params(≤5)`, `check_indicators(≤10)`, `check_optimization_attempt(≤3/strategy)`, `check_experiments_per_oos(≤10)`
- [ ] Implement `src/validation/cost_stress.py` stress sweep `0.5×..2.0×` + slippage/latency stress matrices; fail if fragile
- [ ] Tests: `tests/test_research_budget_less_optimization.py`, `tests/test_cost_stress_robustness.py`

### Week 3 — Bootstrap + PBO/DSR/WRC + Robust Score + Supervisor Lock
- [ ] `src/validation/bootstrap.py` block-bootstrap (block=20, n_iter=500) gated on OOS returns; wire to `robust_oos_edge.py:selection_bias`
- [ ] `src/research/pbo.py` CPCV PBO via `CombinatorialPurgedKFold`, `src/research/dsr.py` DSR, `src/research/white_reality_check.py` WRC — all aliased in `robust_oos_edge._extract_selection_bias`
- [ ] `src/research/robust_oos_edge.py` as final gate in `ResearchIntegrityEngine._gate_robust_oos_edge()` — score>0.70 + critical pass
- [ ] **Supervisor hard constraint:** `quantai_production_runtime_supervisor.py` → raises `ResearchIntegrityError` if `is_robust_edge==False`; infinite search loop allowed, promotion blocked
- [ ] Tests: `tests/test_robust_oos_edge_gating.py`, `tests/test_supervisor_cannot_promote_without_oos_pass.py`

### Week 4 — E2E Validation & Tag
- [ ] Run full pipeline on **fresh OOS** slice never used before (e.g. last 90 days holdout not in `data/experiments/*.json`)
- [ ] Generate `reports/milestone_5_2_oos_report.json` with all 10 gate verdicts + Robust score per champion candidate
- [ ] Nightly regression: `pytest tests/test_walk_forward_*.py tests/test_ml_engine.py -k pbo` + `python -m src.validation.gate` (full) must be PASS
- [ ] Documentation: `docs/validation/regression-gates.md` updated with new Gate 6-10 thresholds; `docs/roadmap/phase-gates.md` linked to this milestone
- [ ] Tag: `git tag quantai-5.2-research-integrity-complete` only after E2E PASS

### Exit Criteria for 5.2
- [ ] All Week 0–4 checkboxes checked and reviewed
- [ ] `python -m src.validation.gate` → `PASS` (no BLOCKED on G1–G5)
- [ ] At least one E2E run produces a `reports/milestone_5_2_oos_report.json` with `robust_oos_edge.score` and `is_robust_edge=true` **or** proves no edge exists (also valid: system correctly rejects overfit PF 1.3)
- [ ] Supervisor integration test demonstrates `BudgetExceeded` and `ResearchIntegrityError` blocks

---

## 7. File & Code References (Canonical)

| Concern | File |
|---------|------|
| Gate orchestration | `src/validation/gate.py` |
| Nested WF isolation | `src/validation/nested_walk_forward.py` |
| OOS tracking | `src/research/experiment_registry.py` |
| Budget caps | `src/research/research_budget.py` |
| Cost stress | `src/validation/cost_stress.py` |
| Bootstrap | `src/validation/bootstrap.py` |
| PBO | `src/research/pbo.py` |
| DSR | `src/research/dsr.py` |
| WRC | `src/research/white_reality_check.py` |
| Robust score | `src/research/robust_oos_edge.py` |
| Integrity engine | `src/champion/research_integrity.py` |
| Champion admission | `src/champion_admission_controller.py` |
| Supervisor | `src/quantai_production_runtime_supervisor.py` |
| Criteria | `docs/validation/acceptance-criteria.md` |

---

## 8. Governance

- **Decision authority:** Quant Researcher + Risk Manager sign G1,G7–G10; ML Engineer signs G2,G5–G6; Senior Dev signs G1–G2.
- **No override:** Supervisor cannot override G10; only fresh OOS and statistically significant resubmission can pass.
- **Audit trail:** Every gate result persisted as `ExperimentRecord` + `RobustOOSResult.to_dict()` JSON for replay.

---

*End of Milestone 5.2 Specification — implementation must satisfy all 10 gates before any capital scale-up (Phase 5).*
