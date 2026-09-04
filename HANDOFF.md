# QuantAI — HANDOFF

Дата: 2026-08-29
Ветка: main (локально, git status — нет git в PATH, но .git существует)
Версия: 5.1.0 (pyproject.toml)

## 1. Что сделано (Done)

### P0 — Validation Integrity (8/8) — ЗАКРЫТЫ
1. **Reproducibility**: `config/__init__.py` + `pyproject.toml` (single source, requires-python >=3.12, deps pinned), `README.md` Quick Start, `pyvenv 3.12.10` восстановлен через `winget Python 3.12`, `pip install -e .` работает.
2. **Probability units**: `MLEngine:787` `0..1`, `SignalGenerator:281` `if >1: /100` guard, `MLFusion:192` `_clamp 0..1`, `FusionConfig min_confidence 0.60` (было 60), `settings:90` `0.60`.
3. **Look-ahead**: `indicators:821 _cleanup_dataframe` `ffill` only (был `bfill().ffill()`), `supertrend bfill` остался на warmup.
4. **PurgedKFold**: `validation/purged_kfold.py:28` `split(X,y, tb_t1=None)` event-based `tb_t1 >= test_t0`, fallback индексный, `CombinatorialPurgedKFold` аналогично.
5. **Triple Barrier**: `labeling.py:66` (`upper/lower/vertical` + cost `0.12%` + tie-break wick), `dataset_builder.py:53` `label_method="triple_barrier"` default, `tb_t1` хранение, `build:516` `drop_last(future_bars)`.
6. **Risk fail-closed**: `trade_engine:413` `risk_orchestrator is None → REJECT`, `evaluate() exception → REJECT` (удалён fallback `calculate_position_size`), импорт `risk_orchestrator` lazy.
7. **FeatureGate v2**: `indicators:782` `bb_width/bb_position/bb_squeeze`, `feature_engine:271` `+14` (trend/adx/macd/bb/supertrend/vol) → `25 фичей` (было 11), `vpin/kyle/lunar` → `return` (skip, не NaN).
8. **Gate единый**: `validation/gate.py:266` `trading_readiness PF≥1.05 DD≥-15% = PromotionRules:49` (было `PF≥1.0 DD≥-35%`).

### Skills / Risk
- 3 YAML созданы: `quantai-execution-analysis.yaml`, `quantai-model-validation.yaml`, `quantai-regime-aware-strategy.yaml` → `7/7`; `20` из 49 deprecated помечены.
- `AGENTS.md:383` `5% per trade` → `1% (max 3%)` (RiskPolicy v1).

### Control Plane — AI Development Supervisor (8 модулей, import OK)
`src/control_plane/` `supervisor.py:20317` (fixed), `task_manager.py`, `agent_router.py` (9 типов, теперь `quant_researcher` реальный), `state_manager.py` (10 stages, `SupervisorState`), `evidence_manager.py` (14 типов), `validation_gate.py` (17 gates), `retry_engine.py` (6 RepairType), `checkpoint_manager.py` (`data/checkpoints/*.pkl`), `audit_logger.py` (`logs/audit/*.jsonl`, buffer 100, 5с flush). `lifecycle startup 0.01с` PASS, `AISupervisor start/stop` PASS.

### Data / Backtest
- `btcusdt_4h_prepared_fixed.parquet 6728 rows` (ffill), `btcusdt_1h 26910`, `btcusdt_15m 107639`.
- `compare_tf`: `4h PF 1.075 (+2.0% DD -2.2% 587 trades)` > `1h 0.86 (-7.5%)` > `15m 0.69 (-10%)`.
- `WF 4h 3000/600/600 6 окон`: `W1 1.04 W2 1.00 W3 1.22 W4 0.76 W5 1.00 W6 0.45` → `median 0.89 <1.15` `profit -6.61$` FAIL. `Threshold tuning 0.60→PF 1.105 full` → `WF -11.83$` overfit, откат к `0.75`.

### Strategy Tournament (4 family, 4h)
- `A Baseline` PF 1.06/1.060, `B Breakout PF 1.658 (+2.6% DD -1.1% 92 trades)` best backtest, `D MeanRev PF 0.718` (-0.8%), `C Cross-Sectional 1h PF 0.998 -44.7% DD -87%` (1h шум).
- `ML` `BalAcc 0.39` (Triple) vs `0.366` Simple ~ random 0.33, `WF 8000` не улучшил.

## 2. Что в работе (In Progress — блокеры)

1. **QuantResearcher bug** `src/control_plane/quant_researcher.py:284` `results` string → `AttributeError: 'str' has no attribute 'get'` в `_make_recommendation`. `_run_tournament` возвращает `dict`, но где-то exception → string. Тест `scripts/test_quant_researcher.py` падает после 3 семей (видно `mean_reversion: BT PF 0.718 | WF PF 0.825`).
2. **Control Plane autonomous loop** — `supervisor_real.log` `Iteration 10 active 8` демо 30с работает, но `AgentRouter._run_quant_researcher` теперь реальный (2-3 мин/цикл). Нужен 1 полный цикл `RESEARCH → WFO` с evidence.
3. **MeanRev** — `bb_position 0..1` уже в `prepared`, `max_adx 60` всё ещё `HOLD`, нужен полный WF 4h.
4. **Cross-sectional** — 1h провален, нужен `4h` `lookback 14 top2 rebalance7` уже `+55.5%` но `DD -70%` без `vol_target/dd_gate`.

## 3. Следующие шаги (Next — по приоритету)

**P0 (1-2 дня, Validation Integrity):**
1. Починить `quant_researcher.py:283` `_make_recommendation` (проверка `isinstance(results, dict)`, `best = max(results.items() ...)`), добавить `try/except` + `return {"success": False}`.
2. Интегрировать реальный турнир в `supervisor._execute_cycle`: `Task(stage=wfo, metadata={family: B})` → `AgentRouter(quant_researcher)` → `EvidenceManager` → `ValidationGate`.

**P1 (1 неделя, Tournament v2 — frozen params):**
3. Заморозить 4 family `ONE params` → один `WF 4h` protocol: `A Baseline, B Breakout(96/20/3.0/12), C Cross-Sectional 4h + vol_target/dd_gate, D MeanRev(60)`.
4. Метрики: `PF, median OOS ret, expectancy, DD, profitable_share` — сравнить, не `full sample`.

**P2:**
5. Breakout + RegimeFilter (TREND/RANGE) — уже `regime_filter wiring` готово, но `_generate_range_reversion` `bb_position 0..1` vs `-0.5..0.5`.
6. Cost-aware TripleBarrier ML (net-of-cost `2*commission`).
7. Cross-sectional 4h full 3-asset `vol_target 0.15` + `dd_soft_stop -25 reentry -12.5`.

## 4. Ключевые решения / файлы

**Decisions:**
- `probability 0..1` единый, `confidence 0..100` отдельно; `ffill` only; `fail-closed`; `event purge` via `tb_t1`; `triple_barrier default`; `RiskPolicy v1 1%/3%/5%/20% 1:20`; `single Gate = PromotionRules PF≥1.05`.

**Файлы:**
- `pyproject.toml:81` (merged pytest), `config/settings.py:90` `min_confidence 0.60`, `config/__init__.py`, `README.md`
- `src/ml_config.py` (вынесен), `src/ml_engine.py:160` `_use_ensemble` порядок, `src/ml_ensemble.py:32` `from src.ml_config`, `src/labeling.py:66`, `src/dataset_builder.py:53/516`, `src/validation/purged_kfold.py:28`, `src/indicators.py:782,821`, `src/feature_engine.py:254,271`, `src/strategy/signal_generator.py:58,187` (RegimeFilter wiring, 3 WeightedGate), `src/strategy/breakout_signal.py`, `src/strategy/mean_reversion_signal.py:100` (`bb_pos 0..1`), `src/trade_engine.py:413`, `src/lifecycle.py:81` `mode PAPER`
- `src/control_plane/*` 8 файлов (supervisor fixed 20317, `self.audit_logger.log` без `await`, black)
- `src/risk/risk_policy.py` (новый), `_bmad/custom/skills/quantai/*.yaml` 7/7
- `scripts/*tournament*, *compare*, *tune*`, `data/*_prepared*.parquet (6728/26910/107639)`, `logs/audit/`, `data/checkpoints/`

**Запуск:**
```bash
pip install -e . && quantai paper run --config config/paper.yaml
python scripts/final_tournament.py
python scripts/run_supervisor.py  # 30с demo, iteration 10
python -m src.validation.gate
```

**Git:** `.git` существует, `git` не в PATH (winget Python 3.12), `pyproject.toml.bak` удалён.

**Оценка:** Архитектура 7.5/10, Edge 2.5/10 — `NO_DEPLOYABLE_CHAMPION` (WF median <1.15).
