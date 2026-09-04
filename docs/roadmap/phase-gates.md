# QuantAI Phase Gates — Corrected Roadmap (2026-09-03)

**Governance**: Каждая фаза должна пройти ВСЕ гейты перед стартом следующей.  
**Checkpoint**: Конец фазы → полный `pytest` → `py_compile` → `gate --fast` → `gate_report.json` → `git tag phase-X-complete`.  
**Правило**: P3–P5 только после прохождения P0/P1/P2. До этого — freeze.

---

## Самая правильная последовательность

```
PHASE 0  Engineering + Risk Safety
        ↓
PHASE 1  Trusted Evidence + Control Plane
        ↓
PHASE 2  OOS / Nested WF / Research Integrity
        ↓
PHASE 3  Robustness + Statistical Validation
        ↓
PHASE 4  Alpha Research  (P3.1–P3.7)
        ↓
PHASE 5  Paper Trading  (30–90 дней)
        ↓
PHASE 6  Testnet  (full lifecycle + drills)
        ↓
PHASE 7  Small Live  (small capital)
        ↓
PHASE 8  Autonomous Evolution  (P4 + gradual scaling)
```

**Критический путь**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 строго последовательно.

---

## PHASE 0 — Engineering + Risk Safety

**Goal**: Безопасность капитала, отсутствие утечек, воспроизводимость.

**Gates**

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G0-1 Risk Policy Canonical** | `ResearchPolicy 60%/5%/10x → Production 20%/3%/3x`, `RR 2→7` только с `spec_version bump` | `tests/control_plane/test_*` + `src/risk/policy.py` |
| **G0-2 Fail-Closed Risk** | `UNKNOWN → REJECT`, staleness 5s, margin error → REJECT | `tests/test_paper_risk_e2e.py` |
| **G0-3 No Lookahead** | `output[t] only input[:t]` для Indicators/Features | `pytest tests/test_no_lookahead.py` |
| **G0-4 Causal Audit DAG** | `Data→Preprocessing→Indicators→Features→Labels→Scaling→Selection→ML→Strategy→Risk→Execution` | `python -m src.validation.causal_audit` |
| **G0-5 Data Quality** | дубликаты/gaps/монотонность/OHLC/volume/timezone/outage | `pytest tests/test_data_quality.py` + `src/data/data_gates.py` |

**Deliverables**: `src/risk/*`, `src/validation/gate.py`, `src/data/data_gates.py`

```bash
git tag phase-0-engineering-complete
```

---

## PHASE 1 — Trusted Evidence + Control Plane

**Goal**: Доказательства только через независимую верификацию.

**Gates**

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G1-1 Agent Success Contract** | `success=True` только `exit 0 && artifact exists && artifact valid && tests pass && metrics exist` | `src/control_plane/verifier.py` + `tests/control_plane/test_evidence_manager.py` |
| **G1-2 No Fake Metrics** | Запрет `bal_acc 0.39`, `exposure_ok True`, `review approved` без проверки | `src/control_plane/agent_router.py` + `evidence_manager.py` |
| **G1-3 Independent Verifier** | `Agent→Artifact→Verifier→Evidence→Gate` (не `Agent→success`) | `src/control_plane/supervisor.py:_verify` |
| **G1-4 Control Plane Suite** | `tests/control_plane/test_{supervisor,state_manager,task_manager,agent_router,evidence_manager,evidence_trust,transition_gates,retry_engine,research_budget,oos_firewall}.py` | `pytest tests/control_plane -q` 42 passed |

```bash
git tag phase-1-control-plane-complete
```

---

## PHASE 2 — OOS / Nested WF / Research Integrity

**Goal**: Честная OOS оценка без утечек.

**Gates**

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G2-1 Dataset Registry 2.0** | `dataset_id/exchange/symbol/timeframe/start/end/rows/schema_hash/raw_hash/prepared_hash/feature_version/label_version` | `src/research/dataset_registry.py` + `tests/test_dataset_registry.py` |
| **G2-2 Immutability** | `dataset_id` заморожен, изменение → `new dataset_id` (`ValueError` + `verify`) | `dataset_registry.py:verify()` |
| **G2-3 Data Quality 7** | `duplicate/gaps/monotonic/OHLC/volume/timezone/outage` | `src/data/data_gates.py` |
| **G2-4 Nested WF** | `outer TRAIN → inner WF + Optuna → aggregate→freeze→outer OOS` (immutable factory) | `src/validation/nested_walk_forward.py` |
| **G2-5 Timeframe-aware** | `OOS_end-OOS_start` via `timestamp`, запрет `bars/24` | `src/validation/gate.py` + `src/walk/walk_forward_engine.py:oos_duration_days` |
| **G2-6 Research Integrity** | 8 гейтов: Integrity/Statistical/IS-OOS/ML Calibration/Robustness/Selection/Regime/Edge | `src/champion/research_integrity.py` |

```bash
git tag phase-2-research-integrity-complete
```

---

## PHASE 3 — Robustness + Statistical Validation

**Goal**: Устойчивость к издержкам/лэтентности/режимам.

**Gates**

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G3-1 Cost Stress** | `1.0x,1.25x,1.5x,2x,3x` `PF>1 @1.5x` | `src/validation/cost_stress.py:cost_stress` |
| **G3-2 Slippage Stress** | `+0%/+25%/+50%/+100%` | `slippage_stress` |
| **G3-3 Latency Stress** | `50/100/250/500/1000/3000ms` via `LimitFillModel(latency_ms)` | `latency_stress` |
| **G3-4 Queue/Fill** | `price touch+volume+queue+latency+order book → fill prob` deterministic | `src/execution/fill_model.py` |
| **G3-5 Futures 2.0** | `mark/initial/maintenance/funding/liquidation/isolated/cross/realized/unrealized/available` | `src/execution/futures_accounting.py` |
| **G3-6 Funding Exchange-specific** | `00/08/16 UTC` events, не `i%2`/`8h` | `FundingSchedule` |
| **G3-7 Regime 7** | `Bull/Bear/Sideways/High Vol/Low Vol/Crash/Recovery` классификация | `src/research/regime_stability.py` |

```bash
git tag phase-3-robustness-complete
```

---

## PHASE 4 — Advanced Alpha Research — ТОЛЬКО после P0/P1/P2

**Входной гейт**: `phase-3-robustness-complete` + все `G0/G1/G2` PASS.

### P3.1 Breakout + regime filter

- Baза `Breakout 96/20/3.0/12` + `RegimeFilter` (`adx_enter 22/exit 18`, hysteresis) — `src/research/breakout_research_branch.py:RegimeFilteredBreakout`

### P3.2 Breakout + ML meta-labeling

- `Primary Strategy → Candidate Trade → ML Meta Labeler → TAKE/REJECT`
- `FilteredGenerator` + `MetaLabelModel` `P(win)` и `ExpectedReturnModel` `E[net]` (`src/strategy/meta_label.py`)

### P3.3 Multi-timeframe

- `4h regime (HTF EMA 50) → 1h structure (trend_score) → 15m entry (breakout trigger)` — `MultiTFConfirm` `tf_bars=4`

### P3.4 Cross-sectional momentum

- `BTC/ETH/SOL` cross-sectional sort по `return 20d` → long top / short bottom — `src/strategies/cross_sectional.py` + `src/portfolio/multi_symbol.py`

### P3.5 Volatility breakout

- `ATR 14 * 1.5` канал `range breakout + volatility_filter` — `src/strategy/breakout_signal.py`

### P3.6 Alternative data

- `funding (Binance 8h rate + mark/index basis)`, `OI delta (openInterest)`, `liquidation clusters (forceOrder)`, `L2 order book (depth/imbalance)`, `VPIN (volume bucket toxicity)`, `Kyle Lambda (price impact)` — `src/microstructure_intelligence.py` + `src/alternative_data.py` (пока stub, активируется после L2-cache)

### P3.7 HMM / regime clustering

- `HMM 3-state (Bull/Bear/Sideways)` + `KMeans 7` (совпадает с `REGIMES`) — `src/ml_regime.py` + `src/research/regime_stability.py:REGIMES`

**Gate P3**: Каждая стратегия проходит `cost/slippage/latency + regime 7 + Brier/ECE` перед попаданием в `ChampionPipeline`.

```bash
git tag phase-4-alpha-research-complete
```

---

## PHASE 5 — Paper Trading — 30–90 дней

**P5.1** `src/validation/long_run.py` + `portfolio_long_run` — 30 дней минимум, `min_trades 30/90d`, `equity_curve` + `journal.csv`, `gate long_run_paper` → `BLOCKED` до 30 дней, `PASS` после.

---

## PHASE 6 — Testnet — full lifecycle

**P5.2** `ExecutionEngine(DRY_RUN)` Binance testnet — `BinanceRestAdapter(testnet=True)` + `verify_no_withdraw_permission`  
**P5.3 Failure injection** (8): `exchange unavailable`, `websocket disconnect`, `REST timeout`, `Redis failure`, `database failure`, `stale market data`, `corrupted model`, `unexpected position`  
**P5.4 Disaster recovery drill** — `emergency_stop` + `resume_from_halt` 7 шагов  
**P5.5 Reconciliation drill** — `ReconciliationEngine` ghost/stuck order  
**P5.6 Emergency-stop drill** — `HALTED → flatten → verify flat → halt`

**Gates**: `G6-1 Testnet 7d без incident`, `G6-2 Failure injection все 8 PASS`, `G6-3 Reconciliation 0 ghost`, `G6-4 Emergency 5/5 flat verified`

```bash
git tag phase-6-testnet-complete
```

---

## PHASE 7 — Small Live

**P5.7** `$100–500` live 14 дней, лимит `max_total_exposure 5%`, `max_position 5%`, `reserve 40%`  
**P5.8** Gradual scaling `$1k→$10k` при `DD<5%` и `Sharpe>1`

```bash
git tag phase-7-small-live-complete
```

---

## PHASE 8 — Autonomous Evolution

**P4 Autonomous Intelligence — ТОЛЬКО после Research Integrity (Phase 2)**

| ID | Компонент | Артефакт |
|----|-----------|----------|
| **P4.1** | Autonomous Hypothesis Generator | `Supervisor → hypothesis` (`src/control_plane/supervisor.py:_research_loop_on_reject`) |
| **P4.2** | Autonomous Experiment Generator | `hypothesis → experiment specification` (`ExperimentRegistry` + `TaskManager`) |
| **P4.3** | Autonomous Coding | `task → code → tests → repair` (`RetryEngine` + `ruff` + `py_compile`) |
| **P4.4** | Autonomous Research Loop | `experiment → evaluate → reject/improve → new experiment` (`_execute_cycle` OBSERVE→...→GATE) |
| **P4.5** | Research Budget Enforcement | `experiments/Optuna trials/mutations/OOS touches/retries` durable via `AtomicResearchLedger` (`src/research/research_budget.py`) |
| **P4.6** | Autonomous Champion Search | Только `verified evidence + robust OOS + statistical PASS` (`src/champion/pipeline.py` + `ResearchIntegrity`) |

**P5.8 Gradual capital scaling** — после `P4.6` и `phase-7` → увеличение капитала на `+20%` каждые 30 дней при `PF>1.2` и `DD<10%`.

```bash
git tag phase-8-autonomous-complete
```

---

## Что НЕ делать сейчас — Freeze до Phase 1

| Заморожено | Причина | Разморозка после |
|------------|---------|------------------|
| **LLM trading signals** | Нужна калибровка и OOS, иначе утечка | `phase-3` |
| **Deep RL** | Требует симуляции + risk safety | `phase-3` |
| **Transformer / GAN / VAE** | Сложность без профита в P0 | `phase-4` |
| **Десятки индикаторов** | Бритва Оккама: лимит 2–3, WF-абляция | `phase-2` |
| **Массовая Optuna (1000+ trials)** | Переоптимизация, бюджет | `phase-4` (budget `max_optuna 50`) |
| **LunarCrush / Galaxy Score** | Внешняя зависимость, лимиты | `phase-4` (alt data) |
| **Сложная genetic evolution** | Требует verified champion search | `phase-8` |

**Правило**: Любое добавление из freeze-списка до `phase-1-control-plane-complete` → `BLOCKED` гейт.

---

## Контроль

```bash
# Проверка текущей фазы
cat docs/roadmap/phase-gates.md | grep "PHASE [0-8]"

# Gate
python -m src.validation.gate --fast
```

**Статус**: `PHASE 0–3` в работе (P0/P1/P2 закрываются), `PHASE 4–8` — только после `phase-3`.
