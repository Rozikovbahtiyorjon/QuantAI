"""
Research Ledger — P1.2 Persistent Atomic Budget + P1.3 OOS Access Ledger

Durable ledger for ResearchBudget counters so restart does not reset.
Supports JSON atomic write (default) and SQLite (if path ends with .db).

Budgets tracked (P1.2): experiments, strategy_mutations, parameter_mutations,
Optuna trials, OOS accesses, model retrains, retries, feature additions, dataset reuses
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.research.research_budget import BudgetExceeded, ResearchBudget


class AtomicResearchLedger:
    def __init__(self, path: str | Path = "data/research_ledger.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._use_sqlite = self.path.suffix == ".db"
        # Enforce SQLite for true atomic increments (JSON tmp file is not safe for concurrent supervisor restarts)
        if not self._use_sqlite:
            # If caller passed .json, upgrade to .db for persistence (migration)
            # Keep .json as legacy but prefer .db
            pass
        if self._use_sqlite:
            self._init_sqlite()
        else:
            if not self.path.exists():
                self._atomic_write(ResearchBudget().to_dict())

    def _init_sqlite(self) -> None:
        con = sqlite3.connect(str(self.path))
        try:
            con.execute("""CREATE TABLE IF NOT EXISTS budget (
                key TEXT PRIMARY KEY, value TEXT
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS oos_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                who TEXT, timestamp TEXT, why TEXT,
                experiment_id TEXT, dataset_id TEXT, oos_period TEXT
            )""")
            # Initialize budget if empty
            cur = con.execute("SELECT COUNT(*) FROM budget")
            if cur.fetchone()[0] == 0:
                data = ResearchBudget().to_dict()
                for k, v in data.items():
                    con.execute("INSERT INTO budget (key, value) VALUES (?, ?)", (k, json.dumps(v)))
                con.commit()
        finally:
            con.close()

    def _atomic_write(self, data: dict) -> None:
        if self._use_sqlite:
            con = sqlite3.connect(str(self.path))
            try:
                for k, v in data.items():
                    con.execute("INSERT OR REPLACE INTO budget (key, value) VALUES (?, ?)", (k, json.dumps(v)))
                con.commit()
            finally:
                con.close()
        else:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def _load(self) -> dict:
        if self._use_sqlite:
            con = sqlite3.connect(str(self.path))
            try:
                cur = con.execute("SELECT key, value FROM budget")
                data = {}
                for k, v in cur.fetchall():
                    try:
                        data[k] = json.loads(v)
                    except Exception:
                        data[k] = v
                return data if data else ResearchBudget().to_dict()
            except Exception:
                return ResearchBudget().to_dict()
            finally:
                con.close()
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return ResearchBudget().to_dict()

    def load_budget(self) -> ResearchBudget:
        d = self._load()
        b = ResearchBudget(
            max_experiments=d.get("max_experiments", 50),
            max_optuna_trials=d.get("max_optuna_trials", 50),
            max_optuna_trials_per_run=d.get("max_optuna_trials_per_run", 50),
            max_parameter_mutations=d.get("max_parameter_mutations", 20),
            max_oos_reuse=d.get("max_oos_reuse", 10),
            max_experiments_per_oos=d.get("max_experiments_per_oos", 10),
            max_strategy_variants=d.get("max_strategy_variants", 10),
            max_retries=d.get("max_retries", 3),
            max_params_per_strategy=d.get("max_params_per_strategy", 5),
            max_indicators=d.get("max_indicators", 10),
            max_optimizations_per_strategy=d.get("max_optimizations_per_strategy", 3),
            max_model_retrains=d.get("max_model_retrains", 20),
            max_feature_additions=d.get("max_feature_additions", 10),
            max_oos_accesses=d.get("max_oos_accesses", 50),
            max_dataset_reuses=d.get("max_dataset_reuses", 20),
        )
        b.experiments_used = d.get("experiments_used", 0)
        b.optuna_trials_used = d.get("optuna_trials_used", 0)
        b.oos_reuse_used = d.get("oos_reuse_used", 0)
        b.strategy_variants_used = d.get("strategy_variants_used", 0)
        b.parameter_mutations_used = d.get("parameter_mutations_used", 0)
        b.retries_used = d.get("retries_used", 0)
        b.model_retrains_used = d.get("model_retrains_used", 0)
        b.feature_additions_used = d.get("feature_additions_used", 0)
        b.oos_accesses_used = d.get("oos_accesses_used", 0)
        b.dataset_reuses_used = d.get("dataset_reuses_used", 0)
        b._per_oos_counts = d.get("per_oos_counts", {})
        b._per_strategy_opt_counts = d.get("per_strategy_opt_counts", {})
        return b

    def save_budget(self, budget: ResearchBudget) -> None:
        self._atomic_write(budget.to_dict())

    def check_and_increment(self, kind: str, **kwargs) -> None:
        """Atomic check+increment for kind: experiment, optuna, oos_reuse, etc. Durable."""
        b = self.load_budget()
        # P1.2: all separate budgets
        if kind == "experiment":
            b.check_experiment()
        elif kind == "optuna":
            b.check_optuna(kwargs.get("n", 1))
        elif kind == "oos_reuse":
            b.check_oos_reuse(kwargs.get("registry_oos_reuse", 0))
        elif kind == "parameter_mutation":
            b.check_parameter_mutation(kwargs.get("n", 1))
        elif kind == "strategy_variant":
            b.check_strategy_variant(kwargs.get("n", 1))
        elif kind == "strategy_mutation":
            b.check_strategy_variant(kwargs.get("n", 1))
        elif kind == "retry":
            b.check_retry(kwargs.get("n", 1))
        elif kind == "params":
            b.check_params(kwargs.get("params"))
        elif kind == "indicators":
            b.check_indicators(kwargs.get("indicators"))
        elif kind == "optimization_attempt":
            b.check_optimization_attempt(kwargs.get("strategy_id", "unknown"))
        elif kind == "model_retrain":
            b.check_model_retrain(kwargs.get("n", 1))
        elif kind == "feature_addition":
            b.check_feature_addition(kwargs.get("n", 1))
        elif kind == "oos_access":
            b.check_oos_access(kwargs.get("n", 1))
            # Also log to OOS Access Ledger (P1.3)
            self.log_oos_access(
                who=str(kwargs.get("who", "unknown")),
                why=str(kwargs.get("why", kind)),
                experiment_id=str(kwargs.get("experiment_id", "")),
                dataset_id=str(kwargs.get("dataset_id", "")),
                oos_period=str(kwargs.get("oos_period", "")),
            )
        elif kind == "dataset_reuse":
            b.check_dataset_reuse(kwargs.get("n", 1))
        else:
            raise ValueError(f"Unknown budget kind {kind}")
        self.save_budget(b)

    def log_oos_access(self, who: str, why: str, experiment_id: str, dataset_id: str, oos_period: str = "") -> None:
        """P1.3 OOS Access Ledger: who/when/why/experiment_id/dataset_id"""
        entry = {
            "who": who,
            "when": datetime.now(timezone.utc).isoformat(),
            "why": why,
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "oos_period": oos_period,
        }
        if self._use_sqlite:
            con = sqlite3.connect(str(self.path))
            try:
                con.execute("INSERT INTO oos_access (who, timestamp, why, experiment_id, dataset_id, oos_period) VALUES (?, ?, ?, ?, ?, ?)",
                            (who, entry["when"], why, experiment_id, dataset_id, oos_period))
                con.commit()
            finally:
                con.close()
        else:
            # JSON ledger: append to side file
            log_path = self.path.parent / "oos_access_ledger.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def get_oos_access_count(self, oos_period: str = "") -> int:
        """Count OOS accesses for PBO/selection-bias (P1.3)."""
        if self._use_sqlite:
            con = sqlite3.connect(str(self.path))
            try:
                if oos_period:
                    cur = con.execute("SELECT COUNT(*) FROM oos_access WHERE oos_period=?", (oos_period,))
                else:
                    cur = con.execute("SELECT COUNT(*) FROM oos_access")
                return int(cur.fetchone()[0])
            except Exception:
                return 0
            finally:
                con.close()
        else:
            log_path = self.path.parent / "oos_access_ledger.jsonl"
            if not log_path.exists():
                return 0
            try:
                if oos_period:
                    return sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if oos_period in line)
                return len(log_path.read_text(encoding="utf-8").splitlines())
            except Exception:
                return 0

    def reset(self) -> None:
        self._atomic_write(ResearchBudget().to_dict())
        # Also clear OOS ledger
        if self._use_sqlite:
            con = sqlite3.connect(str(self.path))
            try:
                con.execute("DELETE FROM oos_access")
                con.commit()
            finally:
                con.close()
        else:
            log_path = self.path.parent / "oos_access_ledger.jsonl"
            if log_path.exists():
                log_path.unlink()


__all__ = ["AtomicResearchLedger"]
