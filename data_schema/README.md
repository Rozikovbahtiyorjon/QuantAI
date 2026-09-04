# Data Schema

Canonical location for QuantAI data contracts and feature schemas.

## Files

- `FEATURE_SCHEMA.json` — full feature schema (25 active + 5 planned) generated from `src/feature_engine.py`
- `FEATURE_SCHEMA.min.json` — minimal schema (active/planned lists) for quick validation
- `.gitkeep` — keeps directory in repo

## Generation

```bash
python scripts/generate_feature_schema.py
```

This reads `src/feature_engine.py` and writes `config/FEATURE_SCHEMA.min.json`.
Canonical copies are kept in `data_schema/` for audit and validation gates.

## Mirroring

`config/FEATURE_SCHEMA.json` is the runtime config location (imported by settings/validation).
`data_schema/` is the canonical audit location. Both are kept in sync — `data_schema` is intentionally NOT gitignored
(see `.gitignore` exceptions `!data_schema/`).

## Validation

- Walk-forward, PurgedKFold and data gates validate feature hashes against this schema.
- See `src/research/experiment_registry.py` (feature_schema_hash) and `src/data/data_gates.py`.
