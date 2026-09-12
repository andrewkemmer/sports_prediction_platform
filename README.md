# sports_prediction_platform

Multi-sport prediction platform (MLB, NFL, NHL, NBA). This repository is the
successor to `sports_prediction_model` (read-only reference).

## Status

Phases 0–7.5 are complete and Phase 7.5 is closed. Phase 7.6 — the
conformance, rebuild, rewrite and cleanup phase — owns every remaining
carry-forward. The authoritative phase ledger, blocker table, deferred
findings and policy changes live in `docs/TRACKER.md`; the platform rules
live in `GUARDRAILS.md`. Phase state is deliberately not duplicated here.

## Layout

```
core/            shared platform code (contracts, config, storage, folds, OOF, optimization)
sports/mlb/      MLB pipeline: ingestion -> features -> training -> artifacts
sports/nfl/      NFL pipeline (same capability set, sport-specific modules documented in GUARDRAILS §22)
sports/nhl/      NHL pipeline
sports/nba/      NBA pipeline
frontend/        dashboard pages + artifact loaders (Streamlit)
experiments/     durable, allowlisted experiment scaffolding (empty by default)
tests/           closed, manifest-enforced production-contract suite
docs/            phase ledger and governance documents (`docs/TRACKER.md`)
store/           local raw / feature / model stores (gitignored, never committed)
```

Per-sport artifacts are written to `sports/<sport>/data_delivery/` (also
gitignored); ingestion and model stores stay under `store/`.

## Requirements

Python 3.10 — the version CI installs and the version every reported number
comes from (the pinned development environment is 3.10.21). A 3.11+
interpreter is neither required nor tested.

Dependencies are exactly pinned in `requirements-dev.txt`, which is the
authoritative manifest; `pyproject.toml` carries advisory floors only.

## Development

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest tests/ --strict-markers -q
```

`tests/` is a closed, manifest-listed suite (`tests/manifest.yaml`): every
file there has an entry, and hygiene plus the guardrail checks run inside the
normal suite. The dashboard needs the optional frontend extras
(`pip install -e ".[frontend]"`); CI exercises it through the loader/adapter
tests, which run without a Streamlit runtime.
