# sports_prediction_platform

Multi-sport prediction platform (MLB, NFL, NHL, NBA). This repository is the
successor scaffold to `sports_prediction_model` (read-only reference).

## Status — Phase 0 (scaffold)

Directories are in place; no pipelines, models, ingestion, or frontend logic
have been implemented yet. The existing MLB frontend/artifact audit lives in
the planning conversation, not in this repo.

## Layout

```
core/            shared platform code (artifact contracts, config, loaders)
sports/mlb/      MLB pipeline + artifacts (to be built)
sports/nfl/      NFL pipeline + artifacts (to be built)
sports/nhl/      NHL pipeline + artifacts (to be built)
sports/nba/      NBA pipeline + artifacts (to be built)
experiments/     ablations, probes, one-off studies
frontend/        dashboard (to be implemented)
tests/           test suite
```

## Development

```bash
pip install -r requirements-dev.txt   # or: pip install -e ".[dev]"
pytest
```
