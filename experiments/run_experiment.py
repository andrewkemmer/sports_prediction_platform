"""The ONE generic, configuration-driven experiment runner (§16).

Usage (Kaggle or local)::

    python experiments/run_experiment.py \
        --sport mlb --config experiments/configs/mlb_search.yaml

The runner reuses production code — walk-forward folds, the point-in-time
feature frame, the locked trainers/calibration, the evaluation and
validation gates — through the declarative ``OptimizationConfig`` +
``ModelScope`` surfaces and the per-sport adapter bindings
(``sports.<sport>.optimization.<sport>_adapters``). It never reimplements
production modeling logic, never writes reports to ``data_delivery/``,
never alters production contracts, and never pushes to GitHub. Every
experiment is a configuration; there are no per-experiment programs.

Scopes without a bound adapter (offline smoke, no durable store) are
reported as DEFERRED in the log — visible, never silently skipped.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

from core.config import load_config, normalize_sport_key
from core.experiments.optimizer import run_optimization
from core.experiments.search_space import load_optimization_config

logger = logging.getLogger("experiments.run_experiment")

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run_config(path: Path) -> dict:
    """Load a YAML/JSON experiment config file (fail loud)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # experiments-only dependency (never imported by tests/)
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: experiment config must be a mapping")
    return raw


def load_study(sport: str):
    """The sport's locked study config (production loader)."""
    mod = importlib.import_module(f"sports.{sport}.config.study_config")
    loader = getattr(mod, f"load_{sport}_study")
    return loader() if sport != "mlb" else mod.load_mlb_study()


def build_adapters(sport: str) -> dict:
    """Bind the production adapters for ``sport``'s scopes.

    Returns {} when the sport's durable store is unavailable (offline
    smoke): the optimizer reports each scope as DEFERRED — visible in
    the log, never silently skipped.
    """
    try:
        mod = importlib.import_module(f"sports.{sport}.optimization")
        factory = getattr(mod, f"{sport}_adapters")
        study = load_study(sport)
        if sport == "mlb":
            return dict(factory(study))
        return dict(factory(study))
    except FileNotFoundError as exc:
        logger.warning("[run_experiment] %s store unavailable: %s",
                       sport, exc)
        return {}


def resolve_scopes(sport: str, config) -> tuple:
    """The run's scopes: declared in the config, or derived from the
    sport's locked study contracts via the production
    ``default_scopes`` binding (moneyline + market, independent)."""
    if config.scopes:
        return config.scopes
    from core.experiments.sports import default_scopes
    study = load_study(sport)
    feats = tuple(dict.fromkeys(
        (*getattr(study, "moneyline_feature_cols"),
         *getattr(study, "market_feature_cols"))))
    return default_scopes(sport, feats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generic configuration-driven experiment runner (§16; "
                    "prints results to the log, writes no artifacts)")
    parser.add_argument("--sport", required=True,
                        choices=("mlb", "nfl", "nhl", "nba"))
    parser.add_argument("--config", required=True,
                        type=Path,
                        help="experiment config YAML/JSON under "
                             "experiments/configs/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(message)s")
    sport = normalize_sport_key(args.sport)
    raw = load_run_config(args.config)
    config = load_optimization_config(raw)
    scopes = resolve_scopes(sport, config)
    if not config.scopes:
        config = load_optimization_config({
            **raw,
            "scopes": [
                {"sport": s.sport, "model": s.model,
                 "raw_fields": list(s.raw_fields),
                 "prod_features": list(s.prod_features),
                 "target": s.target,
                 "horizon_field": s.horizon_field}
                for s in scopes
            ],
        })

    logger.info("[run_experiment] sport=%s config=%s scopes=%d",
                sport, args.config.name, len(config.scopes))
    adapters = build_adapters(sport)
    results = run_optimization(config, adapters)
    for key, summary in results.items():
        if summary.get("deferred"):
            logger.info("[run_experiment] %s: DEFERRED (no adapter bound)",
                        key)
        else:
            logger.info("[run_experiment] %s: %s", key,
                        json.dumps(summary, sort_keys=True, default=str))
    logger.info(
        "[run_experiment] done — results printed only; no "
        "data_delivery/ writes, no contract promotion (§16)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
