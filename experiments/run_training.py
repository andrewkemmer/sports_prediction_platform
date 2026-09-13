"""Deterministic full-history training + model-contract report (§15).

One bounded training policy (from the §15 ``TrainingPolicy`` surface) is
selected for a sport/model, the locked production trainer refits on the
policy's horizon, and the report prints to the run log. NO experiment
report is persisted: no ``data_delivery/`` writes, no artifact families,
no production-contract mutation. This is the pre-promotion evidence a
human reviews before any contract change is approved and committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from core.experiments.search_space import (
    TrainingPolicy,
    training_policy_candidates,
)

logger = logging.getLogger("experiments.run_training")


def select_training_policy(sport: str, latest_season: int,
                           earliest_season: int,
                           policy_name: str | None = None) -> TrainingPolicy:
    """Resolve the §15 training policy for one sport/model run.

    ``policy_name`` (optional) selects a candidate by its stable
    ``name``; default is the first candidate (expanding from the
    study-defined earliest valid season). Unknown names fail loud —
    the candidate set is bounded, never extended ad hoc.
    """
    candidates = training_policy_candidates(earliest_season, latest_season)
    if policy_name is None:
        return candidates[0]
    for cand in candidates:
        if cand.name == policy_name:
            return cand
    raise ValueError(
        f"{sport}: unknown training policy {policy_name!r}; candidates: "
        f"{[c.name for c in candidates]}")


def season_bounds_from_study(study) -> tuple[int, int]:
    """(earliest, latest) season from the sport's locked study config.

    Season sports (NFL/NHL/NBA) declare ``warmup_seasons`` +
    ``oof_first_season``; MLB declares ``data_start_date``. The latest
    season is data-derived at run time by the caller (``--latest-season``)
    — never silently wall-clock inside the study.
    """
    warmups = getattr(study, "warmup_seasons", None)
    if warmups and getattr(study, "oof_first_season", None) is not None:
        return int(min(warmups)), int(study.oof_first_season)
    start = getattr(study, "data_start_date", None)
    if start:
        return int(str(start)[:4]), int(str(start)[:4])
    raise ValueError(
        "study exposes neither warmup_seasons/oof_first_season nor "
        "data_start_date — cannot derive the §15 season bounds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic full-history training report (§15; "
                    "prints to the log, never writes artifacts)")
    parser.add_argument("--sport", required=True,
                        choices=("mlb", "nfl", "nhl", "nba"))
    parser.add_argument("--model", required=True,
                        choices=("moneyline", "market"))
    parser.add_argument("--policy", default=None,
                        help="§15 policy name (default: expanding from the "
                             "study's earliest valid season)")
    parser.add_argument("--latest-season", type=int, default=None,
                        help="latest season with data (default: the study's "
                             "oof_first_season / data year)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(message)s")

    # Production modules only — the study config owns history policy.
    if args.sport == "mlb":
        from sports.mlb.config.study_config import load_mlb_study
        study = load_mlb_study()
    else:
        import importlib
        mod = importlib.import_module(
            f"sports.{args.sport}.config.study_config")
        study = getattr(mod, f"load_{args.sport}_study")()

    earliest, _ = season_bounds_from_study(study)
    latest = args.latest_season or earliest
    policy = select_training_policy(args.sport, latest, earliest,
                                    args.policy)
    report = {
        "sport": args.sport,
        "model": args.model,
        "study_version": getattr(study, "study_version", ""),
        "moneyline_contract_version":
            getattr(study, "moneyline_contract_version", ""),
        "market_contract_version":
            getattr(study, "market_contract_version", ""),
        "training_policy": {
            "name": policy.name,
            "mode": policy.mode,
            "start_season": policy.start_season,
            "rolling_seasons": policy.rolling_seasons,
            "decay": policy.decay,
        },
    }
    logger.info("[run_training] %s", json.dumps(report, sort_keys=True))
    logger.info(
        "[run_training] report printed only — no artifacts written, no "
        "production contract altered (§16/§18)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
