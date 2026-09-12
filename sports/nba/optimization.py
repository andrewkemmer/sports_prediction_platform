"""NBA optimization adapter bindings.

Owns NBA's ``nba_adapters`` — the sport-side half of the optimization
harness. Connects the shared sport-agnostic harness
(``core.optimization.adapters``) to NBA's EXISTING pieces: the schedule
store, the feature engine, and the study.yaml fold geometry. Dependency
direction is sports -> core.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from core.optimization.adapters import (
    STORE,
    matrix_builder,
    member_factory_for,
    raw_columns_fn,
    slate_check,
)
from core.optimization.models import ModelScope


def nba_adapters(study) -> dict[str, dict]:
    """NBA moneyline + market scope adapters on the real NBA store."""
    from core.folds import make_folds
    from sports.nba.features import build_game_features, build_slate_features
    from sports.nba.ingestion import load_schedule_cache
    from sports.nba.study_config import load_nba_study

    study = study or load_nba_study()
    path = STORE / "nba" / "raw" / "schedule.parquet"
    if not path.is_file():
        return {}
    seasons_start = min(study.warmup_seasons)
    schedule = load_schedule_cache(path)
    schedule = schedule[
        (pd.to_numeric(schedule["season"], errors="coerce") >= seasons_start)]

    def load_decided() -> pd.DataFrame:
        decided_all = schedule[schedule["home_score"].notna()
                               & schedule["away_score"].notna()].copy()
        return build_game_features(decided_all)

    def load_slate() -> pd.DataFrame:
        return build_slate_features(schedule)

    def build_folds(df: pd.DataFrame) -> list:
        folds = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col="game_date",
                           min_val_games=study.min_val_games)
        return [(list(f.train_idx), list(f.val_idx)) for f in folds]

    def shared(kind: str, scope: ModelScope) -> dict:
        raws = scope.raw_fields
        return {
            "load_decided": load_decided,
            "load_slate": load_slate,
            "build_folds": build_folds,
            "matrix": matrix_builder(raws),
            "member_factory": member_factory_for(kind),
            "raw_columns": raw_columns_fn(
                ("is_home", "travel_miles_diff", "conf_game", "prime_time")),
            "slate_check": slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "game_date",
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.optimization.sports import default_scopes
    ml, mk = default_scopes("nba", tuple(study.moneyline_feature_cols))
    # Each scope binds its OWN contract's list (Phase 7.5 Task 2 versioned
    # contracts — the market scope never inherits the moneyline's list).
    mk = dataclasses.replace(mk, prod_features=tuple(study.market_feature_cols))
    return {"nba/moneyline": shared("moneyline", ml),
            "nba/market": shared("market", mk)}
