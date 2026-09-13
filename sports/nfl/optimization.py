"""NFL optimization adapter bindings.

Owns NFL's ``nfl_adapters`` — the sport-side half of the optimization
harness. Connects the shared sport-agnostic harness
(``core.experiments.adapters``) to NFL's EXISTING pieces: the nflverse
season caches, the feature engine, the venue lookup, and the study.yaml
fold geometry. Dependency direction is sports -> core.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from core.experiments.adapters import (
    STORE,
    matrix_builder,
    member_factory_for,
    raw_columns_fn,
    slate_check,
)
from core.experiments.search_space import ModelScope


def nfl_adapters(study) -> dict[str, dict]:
    """NFL moneyline + market scope adapters on the real NFL store."""
    from core.folds import make_folds
    from sports.nfl.features.build_frame import build_game_features, build_slate_features
    from sports.nfl.ingestion import eligible_games, load_pbp
    from sports.nfl.run_production import VENUE_CSV
    from sports.nfl.config.study_config import load_nfl_study

    study = study or load_nfl_study()
    cache = STORE / "nfl" / "raw"
    if not cache.is_dir():
        return {}
    seasons = sorted(int(p.stem.split("_")[-1])
                     for p in cache.glob("schedules_*.parquet"))
    if not seasons:
        return {}

    schedule = pd.concat(
        [pd.read_parquet(cache / f"schedules_{s}.parquet")
         for s in seasons], ignore_index=True)
    schedule = eligible_games(schedule, study.oof_first_season,
                              min(study.warmup_seasons))
    pbp = load_pbp(seasons, cache, full_repull=False)
    pbp_rollup = cache / "pbp_rollup.parquet"

    def load_decided() -> pd.DataFrame:
        decided_all = schedule[schedule["home_score"].notna()
                               & schedule["away_score"].notna()].copy()
        df = build_game_features(decided_all, pbp, VENUE_CSV,
                                 pbp_rollup_path=pbp_rollup)
        return df.sort_values("gameday").reset_index(drop=True)

    def load_slate() -> pd.DataFrame:
        return build_slate_features(schedule, pbp, VENUE_CSV)

    def build_folds(df: pd.DataFrame) -> list:
        folds = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col="gameday",
                           val_scope="oof_season", end_of_day=True)
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
                ("is_home", "is_dome_home", "altitude_home",
                 "rest_short_diff", "div_game", "prime_time",
                 "travel_miles_diff", "ewm_ypp_diff", "pace_plays_min_diff")),
            "slate_check": slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "gameday",  # NFL frames key time on gameday
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.experiments.sports import default_scopes
    ml, mk = default_scopes("nfl", tuple(study.moneyline_feature_cols))
    mk = dataclasses.replace(mk, prod_features=tuple(study.market_feature_cols))
    return {"nfl/moneyline": shared("moneyline", ml),
            "nfl/market": shared("market", mk)}
