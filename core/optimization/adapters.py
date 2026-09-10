"""Sport adapter bindings for the optimization runner.

Each adapter connects the shared harness to a sport's EXISTING pieces —
its durable store (store/<sport>/raw), its feature engine (the same
production derivations), its fold module (study.yaml geometry), and its
training member factory (production imputation / NaN routing). No new
ingestion, derivation, or model code is introduced here.

Scope kinds:
* moneyline — binary home_win probability; the linear member path uses
  the production train-only imputation + standard-scaling policy.
* market — continuous margin regressand (the distributional engine's
  target); evaluated through the same walk-forward OOF machinery with a
  ridge member on the same preprocessing. Bounds and gates are identical
  and independent per model.

A sport without a real store returns {} — the runner reports the scope
as DEFERRED (never fabricated, never silently skipped).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.optimization.candidates import (
    spec_from_name,
)
from core.optimization.models import ModelScope

logger = logging.getLogger("core.optimization")

STORE = Path("store")

# Identity/audit columns never eligible as candidate raw fields.
_META_COLS = frozenset({
    "game_id", "game_date", "gameday", "season", "week", "home_team",
    "away_team", "start_time_utc", "venue", "game_state", "game_type",
    "neutral_site", "home_score", "away_score", "home_win", "margin",
    "total", "home_record", "away_record", "game_pk",
})


# ---------------------------------------------------------------------------
# Shared matrix / member machinery (production policy, per scope kind)
# ---------------------------------------------------------------------------
def _fit_linear_preprocessor(X: pd.DataFrame):
    """Production TrainFoldPreprocessor policy: train-median imputation
    + standard scaling, all-NaN columns impute to neutral 0.0."""
    medians = X.median(numeric_only=True)
    means = X.mean(numeric_only=True)
    stds = X.std(numeric_only=True).replace(0.0, 1.0)
    return (medians.fillna(0.0), means.fillna(0.0), stds.fillna(1.0))


def _apply_preprocessor(X: pd.DataFrame, med, mean, std) -> np.ndarray:
    Xf = X.astype(float).fillna(med)
    return ((Xf - mean) / std).to_numpy(dtype=np.float64)


def _matrix_builder(raw_fields: tuple[str, ...]):
    """Matrix builder: incumbent features come from the frame's served
    columns; generated candidates resolve through the shared derivation
    evaluator via their deterministic names (identical evaluator in both
    places). Unknown columns ship true NaN with explicit routing."""

    def matrix(df: pd.DataFrame, feats) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for f in feats:
            if f in df.columns:
                cols[f] = df[f].astype(float)
                continue
            spec = spec_from_name(f, raw_fields)
            if spec is None:
                cols[f] = pd.Series(np.nan, index=df.index)
                continue
            from core.optimization.candidates import derive_column

            cols[f] = derive_column(
                df, spec, team_cols=("home_team", "away_team"),
                date_col="game_date").astype(float)
        return pd.DataFrame(cols, index=df.index)

    return matrix


def _raw_columns_fn(exclude: tuple[str, ...] = ()):
    def raw_columns(df: pd.DataFrame) -> tuple[str, ...]:
        return tuple(c for c in df.columns
                     if c not in _META_COLS and c not in exclude
                     and pd.api.types.is_numeric_dtype(df[c]))
    return raw_columns


def _member_factory(kind: str, seed: int):
    """Production-policy member factory for one fold fit.

    moneyline: logistic regression on train-imputed + scaled values
    (same policy class as the production linear members; trees route NaN
    natively and the harness's linear proxy is the bounded-sweep stand-in
    across all candidates identically). predict returns PROBABILITIES.
    market: ridge regression on the same preprocessing (the distribution
    engine's linear regressand path); predict returns the regressand.
    """
    state: dict = {}

    def fit(X: pd.DataFrame, y: np.ndarray) -> None:
        med, mean, std = _fit_linear_preprocessor(X)
        state["pre"] = (med, mean, std)
        if kind == "market":
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=1.0, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression

            m = LogisticRegression(max_iter=1000, random_state=seed)
        m.fit(_apply_preprocessor(X, med, mean, std), y)
        state["model"] = m
        state["kind"] = kind

    def predict(X: pd.DataFrame) -> np.ndarray:
        med, mean, std = state["pre"]
        Xp = _apply_preprocessor(X, med, mean, std)
        if state["kind"] == "market":
            return state["model"].predict(Xp)
        return state["model"].predict_proba(Xp)[:, 1]

    return {"fit": fit, "predict": predict}


def _member_factory_for(kind: str):
    def factory(seed: int):
        return _member_factory(kind, seed)
    return factory


def _slate_check(slate: pd.DataFrame, feats) -> bool:
    return all(f in slate.columns for f in feats)


# ---------------------------------------------------------------------------
# NBA — schedule.parquet store
# ---------------------------------------------------------------------------
def nba_adapters(study) -> dict[str, dict]:
    """NBA moneyline + market scope adapters on the real NBA store."""
    from sports.nba.features import build_game_features, build_slate_features
    from sports.nba.folds import make_folds
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
            "matrix": _matrix_builder(raws),
            "member_factory": _member_factory_for(kind),
            "raw_columns": _raw_columns_fn(
                ("is_home", "travel_miles_diff", "conf_game", "prime_time")),
            "slate_check": _slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "game_date",
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.optimization.sports import default_scopes
    ml, mk = default_scopes("nba", tuple(study.feature_columns))
    return {"nba/moneyline": shared("moneyline", ml),
            "nba/market": shared("market", mk)}


# ---------------------------------------------------------------------------
# NFL — nflverse season caches
# ---------------------------------------------------------------------------
def nfl_adapters(study) -> dict[str, dict]:
    """NFL moneyline + market scope adapters on the real NFL store."""
    from sports.nfl.features import build_game_features, build_slate_features
    from sports.nfl.folds import make_folds
    from sports.nfl.ingestion import eligible_games, load_pbp
    from sports.nfl.runner import VENUE_CSV
    from sports.nfl.study_config import load_nfl_study

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
                           date_col="gameday")
        return [(list(f.train_idx), list(f.val_idx)) for f in folds]

    def shared(kind: str, scope: ModelScope) -> dict:
        raws = scope.raw_fields
        return {
            "load_decided": load_decided,
            "load_slate": load_slate,
            "build_folds": build_folds,
            "matrix": _matrix_builder(raws),
            "member_factory": _member_factory_for(kind),
            "raw_columns": _raw_columns_fn(
                ("is_home", "is_dome_home", "altitude_home",
                 "rest_short_diff", "div_game", "prime_time",
                 "travel_miles_diff", "ewm_ypp_diff", "pace_plays_min_diff")),
            "slate_check": _slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "gameday",  # NFL frames key time on gameday
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.optimization.sports import default_scopes
    ml, mk = default_scopes("nfl", tuple(study.feature_columns))
    return {"nfl/moneyline": shared("moneyline", ml),
            "nfl/market": shared("market", mk)}


# ---------------------------------------------------------------------------
# NHL — schedule.parquet store
# ---------------------------------------------------------------------------
def nhl_adapters(study) -> dict[str, dict]:
    """NHL moneyline + market scope adapters on the real NHL store."""
    from sports.nhl.features import build_game_features, build_slate_features
    from sports.nhl.folds import make_folds
    from sports.nhl.ingestion import load_schedule_cache
    from sports.nhl.study_config import load_nhl_study

    study = study or load_nhl_study()
    path = STORE / "nhl" / "raw" / "schedule.parquet"
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
            "matrix": _matrix_builder(raws),
            "member_factory": _member_factory_for(kind),
            "raw_columns": _raw_columns_fn(
                ("is_home", "travel_miles_diff", "div_game", "prime_time")),
            "slate_check": _slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "game_date",
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.optimization.sports import default_scopes
    ml, mk = default_scopes("nhl", tuple(study.feature_columns))
    return {"nhl/moneyline": shared("moneyline", ml),
            "nhl/market": shared("market", mk)}


def all_adapters() -> dict[str, dict]:
    """Every sport's adapters for sports with a real store; sports
    without one contribute nothing (their scopes report DEFERRED)."""
    out: dict[str, dict] = {}
    for fn in (nba_adapters, nfl_adapters, nhl_adapters):
        try:
            out.update(fn(None))
        except Exception as exc:  # noqa: BLE001 — a sport's store problems
            logger.warning("adapter binding failed for %s: %s",
                           fn.__name__, exc)
    return out
