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

import dataclasses
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
    ml, mk = default_scopes("nba", tuple(study.moneyline_feature_cols))
    # Each scope binds its OWN contract's list (Phase 7.5 Task 2 versioned
    # contracts — the market scope never inherits the moneyline's list).
    mk = dataclasses.replace(mk, prod_features=tuple(study.market_feature_cols))
    return {"nba/moneyline": shared("moneyline", ml),
            "nba/market": shared("market", mk)}


# ---------------------------------------------------------------------------
# NFL — nflverse season caches
# ---------------------------------------------------------------------------
def nfl_adapters(study) -> dict[str, dict]:
    """NFL moneyline + market scope adapters on the real NFL store."""
    from core.folds import make_folds
    from sports.nfl.features import build_game_features, build_slate_features
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
                           date_col="gameday",
                           val_scope="oof_season", end_of_day=True)
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
    ml, mk = default_scopes("nfl", tuple(study.moneyline_feature_cols))
    mk = dataclasses.replace(mk, prod_features=tuple(study.market_feature_cols))
    return {"nfl/moneyline": shared("moneyline", ml),
            "nfl/market": shared("market", mk)}


# ---------------------------------------------------------------------------
# NHL — schedule.parquet store
# ---------------------------------------------------------------------------
def nhl_adapters(study) -> dict[str, dict]:
    """NHL moneyline + market scope adapters on the real NHL store."""
    from core.folds import make_folds
    from sports.nhl.features import build_game_features, build_slate_features
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
    ml, mk = default_scopes("nhl", tuple(study.moneyline_feature_cols))
    mk = dataclasses.replace(mk, prod_features=tuple(study.market_feature_cols))
    return {"nhl/moneyline": shared("moneyline", ml),
            "nhl/market": shared("market", mk)}


# ---------------------------------------------------------------------------
# MLB — pitches.parquet store (DuckDB game_level + PIT Elo/records)
# ---------------------------------------------------------------------------
def mlb_adapters(study=None) -> dict[str, dict]:
    """MLB moneyline + market scope adapters on the real MLB store.

    Reuses the EXISTING MLB pieces unchanged: the pitches store, the
    DuckDB feature engine (``build_features``), the canonical decided
    frame (``get_decided_frame``), the PIT Elo/records enrichment, and
    the production feature composition (``build_candidate_frame`` — the
    exact frame production training consumes). No MLB production code
    changes.

    Scope frames (INDEPENDENT, mirroring production):
    * moneyline — the frozen 64-column FEATURE_COLS candidate frame; the
      incumbent view is that list, the candidate raw pool is its per-side
      members (the MLB convention: ``home_elo`` not ``elo_home``).
    * market — the frozen 53-column RUN_FEATURE_COLS run-engine frame;
      the incumbent view is that list (minus its per-side members), the
      candidate raw pool is those per-side members. The target is the
      run-engine regressand ``margin`` (home_score - away_score on the
      decided rows only — never attached to slate rows).
    """
    from core.folds import walk_forward_splits
    from sports.mlb.feature_registry import (
        FEATURE_COLS,
        RUN_FEATURE_COLS,
        build_candidate_frame,
        derive_diff_features,
        ensure_feature_columns,
    )
    from sports.mlb.frames import enrich_elo_and_records, get_decided_frame
    from sports.mlb.study_config import load_mlb_study

    study = study or load_mlb_study()
    cache = STORE / "mlb" / "raw" / "pitches.parquet"
    if not cache.is_file():
        return {}

    # Incumbent views: each scope's frozen feature list MINUS its own
    # per-side members (which form the derivation pool instead). The
    # split is TWIN-PAIR based, not suffix-based: a column joins the
    # derivation pool only when it has a complete home/away pair in the
    # frozen list (sp_era_home + sp_era_away -> pool). Its twin-checked
    # diff (sp_era_diff) stays served where production serves it.
    # Columns like is_home (no _away twin) and the enriched home_elo /
    # away_elo (pool members by MLB convention, no served _diff twin)
    # are classified correctly only by pair membership.
    def _split(frozen: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        frozen_set = set(frozen)
        pool, incumbent = [], []
        for f in frozen:
            if f.endswith("_home") and f"{f[:-len('_home')]}_away" in frozen_set:
                pool.append(f)
            elif f.endswith("_away") and f"{f[:-len('_away')]}_home" in frozen_set:
                pool.append(f)
            else:
                incumbent.append(f)
        return tuple(pool), tuple(incumbent)

    ml_side, ml_incumbent = _split(FEATURE_COLS)
    run_side, run_incumbent = _split(RUN_FEATURE_COLS)

    def load_decided() -> pd.DataFrame:
        game_df, _ = _mlb_feature_frames(cache)
        decided = get_decided_frame(game_df)
        df, _cols = build_candidate_frame(decided)
        # Run-engine regressand (decided rows only; the production run
        # engine derives the same quantity from the score columns). Slate
        # rows never carry it.
        df["margin"] = (pd.to_numeric(df["home_score"], errors="coerce")
                        - pd.to_numeric(df["away_score"], errors="coerce"))
        return df.sort_values("game_date").reset_index(drop=True)

    def load_slate() -> pd.DataFrame:
        game_df, _ = _mlb_feature_frames(cache)
        pending = game_df[game_df["home_win"].isna()].copy()
        if pending.empty:
            # No undecided games: an empty slate with the identity columns
            # the harness expects (no fabricated rows, no fabricated fields).
            return pending.assign(**{c: pd.Series(dtype="float64")
                                     for c in ("game_date", "home_team",
                                               "away_team")})
        # Pre-game rows: PIT-carried state only (the production
        # _build_slate -> observed-filter -> derive_diff_features path,
        # runner.py line-for-line). Outcome columns are dropped — a slate
        # frame never carries a game result. Both frozen views' columns
        # are then guaranteed present (missing -> NaN), the production
        # ensure_feature_columns contract: unavailable observations ship
        # true NULLs and are routed (native NaN for trees, train-median
        # for the linear path) — never fabricated.
        observed = [c for c in pending.columns if pending[c].notna().any()]
        slate = derive_diff_features(
            enrich_elo_and_records(pending[observed], rename_team_woba=True))
        slate = slate.drop(columns=[c for c in
                                    ("home_win", "margin", "total",
                                     "home_score", "away_score",
                                     "total_runs")
                                    if c in slate.columns])
        slate = ensure_feature_columns(slate, FEATURE_COLS)
        slate = ensure_feature_columns(slate, RUN_FEATURE_COLS)
        if "game_date" not in slate.columns:
            raise ValueError(
                "mlb adapter: slate lost game_date after composition")
        return slate.sort_values("game_date").reset_index(drop=True)

    def build_folds(df: pd.DataFrame) -> list:
        # Study.yaml geometry, unchanged: weekly validation windows,
        # expanding train, warmup skip — plus the study's OOF rules (min
        # training rows) and min_val_games, exactly the folds production
        # train_moneyline_ensemble keeps.
        splits = walk_forward_splits(
            df, retrain_cadence_days=study.walk_forward.step_days,
            max_eval_folds=study.max_eval_folds,
            min_train_days=study.training.min_train_days_warmup)
        out = []
        for s in splits:
            tr = s["train_games"].index.to_numpy()
            va = s["val_games"].index.to_numpy()
            if len(tr) < study.oof.min_training_rows:
                continue  # study.yaml OOF rule: min training rows per fold
            if len(va) < study.min_val_games:
                continue  # production min-val rule (fold skipped)
            out.append((list(tr), list(va)))
        return out

    def shared(kind: str, scope: ModelScope) -> dict:
        # The scope's declared raw pool IS the candidate catalog source:
        # raw_columns returns exactly the declared per-side members that
        # are present and numeric on the frame (the runner builds the
        # catalog from this selector). Declared-but-absent fields would
        # never become candidates; non-declared frame columns (score
        # totals, record counts, std/delta variants) can never leak in.
        declared = scope.raw_fields

        def raw_columns(df: pd.DataFrame) -> tuple[str, ...]:
            return tuple(f for f in declared
                         if f in df.columns
                         and pd.api.types.is_numeric_dtype(df[f]))

        return {
            "load_decided": load_decided,
            "load_slate": load_slate,
            "build_folds": build_folds,
            "matrix": _matrix_builder(scope.raw_fields),
            "member_factory": _member_factory_for(kind),
            "raw_columns": raw_columns,
            "slate_check": _slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "game_date",
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.optimization.sports import mlb_scopes
    ml, mk = mlb_scopes(ml_incumbent, run_incumbent)
    return {"mlb/moneyline": shared("moneyline", ml),
            "mlb/market": shared("market", mk)}

def _mlb_feature_frames(cache: Path):
    """Build (game_df, pbp_df) from the pitches cache with a frame cache
    so both scopes bind one build per process."""
    key = str(cache)
    if key not in _MLB_FRAME_CACHE:
        from sports.mlb.features import build_features

        _MLB_FRAME_CACHE[key] = build_features(cache,
                                               output_dir=cache.parent)
    return _MLB_FRAME_CACHE[key]


_MLB_FRAME_CACHE: dict[str, tuple] = {}


def all_adapters() -> dict[str, dict]:
    """Every sport's adapters for sports with a real store; sports
    without one contribute nothing (their scopes report DEFERRED)."""
    out: dict[str, dict] = {}
    for fn in (nba_adapters, nfl_adapters, nhl_adapters, mlb_adapters):
        try:
            out.update(fn(None))
        except Exception as exc:  # noqa: BLE001 — a sport's store problems
            logger.warning("adapter binding failed for %s: %s",
                           fn.__name__, exc)
    return out
