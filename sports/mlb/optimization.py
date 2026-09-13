"""MLB optimization adapter bindings.

Owns MLB's ``mlb_adapters`` — the sport-side half of the optimization
harness. Connects the shared sport-agnostic harness
(``core.experiments.adapters``) to MLB's EXISTING pieces: the pitches
store, the DuckDB feature engine, the canonical decided frame, the PIT
Elo/records enrichment, and the frozen feature contracts. Nothing here
reaches back into ``core`` for sport knowledge (policy section 18: the
dependency direction is sports -> core).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.experiments.adapters import (
    STORE,
    matrix_builder,
    member_factory_for,
    slate_check,
)
from core.experiments.search_space import ModelScope


def mlb_adapters(study=None) -> dict[str, dict]:
    """MLB moneyline + market scope adapters on the real MLB store.

    Reuses the EXISTING MLB pieces unchanged: the pitches store, the
    DuckDB feature engine (``build_features``), the canonical decided
    frame (``get_decided_frame``), the PIT Elo/records enrichment, and
    the production feature composition (``build_candidate_frame`` — the
    exact frame production training consumes). No MLB production code
    changes.

    Scope frames (INDEPENDENT, mirroring production):
    * moneyline — the frozen 64-column MONEYLINE_FEATURE_COLS candidate
      frame; the incumbent view is that list, the candidate raw pool is
      its per-side members (the MLB convention: ``home_elo`` not
      ``elo_home``).
    * market — the frozen 53-column MARKET_FEATURE_COLS run-engine frame;
      the incumbent view is that list (minus its per-side members), the
      candidate raw pool is those per-side members. The target is the
      run-engine regressand ``margin`` (home_score - away_score on the
      decided rows only — never attached to slate rows).
    """
    from core.folds import walk_forward_splits
    from sports.mlb.feature_registry import (
        MARKET_FEATURE_COLS,
        MONEYLINE_FEATURE_COLS,
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

    ml_side, ml_incumbent = _split(MONEYLINE_FEATURE_COLS)
    run_side, run_incumbent = _split(MARKET_FEATURE_COLS)

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
        slate = ensure_feature_columns(slate, MONEYLINE_FEATURE_COLS)
        slate = ensure_feature_columns(slate, MARKET_FEATURE_COLS)
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
            "matrix": matrix_builder(scope.raw_fields),
            "member_factory": member_factory_for(kind),
            "raw_columns": raw_columns,
            "slate_check": slate_check,
            "team_cols": ("home_team", "away_team"),
            "date_col": "game_date",
            "leakage_extra": True,
            "_scope": scope,
        }

    from core.experiments.sports import mlb_scopes
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
