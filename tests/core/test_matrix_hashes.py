"""Permanent evidence test: feature-matrix hashes match pinned incumbents.

Recomputes all 8 scope feature-matrix hashes (4 sports x moneyline/market)
from the REAL stores using each scope's versioned contract column order,
and compares EXACTLY against the reviewed in-module constants below.

Canonical hash definition (frozen with the original capture):
fixed row order (game-id mergesort), the scope's exact feature-column
order, float64 values with NaN preserved via a NaN-mask + zero-filled
body, SHA256 over (mask bytes || body bytes || repr(cols)).

The expected values were captured during the Phase 7.5 remediation against
the real stores, reviewed, and frozen here as constants. They are never
regenerated, overwritten, or exported. These tests are the durable
neutrality proof: any change to feature content, column order, or matrix
width for any scope fails here.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# REVIEWED INCUMBENT CONSTANTS (frozen; never regenerate or export)
# ---------------------------------------------------------------------------
EXPECTED_MATRIX_HASHES: dict[str, dict[str, str]] = {
    "mlb": {
        "moneyline": "3b6962f300932b98893f5276cf63a93862c619251f811898dd940c178c32dc2f",
        "market": "9ddfc57aa896e43482b8ff6b36a00888f04395dfa11807187679008e67b1be68",
    },
    "nba": {
        "moneyline": "c333e30852276d3015c0d77826c3e934de48f02a3e495a11190d26f99ecaa34d",
        "market": "c333e30852276d3015c0d77826c3e934de48f02a3e495a11190d26f99ecaa34d",
    },
    "nfl": {
        "moneyline": "8f860e483a27617f9080e4c01f0ee5a16d14023edf4bde6c1907a70b9647b545",
        "market": "8f860e483a27617f9080e4c01f0ee5a16d14023edf4bde6c1907a70b9647b545",
    },
    "nhl": {
        "moneyline": "3f8c0fef4d1dfa1063748e710d99c11edeb62f10bd7ad29abe0cf0a64dba16c0",
        "market": "3f8c0fef4d1dfa1063748e710d99c11edeb62f10bd7ad29abe0cf0a64dba16c0",
    },
}


def matrix_hash(df: pd.DataFrame, cols, id_series: pd.Series) -> str:
    sub = df.reindex(columns=list(cols)).astype("float64")
    order = pd.DataFrame({"id": id_series}).sort_values(
        ["id"], kind="mergesort").index
    sub = sub.loc[order]
    arr = np.ascontiguousarray(sub.to_numpy(dtype=np.float64))
    nan_mask = np.isnan(arr).astype(np.uint8)
    arr_f = np.where(np.isnan(arr), 0.0, arr)
    h = hashlib.sha256()
    h.update(nan_mask.tobytes())
    h.update(arr_f.tobytes())
    h.update(repr(list(cols)).encode())
    return h.hexdigest()


def _mlb():
    from sports.mlb.feature_registry import (MARKET_FEATURE_COLS,
                                             MONEYLINE_FEATURE_COLS,
                                             build_candidate_frame)
    from sports.mlb.features import build_features
    from sports.mlb.frames import get_decided_frame

    game_df, _ = build_features("store/mlb/raw/pitches.parquet",
                                output_dir="store/mlb/raw")
    df = get_decided_frame(game_df).sort_values(
        ["game_date", "game_pk"]).reset_index(drop=True)
    cand, _ = build_candidate_frame(df)
    ids = cand["game_pk"].astype(str)
    return {
        "moneyline": matrix_hash(cand, MONEYLINE_FEATURE_COLS, ids),
        "market": matrix_hash(cand, MARKET_FEATURE_COLS, ids),
    }


def _nba():
    from sports.nba.feature_registry import (MARKET_FEATURE_COLS,
                                             MONEYLINE_FEATURE_COLS)
    from sports.nba.features import build_game_features
    from sports.nba.ingestion import load_schedule_cache
    from sports.nba.study_config import load_nba_study

    study = load_nba_study()
    sched = load_schedule_cache(Path("store/nba/raw/schedule.parquet"))
    sched = sched[(pd.to_numeric(sched["season"], errors="coerce")
                   >= min(study.warmup_seasons))]
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        ["game_date", "game_id"]).reset_index(drop=True)
    ids = df["game_id"].astype(str)
    return {
        "moneyline": matrix_hash(df, MONEYLINE_FEATURE_COLS, ids),
        "market": matrix_hash(df, MARKET_FEATURE_COLS, ids),
    }


def _nfl():
    from sports.nfl.feature_registry import (MARKET_FEATURE_COLS,
                                             MONEYLINE_FEATURE_COLS)
    from sports.nfl.features import build_game_features
    from sports.nfl.ingestion import eligible_games, load_pbp, load_schedule
    from sports.nfl.study_config import load_nfl_study

    study = load_nfl_study()
    cache = Path("store/nfl/raw")
    current_season = date.today().year + 1 if date.today().month >= 3 \
        else date.today().year
    seasons = sorted(set(
        list(range(study.oof_first_season, current_season + 1))
        + list(study.warmup_seasons)))
    schedule = load_schedule(seasons, cache, full_repull=False)
    schedule = eligible_games(schedule, study.oof_first_season,
                              min(study.warmup_seasons))
    pbp = load_pbp(seasons, cache, full_repull=False)
    decided = schedule[schedule["home_score"].notna()
                       & schedule["away_score"].notna()].copy()
    pbp_rollup = cache / "pbp_rollup.parquet"
    df = build_game_features(decided, pbp, None, pbp_rollup_path=pbp_rollup)
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    ids = df["game_id"].astype(str)
    return {
        "moneyline": matrix_hash(df, MONEYLINE_FEATURE_COLS, ids),
        "market": matrix_hash(df, MARKET_FEATURE_COLS, ids),
    }


def _nhl():
    from sports.nhl.feature_registry import (MARKET_FEATURE_COLS,
                                             MONEYLINE_FEATURE_COLS)
    from sports.nhl.features import build_game_features
    from sports.nhl.ingestion import load_schedule_cache
    from sports.nhl.study_config import load_nhl_study

    study = load_nhl_study()
    sched = load_schedule_cache(Path("store/nhl/raw/schedule.parquet"))
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        ["game_date", "game_id"]).reset_index(drop=True)
    ids = df["game_id"].astype(str)
    return {
        "moneyline": matrix_hash(df, MONEYLINE_FEATURE_COLS, ids),
        "market": matrix_hash(df, MARKET_FEATURE_COLS, ids),
    }


_BUILDERS = {"mlb": _mlb, "nba": _nba, "nfl": _nfl, "nhl": _nhl}


@pytest.mark.parametrize("sport", sorted(_BUILDERS))
def test_matrix_hashes_match_pinned_incumbents(sport: str) -> None:
    pinned = EXPECTED_MATRIX_HASHES[sport]
    live = _BUILDERS[sport]()
    for scope in ("moneyline", "market"):
        assert live[scope] == pinned[scope], (
            f"{sport}/{scope}: feature-matrix hash drifted from the pinned "
            f"incumbent (content, column order, or width changed)")
