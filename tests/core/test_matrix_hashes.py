"""Permanent evidence test: feature-matrix hashes match pinned incumbents.

Recomputes all 8 scope feature-matrix hashes (4 sports x moneyline/market)
from the committed bounded raw fixtures (materialized into the gitignored
store by ``tests/raw_store_fixtures.py``) using each scope's versioned
contract column order, and compares EXACTLY against the reviewed in-module
constants below.

Canonical hash definition (frozen with the original capture):
fixed row order (game-id mergesort), the scope's exact feature-column
order, float64 values with NaN preserved via a NaN-mask + zero-filled
body, SHA256 over (mask bytes || body bytes || repr(cols)).

The expected values were re-baselined during the Phase 7.5e-A rebaseline
against ``tests/fixtures/raw_store/*.parquet``, reviewed, and frozen here as
constants. The ORIGINAL Phase 7.5 capture values (from a frozen external
store that never entered the repository) are recorded in docs/TRACKER.md
for provenance. They are never regenerated, overwritten, or exported. These
tests are the durable neutrality proof: any change to feature content,
column order, or matrix width for any scope fails here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Pinned season ceiling (capture-time value). The Phase 7.5 evidence
# capture ran in 2026-09, when the historical ``date.today()``-derived
# ceiling evaluated to 2027; the value is pinned so the test is fully
# deterministic (no wall-clock dependence) and reproduces the
# capture-time season list exactly. Any change requires explicit
# reviewer approval in a phase report.
PINNED_CURRENT_SEASON = 2027

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _raw_store(raw_store):
    """Run every matrix-hash test against the committed raw fixtures."""
    return raw_store


# ---------------------------------------------------------------------------
# REVIEWED INCUMBENT CONSTANTS (frozen; never regenerate or export)
# Phase 7.5e-A rebaseline — see docs/TRACKER.md for the original values.
# ---------------------------------------------------------------------------
EXPECTED_MATRIX_HASHES: dict[str, dict[str, str]] = {
    "mlb": {
        "moneyline": "c9378dd21ff5fbce59416da39de676b3721fec19e853f541280e615b469fb133",
        "market": "b18eed799cc2ed17ceefcd886d42a2f1840cd580a47f5b242565e1ffe432f49b",
    },
    "nba": {
        "moneyline": "f99a7bd6c7ad1b9718b768cdb0aa013cc15ec34d5531fba80eaeac03e9f4cd51",
        "market": "f99a7bd6c7ad1b9718b768cdb0aa013cc15ec34d5531fba80eaeac03e9f4cd51",
    },
    "nfl": {
        "moneyline": "b22565667e2d4b91e93b29bfd9801a55128cb76871730a989560a0ca22ad996d",
        "market": "b22565667e2d4b91e93b29bfd9801a55128cb76871730a989560a0ca22ad996d",
    },
    "nhl": {
        "moneyline": "32a1d1ef8da19414330f10a027ac0dd726659e2002c6f53af1beb8c5b90e3470",
        "market": "32a1d1ef8da19414330f10a027ac0dd726659e2002c6f53af1beb8c5b90e3470",
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
    current_season = PINNED_CURRENT_SEASON
    seasons = sorted(set(
        list(range(study.oof_first_season, current_season + 1))
        + list(study.warmup_seasons)))
    # Deterministic, network-free cache-only loaders: the pinned store is
    # the recorded fixture; the default production transport is never hit.
    from tests.nfl_fixtures import cache_only_loader
    schedule = load_schedule(seasons, cache, full_repull=False,
                             load=cache_only_loader(cache, "schedules"))
    schedule = eligible_games(schedule, study.oof_first_season,
                              min(study.warmup_seasons))
    pbp = load_pbp(seasons, cache, full_repull=False,
                   load=cache_only_loader(cache, "pbp"))
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
