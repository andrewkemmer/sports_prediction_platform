"""Phase 7.5 — feature-matrix hash capture (before/after).

Canonical hash: fixed row order (chronological, game-id tiebreak), the
scope's exact feature-column order, float64 values with NaN preserved,
SHA256 over the row-major bytes. Per sport, both scopes (moneyline +
market) with their incumbent feature lists. Output:
/tmp/phase75/matrices_<tag>.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/daytona/codebase")

TAG = sys.argv[1] if len(sys.argv) > 1 else "before"
OUT = Path(f"/tmp/phase75/matrices_{TAG}.json")


def matrix_hash(df: pd.DataFrame, cols: tuple[str, ...],
                id_series: pd.Series) -> str:
    sub = df.reindex(columns=list(cols)).astype("float64")
    order = pd.DataFrame({"id": id_series}).sort_values(
        ["id"], kind="mergesort").index
    sub = sub.loc[order]
    arr = np.ascontiguousarray(sub.to_numpy(dtype=np.float64))
    # Canonical NaN encoding: bit-pattern is deterministic (quiet NaN), but
    # normalize by hashing bytes with NaN mapped to a fixed sentinel frame.
    nan_mask = np.isnan(arr).astype(np.uint8)
    arr_f = np.where(np.isnan(arr), 0.0, arr)
    h = hashlib.sha256()
    h.update(nan_mask.tobytes())
    h.update(arr_f.tobytes())
    h.update(repr(list(cols)).encode())
    return h.hexdigest()


def mlb():
    from sports.mlb.feature_registry import FEATURE_COLS, RUN_FEATURE_COLS, \
        build_candidate_frame
    from sports.mlb.frames import get_decided_frame
    from sports.mlb.features import build_features

    game_df, _ = build_features("store/mlb/raw/pitches.parquet",
                                output_dir="store/mlb/raw")
    df = get_decided_frame(game_df).sort_values(
        ["game_date", "game_pk"]).reset_index(drop=True)
    cand, _ = build_candidate_frame(df)
    ids = cand["game_pk"].astype(str)
    return {
        "moneyline": {"n_rows": len(cand), "n_cols": len(FEATURE_COLS),
                      "sha": matrix_hash(cand, FEATURE_COLS, ids)},
        "market": {"n_rows": len(cand), "n_cols": len(RUN_FEATURE_COLS),
                   "sha": matrix_hash(cand, RUN_FEATURE_COLS, ids)},
    }


def nba():
    from sports.nba.features import build_game_features
    from sports.nba.ingestion import load_schedule_cache
    from sports.nba.study_config import load_nba_study

    study = load_nba_study()
    ml_cols = tuple(study.feature_columns)
    # Market scope shares the same served pool in production today
    # (independent lists are Task 2's rename/remap, values unchanged).
    mk_cols = ml_cols
    sched = load_schedule_cache(Path("store/nba/raw/schedule.parquet"))
    sched = sched[(pd.to_numeric(sched["season"], errors="coerce")
                   >= min(study.warmup_seasons))]
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        ["game_date", "game_id"]).reset_index(drop=True)
    ids = df["game_id"].astype(str)
    return {
        "moneyline": {"n_rows": len(df), "n_cols": len(ml_cols),
                      "sha": matrix_hash(df, ml_cols, ids)},
        "market": {"n_rows": len(df), "n_cols": len(mk_cols),
                   "sha": matrix_hash(df, mk_cols, ids)},
    }


def nfl():
    from sports.nfl.features import build_game_features
    from sports.nfl.ingestion import eligible_games, load_pbp, load_schedule
    from sports.nfl.study_config import load_nfl_study

    study = load_nfl_study()
    ml_cols = tuple(study.feature_columns)
    mk_cols = ml_cols
    cache = Path("store/nfl/raw")
    from datetime import date
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
        "moneyline": {"n_rows": len(df), "n_cols": len(ml_cols),
                      "sha": matrix_hash(df, ml_cols, ids)},
        "market": {"n_rows": len(df), "n_cols": len(mk_cols),
                   "sha": matrix_hash(df, mk_cols, ids)},
    }


def nhl():
    from sports.nhl.features import build_game_features
    from sports.nhl.ingestion import load_schedule_cache
    from sports.nhl.study_config import load_nhl_study

    study = load_nhl_study()
    ml_cols = tuple(study.feature_columns)
    mk_cols = ml_cols
    sched = load_schedule_cache(Path("store/nhl/raw/schedule.parquet"))
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        ["game_date", "game_id"]).reset_index(drop=True)
    ids = df["game_id"].astype(str)
    return {
        "moneyline": {"n_rows": len(df), "n_cols": len(ml_cols),
                      "sha": matrix_hash(df, ml_cols, ids)},
        "market": {"n_rows": len(df), "n_cols": len(mk_cols),
                   "sha": matrix_hash(df, mk_cols, ids)},
    }


out = {}
for sport, fn in (("mlb", mlb), ("nba", nba), ("nfl", nfl), ("nhl", nhl)):
    try:
        out[sport] = fn()
        print(f"{sport}: " + " ".join(
            f"{k}={v['sha'][:12]}({v['n_rows']}x{v['n_cols']})"
            for k, v in out[sport].items()), flush=True)
    except Exception as exc:  # noqa: BLE001
        out[sport] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{sport}: ERROR {exc}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print("written", OUT)
