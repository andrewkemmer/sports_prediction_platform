"""Phase 7.5 — fold fingerprint capture (before/after).

Captures, per sport, from the REAL stores and the per-sport study.yaml:
  * fold count
  * per-fold train/val row counts
  * SHA256 of the ordered train/val game IDs per fold
  * fold date boundaries

Uses the CURRENT per-sport fold machinery (sports/<sport> fold modules
before Task 1; core.folds after). Output: /tmp/phase75/fingerprints_<tag>.json
(log-only, /tmp).
"""
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/daytona/codebase")

TAG = sys.argv[1] if len(sys.argv) > 1 else "before"
OUT = Path(f"/tmp/phase75/fingerprints_{TAG}.json")


def _sha_ids(ids) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode())
        h.update(b"|")
    return h.hexdigest()


def _fold_records(folds, df, id_col, date_col):
    recs = []
    for tr_idx, va_idx in folds:
        tr = df.iloc[sorted(tr_idx)]
        va = df.iloc[sorted(va_idx)]
        recs.append({
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "train_ids_sha": _sha_ids(tr[id_col].tolist()),
            "val_ids_sha": _sha_ids(va[id_col].tolist()),
            "train_date_min": str(pd.to_datetime(tr[date_col]).min().date()),
            "train_date_max": str(pd.to_datetime(tr[date_col]).max().date()),
            "val_date_min": str(pd.to_datetime(va[date_col]).min().date()),
            "val_date_max": str(pd.to_datetime(va[date_col]).max().date()),
        })
    return recs


def mlb():
    from sports.mlb.features import build_features
    from sports.mlb.frames import get_decided_frame
    from sports.mlb.study_config import load_mlb_study
    from sports.mlb.training import walk_forward_splits

    study = load_mlb_study()
    game_df, _ = build_features("store/mlb/raw/pitches.parquet",
                                output_dir="store/mlb/raw")
    df = get_decided_frame(game_df).sort_values("game_date").reset_index(drop=True)
    splits = walk_forward_splits(
        df, retrain_cadence_days=study.walk_forward.step_days,
        max_eval_folds=study.max_eval_folds,
        min_train_days=study.training.min_train_days_warmup)
    folds = [(s["train_games"].index.to_numpy(), s["val_games"].index.to_numpy())
             for s in splits
             if len(s["train_games"]) >= study.oof.min_training_rows
             and len(s["val_games"]) >= study.min_val_games]
    return {"n_folds": len(folds), "folds": _fold_records(folds, df,
                                                          "game_pk", "game_date")}


def nba():
    from core.folds import make_folds
    from sports.nba.features import build_game_features
    from sports.nba.ingestion import load_schedule_cache
    from sports.nba.study_config import load_nba_study

    study = load_nba_study()
    sched = load_schedule_cache(Path("store/nba/raw/schedule.parquet"))
    sched = sched[(pd.to_numeric(sched["season"], errors="coerce")
                   >= min(study.warmup_seasons))]
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values("game_date").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="game_date",
                    min_val_games=study.min_val_games,
                    val_scope="all", end_of_day=False)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return {"n_folds": len(folds), "folds": _fold_records(folds, df,
                                                          "game_id", "game_date")}


def nfl():
    from core.folds import make_folds
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
    df = build_game_features(decided, pbp, None,
                             pbp_rollup_path=pbp_rollup)
    df = df.sort_values("gameday").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="gameday",
                    val_scope="oof_season", end_of_day=True)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return {"n_folds": len(folds), "folds": _fold_records(folds, df,
                                                          "game_id", "gameday")}


def nhl():
    from core.folds import make_folds
    from sports.nhl.features import build_game_features
    from sports.nhl.ingestion import load_schedule_cache
    from sports.nhl.study_config import load_nhl_study

    study = load_nhl_study()
    sched = load_schedule_cache(Path("store/nhl/raw/schedule.parquet"))
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values("game_date").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="game_date",
                    min_val_games=study.min_val_games,
                    val_scope="all", end_of_day=False)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return {"n_folds": len(folds), "folds": _fold_records(folds, df,
                                                          "game_id", "game_date")}


out = {}
for sport, fn in (("mlb", mlb), ("nba", nba), ("nfl", nfl), ("nhl", nhl)):
    try:
        out[sport] = fn()
        print(f"{sport}: {out[sport]['n_folds']} folds", flush=True)
    except Exception as exc:  # noqa: BLE001
        out[sport] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{sport}: ERROR {exc}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print("written", OUT)
