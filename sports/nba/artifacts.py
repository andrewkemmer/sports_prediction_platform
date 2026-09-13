"""NBA artifact writers.

Every writer validates its output against the immutable schema fixture
(``tests/fixtures/nba_artifact_schemas.json``) — filenames, column names
and order, JSON key structure, dtypes, required fields, and null behavior
pinned by the Phase 5 fixture. Writers fail loudly on any mismatch;
nothing is silently altered.

Market meanings stay NBA-specific: the moneyline is two-way (an NBA game
cannot tie — OT resolves every game, TIE is structurally impossible);
integer spreads/totals carry push mass.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.contracts import require_features_metadata
from core.contracts import validate_record
from sports.nba.feature_registry import MONEYLINE_CONTRACT_VERSION

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" \
    / "nba_artifact_schemas.json"

_MARKETS_COLS_CACHE: list[str] | None = None


class NBAArtifactError(RuntimeError):
    """Raised when an NBA artifact would violate the schema fixture."""


def _fixture() -> dict:
    if not FIXTURE_PATH.is_file():
        raise NBAArtifactError(f"schema fixture missing: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _json_safe(obj):
    """Recursively make a structure JSON-strict (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _r4(v):
    try:
        f = float(v)
        return round(f, 4) if f == f else None
    except (TypeError, ValueError):
        return v


def markets_columns() -> list[str]:
    """The NBA markets schema from the fixture (base + basketball grids)."""
    global _MARKETS_COLS_CACHE
    if _MARKETS_COLS_CACHE is None:
        fx = _fixture()
        cols = fx["artifacts"]["markets"]["columns"]
        _MARKETS_COLS_CACHE = list(cols)
    return list(_MARKETS_COLS_CACHE)


# ---------------------------------------------------------------------------
# JSON families
# ---------------------------------------------------------------------------


def write_moneyline_json(path: Path, slate_df: pd.DataFrame,
                         p_home_cal: np.ndarray,
                         team_names: dict[str, str],
                         config_meta: dict) -> dict:
    """The games[] card contract (fixture: moneyline_json)."""
    fx = _fixture()["artifacts"]["moneyline_json"]
    games = []
    slate = slate_df.reset_index(drop=True)
    for i in range(len(slate)):
        g = slate.iloc[i]
        ph = None
        if i < len(p_home_cal) and np.isfinite(p_home_cal[i]):
            ph = float(p_home_cal[i])
        pick = None
        if ph is not None:
            pick = str(g["home_team"]) if ph >= 0.5 else str(g["away_team"])
        gd = str(g.get("game_date", "") or "")
        st = g.get("start_time_utc")
        start = str(st) if st else \
            (f"{gd}T00:00:00Z" if gd else None)
        games.append({k: _json_safe(v) for k, v in {
            "game_id": g["game_id"],
            "game_date": gd,
            "start_time_utc": start,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_team_name": team_names.get(g["home_team"],
                                             g["home_team"]),
            "away_team_name": team_names.get(g["away_team"],
                                             g["away_team"]),
            "home_record": g.get("home_record") or None,
            "away_record": g.get("away_record") or None,
            "venue": g.get("venue") or None,
            "game_status": "pre",
            "home_score": None,
            "away_score": None,
            "home_win_prob_model": ph,
            "away_win_prob_model": (1.0 - ph) if ph is not None else None,
            "model_pick": pick,
            "model_correct": None,
        }.items()})
    record = {
        "created_utc": _now_utc(),
        "config": config_meta,
        "slate_date": str(pd.to_datetime(slate["game_date"]).min().date())
        if len(slate) else None,
        "n_games": len(games),
        "games": games,
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"moneyline_json key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "moneyline_v1", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


def write_calibration_json(path: Path, moneyline_metrics: dict,
                           calibrated_metrics: dict, buckets: list[dict],
                           daily: list[dict], config_meta: dict,
                           platt: dict | None = None, run_date: str = "",
                           n_games: int = 0,
                           calibrated_buckets: list[dict] | None = None
                           ) -> dict:
    """MLB/NFL/NHL-shaped calibration artifact (fixture: calibration_json)."""
    fx = _fixture()["artifacts"]["calibration_json"]
    cal_sec: dict = {}
    if isinstance(platt, dict) and platt.get("a") is not None:
        n = int(platt.get("n") or n_games or 0)
        cal_sec = {
            "method": "platt",
            "params": {"a": _r4(platt.get("a")), "b": _r4(platt.get("b")),
                       "n": n},
            "metrics_raw": {
                "brier": _r4(moneyline_metrics.get("brier")),
                "logloss": _r4(moneyline_metrics.get("logloss")),
                "ece": _r4(moneyline_metrics.get("ece")),
            },
            "metrics_calibrated": {
                "brier": _r4(calibrated_metrics.get("brier")),
                "logloss": _r4(calibrated_metrics.get("logloss")),
                "ece": _r4(calibrated_metrics.get("ece")),
            },
        }
        if calibrated_buckets:
            cal_sec["calibration_buckets_calibrated"] = calibrated_buckets
    record = {
        "date": run_date,
        "trained_at": _now_utc(),
        "n_games": int(n_games),
        "created_utc": _now_utc(),
        "config": config_meta,
        "metrics": {
            "auc": _r4(moneyline_metrics.get("auc")),
            "brier": _r4(moneyline_metrics.get("brier")),
            "logloss": _r4(moneyline_metrics.get("logloss")),
            "ece": _r4(moneyline_metrics.get("ece")),
            "ece_calibrated": _r4(calibrated_metrics.get("ece")),
            "auc_calibrated": _r4(calibrated_metrics.get("auc")),
            "brier_calibrated": _r4(calibrated_metrics.get("brier")),
            "logloss_calibrated": _r4(calibrated_metrics.get("logloss")),
        },
        "calibration": cal_sec,
        "calibration_buckets": buckets,
        "daily": [{**row, "metrics": {k: _r4(v)
                                      for k, v in (row.get("metrics")
                                                   or {}).items()}}
                  for row in daily],
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"calibration_json key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "calibration", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


def write_player_matchup_json(path: Path, player_df: pd.DataFrame,
                              slate_df: pd.DataFrame) -> dict:
    """Participant enrichment contract (fixture: player_matchup).
    Unavailable players are None (the frontend renders TBD) — never
    fabricated."""
    fx = _fixture()["artifacts"]["player_matchup"]
    from sports.nba.participant_enrichment import PLAYER_FIELDS
    base = slate_df.reset_index(drop=True)
    games = []
    for i, row in base.iterrows():
        rec = {
            "game_id": row["game_id"],
            "game_date": str(pd.to_datetime(
                row.get("game_date")).date())
            if pd.notna(row.get("game_date")) else "",
            "home_team": str(row.get("home_team", "") or ""),
            "away_team": str(row.get("away_team", "") or ""),
        }
        for f in PLAYER_FIELDS:
            rec[f] = _json_safe(player_df.iloc[i][f]) if f in player_df.columns \
                else None
        games.append(rec)
    record = {
        "created_utc": _now_utc(),
        "n_games": len(games),
        "games": games,
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"player_matchup key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "player_matchup", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


def write_feature_json(path: Path, cov: pd.DataFrame, config_meta: dict,
                       fold_info: dict, feature_contract_metadata: dict
                       ) -> dict:
    """Feature manifest record (fixture: feature_json)."""
    fx = _fixture()["artifacts"]["feature_json"]
    record = {
        "created_utc": _now_utc(),
        # Bound from the active moneyline contract carried in config_meta
        # (T3), never a hardcoded legacy literal.
        "feature_set_version": config_meta.get("feature_set_version",
                                               MONEYLINE_CONTRACT_VERSION),
        "manifest": feature_contract_metadata,
        "served_columns": list(feature_contract_metadata.get(
            "features", {}).keys()) or None,
        "coverage": _json_safe(cov.to_dict(orient="records")),
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"feature_json key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "feature_v1", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


def write_model_monitor_json(path: Path, run_date: str, drift: list[dict],
                             cov: list[dict], ensemble: list[dict],
                             rb: list[dict], baseline: float,
                             config_meta: dict, fold_info: dict,
                             metrics: dict | None = None,
                             platt: dict | None = None,
                             features_metadata: dict | None = None) -> dict:
    """Platform-shaped monitor artifact (fixture: model_monitor)."""
    require_features_metadata("nba", features_metadata)
    fx = _fixture()["artifacts"]["model_monitor"]
    iso_date = (f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
                if len(str(run_date)) == 8 and str(run_date).isdigit()
                else str(run_date))
    m = metrics or {}
    cal = (platt if isinstance(platt, dict)
           and platt.get("a") is not None else None)
    version_row: dict = {
        "version": run_date, "date": iso_date,
        "weights": {r["name"]: r["weight"] for r in ensemble},
        "auc": m.get("auc") or (ensemble[0].get("auc") if ensemble
                                else None),
        "logloss": m.get("logloss"),
        "ece_calibrated": m.get("ece_calibrated") or m.get("ece"),
        "note": "rebuild run",
    }
    if cal:
        version_row["calibration"] = {"a": cal["a"], "b": cal["b"]}
    record = {
        "last_retrained": iso_date,
        "last_retrained_note": None,
        "next_retrain": iso_date,
        "next_retrain_note": None,
        "upset_note": None,
        "feature_drift": drift,
        # Canonical feature contract metadata (the same document the
        # feature_v1 artifact carries): one entry per cataloged feature with
        # the full FeatureSpec field set, plus the explicit unavailable-column
        # warnings. This block used to be a stub derived from the coverage
        # rows -- and the only caller passes cov=[], so the Model Monitor
        # received an EMPTY features_metadata for every feature.
        "features_metadata": features_metadata,
        "feature_coverage": cov,
        "ensemble": ensemble,
        "rolling_brier": rb,
        "brier_baseline": _json_safe(baseline),
        "brier_baseline_label": ("Constant home-edge"
                                 if np.isfinite(baseline) else "n/a"),
        "rolling_brier_meta": {
            "window_days": 30,
            "min_games_per_day": 1,
            "excluded_sparse_days": 0,
            "calibrator_is_identity": False,
            "map_scope_note": ("Points use the deployed Platt map (fit on "
                               "all OOF games)."),
        },
        "version_history": [version_row],
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"model_monitor key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "model_monitor", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


def write_markets_monitor_json(path: Path, run_date: str,
                               oof_market_rows: pd.DataFrame,
                               config_meta: dict | None = None,
                               ml_reference: dict | None = None) -> dict:
    """Spread & Totals Monitor artifact (fixture: markets_monitor)."""
    fx = _fixture()["artifacts"]["markets_monitor"]
    iso_date = (f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
                if len(str(run_date)) == 8 and str(run_date).isdigit()
                else str(run_date))

    def _card(df: pd.DataFrame, p_col: str, y_col: str) -> dict:
        if not len(df):
            return {}
        p = pd.to_numeric(df.get(p_col), errors="coerce")
        y = pd.to_numeric(df.get(y_col), errors="coerce")
        ok = p.notna() & y.notna() & y.isin([0, 1])
        if not ok.any():
            return {}
        p, y = p[ok].to_numpy(), y[ok].astype(int).to_numpy()
        from sklearn.metrics import brier_score_loss, log_loss
        return {
            "n": int(len(p)),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "logloss": (round(float(log_loss(y, p, labels=[0, 1])), 4)
                        if len(np.unique(y)) == 2 else None),
            "predicted_mean": round(float(p.mean()), 4),
        }

    cards = {
        "over_under": _card(oof_market_rows, "p_over_fair", "y_over_fair"),
        "spread": _card(oof_market_rows, "p_cover_fair", "y_cover_fair"),
        "derived_ml": _card(oof_market_rows, "p_home_win_derived",
                            "y_home_win"),
    }
    slate_history = []
    for card_key in ("over_under", "spread", "derived_ml"):
        c = cards.get(card_key) or {}
        if not c.get("n"):
            continue
        slate_history.append({"card": card_key, "date": iso_date,
                              **c})
    if ml_reference and cards.get("derived_ml"):
        cards["derived_ml"]["ml_reference"] = dict(ml_reference)
    record = {
        "schema": "run-engine-monitor/v2",
        "date": iso_date,
        "markets_persisted": True,
        "markets_persist_error": None,
        "winner_cards": cards,
        "slate_history": slate_history,
        "fit": {},
        "market_metrics": {},
        "config": config_meta or {},
    }
    if list(record.keys()) != list(fx["keys"]):
        raise NBAArtifactError(
            f"markets_monitor key order mismatch: {list(record.keys())}")
    safe = _json_safe(record)
    validate_record("nba", "run_engine_monitor", safe)
    path.write_text(json.dumps(safe, indent=1, allow_nan=False))
    return record


# ---------------------------------------------------------------------------
# CSV families
# ---------------------------------------------------------------------------


def write_predictions_history_csv(path: Path, oof: pd.DataFrame,
                                  p_cal: np.ndarray | None) -> pd.DataFrame:
    """Decided-game predicted-vs-actual (fixture: predictions_history)."""
    fx = _fixture()["artifacts"]["predictions_history"]
    df = oof.sort_values("game_date").reset_index(drop=True)
    p = df["p_ensemble"].astype(float)
    out = pd.DataFrame({
        "game_id": df["game_id"],
        "game_date": pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d"),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_score": df["home_score"].astype(float),
        "away_score": df["away_score"].astype(float),
        "home_win": df["home_win"].astype(float),
        "home_win_prob_model": p,
        "home_win_prob_model_calibrated":
            p_cal if p_cal is not None and len(p_cal) == len(df) else p,
        "model_pick": np.where(p >= 0.5, df["home_team"], df["away_team"]),
        "actual_winner": np.select(
            [df["home_win"] > 0.5, df["home_win"] < 0.5],
            [df["home_team"], df["away_team"]], default=""),
    })
    out["correct"] = out["model_pick"] == out["actual_winner"]
    if list(out.columns) != list(fx["columns"]):
        raise NBAArtifactError(
            f"predictions_history column mismatch: {list(out.columns)}")
    validate_record("nba", "predictions_history", out)
    out.to_csv(path, index=False)
    return out


def write_power_rankings_csv(path: Path, ratings: dict[str, float],
                             records: dict[str, tuple[int, int]],
                             point_diff: dict[str, int],
                             team_names: dict[str, str]) -> pd.DataFrame:
    """Elo power rankings (fixture: power_rankings)."""
    fx = _fixture()["artifacts"]["power_rankings"]
    rows = []
    for team, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        w, l = records.get(team, (0, 0))
        rows.append({
            "rank": 0, "team": team,
            "team_name": team_names.get(team, team),
            "elo": round(float(elo), 1),
            "wins": int(w), "losses": int(l),
            "record": f"{int(w)}-{int(l)}",
            "pct": round(w / (w + l), 3) if (w + l) else np.nan,
            "run_diff": int(point_diff.get(team, 0)),
            "l10": "", "home_pct": np.nan, "away_pct": np.nan,
        })
    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)
    if list(df.columns) != list(fx["columns"]):
        raise NBAArtifactError(
            f"power_rankings column mismatch: {list(df.columns)}")
    validate_record("nba", "power_rankings", df)
    df.to_csv(path, index=False)
    return df


def write_markets_csv(path: Path, meta_path: Path,
                      oof_rows: pd.DataFrame, slate_rows: pd.DataFrame,
                      config_meta: dict) -> pd.DataFrame:
    """Decided OOF rows + slate rows into the markets artifact
    (fixture: markets)."""
    fx = _fixture()["artifacts"]["markets"]
    meta_fx = _fixture()["artifacts"]["markets_meta"]
    cols = markets_columns()
    out = pd.concat([oof_rows, slate_rows], ignore_index=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    validate_record("nba", "run_engine_markets", out)
    out.to_csv(path, index=False)
    meta = {
        "record": "nba_run_engine_markets",
        "written_utc": _now_utc(),
        "config": config_meta,
        "columns": len(cols),
        "n_rows": int(len(out)),
        "n_oof": int((out["kind"] == "oof").sum()),
        "n_slate": int((out["kind"] == "slate").sum()),
        "grids": meta_fx["grids"],
    }
    if list(meta.keys()) != list(meta_fx["keys"]):
        raise NBAArtifactError(
            f"markets meta key order mismatch: {list(meta.keys())}")
    validate_record("nba", "run_engine_markets.meta", meta)
    meta_path.write_text(json.dumps(meta, indent=1))
    return out


def persist_shap_game(frame: pd.DataFrame, run_date: str, game_key: str,
                      out_dir: Path) -> Path:
    """One SHAP CSV per undecided slate game (fixture: shap_game)."""
    fx = _fixture()["artifacts"]["shap_game"]
    cols = ["feature", "shap_value", "signed_effect", "perspective_team"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise NBAArtifactError(f"shap_game missing columns: {missing}")
    out = frame[cols]
    out_path = out_dir / f"nba_shap_game_{run_date}_{game_key}.csv"
    validate_record("nba", "shap_game", out)
    out.to_csv(out_path, index=False)
    return out_path
