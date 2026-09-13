"""MLB production artifact writers.

Every writer is pinned field-by-field to the immutable schema fixture
``tests/fixtures/mlb_artifact_schemas.json`` (extracted verbatim from the
read-only reference repository's ``mlb-backend/data_delivery/`` production
artifacts, commit 5dce619, dated 2026-09-07). A writer refuses to persist a
frame whose columns/keys/null behavior deviates from the fixture contract —
the exact-artifact rule from Phase 2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.contracts import validate_record
from sports.mlb.models.market.run_engine import (
    RUN_LINE_GRID,
    RUN_LINE_GRID_FULL,
    TOTAL_LINE_GRID,
    rl_col,
)

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" \
    / "mlb_artifact_schemas.json"


def _fixture_columns(artifact: str) -> list[str] | None:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    spec = data["artifacts"].get(artifact)
    if spec is None:
        return None
    return list(spec.get("columns", []))


def _check_columns(frame: pd.DataFrame, artifact: str) -> None:
    expected = _fixture_columns(artifact)
    if expected is None:
        return
    got = list(frame.columns)
    if got != expected:
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise ValueError(
            f"{artifact}: artifact columns deviate from the immutable "
            f"schema fixture — missing={missing} extra={extra} "
            f"(order matters; see tests/fixtures/mlb_artifact_schemas.json)")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Markets (run_engine_markets_<date>.csv + .meta.json)
# ---------------------------------------------------------------------------
# Column ORDER is pinned to the immutable fixture (the real artifact's
# header order): the derived-ML + decided-target + agreement columns come
# immediately after alpha_home/alpha_away, THEN the totals grid (over, push,
# under), home-cover margins, and the run-line 3-way splits.
MARKET_COLUMNS_V3 = (
    ["game_pk", "game_date", "kind", "home_expected_runs",
     "away_expected_runs", "alpha_home", "alpha_away",
     "p_home_win_derived", "p_away_win_derived",
     "home_score", "away_score", "total_runs",
     "ml_win_prob", "agreement_conflict"]
    + [f"p_over_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_push_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_under_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_home_cover_{str(m).replace('.', '_')}" for m in RUN_LINE_GRID]
    + [rl_col(m, side) for m in RUN_LINE_GRID_FULL
       for side in ("home", "push", "away")]
)
assert len(MARKET_COLUMNS_V3) == 78
NULLABLE_MARKET_COLUMNS = {"ml_win_prob"}


def persist_markets(markets: pd.DataFrame, target_date_str: str,
                    summary: dict,
                    out_dir: Path | str | None = None) -> Path:
    """Write run_engine_markets_<date>.csv + .meta.json.

    Column order is pinned to the fixture. Decided-target columns
    (home_score/away_score/total_runs) must be populated on OOF rows; slate
    rows are undecided BY DEFINITION and are exempt from exactly those
    three. ml_win_prob is the only other nullable column.
    """
    out_path = Path(out_dir or ".") / f"run_engine_markets_{target_date_str}.csv"
    missing = [c for c in MARKET_COLUMNS_V3 if c not in markets.columns]
    if missing:
        raise ValueError(f"markets frame missing required columns: {missing}")
    frame = markets[MARKET_COLUMNS_V3]
    target_cols = ["home_score", "away_score", "total_runs"]
    bad = [c for c in MARKET_COLUMNS_V3
           if c not in NULLABLE_MARKET_COLUMNS
           and (frame[c].isna().any() if c not in target_cols
                else frame.loc[frame["kind"] == "oof", c].isna().any())]
    if bad:
        raise ValueError(
            f"markets frame contains NaNs in {bad} — refusing to persist")
    _check_columns(frame, "run_engine_markets")
    validate_record("mlb", "run_engine_markets", frame)
    _atomic_write_csv(frame, out_path)
    meta_path = out_path.with_suffix("")
    meta_path = out_path.parent / \
        f"run_engine_markets_{target_date_str}.meta.json"
    validate_record("mlb", "run_engine_markets.meta", summary)
    _atomic_write_json(summary, meta_path)
    logger.info("Run engine markets: %d rows -> %s (+ meta json)",
                len(frame), out_path.name)
    return out_path


def persist_oof(oof: pd.DataFrame, target_date_str: str,
                out_dir: Path | str | None = None) -> Path:
    """Write run_engine_oof_<date>.csv (retention-exempt, non-frontend)."""
    out_path = Path(out_dir or ".") / f"run_engine_oof_{target_date_str}.csv"
    frame = oof[["game_pk", "game_date", "home_expected_runs",
                 "away_expected_runs", "home_score", "away_score"]].copy()
    frame["game_pk"] = frame["game_pk"].map(_pk_str)
    frame["home_expected_runs"] = frame["home_expected_runs"].round(4)
    frame["away_expected_runs"] = frame["away_expected_runs"].round(4)
    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    _check_columns(frame, "run_engine_oof")
    validate_record("mlb", "run_engine_oof", frame)
    _atomic_write_csv(frame, out_path)
    logger.info("Run engine OOF: %d rows -> %s", len(frame), out_path.name)
    return out_path


def _pk_str(x) -> str:
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(x)


# ---------------------------------------------------------------------------
# predictions_history (moneyline pipeline artifact)
# ---------------------------------------------------------------------------
PREDICTIONS_HISTORY_COLUMNS = [
    "game_id", "game_date", "home_team", "away_team",
    "home_score", "away_score", "home_win",
    "home_win_prob_model", "home_win_prob_model_calibrated",
    "model_pick", "actual_winner", "correct"]


def persist_predictions_history(frame: pd.DataFrame, target_date_str: str,
                                out_dir: Path | str | None = None) -> Path:
    out_path = Path(out_dir or ".") / \
        f"predictions_history_{target_date_str}.csv"
    missing = [c for c in PREDICTIONS_HISTORY_COLUMNS
               if c not in frame.columns]
    if missing:
        raise ValueError(f"predictions_history missing columns: {missing}")
    out = frame[PREDICTIONS_HISTORY_COLUMNS]
    _check_columns(out, "predictions_history")
    validate_record("mlb", "predictions_history", out)
    _atomic_write_csv(out, out_path)
    logger.info("predictions_history: %d rows -> %s", len(out), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# todays_games
# ---------------------------------------------------------------------------
TODAYS_GAMES_COLUMNS = [
    "game_id", "game_date", "start_time_utc", "home_team", "away_team",
    "home_record", "away_record", "home_win_prob_model",
    "away_win_prob_model", "edge_home", "edge_away", "sp_name_home",
    "sp_name_away", "sp_era_home", "sp_k9_home", "sp_era_away",
    "sp_k9_away", "venue", "model_pick", "home_win", "home_score",
    "away_score", "total_runs", "game_state", "game_status_detail"]


def persist_todays_games(frame: pd.DataFrame, target_date_str: str,
                         out_dir: Path | str | None = None) -> Path:
    out_path = Path(out_dir or ".") / f"todays_games_{target_date_str}.csv"
    missing = [c for c in TODAYS_GAMES_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"todays_games missing columns: {missing}")
    out = frame[TODAYS_GAMES_COLUMNS]
    _check_columns(out, "todays_games")
    validate_record("mlb", "todays_games", out)
    _atomic_write_csv(out, out_path)
    logger.info("todays_games: %d rows -> %s", len(out), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# power_rankings
# ---------------------------------------------------------------------------
POWER_RANKINGS_COLUMNS = [
    "rank", "team", "team_name", "elo", "wins", "losses", "record",
    "pct", "run_diff", "l10", "home_pct", "away_pct"]


def persist_power_rankings(frame: pd.DataFrame, target_date_str: str,
                           out_dir: Path | str | None = None) -> Path:
    out_path = Path(out_dir or ".") / f"power_rankings_{target_date_str}.csv"
    missing = [c for c in POWER_RANKINGS_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"power_rankings missing columns: {missing}")
    out = frame[POWER_RANKINGS_COLUMNS].sort_values("rank")
    _check_columns(out, "power_rankings")
    validate_record("mlb", "power_rankings", out)
    _atomic_write_csv(out, out_path)
    logger.info("power_rankings: %d rows -> %s", len(out), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# calibration / model_monitor / rolling_brier (JSON artifacts)
# ---------------------------------------------------------------------------
CALIBRATION_KEYS = ["date", "n_games", "trained_at", "metrics",
                    "calibration_buckets", "calibration", "daily",
                    "league_total", "evening_games_league"]


def persist_calibration(payload: dict, target_date_str: str,
                        out_dir: Path | str | None = None) -> Path:
    missing = [k for k in CALIBRATION_KEYS if k not in payload]
    if missing:
        raise ValueError(f"calibration payload missing keys: {missing}")
    out_path = Path(out_dir or ".") / f"calibration_{target_date_str}.json"
    _check_json_keys(payload, "calibration")
    validate_record("mlb", "calibration", payload)
    _atomic_write_json(payload, out_path)
    logger.info("calibration -> %s", out_path.name)
    return out_path


MODEL_MONITOR_KEYS = ["date", "version", "last_retrained", "next_retrain",
                      "metrics", "drift_summary", "feature_drift",
                      "feature_coverage", "rolling_brier", "brier_baseline",
                      "brier_baseline_label", "rolling_brier_meta",
                      "features_metadata", "run_engine", "ensemble",
                      "model_history", "version_history"]


def persist_model_monitor(payload: dict, target_date_str: str,
                          out_dir: Path | str | None = None) -> Path:
    missing = [k for k in MODEL_MONITOR_KEYS if k not in payload]
    if missing:
        raise ValueError(f"model_monitor payload missing keys: {missing}")
    out_path = Path(out_dir or ".") / f"model_monitor_{target_date_str}.json"
    _check_json_keys(payload, "model_monitor")
    validate_record("mlb", "model_monitor", payload)
    _atomic_write_json(payload, out_path)
    logger.info("model_monitor -> %s", out_path.name)
    return out_path


ROLLING_BRIER_KEYS = ["window_days", "min_games_per_day", "source_column",
                      "calibration", "map_scope_note",
                      "calibrator_is_identity", "n_points",
                      "n_games_total", "excluded_sparse_days",
                      "history_mean_brier", "series", "n_games_in_series"]


def persist_rolling_brier(payload: dict, target_date_str: str,
                          out_dir: Path | str | None = None) -> Path:
    missing = [k for k in ROLLING_BRIER_KEYS if k not in payload]
    if missing:
        raise ValueError(f"rolling_brier payload missing keys: {missing}")
    out_path = Path(out_dir or ".") / f"rolling_brier_{target_date_str}.json"
    validate_record("mlb", "rolling_brier", payload)
    _atomic_write_json(payload, out_path)
    logger.info("rolling_brier -> %s", out_path.name)
    return out_path


FEATURES_METADATA_KEYS = ["generated_for", "n_features", "warnings",
                          "features", "categorical_context"]


def persist_features_metadata(payload: dict, target_date_str: str,
                              out_dir: Path | str | None = None) -> Path:
    missing = [k for k in FEATURES_METADATA_KEYS if k not in payload]
    if missing:
        raise ValueError(f"features_metadata payload missing keys: {missing}")
    out_path = Path(out_dir or ".") / \
        f"features_metadata_{target_date_str}.json"
    validate_record("mlb", "features_metadata", payload)
    _atomic_write_json(payload, out_path)
    logger.info("features_metadata -> %s", out_path.name)
    return out_path


RUN_ENGINE_MONITOR_KEYS = ["schema", "date", "markets_persisted",
                           "markets_persist_error", "winner_cards",
                           "rolling", "fit", "market_metrics", "phase1"]


def persist_run_engine_monitor(payload: dict, target_date_str: str,
                               out_dir: Path | str | None = None) -> Path:
    missing = [k for k in RUN_ENGINE_MONITOR_KEYS if k not in payload]
    if missing:
        raise ValueError(f"run_engine_monitor payload missing keys: {missing}")
    if payload.get("schema") != "run-engine-monitor/v2":
        raise ValueError("run_engine_monitor.schema must be "
                         "'run-engine-monitor/v2'")
    out_path = Path(out_dir or ".") / \
        f"run_engine_monitor_{target_date_str}.json"
    validate_record("mlb", "run_engine_monitor", payload)
    _atomic_write_json(payload, out_path)
    logger.info("run_engine_monitor -> %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# SHAP game artifact
# ---------------------------------------------------------------------------
SHAP_GAME_COLUMNS = ["feature", "shap_value", "signed_effect",
                     "perspective_team"]


def persist_shap_game(frame: pd.DataFrame, target_date_str: str,
                      game_key: str,
                      out_dir: Path | str | None = None) -> Path:
    """game_key embeds the matchup, e.g. ``ARI@KC``."""
    out_path = Path(out_dir or ".") / \
        f"shap_game_{target_date_str}_{game_key}.csv"
    missing = [c for c in SHAP_GAME_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"shap_game missing columns: {missing}")
    out = frame[SHAP_GAME_COLUMNS]
    _check_columns(out, "shap_game")
    validate_record("mlb", "shap_game", out)
    _atomic_write_csv(out, out_path)
    logger.info("shap_game -> %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# feature_drift / feature_coverage
# ---------------------------------------------------------------------------
FEATURE_DRIFT_COLUMNS = ["feature", "current_mean", "baseline_mean", "psi",
                         "psi_adjusted", "noise_floor", "mean_shift",
                         "shift_se", "location_shift", "status",
                         "weight_pct", "n_baseline", "n_current"]

FEATURE_COVERAGE_COLUMNS = ["feature", "window", "n_games", "n_nonnull",
                            "pct_nonnull", "n_measured", "pct_measured",
                            "n_default_zero", "status"]


def persist_feature_drift(frame: pd.DataFrame, target_date_str: str,
                          out_dir: Path | str | None = None) -> Path:
    out_path = Path(out_dir or ".") / f"feature_drift_{target_date_str}.csv"
    missing = [c for c in FEATURE_DRIFT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"feature_drift missing columns: {missing}")
    out = frame[FEATURE_DRIFT_COLUMNS]
    _check_columns(out, "feature_drift")
    validate_record("mlb", "feature_drift", out)
    _atomic_write_csv(out, out_path)
    logger.info("feature_drift -> %s", out_path.name)
    return out_path


def persist_feature_coverage(frame: pd.DataFrame, target_date_str: str,
                             out_dir: Path | str | None = None) -> Path:
    out_path = Path(out_dir or ".") / \
        f"feature_coverage_{target_date_str}.csv"
    missing = [c for c in FEATURE_COVERAGE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"feature_coverage missing columns: {missing}")
    out = frame[FEATURE_COVERAGE_COLUMNS]
    _check_columns(out, "feature_coverage")
    validate_record("mlb", "feature_coverage", out)
    _atomic_write_csv(out, out_path)
    logger.info("feature_coverage -> %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# game_level_features (§22 Amendment 11 support artifact)
# ---------------------------------------------------------------------------
GAME_LEVEL_FEATURES_COLUMNS = ["game_pk", "game_date", "home_team",
                               "away_team", "home_score", "away_score",
                               "home_win"]


def persist_game_level_features(frame: pd.DataFrame, target_date_str: str,
                                out_dir: Path | str | None = None) -> Path:
    """The decided game-level frame: reconciles stale board state against
    authoritative finals and gives the markets page its game_pk →
    team-names bridge (markets rows carry game_pk only)."""
    out_path = Path(out_dir or ".") / \
        f"game_level_features_{target_date_str}.csv"
    missing = [c for c in GAME_LEVEL_FEATURES_COLUMNS
               if c not in frame.columns]
    if missing:
        raise ValueError(f"game_level_features missing columns: {missing}")
    out = frame[GAME_LEVEL_FEATURES_COLUMNS].copy()
    out = out.dropna(subset=["game_pk"]).drop_duplicates(
        subset=["game_pk"], keep="last")
    out = out.sort_values("game_pk").reset_index(drop=True)
    _check_columns(out, "game_level_features")
    validate_record("mlb", "game_level_features", out)
    _atomic_write_csv(out, out_path)
    logger.info("game_level_features: %d rows -> %s", len(out), out_path.name)
    return out_path


def _check_json_keys(payload: dict, artifact: str) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    spec = fixture["artifacts"].get(artifact)
    if spec is None:
        return
    expected = set(spec.get("keys", []))
    missing = expected - set(payload.keys())
    if missing:
        raise ValueError(
            f"{artifact}: JSON keys deviate from the immutable schema "
            f"fixture — missing={sorted(missing)}")


def write_all_artifacts(target_date_str: str, out_dir: Path | str,
                        *, markets: pd.DataFrame, markets_summary: dict,
                        oof: pd.DataFrame,
                        todays_games: pd.DataFrame,
                        power_rankings: pd.DataFrame,
                        predictions_history: pd.DataFrame,
                        calibration: dict,
                        model_monitor: dict,
                        rolling_brier: dict,
                        features_metadata: dict,
                        run_engine_monitor: dict,
                        shap_game: pd.DataFrame | None = None,
                        shap_game_key: str = "",
                        feature_drift: pd.DataFrame,
                        feature_coverage: pd.DataFrame,
                        game_level_features: pd.DataFrame | None = None,
                        ) -> list[Path]:
    """Write the full frontend artifact set for one production date.

    Every writer validates against the immutable fixture before persisting;
    any failure raises BEFORE retention is allowed to run (the runner's
    generation_succeeded gate stays honest).
    """
    out = Path(out_dir)
    written = [
        persist_todays_games(todays_games, target_date_str, out),
        persist_power_rankings(power_rankings, target_date_str, out),
        persist_markets(markets, target_date_str, markets_summary, out),
        persist_oof(oof, target_date_str, out),
        persist_predictions_history(predictions_history, target_date_str, out),
        persist_calibration(calibration, target_date_str, out),
        persist_model_monitor(model_monitor, target_date_str, out),
        persist_rolling_brier(rolling_brier, target_date_str, out),
        persist_features_metadata(features_metadata, target_date_str, out),
        persist_run_engine_monitor(run_engine_monitor, target_date_str, out),
        persist_feature_drift(feature_drift, target_date_str, out),
        persist_feature_coverage(feature_coverage, target_date_str, out),
        persist_game_level_features(
            game_level_features if game_level_features is not None
            else pd.DataFrame(columns=GAME_LEVEL_FEATURES_COLUMNS),
            target_date_str, out),
    ]
    if shap_game is not None and not shap_game.empty:
        written.append(persist_shap_game(shap_game, target_date_str,
                                         shap_game_key, out))
    return written
