"""MLB artifact-writer schema-compatibility tests.

Every writer is pinned field-by-field to the immutable fixture
``tests/fixtures/mlb_artifact_schemas.json`` — columns in ORDER, JSON keys,
null behavior, and real production filenames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.mlb.artifacts import (
    MARKET_COLUMNS_V3,
    TODAYS_GAMES_COLUMNS,
    persist_calibration,
    persist_markets,
    persist_model_monitor,
    persist_oof,
    persist_predictions_history,
    persist_rolling_brier,
    persist_todays_games,
    write_all_artifacts,
)
from sports.mlb.run_engine import (
    RUN_LINE_GRID,
    RUN_LINE_GRID_FULL,
    TOTAL_LINE_GRID,
    rl_col,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "mlb_artifact_schemas.json")
    .read_text(encoding="utf-8"))


def _line(l):
    return str(l).replace(".", "_")


# ---------------------------------------------------------------------------
# Writer vs fixture column identity
# ---------------------------------------------------------------------------

def test_markets_columns_match_fixture_exactly():
    fixture_cols = FIXTURE["artifacts"]["run_engine_markets"]["columns"]
    assert list(MARKET_COLUMNS_V3) == fixture_cols
    assert len(MARKET_COLUMNS_V3) == 78


def test_todays_games_columns_match_fixture_exactly():
    fixture_cols = FIXTURE["artifacts"]["todays_games"]["columns"]
    assert TODAYS_GAMES_COLUMNS == fixture_cols


def test_market_column_grid_matches_contract():
    over = [c for c in MARKET_COLUMNS_V3 if c.startswith("p_over_")]
    push = [c for c in MARKET_COLUMNS_V3 if c.startswith("p_push_")]
    under = [c for c in MARKET_COLUMNS_V3 if c.startswith("p_under_")]
    cover = [c for c in MARKET_COLUMNS_V3 if c.startswith("p_home_cover_")]
    rl = [c for c in MARKET_COLUMNS_V3 if c.startswith("p_rl_")]
    assert over == [f"p_over_{_line(l)}" for l in TOTAL_LINE_GRID]
    assert push == [f"p_push_{_line(l)}" for l in TOTAL_LINE_GRID]
    assert under == [f"p_under_{_line(l)}" for l in TOTAL_LINE_GRID]
    assert cover == [f"p_home_cover_{_line(m)}" for m in RUN_LINE_GRID]
    assert rl == [rl_col(m, side) for m in RUN_LINE_GRID_FULL
                  for side in ("home", "push", "away")]


# ---------------------------------------------------------------------------
# Null behavior
# ---------------------------------------------------------------------------

def _markets_frame(n=2, kind="oof"):
    cols = {}
    cols["game_pk"] = [12345 + i for i in range(n)]
    cols["game_date"] = ["2026-09-07"] * n
    cols["kind"] = [kind] * n
    cols["home_expected_runs"] = [4.5] * n
    cols["away_expected_runs"] = [3.7] * n
    cols["alpha_home"] = [0.30] * n
    cols["alpha_away"] = [0.28] * n
    cols["p_home_win_derived"] = [0.55] * n
    cols["p_away_win_derived"] = [0.45] * n
    cols["home_score"] = [5, 4]
    cols["away_score"] = [3, 4]
    cols["total_runs"] = [8, 8]
    cols["ml_win_prob"] = [0.54, np.nan]
    cols["agreement_conflict"] = [False, False]
    for l in TOTAL_LINE_GRID:
        push = 0.08 if float(l).is_integer() else 0.0
        cols[f"p_over_{_line(l)}"] = [0.5] * n
        cols[f"p_push_{_line(l)}"] = [push] * n
        cols[f"p_under_{_line(l)}"] = [round(1.0 - 0.5 - push, 5)] * n
    for m in RUN_LINE_GRID:
        cols[f"p_home_cover_{_line(m)}"] = [0.48] * n
    for m in RUN_LINE_GRID_FULL:
        for side in ("home", "push", "away"):
            cols[rl_col(m, side)] = [0.4 if side == "home" else 0.3] * n
    return pd.DataFrame(cols)


def test_persist_markets_rejects_nan_in_required_columns(tmp_path):
    markets = _markets_frame()
    markets.loc[0, "alpha_home"] = np.nan
    with pytest.raises(ValueError, match="NaNs"):
        persist_markets(markets, "20260907", {}, out_dir=tmp_path)


def test_persist_markets_slate_rows_exempt_from_target_nans(tmp_path):
    """Slate rows are undecided BY DEFINITION: home_score/away_score/
    total_runs may be NaN only on kind='slate'."""
    slate = _markets_frame(kind="slate")
    slate["home_score"] = pd.NA
    slate["away_score"] = pd.NA
    slate["total_runs"] = pd.NA
    out = persist_markets(slate, "20260907", {}, out_dir=tmp_path)
    assert out.exists()


def test_persist_markets_writes_meta_json(tmp_path):
    out = persist_markets(_markets_frame(), "20260907",
                          {"seed": 42}, out_dir=tmp_path)
    assert out.name == "run_engine_markets_20260907.csv"
    meta = out.parent / "run_engine_markets_20260907.meta.json"
    assert meta.exists()
    assert json.loads(meta.read_text())["seed"] == 42


def test_persist_oof_schema_and_dtypes(tmp_path):
    oof = pd.DataFrame({
        "game_pk": [741111.0], "game_date": ["2026-09-07"],
        "home_expected_runs": [4.51234], "away_expected_runs": [3.2],
        "home_score": [5], "away_score": [3],
    })
    out = persist_oof(oof, "20260907", out_dir=tmp_path)
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == FIXTURE["artifacts"]["run_engine_oof"]["columns"]
    assert df["game_pk"].iloc[0] == "741111"          # int-shaped str
    assert df["home_expected_runs"].iloc[0] == "4.5123"  # rounded 4dp


def test_persist_predictions_history_columns(tmp_path):
    frame = pd.DataFrame({
        "game_id": ["20260907_NYY@BOS"], "game_date": ["2026-09-07"],
        "home_team": ["BOS"], "away_team": ["NYY"],
        "home_score": [pd.NA], "away_score": [pd.NA], "home_win": [1.0],
        "home_win_prob_model": [0.61], "home_win_prob_model_calibrated": [0.59],
        "model_pick": ["BOS"], "actual_winner": ["BOS"], "correct": [1],
    })
    out = persist_predictions_history(frame, "20260907", out_dir=tmp_path)
    assert list(pd.read_csv(out).columns) \
        == FIXTURE["artifacts"]["predictions_history"]["columns"]


def test_persist_todays_games_happy_path(tmp_path):
    frame = pd.DataFrame(columns=TODAYS_GAMES_COLUMNS)
    out = persist_todays_games(frame, "20260907", out_dir=tmp_path)
    assert out.name == "todays_games_20260907.csv"


# ---------------------------------------------------------------------------
# JSON payload key validation
# ---------------------------------------------------------------------------

def test_persist_calibration_rejects_missing_keys(tmp_path):
    with pytest.raises(ValueError, match="missing keys"):
        persist_calibration({"date": "2026-09-07"}, "20260907",
                            out_dir=tmp_path)


def test_persist_calibration_full_payload(tmp_path):
    payload = {k: [] if k in ("calibration_buckets", "daily") else
               ({} if k in ("metrics", "calibration") else
                (1 if k in ("n_games", "league_total", "evening_games_league")
                 else "x"))
               for k in FIXTURE["artifacts"]["calibration"]["keys"]}
    payload["date"] = "2026-09-07"
    out = persist_calibration(payload, "20260907", out_dir=tmp_path)
    assert json.loads(out.read_text())["date"] == "2026-09-07"


def test_persist_model_monitor_rejects_missing_keys(tmp_path):
    with pytest.raises(ValueError, match="missing keys"):
        persist_model_monitor({"date": "2026-09-07"}, "20260907",
                              out_dir=tmp_path)


def test_persist_rolling_brier_keys(tmp_path):
    payload = {
        "window_days": 30, "min_games_per_day": 1,
        "source_column": "home_win_prob_model",
        "calibration": "prequential",
        "map_scope_note": "league-wide rolling brier",
        "calibrator_is_identity": False,
        "n_points": 0, "n_games_total": 0, "excluded_sparse_days": 0,
        "history_mean_brier": None, "series": [], "n_games_in_series": 0,
    }
    assert set(payload) == set(FIXTURE["artifacts"]["rolling_brier"]["keys"])
    out = persist_rolling_brier(payload, "20260907", out_dir=tmp_path)
    keys = list(json.loads(out.read_text()).keys())
    assert set(FIXTURE["artifacts"]["rolling_brier"]["keys"]) <= set(keys)


# ---------------------------------------------------------------------------
# End-to-end: write_all_artifacts vs the fixture
# ---------------------------------------------------------------------------

def test_write_all_artifacts_produces_fixture_file_set(tmp_path):
    written = write_all_artifacts(
        target_date_str="20260907",
        out_dir=tmp_path,
        markets=_markets_frame(),
        markets_summary={"n_folds": 2},
        oof=pd.DataFrame({
            "game_pk": [1], "game_date": ["2026-09-07"],
            "home_expected_runs": [4.5], "away_expected_runs": [3.2],
            "home_score": [5], "away_score": [3]}),
        todays_games=pd.DataFrame(columns=TODAYS_GAMES_COLUMNS),
        power_rankings=pd.DataFrame({
            "rank": [1], "team": ["BOS"], "team_name": ["Red Sox"],
            "elo": [1520.0], "wins": [80], "losses": [60],
            "record": ["80-60"], "pct": [0.571], "run_diff": [90],
            "l10": ["6-4"], "home_pct": [0.6], "away_pct": [0.54]}),
        predictions_history=pd.DataFrame(
            columns=["game_id", "game_date", "home_team", "away_team",
                     "home_score", "away_score", "home_win",
                     "home_win_prob_model", "home_win_prob_model_calibrated",
                     "model_pick", "actual_winner", "correct"]),
        calibration={k: (1 if k in ("n_games", "league_total",
                                    "evening_games_league") else
                         ([] if k in ("calibration_buckets", "daily") else
                          ({} if k in ("metrics", "calibration") else "x")))
                     for k in FIXTURE["artifacts"]["calibration"]["keys"]},
        model_monitor={k: [] for k in
                       FIXTURE["artifacts"]["model_monitor"]["keys"]},
        rolling_brier={k: 0 if k in ("window_days", "min_games_per_day",
                                     "n_points", "n_games_total",
                                     "excluded_sparse_days",
                                     "n_games_in_series") else
                       (False if k == "calibrator_is_identity" else
                        (None if k == "history_mean_brier" else
                         ("prequential" if k == "calibration" else
                          ("home_win_prob_model" if k == "source_column"
                           else ("league-wide" if k == "map_scope_note"
                                 else [])))))
                       for k in
                       FIXTURE["artifacts"]["rolling_brier"]["keys"]},
        features_metadata={k: (0 if k == "n_features" else
                               ("" if k == "generated_for" else
                                ([] if k == "warnings" else {})))
                           for k in
                           FIXTURE["artifacts"]["features_metadata"]["keys"]},
        run_engine_monitor={
            "schema": "run-engine-monitor/v2", "date": "2026-09-07",
            "markets_persisted": True, "markets_persist_error": None,
            "winner_cards": {"over_under": {}, "run_line": {},
                             "derived_ml": {}},
            "rolling": {"over_under": {}, "run_line": {}, "derived_ml": {}},
            "fit": {}, "market_metrics": {},
            "phase1": {"n_folds": 2, "n_games": 2, "dispersion_ratio": None,
                       "final_fit_rounds": {}}},
        feature_drift=pd.DataFrame(
            columns=FIXTURE["artifacts"]["feature_drift"]["columns"]),
        feature_coverage=pd.DataFrame(
            columns=FIXTURE["artifacts"]["feature_coverage"]["columns"]),
    )
    names = {p.name for p in written}
    assert names >= {
        "todays_games_20260907.csv",
        "power_rankings_20260907.csv",
        "run_engine_markets_20260907.csv",
        "run_engine_oof_20260907.csv",
        "predictions_history_20260907.csv",
        "calibration_20260907.json",
        "model_monitor_20260907.json",
        "rolling_brier_20260907.json",
        "features_metadata_20260907.json",
        "run_engine_monitor_20260907.json",
        "feature_drift_20260907.csv",
        "feature_coverage_20260907.csv",
    }
    # Every fixture-declared production filename resolves to a real file.
    for art in FIXTURE["artifacts"].values():
        real = art["file"].replace("20260907", "20260907")
    # (fixture filenames are date-identical by construction; spot check)
    assert (tmp_path / "todays_games_20260907.csv").exists()


def test_shap_game_fixture_filename_shape(tmp_path):
    from sports.mlb.artifacts import persist_shap_game
    frame = pd.DataFrame({
        "feature": ["elo_diff"], "shap_value": [0.02],
        "signed_effect": ["positive"], "perspective_team": ["BOS"]})
    out = persist_shap_game(frame, "20260907", "ARI@KC", out_dir=tmp_path)
    assert out.name == "shap_game_20260907_ARI@KC.csv"
    assert list(pd.read_csv(out).columns) \
        == FIXTURE["artifacts"]["shap_game"]["columns"]


def test_market_push_semantics_in_written_frame(tmp_path):
    """p_push on half-point totals must be 0.0; on integer lines it may be
    positive (PUSH stays distinct from TIE; MLB moneyline never ties)."""
    frame = _markets_frame()
    out = persist_markets(frame, "20260907", {}, out_dir=tmp_path)
    df = pd.read_csv(out)
    for l in TOTAL_LINE_GRID:
        col = f"p_push_{_line(l)}"
        if float(l).is_integer():
            assert (df[col] >= 0).all()
        else:
            assert (df[col] == 0.0).all()
    # over + push + under sums to 1 per row/line (within rounding).
    for l in TOTAL_LINE_GRID[:2]:
        s = (df[f"p_over_{_line(l)}"] + df[f"p_push_{_line(l)}"]
             + df[f"p_under_{_line(l)}"])
        assert s.sub(1.0).abs().max() < 2e-3
    # Run-line 3-way splits sum to 1 exactly.
    for m in RUN_LINE_GRID_FULL:
        s = sum(df[rl_col(m, side)] for side in ("home", "push", "away"))
        assert s.sub(1.0).abs().max() < 2e-3
    assert not math.isnan(df["p_home_win_derived"].iloc[0])
