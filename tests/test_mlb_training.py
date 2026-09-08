"""Warmup eligibility, fold/OOF determinism, and study-config tests.

The warmup contract (30 completed games per team), fold geometry, and OOF
rules all come from the study config — never from the run window.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from core.config import default_config
from core.warmup import evaluate_warmup
from sports.mlb.feature_registry import (
    FEATURE_COLS,
    RUN_FEATURE_COLS,
    validate_registry,
)
from sports.mlb.study_config import (
    MLBStudyError,
    load_mlb_study,
)
from sports.mlb.training import train_moneyline_ensemble, walk_forward_splits
from tests.mlb_fixtures import make_statcast_games

D0 = date(2026, 4, 1)


# ---------------------------------------------------------------------------
# Warmup eligibility
# ---------------------------------------------------------------------------

def test_warmup_gate_30_games_per_team():
    cfg = default_config().sport("mlb").warmup
    assert cfg.min_games_per_team == 30
    ok = evaluate_warmup(
        sport_warmup=cfg, first_game_date="20260401",
        run_date="20260601",
        team_game_counts={t: 30 for t in ("BOS", "NYY")})
    assert ok.ready
    short = evaluate_warmup(
        sport_warmup=cfg, first_game_date="20260401",
        run_date="20260601",
        team_game_counts={"BOS": 30, "NYY": 29})
    assert not short.ready
    assert "NYY" in short.teams_below_minimum


def test_study_yaml_pins_the_agreed_defaults():
    study = load_mlb_study()
    assert study.warmup.min_games_per_team == 30
    assert study.warmup.min_history_days == 30
    assert study.walk_forward.step_days == 7
    assert study.oof.min_training_rows == 200
    assert study.frontend_days == 10
    assert "run_engine_oof" in study.retention_exempt_families
    assert "run_engine_oof" not in study.allowlisted_families


def test_study_rejects_warmup_drift(tmp_path):
    p = tmp_path / "study.yaml"
    p.write_text(
        "sport: mlb\nkind: production\nstudy_id: x\ndata_start_date: '20240101'\n"
        "warmup:\n  min_history_days: 30\n  min_games_per_team: 10\n"
        "walk_forward:\n  min_train_games: 200\n  step_days: 7\n"
        "oof:\n  min_training_rows: 200\n", encoding="utf-8")
    with pytest.raises(MLBStudyError, match="must be 30"):
        load_mlb_study(p)


def test_study_rejects_unknown_fields(tmp_path):
    p = tmp_path / "study.yaml"
    p.write_text(
        "sport: mlb\nkind: production\nstudy_id: x\ndata_start_date: '20240101'\n"
        "warmup:\n  min_history_days: 30\n  min_games_per_team: 30\n"
        "walk_forward:\n  min_train_games: 200\n  step_days: 7\n"
        "oof:\n  min_training_rows: 200\n"
        "dynamic_optimization: true\n", encoding="utf-8")
    with pytest.raises(MLBStudyError, match="unknown study fields"):
        load_mlb_study(p)


# ---------------------------------------------------------------------------
# Fold / OOF determinism
# ---------------------------------------------------------------------------

def _decided(n_days=60, seed=11):
    df = make_statcast_games(D0, n_days, games_per_day=4, seed=seed,
                             include_slate_day=False)
    rows = (df[df["description"] == "final"]
            .groupby("game_pk", as_index=False)
            .first())
    return rows


def test_walk_forward_non_overlapping_chronological():
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    splits = walk_forward_splits(games, retrain_cadence_days=7)
    assert splits
    prev_end = None
    for s in splits:
        assert s["train_games"]["game_date"].max() < s["val_start"]
        if prev_end is not None:
            assert s["val_start"] > prev_end
        prev_end = s["val_end"]
        assert len(s["val_games"]) > 0


def test_walk_forward_partial_tail_flagged():
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    splits = walk_forward_splits(games, retrain_cadence_days=7)
    tail = splits[-1]
    # The final fold may run partially into the tail; the flag is honest.
    assert isinstance(tail["is_partial_tail"], bool)


def test_walk_forward_deterministic():
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    a = walk_forward_splits(games, retrain_cadence_days=7)
    b = walk_forward_splits(games, retrain_cadence_days=7)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa["val_start"] == sb["val_start"]
        assert sa["val_end"] == sb["val_end"]
        assert sa["train_games"]["game_pk"].tolist() \
            == sb["train_games"]["game_pk"].tolist()


def test_min_train_days_warmup_skips_early_folds():
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    no_warm = walk_forward_splits(games, retrain_cadence_days=7,
                                  min_train_days=0)
    with_warm = walk_forward_splits(games, retrain_cadence_days=7,
                                    min_train_days=30)
    assert with_warm[0]["val_start"] > no_warm[0]["val_start"]


def test_moneyline_training_deterministic():
    study = load_mlb_study()
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["game_id"] = games["game_pk"].astype(str)
    games["home_team"] = "BOS"
    games["away_team"] = "NYY"
    # Use a small, all-NaN-free feature subset so every member trains fast.
    feature_cols = ("is_home", "elo_diff", "rest_days_diff")
    games["is_home"] = 1.0
    games["elo_diff"] = 10.0
    games["rest_days_diff"] = 1.0
    r1 = train_moneyline_ensemble(games, feature_cols, study,
                                  min_val_games=1, min_train_games=1)
    r2 = train_moneyline_ensemble(games, feature_cols, study,
                                  min_val_games=1, min_train_games=1)
    pd.testing.assert_series_equal(r1["oof"]["home_win_prob_model"],
                                   r2["oof"]["home_win_prob_model"])
    assert r1["metrics"]["n"] == r2["metrics"]["n"]


def test_oof_probs_strictly_inside():
    study = load_mlb_study()
    games = _decided()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["game_id"] = games["game_pk"].astype(str)
    games["home_team"] = "BOS"
    games["away_team"] = "NYY"
    games["is_home"] = 1.0
    games["elo_diff"] = 10.0
    games["rest_days_diff"] = 1.0
    res = train_moneyline_ensemble(games, ("is_home", "elo_diff",
                                           "rest_days_diff"), study,
                                   min_val_games=1, min_train_games=1)
    p = res["oof"]["home_win_prob_model"]
    assert ((p > 0) & (p < 1)).all()
    pc = res["oof"]["home_win_prob_model_calibrated"]
    assert ((pc > 0) & (pc < 1)).all()


def test_registry_consistent_with_frozen_views():
    validate_registry()  # raises FeatureRegistryError on drift
    assert len(RUN_FEATURE_COLS) == 53
    assert len(FEATURE_COLS) > len(RUN_FEATURE_COLS)
