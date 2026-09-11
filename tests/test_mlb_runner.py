"""MLB production-runner tests: mandatory env vars, retention gating, and
the runner's schema-honest payload helpers.

``run_mlb_production`` fails loudly when MLB_START_DATE / MLB_END_DATE /
MLB_FULL_REPULL are missing or invalid; retention executes only after
successful artifact generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.mlb.runner import (
    ENV_END_DATE,
    ENV_FULL_REPULL,
    ENV_START_DATE,
    MLBRunnerError,
    run_mlb_production,
)

VALID_ENV = {ENV_START_DATE: "2026-09-01", ENV_END_DATE: "2026-09-07",
             ENV_FULL_REPULL: "0"}


# ---------------------------------------------------------------------------
# Mandatory environment variables (no silent defaults)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", [ENV_START_DATE, ENV_END_DATE,
                                     ENV_FULL_REPULL])
def test_missing_env_var_fails_loudly(missing):
    env = dict(VALID_ENV)
    env.pop(missing)
    with pytest.raises(MLBRunnerError, match="Missing required production"):
        run_mlb_production(env=env, dry_run_ingestion=True)


@pytest.mark.parametrize("bad", ["true", "yes", "2", "1.0", ""])
def test_invalid_full_repull_fails_loudly(bad):
    env = dict(VALID_ENV, **{ENV_FULL_REPULL: bad})
    with pytest.raises(Exception, match="FULL_REPULL"):
        run_mlb_production(env=env, dry_run_ingestion=True)


def test_invalid_date_format_fails_loudly():
    env = dict(VALID_ENV, **{ENV_START_DATE: "20260901"})
    with pytest.raises(Exception, match="YYYY-MM-DD"):
        run_mlb_production(env=env, dry_run_ingestion=True)


def test_end_before_start_fails_loudly():
    env = dict(VALID_ENV, **{ENV_END_DATE: "2026-08-31"})
    with pytest.raises(Exception, match="END_DATE"):
        run_mlb_production(env=env, dry_run_ingestion=True)


def test_no_silent_today_default_for_end_date():
    env = {ENV_START_DATE: "2026-09-01", ENV_FULL_REPULL: "0"}
    with pytest.raises(MLBRunnerError):
        run_mlb_production(env=env, dry_run_ingestion=True)


def test_whitespace_only_value_is_not_a_value():
    env = dict(VALID_ENV, **{ENV_START_DATE: "   "})
    with pytest.raises(MLBRunnerError, match="Missing required production"):
        run_mlb_production(env=env, dry_run_ingestion=True)


def test_env_names_are_mlb_prefixed():
    """No aliasing: the strict names are MLB_START_DATE/MLB_END_DATE/
    MLB_FULL_REPULL (not the generic core names)."""
    assert ENV_START_DATE == "MLB_START_DATE"
    assert ENV_END_DATE == "MLB_END_DATE"
    assert ENV_FULL_REPULL == "MLB_FULL_REPULL"


# ---------------------------------------------------------------------------
# Retention gating inside the runner path
# ---------------------------------------------------------------------------

def test_runconfig_training_window_independent_of_run_window():
    """The run window scopes ingestion/serving; training history comes from
    the study config — the runner never derives training dates from the env
    values."""
    from core.runconfig import resolve_run_window
    from sports.mlb.study_config import load_mlb_study
    study = load_mlb_study()
    w = resolve_run_window(
        start_date=VALID_ENV[ENV_START_DATE],
        end_date=VALID_ENV[ENV_END_DATE],
        full_repull=VALID_ENV[ENV_FULL_REPULL])
    # The study's training geometry is a constant of the config, not a
    # function of the run window.
    assert study.walk_forward.min_train_games == 200
    assert study.training.retrain_cadence_days == 7
    assert w.start_date == "2026-09-01"  # serving scope only


def test_retention_exemption_for_run_engine_oof(tmp_path):
    """run_engine_oof_* is NOT frontend-consumed: it must never be a
    retention candidate even when dated and old."""
    from core.config import default_config
    from core.retention import run_production_retention
    sink = tmp_path / "sports" / "mlb" / "data_delivery"
    sink.mkdir(parents=True)
    (sink / "run_engine_oof_20260801.csv").write_text("game_pk\n1\n")
    (sink / "todays_games_20260801.csv").write_text("game_id\nx\n")
    plan = run_production_retention(
        sink, default_config(), "mlb", generation_succeeded=True,
        reference_date="20260907")
    deleted = {p.name for p in plan.deleted}
    assert "todays_games_20260801.csv" in deleted
    assert "run_engine_oof_20260801.csv" not in deleted
    assert (sink / "run_engine_oof_20260801.csv").exists()


def test_study_yaml_exempt_family_never_allowlisted():
    study = load_study_guarded()
    assert "run_engine_oof" in study.retention_exempt_families
    assert "run_engine_oof" not in study.allowlisted_families


def load_study_guarded():
    from sports.mlb.study_config import load_mlb_study
    return load_mlb_study()


# ---------------------------------------------------------------------------
# Payload helpers (schema-honest)
# ---------------------------------------------------------------------------

def test_todays_games_helper_never_fabricates(tmp_path):
    from sports.mlb.runner import _todays_games_frame
    slate = pd.DataFrame({"game_pk": [1], "game_date": ["2026-09-07"]})
    frame = _todays_games_frame(slate, {})
    # Fixture columns all present; model probability stays NA (no training
    # in this helper — never a fabricated number).
    assert "home_win_prob_model" in frame.columns
    assert frame["home_win_prob_model"].isna().all()


def test_empty_slate_yields_empty_typed_frame():
    from sports.mlb.runner import _todays_games_frame
    frame = _todays_games_frame(pd.DataFrame(), {})
    assert len(frame) == 0
    assert "game_id" in frame.columns


def test_power_rankings_helper_on_empty_decided():
    from sports.mlb.runner import _power_rankings_frame
    frame = _power_rankings_frame(pd.DataFrame())
    assert len(frame) == 0
    assert "elo" in frame.columns


# ---------------------------------------------------------------------------
# Phase 2.1 additions: SHAP count rule, imputation policy, unavailable columns
# ---------------------------------------------------------------------------

def test_shap_count_rule_exactly_one_per_slate_game():
    """The SHAP contract: EXACTLY one shap_game file per undecided slate
    game — no fixed minimum (the old 'six files' rule is removed)."""
    from sports.mlb.runner import _write_slate_shap
    from sports.mlb.feature_registry import MONEYLINE_FEATURE_COLS
    rng = np.random.default_rng(3)
    n = 120
    decided = pd.DataFrame({
        "home_win": rng.integers(0, 2, n).astype(float)})
    for c in MONEYLINE_FEATURE_COLS:
        decided[c] = rng.normal(size=n)
    slate = decided.iloc[:4].drop(columns=["home_win"]).copy()
    slate["home_team"] = ["BOS", "NYY", "LAD", "ATL"]
    slate["away_team"] = ["NYY", "BOS", "SD", "NYM"]
    with tempfile.TemporaryDirectory() as td:
        written = _write_slate_shap(Path(td), "20260907", slate, decided,
                                    MONEYLINE_FEATURE_COLS)
        assert len(written) == len(slate), "one file per slate game"
        keys = [p.name.split("_", 2)[2] for p in written]
        assert len(set(keys)) == len(keys), "unique per game"
        # Empty slate -> zero files (0 == len(slate), no minimum).
        empty = _write_slate_shap(Path(td), "20260907",
                                  slate.iloc[:0], decided, MONEYLINE_FEATURE_COLS)
        assert empty == []


def test_imputation_is_train_only():
    """Apply-time imputation reuses TRAIN medians: an apply row with an
    extreme NaN pattern is filled from train statistics, not refit."""
    from sports.mlb.training import _impute_train_median
    rng = np.random.default_rng(5)
    X_tr = rng.normal(loc=10.0, scale=2.0, size=(80, 3))
    X_tr[:10, 1] = np.nan  # partial missingness in train
    X_apply = rng.normal(loc=10.0, scale=2.0, size=(5, 3))
    X_apply[:, 1] = np.nan
    out = _impute_train_median(X_tr, X_apply)
    assert not np.isnan(out).any()
    # Fill value equals the train-fold median (~10.0), not the apply-frame's
    # own (degenerate) statistic.
    assert np.allclose(out[:, 1], np.nanmedian(X_tr[:, 1]), atol=1e-9)


def test_all_nan_training_column_routed_explicitly():
    """A column entirely unavailable in training is routed as explicitly
    unavailable (documented 0.0 fallback for the linear members only) —
    it never fabricates a pseudo-statistic, and tree members keep NaN."""
    from sports.mlb.training import _impute_train_median
    rng = np.random.default_rng(6)
    X_tr = rng.normal(size=(60, 2))
    X_tr[:, 1] = np.nan
    X_apply = rng.normal(size=(4, 2))
    X_apply[:, 1] = np.nan  # slate rows also lack the feature entirely
    out = _impute_train_median(X_tr, X_apply)
    # Available column: median fill. Unavailable column: explicit 0.0.
    assert np.allclose(out[:, 1], 0.0)
    # Tree path is untouched: raw matrix still carries NaN.
    assert np.isnan(X_tr[:, 1]).all()


def test_features_metadata_marks_unavailable_columns():
    """features_metadata.warnings explicitly lists every entirely-
    unavailable feature — never a silent drop."""
    from sports.mlb.runner import _features_metadata_payload
    decided = pd.DataFrame({
        "real_feature": [1.0, 2.0, 3.0],
        "ghost_feature": [np.nan, np.nan, np.nan],
    })
    payload = _features_metadata_payload(
        ("real_feature", "ghost_feature", "absent_feature"),
        "20260907", decided)
    assert any("ghost_feature" in w and "unavailable" in w
               for w in payload["warnings"])
    assert any("absent_feature" in w for w in payload["warnings"])
    assert not any("real_feature" in w for w in payload["warnings"])


def test_five_member_stack_all_execute():
    """The complete intended five-member moneyline stack (xgboost,
    lightgbm, logistic, randomforest, mlp) must all fit and predict —
    xgboost is no longer skippable-by-absence in the certified runtime."""
    from sports.mlb.training import ENSEMBLE_WEIGHTS, _fit_member, \
        _member_predict
    rng = np.random.default_rng(9)
    X = rng.normal(size=(150, 4)); X[rng.random(X.shape) < 0.05] = np.nan
    y = (X[:, 0] > 0).astype(float)
    executed = []
    for name in ENSEMBLE_WEIGHTS:
        m = _fit_member(name, X, y, X, y)
        if m is not None:
            _member_predict(name, m, X)
            executed.append(name)
    assert set(executed) == set(ENSEMBLE_WEIGHTS), \
        f"missing members: {set(ENSEMBLE_WEIGHTS) - set(executed)}"
