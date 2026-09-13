"""MLB production-runner tests: mandatory env vars, retention gating, and
the runner's schema-honest payload helpers.

``run_mlb_production`` fails loudly when MLB_START_DATE / MLB_END_DATE /
MLB_FULL_REPULL are missing or invalid; retention executes only after
successful artifact generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.mlb.run_production import (
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
    from core.config import resolve_run_window
    from sports.mlb.config.study_config import load_mlb_study
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
    from core.artifacts import run_production_retention
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
    from sports.mlb.config.study_config import load_mlb_study
    return load_mlb_study()


# ---------------------------------------------------------------------------
# Payload helpers (schema-honest)
# ---------------------------------------------------------------------------

def test_todays_games_helper_never_fabricates(tmp_path):
    from sports.mlb.run_production import _todays_games_frame
    slate = pd.DataFrame({"game_pk": [1], "game_date": ["2026-09-07"]})
    frame = _todays_games_frame(slate, {})
    # Fixture columns all present; model probability stays NA (no training
    # in this helper — never a fabricated number).
    assert "home_win_prob_model" in frame.columns
    assert frame["home_win_prob_model"].isna().all()


def test_empty_slate_yields_empty_typed_frame():
    from sports.mlb.run_production import _todays_games_frame
    frame = _todays_games_frame(pd.DataFrame(), {})
    assert len(frame) == 0
    assert "game_id" in frame.columns


def test_power_rankings_helper_on_empty_decided():
    from sports.mlb.run_production import _power_rankings_frame
    frame = _power_rankings_frame(pd.DataFrame())
    assert len(frame) == 0
    assert "elo" in frame.columns


# ---------------------------------------------------------------------------
# Phase 2.1 additions: SHAP count rule, imputation policy, unavailable columns
# ---------------------------------------------------------------------------

def test_shap_count_rule_exactly_one_per_slate_game():
    """The SHAP contract: EXACTLY one shap_game file per undecided slate
    game — no fixed minimum (the old 'six files' rule is removed)."""
    from sports.mlb.run_production import _write_slate_shap
    from sports.mlb.features.registry import MONEYLINE_FEATURE_COLS
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
    from sports.mlb.models.moneyline.training import _impute_train_median
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
    from sports.mlb.models.moneyline.training import _impute_train_median
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
    from sports.mlb.run_production import _features_metadata_payload
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
    from sports.mlb.models.moneyline.training import ENSEMBLE_WEIGHTS, _fit_member, \
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


# ---------------------------------------------------------------------------
# rolling_brier payload: the persisted series must be COMPUTED, not discarded
# ---------------------------------------------------------------------------

_MLB_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
     / "mlb_artifact_schemas.json").read_text(encoding="utf-8"))


def _markets_frame(dates, probs, scores):
    return pd.DataFrame({
        "game_pk": [f"g{i}" for i in range(len(dates))],
        "game_date": dates,
        "ml_win_prob": probs,
        "home_score": [s[0] for s in scores],
        "away_score": [s[1] for s in scores],
    })


def test_rolling_brier_payload_known_value():
    """KNOWN-VALUE Brier: one day of p=[0.8, 0.6, 0.3] against actual
    outcomes [1, 0, 1] must give mean((p - y)^2) = 0.29667, i.e. a real float
    derived from predictions vs outcomes — never a placeholder."""
    from sports.mlb.run_production import _rolling_brier_payload
    markets = _markets_frame(["2026-09-01"] * 3, [0.8, 0.6, 0.3],
                             [(5, 3), (2, 4), (7, 1)])
    payload = _rolling_brier_payload({"markets": markets})
    expected = ((0.8 - 1.0) ** 2 + (0.6 - 0.0) ** 2
                + (0.3 - 1.0) ** 2) / 3
    assert payload["series"] == [{"date": "2026-09-01",
                                  "brier": round(expected, 5),
                                  "games": 3}]
    assert payload["n_points"] == 1
    assert payload["n_games_total"] == 3
    assert payload["n_games_in_series"] == 3
    assert isinstance(payload["history_mean_brier"], float)
    assert payload["history_mean_brier"] == pytest.approx(round(expected, 5))
    assert 0.0 <= payload["history_mean_brier"] <= 1.0


def test_rolling_brier_payload_multi_day_series_is_real():
    """Two days, three games: per-day means must be the true Brier values and
    the series must be date-sorted."""
    from sports.mlb.run_production import _rolling_brier_payload
    markets = _markets_frame(["2026-09-01", "2026-09-01", "2026-09-02"],
                             [0.7, 0.4, 0.9],
                             [(3, 1), (0, 2), (6, 5)])
    payload = _rolling_brier_payload({"markets": markets})
    assert payload["series"] == [
        {"date": "2026-09-01", "brier": 0.125, "games": 2},
        {"date": "2026-09-02", "brier": 0.01, "games": 1},
    ]
    assert payload["history_mean_brier"] == pytest.approx(0.0675)
    assert payload["n_points"] == 2
    assert payload["n_games_total"] == 3


def test_rolling_brier_payload_excludes_unavailable_rows():
    """A game with no moneyline probability is EXCLUDED from the series, not
    fabricated as a 0.0/0.5 guess."""
    from sports.mlb.run_production import _rolling_brier_payload
    markets = _markets_frame(["2026-09-01"] * 3, [0.8, np.nan, 0.2],
                             [(1, 0), (2, 1), (0, 3)])
    payload = _rolling_brier_payload({"markets": markets})
    assert payload["series"] == [{"date": "2026-09-01", "brier": 0.04,
                                  "games": 2}]
    assert payload["n_games_total"] == 2


def test_rolling_brier_payload_preserves_contract_keys_and_order():
    """The schema the fixture pins is unchanged: same 12 keys, same order."""
    from sports.mlb.run_production import _rolling_brier_payload
    keys = _MLB_FIXTURE["artifacts"]["rolling_brier"]["keys"]
    payload = _rolling_brier_payload(
        {"markets": _markets_frame(["2026-09-01"], [0.5], [(1, 0)])})
    assert list(payload) == keys
    assert set(payload) == set(keys)


def test_rolling_brier_payload_empty_markets_is_contract_shaped():
    """With no decided rows the payload is empty but still contract-shaped;
    history_mean_brier is null only here (the one contract-nullable key)."""
    from sports.mlb.run_production import _rolling_brier_payload
    keys = _MLB_FIXTURE["artifacts"]["rolling_brier"]["keys"]
    payload = _rolling_brier_payload({"markets": pd.DataFrame()})
    assert list(payload) == keys
    assert payload["series"] == []
    assert payload["n_points"] == 0
    assert payload["n_games_total"] == 0
    assert payload["n_games_in_series"] == 0
    assert payload["history_mean_brier"] is None


def test_rolling_brier_uses_moneyline_semantics_not_the_totals_helper():
    """rolling_brier is contractually a MONEYLINE artifact. It must score
    ``home_win_prob_model`` (carried as ``ml_win_prob``) against settled
    ``home_score > away_score`` results — and must NOT accidentally consume
    the run engine's ``compute_rolling_totals_brier``, which scores the
    OVER/UNDER TOTALS market (``p_over_9_0`` vs ``total_runs >= 9.5``).
    Wiring that helper here would be a market-semantics regression."""
    import sports.mlb.models.market.run_engine as run_engine
    from sports.mlb.run_production import _rolling_brier_payload

    # A day where the two markets DISAGREE, so the returned value identifies
    # which semantics were used. Totals column is wrong on both games.
    markets = pd.DataFrame({
        "game_pk": ["a", "b"],
        "game_date": ["2026-09-01", "2026-09-01"],
        "ml_win_prob": [0.8, 0.2],
        "p_over_9_0": [0.0, 1.0],
        "home_score": [1, 0],
        "away_score": [0, 3],
        "total_runs": [1, 3],
    })
    # The totals helper is a DIFFERENT market (over/under) and is currently
    # defective as well as unwired: it builds ``brier_by_day`` as a numpy
    # ndarray and then indexes it with ``.loc``. Pinning that defect keeps a
    # future "just wire it up" change from looking harmless.
    with pytest.raises(AttributeError, match="loc"):
        run_engine.compute_rolling_totals_brier(markets)

    payload = _rolling_brier_payload({"markets": markets})
    assert payload["source_column"] == "home_win_prob_model"
    # moneyline: ((0.8-1)^2 + (0.2-0)^2) / 2 = 0.04
    assert payload["series"] == [{"date": "2026-09-01", "brier": 0.04,
                                  "games": 2}]
    assert payload["history_mean_brier"] == pytest.approx(0.04)
    # (If the totals market WERE scored here the same day would read 0.5:
    # p_over=[0.0, 1.0] against total_runs=[1, 3] -> y_over=[0, 0].)
    assert payload["history_mean_brier"] != pytest.approx(0.5)

    # The totals column is irrelevant to this artifact: removing it, and
    # removing the totals helper entirely, cannot change the payload.
    assert _rolling_brier_payload(
        {"markets": markets.drop(columns=["p_over_9_0"])}) == payload

    def _tripwire(*_a, **_k):
        raise AssertionError(
            "rolling_brier must not consume the totals-market helper")
    original = run_engine.compute_rolling_totals_brier
    run_engine.compute_rolling_totals_brier = _tripwire
    try:
        assert _rolling_brier_payload({"markets": markets}) == payload
    finally:
        run_engine.compute_rolling_totals_brier = original
