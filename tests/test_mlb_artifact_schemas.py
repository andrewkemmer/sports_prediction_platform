"""MLB artifact schema verification against immutable real-sample fixtures.

The fixture (``tests/fixtures/mlb_artifact_schemas.json``) was extracted
from the read-only reference repository's ``mlb-backend/data_delivery/``
production artifacts (commit 5dce619, files dated 2026-09-07). These tests
pin the MLB artifact contract to the EXACT real columns/keys — not
remembered schemas. If a fixture and the contract disagree, the test
fails and the difference must be resolved deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.contracts import default_artifact_contract

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mlb_artifact_schemas.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mlb_contract() -> object:
    return default_artifact_contract("mlb")


# Map fixture artifact name -> contract family name.
FIXTURE_TO_FAMILY = {
    "todays_games": "todays_games",
    "power_rankings": "power_rankings",
    "run_engine_markets": "markets",
    "run_engine_markets_meta": "markets",
    "calibration": "calibration",
    "model_monitor": "model_monitor",
    "rolling_brier": "rolling_brier",
    "features_metadata": "features_metadata",
    "run_engine_monitor": "markets_monitor",
    "shap_game": "shap_game",
    "predictions_history": "predictions_history",
    "feature_coverage": "feature_coverage",
    "feature_drift": "feature_drift",
    "run_engine_oof": "run_engine_oof",
}


def test_fixture_exists_and_is_immutable_source(fixture):
    assert fixture["source"].startswith("Read-only reference repository")
    assert fixture["extracted_from_commit"] == "5dce619"
    assert len(fixture["artifacts"]) >= 13


def test_every_fixture_artifact_maps_to_a_contract_family(
        fixture, mlb_contract):
    for name in fixture["artifacts"]:
        fam = FIXTURE_TO_FAMILY[name]
        # Must resolve without KeyError — the contract declares the family.
        mlb_contract.family(fam)


def test_contract_family_patterns_match_real_filenames(fixture, mlb_contract):
    real = {
        "todays_games": "todays_games_20260907.csv",
        "power_rankings": "power_rankings_20260907.csv",
        "markets": "run_engine_markets_20260907.csv",
        "calibration": "calibration_20260907.json",
        "model_monitor": "model_monitor_20260907.json",
        "rolling_brier": "rolling_brier_20260907.json",
        "features_metadata": "features_metadata_20260907.json",
        "markets_monitor": "run_engine_monitor_20260907.json",
        "shap_game": "shap_game_20260907_ARI@KC.csv",
        "predictions_history": "predictions_history_20260907.csv",
        "feature_coverage": "feature_coverage_20260907.csv",
        "feature_drift": "feature_drift_20260907.csv",
        "run_engine_oof": "run_engine_oof_20260907.csv",
    }
    for fam_name, filename in real.items():
        fam = mlb_contract.family(fam_name)
        # The family pattern must accept the real production filename.
        date = fam.date_from_name(filename)
        if fam.date_stamped:
            assert date == "20260907", (
                f"family {fam_name} pattern {fam.pattern} does not match "
                f"real filename {filename}")


def test_todays_games_columns_exact(fixture, mlb_contract):
    cols = fixture["artifacts"]["todays_games"]["columns"]
    assert cols == [
        "game_id", "game_date", "start_time_utc", "home_team", "away_team",
        "home_record", "away_record", "home_win_prob_model",
        "away_win_prob_model", "edge_home", "edge_away", "sp_name_home",
        "sp_name_away", "sp_era_home", "sp_k9_home", "sp_era_away",
        "sp_k9_away", "venue", "model_pick", "home_win", "home_score",
        "away_score", "total_runs", "game_state", "game_status_detail",
    ]
    assert len(cols) == 25


def test_run_engine_markets_column_groups(fixture):
    cols = fixture["artifacts"]["run_engine_markets"]["columns"]
    assert len(cols) == 78
    over = [c for c in cols if c.startswith("p_over_")]
    push = [c for c in cols if c.startswith("p_push_")]
    under = [c for c in cols if c.startswith("p_under_")]
    cover = [c for c in cols if c.startswith("p_home_cover_")]
    rl = [c for c in cols if c.startswith("p_rl_")]
    base = [c for c in cols if not c.startswith((
        "p_over_", "p_push_", "p_under_", "p_home_cover_", "p_rl_"))]
    assert len(over) == len(push) == len(under) == 13
    assert len(cover) == 4
    assert len(rl) == 21  # 7 lines x (home, push, away)
    assert base == [
        "game_pk", "game_date", "kind", "home_expected_runs",
        "away_expected_runs", "alpha_home", "alpha_away",
        "p_home_win_derived", "p_away_win_derived", "home_score",
        "away_score", "total_runs", "ml_win_prob", "agreement_conflict",
    ]
    # Totals grid 6.5..12.5 in 0.5 steps; run lines 1.0..4.0 in 0.5 steps.
    total_lines = [c[len("p_over_"):] for c in over]
    assert total_lines[0] == "6_5" and total_lines[-1] == "12_5"
    rl_lines = sorted({c.split("_")[2] + "_" + c.split("_")[3] for c in rl})
    assert rl_lines == ["1_0", "1_5", "2_0", "2_5", "3_0", "3_5", "4_0"]


def test_model_monitor_top_level_keys_exact(fixture):
    keys = fixture["artifacts"]["model_monitor"]["keys"]
    assert keys == [
        "date", "version", "last_retrained", "next_retrain", "metrics",
        "drift_summary", "feature_drift", "feature_coverage",
        "rolling_brier", "brier_baseline", "brier_baseline_label",
        "rolling_brier_meta", "features_metadata", "run_engine",
        "ensemble", "model_history", "version_history",
    ]


def test_calibration_top_level_keys_exact(fixture):
    keys = fixture["artifacts"]["calibration"]["keys"]
    assert keys == [
        "date", "n_games", "trained_at", "metrics", "calibration_buckets",
        "calibration", "daily", "league_total", "evening_games_league",
    ]


def test_metrics_keys_match_core_validation(fixture):
    """The seven metric keys in every monitor/calibration artifact."""
    metrics = fixture["artifacts"]["model_monitor"]["subkeys"]["metrics"]
    assert metrics == ["auc", "brier", "logloss", "ece",
                       "brier_calibrated", "logloss_calibrated",
                       "ece_calibrated"]


def test_features_metadata_shape_matches_feature_contract(fixture):
    keys = fixture["artifacts"]["features_metadata"]["keys"]
    assert keys == ["generated_for", "n_features", "warnings", "features",
                    "categorical_context"]
    entry_keys = fixture["artifacts"]["features_metadata"]["subkeys"][
        "features[<name>]"]
    assert entry_keys == ["name", "summary", "definition", "formula",
                          "source", "window", "units", "direction",
                          "members", "tooltip"]
    # The FeatureSpec dataclass serializes exactly this shape.
    from core.contracts import FeatureContract, FeatureSpec
    spec = FeatureSpec(name="is_home", summary="s", definition="d",
                       formula="1", source="static", window="n/a",
                       units="binary", direction="n/a",
                       members=("xgboost",))
    meta = FeatureContract(sport="mlb", version="v1",
                           features=(spec,)).to_metadata_json("20260907")
    assert list(next(iter(meta["features"].values())).keys()) == entry_keys


def test_frontend_required_fields_present_in_columns(fixture):
    """Every frontend_required field for CSV artifacts is a real column."""
    for name, art in fixture["artifacts"].items():
        if art["ext"] != "csv":
            continue
        required = art.get("frontend_required", [])
        if required == ["all"] or not required:
            continue
        missing = set(required) - set(art["columns"])
        assert not missing, f"{name}: frontend requires {missing}"


def test_participant_fields_map_to_participant_box(fixture):
    """The MLB participant box fields come straight from todays_games."""
    cols = fixture["artifacts"]["todays_games"]["columns"]
    for f in ("sp_name_home", "sp_era_home", "sp_k9_home",
              "sp_name_away", "sp_era_away", "sp_k9_away"):
        assert f in cols
    # The reference frontend reads these with fallback aliases
    # (starting_pitcher_home / sp_home_era) — the primary names are fixed.
    from core.contracts import ParticipantBox
    box = ParticipantBox(role="pitcher", name="Jesús Luzardo",
                         stats=(("ERA", "3.20"), ("K/9", "11.81")))
    assert box.stats == (("ERA", "3.20"), ("K/9", "11.81"))


def test_durable_families_are_retention_exempt(fixture, mlb_contract):
    # features_metadata is durable in the contract; the real repo also
    # emits a dated copy, but the contract treats the family as exempt.
    fam = mlb_contract.family("features_metadata")
    assert fam.retention_exempt
    assert not fam.date_stamped


def test_run_engine_oof_has_no_frontend_reader(fixture):
    art = fixture["artifacts"]["run_engine_oof"]
    assert art["frontend_required"] == []
    assert art["note"].startswith("No frontend reader")
