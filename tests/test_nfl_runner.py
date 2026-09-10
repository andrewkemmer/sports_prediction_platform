"""NFL runner, artifact-schema, adapter, and policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import default_config
from core.retention import plan_retention
from sports.nfl.adapter import NFLAdapter
from sports.nfl.artifacts import (
    NFLArtifactError,
    persist_shap_game,
    write_predictions_history_csv,
    write_power_rankings_csv,
)
from sports.nfl.runner import (
    FORBIDDEN_ENV_VARS,
    run_nfl_production,
)
from sports.nfl.study_config import load_nfl_study
from tests.nfl_fixtures import make_schedule


# ---------------------------------------------------------------------------
# Mandatory env vars
# ---------------------------------------------------------------------------


class TestMandatoryEnv:
    def _base(self, **over):
        env = {"NFL_START_DATE": "2026-09-06",
               "NFL_END_DATE": "2026-09-07",
               "NFL_FULL_REPULL": "0"}
        env.update(over)
        return env

    def test_missing_all_raises(self):
        with pytest.raises(Exception, match="Missing required"):
            run_nfl_production(env={}, dry_run_ingestion=True)

    def test_whitespace_counts_as_missing(self):
        env = self._base(NFL_FULL_REPULL="   ")
        with pytest.raises(Exception, match="Missing required"):
            run_nfl_production(env=env, dry_run_ingestion=True)

    def test_bad_format_raises(self):
        with pytest.raises(Exception, match="YYYY-MM-DD"):
            run_nfl_production(env=self._base(NFL_START_DATE="20260906"),
                               dry_run_ingestion=True)

    def test_end_before_start_raises(self):
        with pytest.raises(Exception, match=">="):
            run_nfl_production(env=self._base(NFL_END_DATE="2026-09-05"),
                               dry_run_ingestion=True)

    def test_repull_must_be_0_or_1(self):
        with pytest.raises(Exception, match="0 or 1"):
            run_nfl_production(env=self._base(NFL_FULL_REPULL="true"),
                               dry_run_ingestion=True)

    def test_season_variables_forbidden(self):
        env = self._base(NFL_SLATE_SEASON="2026")
        with pytest.raises(Exception, match="Forbidden"):
            run_nfl_production(env=env, dry_run_ingestion=True)
        assert FORBIDDEN_ENV_VARS == ("NFL_START_SEASON", "NFL_END_SEASON",
                                      "NFL_SLATE_SEASON")


# ---------------------------------------------------------------------------
# Schema fixture + artifact writers
# ---------------------------------------------------------------------------


class TestSchemaFixture:
    def test_fixture_exists_and_has_provenance(self):
        fx = Path("tests/fixtures/nfl_artifact_schemas.json")
        data = json.loads(fx.read_text())
        assert "5dce619" in data["extracted_from_commit"]
        expected = {"moneyline_json", "calibration_json",
                    "predictions_history", "power_rankings", "markets",
                    "markets_meta", "markets_monitor", "qb_matchup",
                    "feature_json", "model_monitor", "shap_game"}
        assert expected <= set(data["artifacts"].keys())

    def test_markets_241_columns(self):
        from sports.nfl.artifacts import markets_columns
        cols = markets_columns()
        assert len(cols) == 241
        assert cols[0] == "kind" and cols[1] == "game_id"

    def test_predictions_history_writer_matches_fixture(self, tmp_path):
        fx = json.loads(Path(
            "tests/fixtures/nfl_artifact_schemas.json").read_text())
        oof = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "gameday": pd.to_datetime(["2026-09-06", "2026-09-06"]),
            "home_team": ["KC", "BUF"], "away_team": ["BAL", "NYJ"],
            "home_score": [27.0, 10.0], "away_score": [20.0, 23.0],
            "home_win": [1.0, 0.0],
            "p_ensemble": [0.65, 0.42],
            "p_ensemble_calibrated": [0.66, 0.41],
        })
        out = write_predictions_history_csv(
            tmp_path / "nfl_predictions_history_20260907.csv", oof,
            oof["p_ensemble_calibrated"].to_numpy())
        assert list(out.columns) == fx["artifacts"]["predictions_history"][
            "columns"]
        assert (out["model_pick"] == ["KC", "NYJ"]).all()
        assert (out["actual_winner"] == ["KC", "NYJ"]).all()
        assert out["correct"].all()

    def test_power_rankings_writer_matches_fixture(self, tmp_path):
        fx = json.loads(Path(
            "tests/fixtures/nfl_artifact_schemas.json").read_text())
        out = write_power_rankings_csv(
            tmp_path / "nfl_power_rankings_20260907.csv",
            {"KC": 1650.0, "BUF": 1600.0},
            {"KC": (5, 1), "BUF": (4, 2)},
            {"KC": "Kansas City Chiefs", "BUF": "Buffalo Bills"})
        assert list(out.columns) == fx["artifacts"]["power_rankings"][
            "columns"]
        assert out["rank"].tolist() == [1, 2]

    def test_shap_game_persists_fixture_columns(self, tmp_path):
        fx = json.loads(Path(
            "tests/fixtures/nfl_artifact_schemas.json").read_text())
        frame = pd.DataFrame({
            "feature": ["elo_diff", "win_pct_diff"],
            "shap_value": [0.12, -0.05],
            "signed_effect": ["positive", "negative"],
            "perspective_team": ["KC", "KC"],
        })
        p = persist_shap_game(frame, "20260907", "2026_01_BAL_KC", tmp_path)
        assert p.name == "nfl_shap_game_20260907_2026_01_BAL_KC.csv"
        out = pd.read_csv(p)
        assert list(out.columns) == fx["artifacts"]["shap_game"]["columns"]

    def test_shap_writer_rejects_missing_columns(self, tmp_path):
        with pytest.raises(NFLArtifactError):
            persist_shap_game(pd.DataFrame({"feature": ["x"]}),
                              "20260907", "g", tmp_path)


# ---------------------------------------------------------------------------
# Adapter semantics
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_settlement_configs(self):
        a = NFLAdapter()
        ml = a.settlement_config("moneyline")
        assert ml.tie_allowed and not ml.push_allowed
        for kind in ("spread", "total"):
            cfg = a.settlement_config(kind)
            assert cfg.push_allowed and not cfg.tie_allowed
        with pytest.raises(KeyError):
            a.settlement_config("run_line")

    def test_normalize_games_shared_fields(self):
        a = NFLAdapter()
        raw = pd.DataFrame({
            "game_id": ["g1", "g2"], "gameday": ["2026-09-06"] * 2,
            "gametime": ["20:15", "13:00"],
            "home_team": ["KC", "BUF"], "away_team": ["BAL", "NYJ"],
            "home_score": [27.0, np.nan], "away_score": [20.0, np.nan],
            "home_win_prob_model": [0.7, 0.51],
        })
        out = a.normalize_games(raw)
        for col in ("game_status", "evening_game", "day_game",
                    "home_team_name", "away_team_name", "model_pick",
                    "model_correct", "is_upset", "is_coin_flip"):
            assert col in out.columns
        assert out["game_status"].tolist() == ["Final", "Scheduled"]
        assert out["evening_game"].tolist() == [1.0, 0.0]
        assert out["model_pick"].tolist()[:1] == ["KC"]
        # decided + correct pick -> model_correct 1.0
        assert out["model_correct"].iloc[0] == 1.0
        # undecided row never gets a fabricated correctness
        assert pd.isna(out["model_correct"].iloc[1])
        # coin flip: 0.51 within the ±0.02 threshold
        assert out["is_coin_flip"].iloc[1] == 1.0

    def test_participant_box_tbd(self):
        a = NFLAdapter()
        row = pd.Series({"qb_home_name": None, "qb_home_rating": np.nan,
                         "qb_home_td_per_game": np.nan})
        box = a.participant_box(row, "home")
        assert box.is_tbd and box.role == "qb" and box.name == ""
        row2 = pd.Series({"qb_home_name": "Patrick Mahomes",
                          "qb_home_rating": 105.2,
                          "qb_home_td_per_game": 2.1})
        box2 = a.participant_box(row2, "home")
        assert not box2.is_tbd
        assert box2.name == "Patrick Mahomes"
        labels = [s[0] for s in box2.stats]
        assert any("Rating" in l for l in labels)
        assert any("TD/game" in l for l in labels)

    def test_market_probs_preserve_tie(self):
        a = NFLAdapter()
        row = pd.Series({"p_home_win_derived": 0.45,
                         "p_away_win_derived": 0.53, "p_tie": 0.02})
        probs = a.market_probs(row)
        assert abs(probs["p_home"] + probs["p_tie"] + probs["p_away"]
                   - 1.0) < 1e-9

    def test_adapter_implements_protocol(self):
        from core.contracts import SportAdapter
        assert isinstance(NFLAdapter(), SportAdapter)


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    def test_frontend_families_only(self, tmp_path):
        sink = tmp_path / "data_delivery"
        sink.mkdir(parents=True)
        # dated frontend artifact (past cutoff)
        (sink / "nfl_power_rankings_20200101.csv").touch()
        # dated frontend artifact (recent)
        (sink / "nfl_power_rankings_20260907.csv").touch()
        # retention-exempt OOF store (never deleted)
        models = sink / "models"
        models.mkdir()
        (models / "nfl_oof_moneyline_20200101.csv").touch()
        cfg = default_config()
        plan = plan_retention(sink, cfg, "nfl",
                              reference_date="20260910")
        names = [p.name for p in plan.candidates]
        assert names == ["nfl_power_rankings_20200101.csv"]
        protected = [p.name for p in plan.protected]
        assert "nfl_oof_moneyline_20200101.csv" in protected

    def test_undated_files_protected(self, tmp_path):
        sink = tmp_path / "data_delivery"
        sink.mkdir(parents=True)
        (sink / "nfl_features_metadata.json").touch()
        cfg = default_config()
        plan = plan_retention(sink, cfg, "nfl", reference_date="20260910")
        assert plan.n_candidates == 0


# ---------------------------------------------------------------------------
# Determinism of the study + fold geometry
# ---------------------------------------------------------------------------


class TestStudyDeterminism:
    def test_study_load_is_deterministic(self):
        a = load_nfl_study()
        b = load_nfl_study()
        assert a == b

    def test_fold_cadence_is_calendar_day_based(self):
        """7-calendar-day windows — never week-ID based."""
        sched = make_schedule(2019, 1)
        df = sched[sched["home_score"].notna()].reset_index(drop=True)
        from core.folds import make_folds
        folds = make_folds(df, 2019, cadence_days=7, date_col="gameday",
                           val_scope="oof_season", end_of_day=True)
        for f in folds:
            assert (f.val_end - f.val_start) == pd.Timedelta(days=6)
