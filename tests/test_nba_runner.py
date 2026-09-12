"""NBA runner, adapter, artifact-schema, SHAP count-rule, retention, and
env-var gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.nba.adapter import NBAAdapter
from sports.nba.runner import (
    ENV_END_DATE,
    ENV_FULL_REPULL,
    ENV_START_DATE,
    NBARunnerError,
    run_nba_production,
)
from sports.nba.study_config import load_nba_study
from tests.nba_fixtures import make_player_cache, make_schedule

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
     / "nba_artifact_schemas.json").read_text(encoding="utf-8"))


def _relaxed_study():
    """Production study with the minimum-training-rows gate relaxed for
    the tiny synthetic fixture (production stays strict at 50)."""
    import dataclasses
    s = load_nba_study()
    return dataclasses.replace(
        s,
        oof=dataclasses.replace(s.oof, min_training_rows=12),
        walk_forward=dataclasses.replace(
            s.walk_forward, min_train_games=12))


@pytest.fixture(scope="module")
def relaxed_study():
    return _relaxed_study()


@pytest.fixture(autouse=True)
def _relax_study_gate(monkeypatch):
    """Point every runner-path study load at the relaxed study."""
    monkeypatch.setattr("sports.nba.runner.load_nba_study",
                        _relaxed_study)


def _env(start="2025-10-20", end="2025-10-23", repull="0"):
    return {ENV_START_DATE: start, ENV_END_DATE: end,
            ENV_FULL_REPULL: repull}


def _prime_cache(cache_dir: Path, sched: pd.DataFrame):
    cache_dir.mkdir(parents=True, exist_ok=True)
    sched.to_parquet(cache_dir / "schedule.parquet", index=False)


class TestEnvGate:
    def test_missing_vars_fail_loudly(self):
        with pytest.raises(NBARunnerError, match="Missing required"):
            run_nba_production(env={}, dry_run_ingestion=True)

    def test_forbidden_season_vars_rejected(self):
        bad = _env()
        bad["NBA_START_SEASON"] = "2025"
        with pytest.raises(NBARunnerError, match="Forbidden"):
            run_nba_production(env=bad, dry_run_ingestion=True)

    def test_invalid_repull_rejected(self):
        with pytest.raises(Exception):
            run_nba_production(env=_env(repull="yes"),
                               dry_run_ingestion=True)

    def test_end_before_start_rejected(self):
        with pytest.raises(Exception):
            run_nba_production(env=_env(start="2025-10-23",
                                        end="2025-10-20"),
                               dry_run_ingestion=True)

    def test_bad_date_format_rejected(self):
        with pytest.raises(Exception):
            run_nba_production(env=_env(start="10/20/2025"),
                               dry_run_ingestion=True)


class TestAdapter:
    def test_artifact_contract_families(self):
        ac = NBAAdapter().artifact_contract()
        assert ac.sport == "nba"
        fams = {f.family for f in ac.families}
        assert "nba_moneyline_json" in fams
        assert "nba_shap_game" in fams
        assert "nba_oof_store" in fams
        exempt = {f.family for f in ac.families if f.retention_exempt}
        assert "nba_oof_store" in exempt
        assert "nba_shap_game" not in exempt

    def test_feature_contract(self):
        from sports.nba.feature_registry import MONEYLINE_CONTRACT_VERSION

        fc = NBAAdapter().feature_contract()
        assert fc.sport == "nba"
        assert fc.version == MONEYLINE_CONTRACT_VERSION
        assert len(fc.features) > 0

    def test_normalize_games_semantics(self):
        raw = pd.DataFrame([{
            "game_id": "g1", "home_team": "BOS", "away_team": "DEN",
            "home_score": 112.0, "away_score": 105.0,
            "start_time_utc": "2025-10-20T23:30:00Z",
            "home_win_prob_model": 0.62,
            "game_state": "Final",
        }, {
            "game_id": "g2", "home_team": "MIA", "away_team": "NYK",
            "home_score": np.nan, "away_score": np.nan,
            "start_time_utc": "2025-10-20T15:00:00Z",
            "home_win_prob_model": 0.51,
            "game_state": "7:30 pm ET",
        }])
        out = NBAAdapter().normalize_games(raw)
        assert list(out["game_status"]) == ["Final", "Scheduled"]
        assert list(out["evening_game"]) == [1.0, 0.0]
        assert out["model_pick"].iloc[0] == "BOS"
        assert out["is_coin_flip"].iloc[1] == 1.0
        assert out["model_correct"].iloc[0] == 1.0
        assert out["is_upset"].iloc[0] == 0.0

    def test_participant_box_tbd_and_filled(self):
        row = pd.Series({
            "player_home_name": "Jayson Tatum",
            "player_home_ppg": 27.4,
            "player_home_apg": 4.8,
        })
        box = NBAAdapter().participant_box(row, "home")
        assert box.is_tbd is False
        assert box.name == "Jayson Tatum"
        assert len(box.stats) == 2
        empty = NBAAdapter().participant_box(pd.Series({}), "away")
        assert empty.is_tbd is True
        assert empty.name == ""

    def test_market_probs_no_tie_ever(self):
        row = pd.Series({
            "p_home_win_derived": 0.6,
            "p_away_win_derived": 0.4,
            "p_push_offered": 0.05,
        })
        probs = NBAAdapter().market_probs(row)
        assert "p_tie" not in probs  # structurally impossible in the NBA
        assert np.isclose(probs["p_home"] + probs["p_away"], 1.0)


class TestParticipantEnrichment:
    def test_last10_strictly_prior(self):
        sched = make_schedule(2015, 2)
        panel = make_player_cache(sched)
        from sports.nba.participant_enrichment import enrich_slate
        slate = sched[sched["home_score"].isna()].copy()
        if slate.empty:
            pytest.skip("fixture has no undecided rows")
        out = enrich_slate(slate, panel)
        # named players derived from strictly-prior appearances
        assert "player_home_ppg" in out.columns
        # no player row for an undecided game can reference itself
        gd = pd.to_datetime(out["game_date"]).iloc[0]
        panel_gd = pd.to_datetime(panel["game_date"])
        assert (panel_gd < gd).any()

    def test_missing_cache_renders_tbd(self):
        sched = make_schedule(2015, 2)
        slate = sched[sched["home_score"].isna()].head(2).copy()
        if slate.empty:
            pytest.skip("fixture has no undecided rows")
        from sports.nba.participant_enrichment import enrich_slate
        out = enrich_slate(slate, None)
        for c in ("player_home_name", "player_away_name",
                  "player_home_ppg", "player_away_apg"):
            assert out[c].isna().all()


class TestSchemaFixture:
    def test_fixture_loads_and_pins_families(self):
        fams = set(FIXTURE["artifacts"].keys())
        assert {"moneyline_json", "calibration_json", "predictions_history",
                "power_rankings", "markets", "markets_meta",
                "markets_monitor", "player_matchup", "feature_json",
                "model_monitor", "shap_game"} <= fams

    def test_shap_count_rule(self):
        rule = FIXTURE["artifacts"]["shap_game"]["count_rule"]
        assert "exactly one" in rule
        assert "no fixed minimum" in rule

    def test_markets_grid_is_nba_sized(self):
        grids = FIXTURE["artifacts"]["markets_meta"]["grids"]
        assert grids["spread"] == [-20, 20]
        assert grids["total"] == [200, 250]


class TestRunnerEndToEnd:
    def test_bounded_run_writes_valid_artifacts(self, tmp_path):
        sched = make_schedule(2015, 2)
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        # run window over the undecided tail of the fixture
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        window_start = str(undecided_dates.min().date())
        window_end = str(undecided_dates.max().date())
        result = run_nba_production(
            env=_env(window_start, window_end, repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        assert result.n_decided_games > 0
        assert result.n_folds > 0
        assert result.n_oof_rows > 0
        names = set(result.artifacts_written)
        assert "nba_moneyline_v1_" + window_end.replace("-", "") \
            + ".json" in names
        assert any(n.startswith("nba_shap_game_") for n in names) \
            or result.n_prediction_games == 0
        assert result.retention_executed is True

        # ---- schema validation against the immutable fixture ----
        date_c = window_end.replace("-", "")
        ml = json.loads(
            (sink / f"nba_moneyline_v1_{date_c}.json").read_text())
        fx_ml = FIXTURE["artifacts"]["moneyline_json"]
        assert list(ml.keys()) == fx_ml["keys"]
        for g in ml["games"]:
            assert list(g.keys()) == fx_ml["game_keys"]
            assert g["home_score"] is None  # slate rows never fabricate

        cal = json.loads(
            (sink / f"nba_calibration_{date_c}.json").read_text())
        assert list(cal.keys()) == FIXTURE["artifacts"][
            "calibration_json"]["keys"]

        mon = json.loads(
            (sink / f"nba_model_monitor_{date_c}.json").read_text())
        assert list(mon.keys()) == FIXTURE["artifacts"][
            "model_monitor"]["keys"]

        mm = json.loads(
            (sink / f"nba_run_engine_monitor_{date_c}.json").read_text())
        assert list(mm.keys()) == FIXTURE["artifacts"][
            "markets_monitor"]["keys"]

        fj = json.loads((sink / f"nba_feature_v1_{date_c}.json").read_text())
        assert list(fj.keys()) == FIXTURE["artifacts"]["feature_json"][
            "keys"]
        assert fj["feature_set_version"] == "nba-prod-v1"

        ph = pd.read_csv(sink / f"nba_predictions_history_{date_c}.csv")
        assert list(ph.columns) == FIXTURE["artifacts"][
            "predictions_history"]["columns"]

        pr = pd.read_csv(sink / f"nba_power_rankings_{date_c}.csv")
        assert list(pr.columns) == FIXTURE["artifacts"]["power_rankings"][
            "columns"]

        mk = pd.read_csv(sink / f"nba_run_engine_markets_{date_c}.csv")
        fx_cols = FIXTURE["artifacts"]["markets"]["columns"]
        assert list(mk.columns) == fx_cols

        meta = json.loads(
            (sink / f"nba_run_engine_markets_{date_c}.meta.json")
            .read_text())
        assert list(meta.keys()) == FIXTURE["artifacts"]["markets_meta"][
            "keys"]

        pm = json.loads(
            (sink / f"nba_player_matchup_{date_c}.json").read_text())
        assert list(pm.keys()) == FIXTURE["artifacts"][
            "player_matchup"]["keys"]

        # SHAP count rule: exactly one file per undecided slate game
        shap_files = list(sink.glob(f"nba_shap_game_{date_c}_*.csv"))
        assert len(shap_files) == result.n_prediction_games
        for f in shap_files:
            df = pd.read_csv(f)
            assert list(df.columns) == FIXTURE["artifacts"]["shap_game"][
                "columns"]

        # OOF store is retention-exempt (models/ dir untouched by policy)
        assert (sink / "models" /
                f"nba_oof_moneyline_{date_c}.csv").exists()

    def test_determinism(self, tmp_path):
        sched = make_schedule(2015, 2)
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        window_start = str(undecided_dates.min().date())
        window_end = str(undecided_dates.max().date())
        outs = []
        for i in range(2):
            cache = tmp_path / f"cache{i}"
            sink = tmp_path / f"sink{i}"
            _prime_cache(cache, sched)
            run_nba_production(
                env=_env(window_start, window_end, repull="1"),
                data_delivery_dir=sink, cache_dir=cache,
                dry_run_ingestion=True)
            date_c = window_end.replace("-", "")
            oof = pd.read_csv(
                sink / "models" / f"nba_oof_moneyline_{date_c}.csv")
            outs.append(oof["p_ensemble"].to_numpy(dtype=float))
        np.testing.assert_allclose(outs[0], outs[1], equal_nan=True)

    def test_retention_only_after_success(self, tmp_path, monkeypatch):
        sched = make_schedule(2015, 2)
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        calls = {"n": 0}
        import core.retention as retention_mod
        orig = retention_mod.run_production_retention

        def spy(*a, **kw):
            calls["n"] += 1
            assert kw.get("generation_succeeded") is True
            return orig(*a, **kw)

        monkeypatch.setattr(
            "sports.nba.runner.run_production_retention", spy)
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        run_nba_production(
            env=_env(str(undecided_dates.min().date()),
                     str(undecided_dates.max().date()), repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        assert calls["n"] == 1

    def test_retention_failure_when_generation_fails(self, tmp_path,
                                                     monkeypatch):
        sched = make_schedule(2015, 2)
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        calls = {"n": 0}

        def spy(*a, **kw):
            calls["n"] += 1
            raise AssertionError("retention must not run on failure")

        monkeypatch.setattr(
            "sports.nba.runner.run_production_retention", spy)
        # break the writers via an invalid slate that trips the fixture gate
        monkeypatch.setattr(
            "sports.nba.artifacts.write_moneyline_json",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        with pytest.raises(RuntimeError, match="boom"):
            run_nba_production(
                env=_env(str(undecided_dates.min().date()),
                         str(undecided_dates.max().date()), repull="1"),
                data_delivery_dir=sink, cache_dir=cache,
                dry_run_ingestion=True)
        assert calls["n"] == 0


class TestFiveMemberCertified:
    def test_all_five_members_execute_on_real_geometry(self, tmp_path):
        sched = make_schedule(2015, 2)
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        result = run_nba_production(
            env=_env(str(undecided_dates.min().date()),
                     str(undecided_dates.max().date()), repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        date_c = str(undecided_dates.max().date()).replace("-", "")
        mon = json.loads(
            (sink / f"nba_model_monitor_{date_c}.json").read_text())
        ensemble = {r["name"]: r for r in mon["ensemble"]}
        assert set(ensemble.keys()) == {
            "xgboost", "lightgbm", "logistic", "randomforest", "mlp"}
        # every member produced OOF predictions (auc present)
        for name, row in ensemble.items():
            assert row["auc"] is not None, f"{name} did not execute"


# ---------------------------------------------------------------------------
# B-005: pre-write record validation — zero-byte-on-failure proofs
# ---------------------------------------------------------------------------

def test_write_predictions_history_rejects_out_of_range_probability(
        tmp_path):
    """p columns outside [0, 1] raise before serialization; no CSV."""
    from core.record_validation import RecordValidationError
    from sports.nba.artifacts import write_predictions_history_csv
    oof = pd.DataFrame({
        "game_id": ["g1", "g2"], "game_date": ["2027-01-07"] * 2,
        "season": [2026] * 2, "home_team": ["A"] * 2, "away_team": ["B"] * 2,
        "fold_id": [1, 2], "home_win": [1.0, 0.0],
        "p_ensemble": [1.2, 0.3],  # 1.2 is out of range
        "home_score": [103.0, 99.0], "away_score": [99.0, 101.0],
    })
    with pytest.raises(RecordValidationError, match="outside"):
        write_predictions_history_csv(tmp_path / "ph.csv", oof, None)
    assert not (tmp_path / "ph.csv").exists()


def test_persist_shap_game_rejects_missing_columns(tmp_path):
    """A shap frame missing a required column raises pre-write."""
    from core.record_validation import RecordValidationError
    from sports.nba.artifacts import persist_shap_game
    frame = pd.DataFrame({"feature": ["f1"], "shap_value": [0.1]})
    with pytest.raises((RecordValidationError, Exception), match="missing columns"):
        persist_shap_game(frame, "20270107", "LAL@BOS", out_dir=tmp_path)
    assert not list(tmp_path.glob("nba_shap_game_*.csv"))
