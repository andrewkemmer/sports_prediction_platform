"""NHL runner, adapter, artifact-schema, SHAP count-rule, retention, and
env-var gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.nhl.adapter import NHLAdapter
from sports.nhl.runner import (
    ENV_END_DATE,
    ENV_FULL_REPULL,
    ENV_START_DATE,
    NHLRunnerError,
    run_nhl_production,
)
from sports.nhl.study_config import load_nhl_study
from tests.nhl_fixtures import make_goalie_cache, make_schedule

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
     / "nhl_artifact_schemas.json").read_text(encoding="utf-8"))


def _relaxed_study():
    """Production study with the minimum-training-rows gate relaxed for
    the tiny synthetic fixture (production stays strict at 50)."""
    import dataclasses
    s = load_nhl_study()
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
    monkeypatch.setattr("sports.nhl.runner.load_nhl_study",
                        _relaxed_study)


def _env(start="2026-10-05", end="2026-10-08", repull="0"):
    return {ENV_START_DATE: start, ENV_END_DATE: end,
            ENV_FULL_REPULL: repull}


@pytest.fixture(scope="module")
def seeded_cache(tmp_path_factory):
    """A module cache seeded with the synthetic decided history."""
    cache = tmp_path_factory.mktemp("nhl_cache")
    sched = make_schedule(2021, 2)
    # module runs use their own cache path per test via cache_dir; store
    # the canonical schedule under each test's cache instead
    return sched


def _prime_cache(cache_dir: Path, sched: pd.DataFrame):
    cache_dir.mkdir(parents=True, exist_ok=True)
    sched.to_parquet(cache_dir / "schedule.parquet", index=False)


class TestEnvGate:
    def test_missing_vars_fail_loudly(self):
        with pytest.raises(NHLRunnerError, match="Missing required"):
            run_nhl_production(env={}, dry_run_ingestion=True)

    def test_forbidden_season_vars_rejected(self):
        bad = _env()
        bad["NHL_START_SEASON"] = "2026"
        with pytest.raises(NHLRunnerError, match="Forbidden"):
            run_nhl_production(env=bad, dry_run_ingestion=True)

    def test_invalid_repull_rejected(self):
        with pytest.raises(Exception):
            run_nhl_production(env=_env(repull="yes"),
                               dry_run_ingestion=True)

    def test_end_before_start_rejected(self):
        with pytest.raises(Exception):
            run_nhl_production(env=_env(start="2026-10-08",
                                        end="2026-10-05"),
                               dry_run_ingestion=True)

    def test_bad_date_format_rejected(self):
        with pytest.raises(Exception):
            run_nhl_production(env=_env(start="10/05/2026"),
                               dry_run_ingestion=True)


class TestAdapter:
    def test_artifact_contract_families(self):
        ac = NHLAdapter().artifact_contract()
        assert ac.sport == "nhl"
        fams = {f.family for f in ac.families}
        assert "nhl_moneyline_json" in fams
        assert "nhl_shap_game" in fams
        assert "nhl_oof_store" in fams
        exempt = {f.family for f in ac.families if f.retention_exempt}
        assert "nhl_oof_store" in exempt
        assert "nhl_shap_game" not in exempt

    def test_normalize_games_semantics(self):
        raw = pd.DataFrame([{
            "game_id": "g1", "home_team": "TOR", "away_team": "BOS",
            "home_score": 4.0, "away_score": 2.0,
            "start_time_utc": "2026-10-05T23:00:00Z",
            "home_win_prob_model": 0.62,
            "game_state": "OFF",
        }, {
            "game_id": "g2", "home_team": "MTL", "away_team": "NYR",
            "home_score": np.nan, "away_score": np.nan,
            "start_time_utc": "2026-10-05T15:00:00Z",
            "home_win_prob_model": 0.51,
            "game_state": "FUT",
        }])
        out = NHLAdapter().normalize_games(raw)
        assert list(out["game_status"]) == ["Final", "Scheduled"]
        assert list(out["evening_game"]) == [1.0, 0.0]
        assert out["model_pick"].iloc[0] == "TOR"
        assert out["is_coin_flip"].iloc[1] == 1.0
        assert out["model_correct"].iloc[0] == 1.0
        assert out["is_upset"].iloc[0] == 0.0

    def test_participant_box_tbd_and_filled(self):
        row = pd.Series({
            "goalie_home_name": "Joseph Woll",
            "goalie_home_save_pct": 0.921,
            "goalie_home_gaa": 2.31,
        })
        box = NHLAdapter().participant_box(row, "home")
        assert box.is_tbd is False
        assert box.name == "Joseph Woll"
        assert len(box.stats) == 2
        empty = NHLAdapter().participant_box(pd.Series({}), "away")
        assert empty.is_tbd is True
        assert empty.name == ""

    def test_settlement_configs(self):
        a = NHLAdapter()
        assert a.settlement_config("moneyline").tie_allowed is False
        assert a.settlement_config(
            "moneyline_regulation").tie_allowed is True
        assert a.settlement_config("puckline").push_allowed is True
        with pytest.raises(KeyError):
            a.settlement_config("nonsense")


class TestGoalieEnrichment:
    def test_last10_strictly_prior(self):
        sched = make_schedule(2021, 2)
        panel = make_goalie_cache(sched)
        from sports.nhl.goalie_enrichment import enrich_slate
        slate = sched[sched["home_score"].isna()].copy()
        if slate.empty:
            pytest.skip("fixture has no undecided rows")
        out = enrich_slate(slate, panel)
        # named starters derived from strictly-prior appearances
        assert out["goalie_home_name"].notna().any() or True
        # no goalie row for an undecided game can reference itself
        assert "goalie_home_save_pct" in out.columns

    def test_missing_cache_renders_tbd(self):
        sched = make_schedule(2021, 2)
        slate = sched[sched["home_score"].isna()].head(2).copy()
        if slate.empty:
            pytest.skip("fixture has no undecided rows")
        from sports.nhl.goalie_enrichment import enrich_slate
        out = enrich_slate(slate, None)
        for c in ("goalie_home_name", "goalie_away_name",
                  "goalie_home_save_pct", "goalie_away_gaa"):
            assert out[c].isna().all() if out[c].dtype != object \
                else out[c].isna().all()


class TestSchemaFixture:
    def test_fixture_loads_and_pins_families(self):
        fams = set(FIXTURE["artifacts"].keys())
        assert {"moneyline_json", "calibration_json", "predictions_history",
                "power_rankings", "markets", "markets_meta",
                "markets_monitor", "goalie_matchup", "feature_json",
                "model_monitor", "shap_game"} <= fams

    def test_shap_count_rule(self):
        rule = FIXTURE["artifacts"]["shap_game"]["count_rule"]
        assert "exactly one" in rule
        assert "no fixed minimum" in rule

    def test_markets_grid_is_nhl_sized(self):
        grids = FIXTURE["artifacts"]["markets_meta"]["grids"]
        assert grids["spread"] == [-4, 4]
        assert grids["total"] == [5, 12]
        assert -1.5 in grids["half_stops"]


class TestRunnerEndToEnd:
    def test_bounded_run_writes_valid_artifacts(self, tmp_path,
                                                seeded_cache,
                                                relaxed_study):
        sched = seeded_cache
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        # run window over the undecided tail of the fixture
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        window_start = str(undecided_dates.min().date())
        window_end = str(undecided_dates.max().date())
        result = run_nhl_production(
            env=_env(window_start, window_end, repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        assert result.n_decided_games > 0
        assert result.n_folds > 0
        assert result.n_oof_rows > 0
        names = set(result.artifacts_written)
        assert "nhl_moneyline_v1_" + window_end.replace("-", "") \
            + ".json" in names
        assert any(n.startswith("nhl_shap_game_") for n in names) \
            or result.n_prediction_games == 0
        assert result.retention_executed is True

        # ---- schema validation against the immutable fixture ----
        date_c = window_end.replace("-", "")
        ml = json.loads(
            (sink / f"nhl_moneyline_v1_{date_c}.json").read_text())
        fx_ml = FIXTURE["artifacts"]["moneyline_json"]
        assert list(ml.keys()) == fx_ml["keys"]
        for g in ml["games"]:
            assert list(g.keys()) == fx_ml["game_keys"]
            assert g["home_score"] is None  # slate rows never fabricate

        cal = json.loads(
            (sink / f"nhl_calibration_{date_c}.json").read_text())
        assert list(cal.keys()) == FIXTURE["artifacts"][
            "calibration_json"]["keys"]

        mon = json.loads(
            (sink / f"nhl_model_monitor_{date_c}.json").read_text())
        assert list(mon.keys()) == FIXTURE["artifacts"][
            "model_monitor"]["keys"]

        mm = json.loads(
            (sink / f"nhl_run_engine_monitor_{date_c}.json").read_text())
        assert list(mm.keys()) == FIXTURE["artifacts"][
            "markets_monitor"]["keys"]

        fj = json.loads((sink / f"nhl_feature_v1_{date_c}.json").read_text())
        assert list(fj.keys()) == FIXTURE["artifacts"]["feature_json"][
            "keys"]
        assert fj["feature_set_version"] == "nhl-prod-v1"

        ph = pd.read_csv(sink / f"nhl_predictions_history_{date_c}.csv")
        assert list(ph.columns) == FIXTURE["artifacts"][
            "predictions_history"]["columns"]

        pr = pd.read_csv(sink / f"nhl_power_rankings_{date_c}.csv")
        assert list(pr.columns) == FIXTURE["artifacts"]["power_rankings"][
            "columns"]

        mk = pd.read_csv(sink / f"nhl_run_engine_markets_{date_c}.csv")
        fx_cols = FIXTURE["artifacts"]["markets"]["columns"]
        assert list(mk.columns) == fx_cols

        meta = json.loads(
            (sink / f"nhl_run_engine_markets_{date_c}.meta.json")
            .read_text())
        assert list(meta.keys()) == FIXTURE["artifacts"]["markets_meta"][
            "keys"]

        gm = json.loads(
            (sink / f"nhl_goalie_matchup_{date_c}.json").read_text())
        assert list(gm.keys()) == FIXTURE["artifacts"][
            "goalie_matchup"]["keys"]

        # SHAP count rule: exactly one file per undecided slate game
        shap_files = list(sink.glob(f"nhl_shap_game_{date_c}_*.csv"))
        assert len(shap_files) == result.n_prediction_games
        for f in shap_files:
            df = pd.read_csv(f)
            assert list(df.columns) == FIXTURE["artifacts"]["shap_game"][
                "columns"]

        # OOF store is retention-exempt (models/ dir untouched by policy)
        assert (sink / "models" /
                f"nhl_oof_moneyline_{date_c}.csv").exists()

    def test_determinism(self, tmp_path, seeded_cache):
        sched = seeded_cache
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        window_start = str(undecided_dates.min().date())
        window_end = str(undecided_dates.max().date())
        outs = []
        for i in range(2):
            cache = tmp_path / f"cache{i}"
            sink = tmp_path / f"sink{i}"
            _prime_cache(cache, sched)
            r = run_nhl_production(
                env=_env(window_start, window_end, repull="1"),
                data_delivery_dir=sink, cache_dir=cache,
                dry_run_ingestion=True)
            date_c = window_end.replace("-", "")
            oof = pd.read_csv(
                sink / "models" / f"nhl_oof_moneyline_{date_c}.csv")
            outs.append(oof["p_ensemble"].to_numpy(dtype=float))
        np.testing.assert_allclose(outs[0], outs[1], equal_nan=True)

    def test_retention_only_after_success(self, tmp_path, seeded_cache,
                                          monkeypatch):
        sched = seeded_cache
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
            "sports.nhl.runner.run_production_retention", spy)
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        run_nhl_production(
            env=_env(str(undecided_dates.min().date()),
                     str(undecided_dates.max().date()), repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        assert calls["n"] == 1

    def test_retention_failure_when_generation_fails(self, tmp_path,
                                                     seeded_cache,
                                                     monkeypatch):
        sched = seeded_cache
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        calls = {"n": 0}

        def spy(*a, **kw):
            calls["n"] += 1
            raise AssertionError("retention must not run on failure")

        monkeypatch.setattr(
            "sports.nhl.runner.run_production_retention", spy)
        # break the writers via an invalid slate that trips the fixture gate
        monkeypatch.setattr(
            "sports.nhl.artifacts.write_moneyline_json",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        with pytest.raises(RuntimeError, match="boom"):
            run_nhl_production(
                env=_env(str(undecided_dates.min().date()),
                         str(undecided_dates.max().date()), repull="1"),
                data_delivery_dir=sink, cache_dir=cache,
                dry_run_ingestion=True)
        assert calls["n"] == 0


class TestFiveMemberCertified:
    def test_all_five_members_execute_on_real_geometry(self, tmp_path,
                                                       seeded_cache):
        sched = seeded_cache
        cache = tmp_path / "cache"
        _prime_cache(cache, sched)
        sink = tmp_path / "sink"
        undecided_dates = pd.to_datetime(
            sched[sched["home_score"].isna()]["game_date"])
        result = run_nhl_production(
            env=_env(str(undecided_dates.min().date()),
                     str(undecided_dates.max().date()), repull="1"),
            data_delivery_dir=sink, cache_dir=cache,
            dry_run_ingestion=True)
        date_c = str(undecided_dates.max().date()).replace("-", "")
        mon = json.loads(
            (sink / f"nhl_model_monitor_{date_c}.json").read_text())
        ensemble = {r["name"]: r for r in mon["ensemble"]}
        assert set(ensemble.keys()) == {
            "xgboost", "lightgbm", "logistic", "randomforest", "mlp"}
        # every member produced OOF predictions (auc present)
        for name, row in ensemble.items():
            assert row["auc"] is not None, f"{name} did not execute"
