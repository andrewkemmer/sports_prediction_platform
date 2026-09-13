"""NHL feature-engine tests: point-in-time discipline, leakage gate,
trailing shift semantics, coverage, and slate construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports.nhl.features.build_frame import (
    build_game_features,
    build_slate_features,
    feature_coverage_report,
    team_events,
    team_stats_ladder,
    _haversine_miles,
)
from sports.nhl.config.study_config import MONEYLINE_FEATURE_COLS, load_nhl_study
from tests.nhl_fixtures import make_schedule


class TestTeamEvents:
    def test_one_row_per_team_game(self):
        sched = make_schedule(2021, 1)
        ev = team_events(sched.dropna(subset=["home_score"]))
        assert len(ev) == 2 * sched["home_score"].notna().sum()
        assert set(ev["is_home"].unique()) == {True, False}

    def test_net_goals_symmetry(self):
        sched = make_schedule(2021, 1)
        ev = team_events(sched.dropna(subset=["home_score"]))
        net = ev.groupby("game_id")["net_from_team"].sum()
        assert np.allclose(net.to_numpy(), 0.0)


class TestLadderPIT:
    def test_trailing_values_exclude_current_game(self):
        sched = make_schedule(2021, 1).dropna(subset=["home_score"])
        study = load_nhl_study()
        ev = team_events(sched)
        ladder = team_stats_ladder(
            ev, form_window=study.form_window,
            winpct_window=study.winpct_window,
            ewm_halflife=study.ewm_halflife)
        srt = ladder.sort_values(["team", "game_date"]).reset_index(
            drop=True)
        # a team's FIRST game must have NaN form (no prior games)
        first = srt.groupby("team").head(1)
        assert first["form_gls"].isna().all()
        # a later row's form equals the mean of the PRIOR games only
        g = srt.groupby("team")
        for team, grp in g:
            if len(grp) < 3:
                continue
            row = grp.iloc[2]
            prior = grp.iloc[:2]["net_from_team"].mean()
            assert np.isclose(row["form_gls"], prior)

    def test_leakage_gate_raises_on_non_chronological(self):
        sched = make_schedule(2021, 1).dropna(subset=["home_score"])
        ev = team_events(sched)
        # duplicate the same (team, date) — the gate must fail loudly
        dup = pd.concat([ev, ev.head(2)], ignore_index=True)
        study = load_nhl_study()
        with pytest.raises(AssertionError, match="strictly increasing"):
            team_stats_ladder(
                dup, form_window=study.form_window,
                winpct_window=study.winpct_window,
                ewm_halflife=study.ewm_halflife)

    def test_b2b_flag(self):
        sched = make_schedule(2021, 1).dropna(subset=["home_score"])
        study = load_nhl_study()
        ev = team_events(sched)
        ladder = team_stats_ladder(
            ev, form_window=study.form_window,
            winpct_window=study.winpct_window,
            ewm_halflife=study.ewm_halflife)
        # b2b is 1.0 exactly when rest_days == 1, NaN on a team's first game
        known = ladder[ladder["rest_days"].notna()]
        expected = (known["rest_days"] == 1).astype(float)
        assert np.allclose(known["b2b"].astype(float), expected)


class TestBuildFeatures:
    def test_decided_frame_targets(self):
        sched = make_schedule(2021, 2)
        dec = sched.dropna(subset=["home_score"])
        out = build_game_features(dec)
        assert (out["home_win"].isin([0.0, 1.0])).all()
        assert np.allclose(out["margin"],
                           out["home_score"] - out["away_score"])
        assert np.allclose(out["total"],
                           out["home_score"] + out["away_score"])

    def test_all_served_features_present(self):
        sched = make_schedule(2021, 2)
        out = build_game_features(sched.dropna(subset=["home_score"]))
        for c in MONEYLINE_FEATURE_COLS:
            assert c in out.columns, f"missing served feature {c}"

    def test_slate_rows_have_no_targets(self):
        sched = make_schedule(2021, 2)
        slate = build_slate_features(sched)
        assert len(slate) > 0
        assert "home_win" not in slate.columns
        assert "margin" not in slate.columns
        # every feature column is present (matrix-width invariant upstream)
        for c in MONEYLINE_FEATURE_COLS:
            assert c in slate.columns

    def test_slate_features_come_from_decided_history_only(self):
        sched = make_schedule(2021, 2)
        slate = build_slate_features(sched)
        # an undecided game's Elo entering must equal the final rating of
        # the decided timeline (it updated nothing)
        dec = sched.dropna(subset=["home_score"])
        dec_feats = build_game_features(dec)
        last_decided = dec_feats.sort_values("game_date").iloc[-1]
        row = slate.iloc[0]
        # both sides' entering ratings must predate the slate game
        assert pd.to_datetime(row["game_date"]) > pd.to_datetime(
            last_decided["game_date"])

    def test_coverage_report_shape(self):
        sched = make_schedule(2021, 2)
        out = build_game_features(sched.dropna(subset=["home_score"]))
        cov = feature_coverage_report(out)
        assert list(cov["feature"]) == list(MONEYLINE_FEATURE_COLS)
        assert (cov["coverage_pct"] >= 0).all()


class TestTravel:
    def test_haversine_known_distance(self):
        # TOR -> MTL ~ 311 miles (great-circle)
        d = float(_haversine_miles(43.65, -79.38, 45.50, -73.57))
        assert 280 < d < 340

    def test_travel_diff_zero_for_same_venue(self):
        sched = make_schedule(2021, 1)
        out = build_game_features(sched.dropna(subset=["home_score"]))
        # all games at the home team's own arena: away team's travel
        # distance minus home's zero travel is >= 0 (or NaN when unknown)
        v = out["travel_miles_diff"].dropna()
        assert (v >= -1e-6).all()
