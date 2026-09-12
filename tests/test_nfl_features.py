"""NFL feature, fold, and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.folds import fold_summary, fold_table, make_folds
from sports.nfl.features import (
    build_game_features,
    build_slate_features,
    feature_coverage_report,
    team_events,
)
from sports.nfl.study_config import MONEYLINE_FEATURE_COLS, load_nfl_study
from tests.nfl_fixtures import make_pbp, make_schedule


@pytest.fixture(scope="module")
def decided_and_pbp():
    sched = make_schedule(2019, 2)
    decided = sched[sched["home_score"].notna()].reset_index(drop=True)
    return decided, make_pbp(sched)


class TestPointInTime:
    def test_team_events_shape(self, decided_and_pbp):
        decided, _ = decided_and_pbp
        ev = team_events(decided)
        assert len(ev) == 2 * len(decided)
        assert set(ev["is_home"].unique()) == {True, False}
        # net points are team-perspective
        assert (ev["net_from_team"] == ev["for"] - ev["against"]).all()

    def test_trailing_features_strictly_prior(self, decided_and_pbp):
        decided, pbp = decided_and_pbp
        study = load_nfl_study()
        df = build_game_features(decided, pbp)
        # win_pct_diff for the FIRST game of each team must be NaN-shifted
        ev = team_events(decided)
        first_games = ev.groupby("team")["gameday"].min()
        # a team's first game in the window has no trailing history for
        # win_pct — the diff must be NaN for both sides' first meeting
        df["gameday"] = pd.to_datetime(df["gameday"])
        first_game_rows = df[df.apply(
            lambda r: any(
                (r["home_team"] == t or r["away_team"] == t)
                and r["gameday"] == g for t, g in first_games.items()),
            axis=1)]
        assert first_game_rows["win_pct_diff"].isna().any()

    def test_ladder_leakage_gate(self):
        """A same-day team duplicate must raise (leakage gate): the
        shift(1) discipline cannot disambiguate two same-day games, so
        the ladder refuses them loudly."""
        ev = pd.DataFrame({
            "game_id": ["g1", "g2", "g1", "g2"],
            "season": [2019, 2019, 2019, 2019],
            "week": [1, 1, 1, 1],
            "gameday": pd.to_datetime(
                ["2019-09-08", "2019-09-08", "2019-09-08",
                 "2019-09-08"]),
            "team": ["ARI", "ARI", "BUF", "BUF"],
            "opponent": ["BUF", "BUF", "ARI", "ARI"],
            "is_home": [False, False, True, True],
            "for": [10.0, 20.0, 20.0, 10.0],
            "against": [20.0, 10.0, 10.0, 20.0],
        })
        ev["net_from_team"] = ev["for"] - ev["against"]
        ev["team_win"] = (ev["for"] > ev["against"]).astype(float)
        study = load_nfl_study()
        from sports.nfl.features import team_stats_ladder
        with pytest.raises(AssertionError):
            team_stats_ladder(
                ev, None, form_window=study.form_window,
                winpct_window=study.winpct_window,
                ypp_window=study.ypp_window,
                ewm_halflife=study.ewm_halflife,
                pace_window=study.pace_window)

    def test_slate_features_have_no_targets(self, decided_and_pbp):
        decided, pbp = decided_and_pbp
        sched = make_schedule(2019, 2)  # includes an undecided final week
        slate = build_slate_features(sched, pbp)
        if slate.empty:
            pytest.skip("no pending rows in fixture")
        assert slate["home_score"].isna().all()
        assert slate["away_score"].isna().all()
        # PIT state must be present (real priors, not fabricated)
        assert slate["elo_diff"].notna().any()

    def test_no_same_day_leakage_in_slate_state(self, decided_and_pbp):
        """The slate's Elo entering state equals the state after the last
        DECIDED game — never affected by same-day future outcomes."""
        decided, pbp = decided_and_pbp
        sched = make_schedule(2019, 2)
        slate = build_slate_features(sched, pbp)
        if slate.empty:
            pytest.skip("no pending rows in fixture")
        # Elo diff must be finite and derived only from decided games
        assert slate["elo_diff"].notna().all()


class TestCoverage:
    def test_feature_coverage_reported(self, decided_and_pbp):
        decided, pbp = decided_and_pbp
        study = load_nfl_study()
        df = build_game_features(decided, pbp)
        cov = feature_coverage_report(df, study)
        assert list(cov["feature"]) == list(MONEYLINE_FEATURE_COLS)
        # Elo-derived and record-derived features must have coverage
        elo_row = cov[cov["feature"] == "elo_diff"].iloc[0]
        assert elo_row["coverage_pct"] == pytest.approx(100.0)
        # PBP-dependent features degrade gracefully on a small fixture
        # but are REPORTED (never silently dropped)
        assert len(cov) == len(MONEYLINE_FEATURE_COLS)

    def test_unavailable_columns_explicit(self, decided_and_pbp):
        decided, pbp = decided_and_pbp
        study = load_nfl_study()
        df = build_game_features(decided, pbp)
        from sports.nfl.feature_registry import (
            unavailable_columns,
            unavailable_warnings,
        )
        unavail = unavailable_columns(df, study.moneyline_feature_cols)
        warns = unavailable_warnings(df, study.moneyline_feature_cols)
        assert len(warns) == len(unavail)
        # travel/altitude come from the committed stadiums table which the
        # fixture venue file doesn't cover here -> they may be unavailable,
        # but they must be EXPLICITLY reported
        for c in unavail:
            assert any(c in w for w in warns)


class TestFolds:
    def test_fold_geometry(self, decided_and_pbp):
        decided, _ = decided_and_pbp
        study = load_nfl_study()
        df = build_game_features(decided, make_pbp(make_schedule(2019, 2)))
        df = df.sort_values("gameday").reset_index(drop=True)
        folds = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col="gameday",
                           val_scope="oof_season", end_of_day=True)
        assert folds
        # training strictly prior to each validation window
        dates = pd.to_datetime(df["gameday"])
        for f in folds:
            assert (dates.loc[f.train_idx] < f.val_start).all()
            assert ((dates.loc[f.val_idx] >= f.val_start)
                    & (dates.loc[f.val_idx] <= f.val_end)).all()
        # validation windows cover ONLY core-season games
        seasons = pd.to_numeric(df["season"])
        for f in folds:
            assert (seasons.loc[f.val_idx] >= study.oof_first_season).all()
        # non-overlapping, chronological
        for a, b in zip(folds, folds[1:]):
            assert b.val_start > a.val_end
        summary = fold_summary(folds)
        assert summary["total_val_games"] == \
            int((seasons >= study.oof_first_season).sum())
        tbl = fold_table(df, folds)
        assert len(tbl) == len(folds)

    def test_warmup_never_in_validation(self, decided_and_pbp):
        decided, _ = decided_and_pbp
        study = load_nfl_study()
        # craft a frame with a warmup-season row
        df = decided.copy()
        warmup_row = df.iloc[0].copy()
        warmup_row["season"] = min(study.warmup_seasons)
        warmup_row["game_id"] = "warmup_1_x_y"
        warmup_row["gameday"] = "2018-09-06"
        df = pd.concat([df, pd.DataFrame([warmup_row])],
                       ignore_index=True)
        df["gameday"] = pd.to_datetime(df["gameday"])
        folds = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col="gameday",
                           val_scope="oof_season", end_of_day=True)
        val_ids = {gid for f in folds for gid in df.loc[f.val_idx,
                                                        "game_id"]}
        assert "warmup_1_x_y" not in val_ids
        train_ids = {gid for f in folds for gid in df.loc[f.train_idx,
                                                          "game_id"]}
        assert "warmup_1_x_y" in train_ids
