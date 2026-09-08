"""NFL ingestion tests: caching, incremental vs full rebuild, equivalence,
and loud failure behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from sports.nfl.ingestion import (
    NFLIngestionError,
    eligible_games,
    load_schedule,
    load_team_names,
)
from tests.nfl_fixtures import make_schedule, make_player_stats


def _fake_loader(frames_by_season: dict[int, pd.DataFrame],
                 fail_seasons: set[int] | None = None):
    fail = fail_seasons or set()

    def load(seasons):
        s = seasons[0] if isinstance(seasons, list) else seasons
        if s in fail:
            raise RuntimeError(f"network down for {s}")
        return frames_by_season[s]

    return load


class TestScheduleIngestion:
    def test_cache_first_and_incremental(self, tmp_path):
        sched_2019 = make_schedule(2019, 1)
        sched_2020 = make_schedule(2020, 1)
        frames = {2019: sched_2019, 2020: sched_2020}
        load = _fake_loader(frames)
        # first pull writes the cache
        df1 = load_schedule([2019, 2020], tmp_path, load=load)
        assert len(df1) == len(sched_2019) + len(sched_2020)
        # second pull (incremental): past seasons come from cache; only the
        # latest is re-pulled
        calls = {"n": 0}

        def counting(seasons):
            calls["n"] += 1
            return frames[seasons[0]]

        df2 = load_schedule([2019, 2020], tmp_path, load=counting)
        assert calls["n"] == 1  # only the latest season re-pulled
        assert len(df2) == len(df1)

    def test_full_repull_discards_cache(self, tmp_path):
        frames = {2019: make_schedule(2019, 1), 2020: make_schedule(2020, 1)}
        load = _fake_loader(frames)
        load_schedule([2019, 2020], tmp_path, load=load)
        calls = {"n": 0}

        def counting(seasons):
            calls["n"] += 1
            return frames[seasons[0]]

        load_schedule([2019, 2020], tmp_path, full_repull=True,
                      load=counting)
        assert calls["n"] == 2  # every season re-pulled

    def test_past_season_failure_is_loud(self, tmp_path):
        frames = {2019: make_schedule(2019, 1), 2020: make_schedule(2020, 1)}
        load = _fake_loader(frames, fail_seasons={2019})
        with pytest.raises(NFLIngestionError, match="2019"):
            load_schedule([2019, 2020], tmp_path, load=load)

    def test_latest_season_failure_degrades_to_cache(self, tmp_path):
        frames = {2019: make_schedule(2019, 1), 2020: make_schedule(2020, 1)}
        load = _fake_loader(frames)
        load_schedule([2019, 2020], tmp_path, load=load)
        failing_latest = _fake_loader(frames, fail_seasons={2020})
        df = load_schedule([2019, 2020], tmp_path, load=failing_latest)
        assert len(df) == len(frames[2019]) + len(frames[2020])

    def test_dedupe_on_game_id(self, tmp_path):
        sched = make_schedule(2019, 1)
        frames = {2019: sched, 2020: sched.assign(season=2020)}
        load = _fake_loader(frames)
        df = load_schedule([2019, 2020], tmp_path, load=load)
        assert df["game_id"].is_unique


class TestEligibility:
    def test_regular_season_filter_and_window(self):
        sched = make_schedule(2019, 2)
        sched = pd.concat([
            sched,
            pd.DataFrame([{
                "game_id": "2019_99_x_y", "season": 2019, "week": 99,
                "game_type": "SB", "gameday": "2020-02-02",
                "gametime": "18:30", "home_team": "ARI",
                "away_team": "BUF", "home_score": 10, "away_score": 20,
            }]),
        ], ignore_index=True)
        out = eligible_games(sched, oof_first_season=2019,
                             warmup_first_season=2018)
        assert (out["game_type"] == "REG").all()
        assert out["game_id"].is_unique

    def test_undecided_rows_kept(self):
        sched = make_schedule(2019, 2)  # final week undecided
        out = eligible_games(sched, oof_first_season=2019,
                             warmup_first_season=2018)
        assert out["home_score"].isna().any()


class TestPlayerStatsAndTeams:
    def test_player_stats_cache_and_qb_filter(self, tmp_path):
        stats = make_player_stats(2020)
        # inject a non-QB row that must be filtered
        stats = pd.concat([stats, pd.DataFrame([{
            "player_id": "x", "player_name": "RB Guy", "position": "RB",
            "team": "ARI", "season": 2020, "week": 1,
            "season_type": "REG", "attempts": 0, "completions": 0,
            "passing_yards": 0, "passing_tds": 0,
            "passing_interceptions": 0,
        }])], ignore_index=True)
        load = _fake_loader({2020: stats})
        out = load_player_stats_cache([2020], tmp_path, load=load)
        assert set(out[2020]["position"].unique()) == {"QB"}

    def test_team_names(self, tmp_path):
        from sports.nfl.ingestion import load_team_names as ltn
        teams = pd.DataFrame({
            "team_abbr": ["ARI", "BUF"],
            "team_name": ["Arizona Cardinals", "Buffalo Bills"],
        })
        out = ltn(tmp_path, load=lambda: teams)
        assert out["ARI"] == "Arizona Cardinals"


def load_player_stats_cache(seasons, cache_dir, *, load):
    from sports.nfl.ingestion import load_player_stats
    return load_player_stats(seasons, cache_dir, load=load)
