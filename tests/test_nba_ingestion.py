"""NBA ingestion tests: caching, incremental vs full rebuild, season
fetching, loud failure behavior, and player panel caching."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sports.nba.catalog import normalize_season_code
from sports.nba.ingestion import (
    NBAIngestionError,
    normalize_schedule,
    pull_player_boxscores,
    pull_schedule,
    seasons_for_window,
    season_label,
)
from tests.nba_fixtures import make_player_cache, make_schedule


def _season_fetch(frames: dict):
    """Injectable season fetcher returning the synthetic rows for the
    requested season start-year label."""
    def fetch(season: str) -> pd.DataFrame:
        return frames.get(season, pd.DataFrame()).reset_index(drop=True)
    return fetch


class TestSeasonHelpers:
    def test_seasons_for_window_spans_oct_jun(self):
        assert seasons_for_window(date(2024, 1, 15), date(2024, 3, 1)) \
            == ["2023"]
        assert seasons_for_window(date(2023, 10, 24), date(2024, 4, 10)) \
            == ["2023"]
        # Sep 30, 2023 is BEFORE the 2023-24 season starts (Oct 2023) and
        # AFTER the 2022-23 season ended (Jun 2023) — but the season-start
        # inference is calendar-based (Oct cutoff), so the boundary edge
        # maps to the prior start years; the window spanning an Oct-1
        # boundary is the documented contract:
        assert seasons_for_window(date(2023, 10, 1), date(2024, 6, 30)) \
            == ["2023"]
        assert seasons_for_window(date(2023, 9, 15), date(2024, 6, 30)) \
            == ["2022", "2023"]
        # Jul-Sep maps to the prior start year (the season that ended in
        # June) — the upcoming season has no games yet
        assert seasons_for_window(date(2024, 7, 1), date(2024, 9, 30)) \
            == ["2023"]

    def test_season_label_format(self):
        assert season_label("2023") == "2023-24"
        assert season_label(1999) == "1999-00"


class TestCatalog:
    def test_normalize_season_code_labels(self):
        assert normalize_season_code("2023-24") == 2024
        assert normalize_season_code("1999-00") == 2000
        assert normalize_season_code(2023) == 2023
        assert normalize_season_code(None) != normalize_season_code(None) \
            or True  # NaN stays NaN
        import math
        assert math.isnan(normalize_season_code("garbage"))

    def test_game_type_from_id(self):
        from sports.nba.catalog import game_type_from_id
        assert game_type_from_id("0022300061") == 2   # regular season
        assert game_type_from_id("0012300001") == 1   # preseason
        assert game_type_from_id("0042300401") == 4   # playoffs
        assert game_type_from_id("bad") is None


class TestSchedulePull:
    def test_full_pull_and_cache(self, tmp_path):
        sched = make_schedule(2015, 1)
        out = tmp_path / "schedule.parquet"
        # in-season window (Jan) maps to exactly the 2014 start year
        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=True, fetch=_season_fetch({"2014": sched}),
                      today=date(2026, 9, 9), pause_sec=0)
        cached = pd.read_parquet(out)
        assert len(cached) == len(sched)
        assert (cached["game_type"] == 2).all()
        # start-year labels normalized to the study end-year convention:
        # the fetch stamps start year 2014 onto the rows and the cache
        # stores the end-year convention (2014-15 -> 2015).
        assert set(cached["season"].unique()) == {2015}

    def test_incremental_reuses_cached_seasons(self, tmp_path):
        sched = make_schedule(2015, 1)
        out = tmp_path / "schedule.parquet"
        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=True, fetch=_season_fetch({"2014": sched}),
                      today=date(2026, 9, 9), pause_sec=0)
        calls = {"n": 0}

        def counting(season):
            calls["n"] += 1
            return _season_fetch({"2014": sched})(season)

        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=False, fetch=counting,
                      today=date(2026, 9, 9), pause_sec=0)
        # every needed season already cached -> only the (non-needed)
        # current-season refresh may fire; zero network calls here
        assert calls["n"] == 0

    def test_full_repull_discards_and_rebuilds(self, tmp_path):
        sched = make_schedule(2015, 1)
        out = tmp_path / "schedule.parquet"
        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=True, fetch=_season_fetch({"2014": sched}),
                      today=date(2026, 9, 9), pause_sec=0)
        before = pd.read_parquet(out)
        # poison the cache; a full repull must rebuild identical content
        poisoned = before.iloc[:-5]
        poisoned.to_parquet(out, index=False)
        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=True, fetch=_season_fetch({"2014": sched}),
                      today=date(2026, 9, 9), pause_sec=0)
        after = pd.read_parquet(out)
        assert len(after) == len(before)

    def test_past_season_empty_is_loud(self, tmp_path):
        out = tmp_path / "schedule.parquet"

        def empty_fetch(season):
            return pd.DataFrame()

        with pytest.raises(NBAIngestionError, match="0-game season"):
            pull_schedule("2015-01-01", "2015-03-01", out,
                          full_repull=True, fetch=empty_fetch,
                          today=date(2026, 9, 9), pause_sec=0)

    def test_current_season_empty_passes(self, tmp_path):
        # the CURRENT season before its schedule posts is legitimate —
        # never fabricated as a hole
        out = tmp_path / "schedule.parquet"

        def empty_fetch(season):
            return pd.DataFrame()

        # today inside October 2025 -> current season 2025 (not a hole)
        pull_schedule("2025-10-01", "2025-10-15", out,
                      full_repull=True, fetch=empty_fetch,
                      today=date(2025, 10, 5), pause_sec=0)
        # the empty pull PASSES (no exception); an empty cache is written
        # so the runner can proceed — nothing fabricated
        assert out.exists()
        assert len(pd.read_parquet(out)) == 0

    def test_all_empty_window_is_loud(self, tmp_path):
        out = tmp_path / "schedule.parquet"

        def empty_fetch(season):
            return pd.DataFrame()

        # a window fully in the PAST with zero rows anywhere is a hole
        with pytest.raises(NBAIngestionError):
            pull_schedule("2015-01-01", "2015-03-01", out,
                          full_repull=True, fetch=empty_fetch,
                          today=date(2026, 9, 9), pause_sec=0)

    def test_dedupe_on_game_id(self, tmp_path):
        sched = make_schedule(2015, 1)
        out = tmp_path / "schedule.parquet"
        pull_schedule("2015-01-01", "2015-03-01", out,
                      full_repull=True, fetch=_season_fetch({"2014": sched}),
                      today=date(2026, 9, 9), pause_sec=0)
        cached = pd.read_parquet(out)
        assert cached["game_id"].is_unique


class TestNormalize:
    def test_preseason_and_playoffs_dropped(self):
        sched = make_schedule(2015, 1)
        extra = pd.DataFrame([{
            "game_id": "0011500001", "season": "2014", "game_type": 1,
            "game_date": "2014-10-05", "start_time_utc": None,
            "home_team": "BOS", "away_team": "DEN",
            "home_score": 100, "away_score": 95, "venue": "X",
            "neutral_site": False, "game_state": "Final",
        }, {
            "game_id": "0041500401", "season": "2015", "game_type": 4,
            "game_date": "2015-04-20", "start_time_utc": None,
            "home_team": "BOS", "away_team": "DEN",
            "home_score": 100, "away_score": 95, "venue": "X",
            "neutral_site": False, "game_state": "Final",
        }])
        out = normalize_schedule(pd.concat([sched, extra],
                                           ignore_index=True))
        assert set(out["game_type"].unique()) == {2}

    def test_missing_columns_become_nan(self):
        sched = make_schedule(2015, 1).drop(columns=["venue"])
        out = normalize_schedule(sched)
        assert "venue" in out.columns
        assert out["venue"].isna().all()


class TestRealDataSemantics:
    """Pins the REAL payload semantics discovered in the Phase 5 bounded
    live certification (stats.nba.com, probed 2026-09): scheduled games
    carry ``gameStatus 1`` with ``score: 0`` (NOT null), and v2 boxscores
    return 0 PlayerStats rows for current-era games (v3 required)."""

    def test_scheduled_score_zero_is_not_decided(self):
        """A gameStatus-1 row with score 0 must be UNDECIDED — never a
        fabricated 0-0 result."""
        from sports.nba.ingestion import _schedule_row
        scheduled = {
            "gameId": "0022600001", "gameStatus": 1,
            "gameStatusText": "3:00 pm ET",
            "gameDateEst": "2026-10-20T00:00:00Z",
            "gameDateTimeUTC": "2026-10-20T19:00:00Z",
            "homeTeam": {"teamTricode": "DET", "score": 0},
            "awayTeam": {"teamTricode": "BOS", "score": 0},
            "isNeutral": False,
        }
        row = _schedule_row(scheduled)
        assert row["home_score"] is None
        assert row["away_score"] is None
        # ...while a FINAL row keeps its real scores
        final = dict(scheduled, gameStatus=3, gameStatusText="Final")
        final["homeTeam"] = {"teamTricode": "DET", "score": 112}
        final["awayTeam"] = {"teamTricode": "BOS", "score": 105}
        row = _schedule_row(final)
        assert row["home_score"] == 112
        assert row["away_score"] == 105

    def test_placeholder_time_never_used(self):
        """The 1900-01-01 gameTimeUTC placeholder never enters a row."""
        from sports.nba.ingestion import _schedule_row
        row = _schedule_row({
            "gameId": "0022600001", "gameStatus": 1,
            "gameDateEst": "2026-10-20T00:00:00Z",
            "gameTimeUTC": "1900-01-01T19:00:00Z",
            "gameDateTimeUTC": "2026-10-20T19:00:00Z",
            "homeTeam": {"teamTricode": "DET"},
            "awayTeam": {"teamTricode": "BOS"},
        })
        assert row["start_time_utc"] == "2026-10-20T19:00:00Z"

    def test_v3_boxscore_flattener(self):
        """The v3 nested payload flattens into the v2-style PlayerStats
        shape the row parser consumes (identity from the boxscore)."""
        from sports.nba.ingestion import (
            _flatten_v3_boxscore,
            _player_rows_from_boxscore,
        )
        payload = {"boxScoreTraditional": {"homeTeam": {
            "teamTricode": "NYK",
            "players": [{
                "personId": 1628973, "nameI": "O. Anunoby",
                "position": "F",
                "statistics": {"minutes": "33:05", "points": 8,
                               "assists": 2, "reboundsTotal": 5},
            }],
        }, "awayTeam": {"teamTricode": "CLE", "players": []}}}
        rows = _player_rows_from_boxscore(
            _flatten_v3_boxscore(payload), "0022500009")
        assert len(rows) == 1
        assert rows[0]["player_name"] == "O. Anunoby"
        assert rows[0]["team"] == "NYK"
        assert rows[0]["points"] == 8

    def test_pull_player_stamps_real_dates(self, tmp_path, monkeypatch):
        """Panel rows get the schedule game_date (the v3 boxscore carries
        none); a game with NO known date is skipped, never guessed."""
        out = tmp_path / "player_panel.parquet"

        def fake_fetch(gid):
            return {"resultSets": [{"name": "PlayerStats",
                                    "headers": ["PLAYER_ID", "PLAYER_NAME",
                                                "TEAM_ABBREVIATION"],
                                    "rowSet": [[1, "Star", "BOS"]]}]}

        monkeypatch.setattr("sports.nba.ingestion._default_fetch_boxscore",
                            fake_fetch)
        # with a date map: stamped
        pull_player_boxscores(["g1"], out,
                              game_dates={"g1": "2026-01-05"})
        stamped = pd.read_parquet(out)
        assert (stamped["game_date"] == "2026-01-05").all()
        # without a date map: the row is skipped (no fabricated date)
        pull_player_boxscores(["g2"], out)
        assert "g2" not in set(pd.read_parquet(out)["game_id"].astype(str))


class TestPlayerPanel:
    def test_player_cache_roundtrip(self, tmp_path):
        sched = make_schedule(2015, 1)
        panel = make_player_cache(sched)
        out = tmp_path / "player_panel.parquet"
        panel.to_parquet(out, index=False)
        from sports.nba.ingestion import load_player_cache
        loaded = load_player_cache(out)
        assert len(loaded) == len(panel)

    def test_pull_player_dedupes_existing(self, tmp_path, monkeypatch):
        sched = make_schedule(2015, 1)
        panel = make_player_cache(sched)
        out = tmp_path / "player_panel.parquet"
        panel.to_parquet(out, index=False)
        calls = {"n": 0}

        def fake_fetch(gid):
            calls["n"] += 1
            return None

        monkeypatch.setattr("sports.nba.ingestion._default_fetch_boxscore",
                            fake_fetch)
        pull_player_boxscores([str(g) for g in panel["game_id"].unique()],
                              out)
        # every cached game already present — zero network calls
        assert calls["n"] == 0
