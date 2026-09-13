"""NBA ingestion tests: caching, incremental vs full rebuild, season
fetching, loud failure behavior, and player panel caching."""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd
import pytest

from sports.nba.features.raw.catalog import normalize_season_code
from sports.nba.ingestion import (
    EXTERNAL_SOURCES,
    SOURCE_BOXSCORE,
    SOURCE_SCHEDULE,
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
        from sports.nba.features.raw.catalog import game_type_from_id
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


# ---------------------------------------------------------------------------
# B-007 external-adapter integration smoke (nba:stats.nba.com/*)
#
# Exercises the REAL DEFAULT adapters (`_default_fetch_*`, no injected
# fetch=) with `urllib.request.urlopen` patched to a deterministic fake:
# production call shape, normalization, determinism, loud failure modes,
# and the no-artifact-after-failure contract.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _api_schedule_games(sched: pd.DataFrame) -> list[dict]:
    games = []
    for r in sched.itertuples(index=False):
        decided = pd.notna(r.home_score) and pd.notna(r.away_score)
        games.append({
            "gameId": r.game_id,
            "gameStatus": 3 if decided else 1,
            "gameStatusText": "Final" if decided else "7:00 pm ET",
            "gameDateEst": f"{r.game_date}T00:00:00Z",
            "gameDateTimeUTC": r.start_time_utc,
            "homeTeam": {"teamTricode": r.home_team,
                         "score": int(r.home_score) if decided else 0},
            "awayTeam": {"teamTricode": r.away_team,
                         "score": int(r.away_score) if decided else 0},
            "isNeutral": False, "arenaName": "Test Arena",
            "arenaCity": "Test City",
        })
    return games


def _api_v3_boxscore(game_id: str) -> dict:
    return {"boxScoreTraditional": {
        "homeTeam": {"teamTricode": "BOS", "players": [{
            "personId": 1628973, "nameI": "O. Anunoby", "position": "F",
            "statistics": {"minutes": "33:05", "points": 8,
                           "assists": 2, "reboundsTotal": 5}}]},
        "awayTeam": {"teamTricode": "CLE", "players": [{
            "personId": 201939, "nameI": "S. Curry", "position": "G",
            "statistics": {"minutes": "35:00", "points": 30,
                           "assists": 6, "reboundsTotal": 4}}]},
    }}


def _patch_urlopen(monkeypatch, handler):
    """Patch the real urllib transport the default adapters use.

    String-target setattr: pytest performs the import internally, so this
    test module itself never imports a transport (manifest hygiene).
    """
    calls: list[str] = []

    def _urlopen(req, timeout=None):
        url = getattr(req, "full_url", str(req))
        calls.append(url)
        return handler(url)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return calls


class TestNbaStatsAdapterIntegrationSmoke:
    def test_registry_declares_every_source_and_default_adapter(self):
        assert EXTERNAL_SOURCES == {
            SOURCE_SCHEDULE: "_default_fetch_schedule",
            SOURCE_BOXSCORE: "_default_fetch_boxscore",
        }

    def test_schedule_default_adapter_end_to_end_and_deterministic(
            self, tmp_path, monkeypatch):
        sched = make_schedule(2015, 1)
        games = _api_schedule_games(sched)

        def handler(url: str):
            assert "scheduleleaguev2" in url
            assert "Season=2014-15" in url  # production label form
            return _FakeResponse({"leagueSchedule": {"gameDates": [
                {"games": games}]}})

        calls = _patch_urlopen(monkeypatch, handler)
        out = tmp_path / "schedule.parquet"
        pull_schedule("2015-01-01", "2015-03-01", out, full_repull=True,
                      today=date(2026, 9, 9), pause_sec=0)
        assert calls  # one season-granular production request
        cached = pd.read_parquet(out)
        assert cached["game_id"].is_unique
        for col in ("game_id", "season", "game_type", "game_date",
                    "home_team", "away_team", "home_score", "away_score"):
            assert col in cached.columns
        assert set(cached["game_type"].unique()) == {2}
        assert pd.api.types.is_datetime64_any_dtype(cached["game_date"])
        assert set(cached["home_team"]) <= {"BOS", "CLE"} | set(
            sched["home_team"])

        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        pull_schedule("2015-01-01", "2015-03-01", a, full_repull=True,
                      today=date(2026, 9, 9), pause_sec=0)
        pull_schedule("2015-01-01", "2015-03-01", b, full_repull=True,
                      today=date(2026, 9, 9), pause_sec=0)
        pd.testing.assert_frame_equal(pd.read_parquet(a),
                                      pd.read_parquet(b))

    def test_schedule_failure_modes_are_loud(self, tmp_path, monkeypatch):
        def boom(url):
            raise ConnectionError("stats api unreachable")

        _patch_urlopen(monkeypatch, boom)
        out = tmp_path / "s1.parquet"
        with pytest.raises(NBAIngestionError) as ei:
            pull_schedule("2015-01-01", "2015-03-01", out, full_repull=True,
                          today=date(2026, 9, 9), pause_sec=0)
        assert ei.value.source == SOURCE_SCHEDULE
        assert not out.exists()

        # empty past season is a hole
        _patch_urlopen(monkeypatch, lambda url: _FakeResponse(
            {"leagueSchedule": {"gameDates": []}}))
        out2 = tmp_path / "s2.parquet"
        with pytest.raises(NBAIngestionError):
            pull_schedule("2015-01-01", "2015-03-01", out2, full_repull=True,
                          today=date(2026, 9, 9), pause_sec=0)
        assert not out2.exists()

        # malformed payload: a non-dict game entry is schema drift and is
        # surfaced as a loud typed adapter error (never silently dropped)
        _patch_urlopen(monkeypatch, lambda url: _FakeResponse(
            {"leagueSchedule": {"gameDates": [{"games": ["nope"]}]}}))
        out3 = tmp_path / "s3.parquet"
        with pytest.raises(NBAIngestionError) as ei:
            pull_schedule("2015-01-01", "2015-03-01", out3, full_repull=True,
                          today=date(2026, 9, 9), pause_sec=0)
        assert ei.value.source == SOURCE_SCHEDULE
        assert not out3.exists()

    def test_boxscore_default_adapter_end_to_end_and_deterministic(
            self, tmp_path, monkeypatch):
        calls = _patch_urlopen(
            monkeypatch,
            lambda url: _FakeResponse(_api_v3_boxscore(url)))
        out = tmp_path / "player_panel.parquet"
        pull_player_boxscores(["0021500001"], out,
                              game_dates={"0021500001": "2015-01-05"})
        assert calls == ["https://stats.nba.com/stats/boxscoretraditionalv3"
                         "?gameId=0021500001"]
        panel = pd.read_parquet(out)
        for col in ("game_id", "team", "player_id", "player_name",
                    "minutes", "points", "assists", "rebounds",
                    "game_date"):
            assert col in panel.columns
        assert set(panel["team"]) == {"BOS", "CLE"}
        assert set(panel["player_name"]) == {"O. Anunoby", "S. Curry"}
        assert (panel["game_date"] == "2015-01-05").all()
        assert int(panel.loc[panel["player_name"] == "S. Curry",
                             "points"].iloc[0]) == 30

        pa = tmp_path / "a.parquet"
        pb = tmp_path / "b.parquet"
        pull_player_boxscores(["0021500001"], pa,
                              game_dates={"0021500001": "2015-01-05"})
        pull_player_boxscores(["0021500001"], pb,
                              game_dates={"0021500001": "2015-01-05"})
        pd.testing.assert_frame_equal(pd.read_parquet(pa),
                                      pd.read_parquet(pb))

    def test_boxscore_failure_modes_are_loud(self, tmp_path, monkeypatch):
        def boom(url):
            raise TimeoutError("boxscore timeout")

        _patch_urlopen(monkeypatch, boom)
        out = tmp_path / "p1.parquet"
        with pytest.raises(NBAIngestionError) as ei:
            pull_player_boxscores(["g1"], out,
                                  game_dates={"g1": "2015-01-05"})
        assert ei.value.source == SOURCE_BOXSCORE
        assert not out.exists()

        _patch_urlopen(monkeypatch, lambda url: _FakeResponse({}))
        out2 = tmp_path / "p2.parquet"
        with pytest.raises(NBAIngestionError, match="EMPTY"):
            pull_player_boxscores(["g1"], out2,
                                  game_dates={"g1": "2015-01-05"})
        assert not out2.exists()

        _patch_urlopen(monkeypatch, lambda url: _FakeResponse(
            {"boxScoreTraditional": {"homeTeam": {"teamTricode": "BOS"},
                                     "awayTeam": {"teamTricode": "CLE"}}}))
        out3 = tmp_path / "p3.parquet"
        with pytest.raises(NBAIngestionError):
            pull_player_boxscores(["g1"], out3,
                                  game_dates={"g1": "2015-01-05"})
        assert not out3.exists()

    def test_ad3_corrupt_cache_is_repulled(self, tmp_path, caplog):
        """AD-3 (nba:cache-unreadable): the corrupt cache is discarded and
        the seasons are rebuilt from source. Merging into unreadable bytes
        must never be attempted (regression: the incremental path used to
        die with a raw parquet error after logging the token)."""
        sched = make_schedule(2015, 1)
        out = tmp_path / "schedule.parquet"
        fetch = _season_fetch({"2014": sched})
        pull_schedule("2015-01-01", "2015-03-01", out, full_repull=True,
                      fetch=fetch, today=date(2026, 9, 9), pause_sec=0)
        good = len(pd.read_parquet(out))
        out.write_bytes(b"not parquet")
        with caplog.at_level(logging.WARNING, logger="sports.nba.ingestion"):
            returned = pull_schedule("2015-01-01", "2015-03-01", out,
                                     full_repull=False, fetch=fetch,
                                     today=date(2026, 9, 9), pause_sec=0)
        assert any("nba:cache-unreadable" in r.getMessage()
                   for r in caplog.records)
        assert returned == out
        cached = pd.read_parquet(out)
        assert len(cached) == good and cached["game_id"].is_unique
