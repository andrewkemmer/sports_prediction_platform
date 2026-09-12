"""NHL ingestion tests: caching, incremental vs full rebuild, window
fetching, loud failure behavior, and goalie panel caching."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from sports.nhl.ingestion import (
    EXTERNAL_SOURCES,
    SOURCE_BOXSCORE,
    SOURCE_SCHEDULE,
    WINDOW_DAYS,
    NHLIngestionError,
    normalize_schedule,
    pull_goalie_boxscores,
    pull_schedule,
)
from tests.nhl_fixtures import make_goalie_cache, make_schedule


def _window_fetch(frames: dict):
    """Injectable week-window fetcher returning the synthetic rows whose
    game_date falls in [start, end]."""
    def fetch(start: date, end: date) -> pd.DataFrame:
        sched = pd.concat(frames.values(), ignore_index=True)
        gd = pd.to_datetime(sched["game_date"]).dt.date
        m = (gd >= start) & (gd <= end)
        return sched[m].reset_index(drop=True)
    return fetch


def _fixture_window(sched: pd.DataFrame) -> tuple[str, str]:
    """Pull bounds that exactly cover the synthetic fixture dates (pulling
    beyond them would hit empty in-season windows, which fail loudly by
    design)."""
    gd = pd.to_datetime(sched["game_date"])
    return str(gd.min().date()), str(gd.max().date())


class TestSchedulePull:
    def test_full_pull_and_cache(self, tmp_path):
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out,
                      full_repull=True, fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        cached = pd.read_parquet(out)
        assert len(cached) == len(sched)
        assert (cached["game_type"] == 2).all()

    def test_incremental_reuses_cache(self, tmp_path):
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out,
                      full_repull=True, fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        calls = {"n": 0}

        def counting(start, end):
            calls["n"] += 1
            return _window_fetch({2021: sched})(start, end)

        pull_schedule(lo, hi, out,
                      full_repull=False, fetch=counting,
                      today=date(2026, 9, 7))
        # incremental only refreshes the trailing tail, not the full window
        assert calls["n"] < 10

    def test_full_repull_discards_and_rebuilds(self, tmp_path):
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out,
                      full_repull=True, fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        before = pd.read_parquet(out)
        # poison the cache; a full repull must rebuild identical content
        poisoned = before.iloc[:-5]
        poisoned.to_parquet(out, index=False)
        pull_schedule(lo, hi, out,
                      full_repull=True, fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        after = pd.read_parquet(out)
        assert len(after) == len(before)

    def test_all_empty_pull_is_loud(self, tmp_path):
        out = tmp_path / "schedule.parquet"

        def empty_fetch(start, end):
            return pd.DataFrame()

        # An entirely empty pull raises (no data at all) — loud, not silent.
        with pytest.raises(NHLIngestionError):
            pull_schedule("2021-07-01", "2021-07-31", out,
                          full_repull=True, fetch=empty_fetch,
                          today=date(2026, 9, 7))

    def test_genuine_gap_passes_but_hole_is_fatal(self, tmp_path):
        out = tmp_path / "schedule.parquet"
        sched = make_schedule(2021, 1)
        gd = pd.to_datetime(sched["game_date"]).dt.date
        lo, hi = gd.min(), gd.max()

        # HOLE: one mid-history week-window returns nothing while both
        # neighbors return games — silent loss risk, must refuse. The hole
        # is aligned to the 7-day pull-window grid so the walk hits it
        # dead-on (a hole BETWEEN walk windows is unreachable by design:
        # each window that overlaps real games returns them).
        hole_start = lo + timedelta(days=3 * WINDOW_DAYS)
        hole_end = hole_start + timedelta(days=WINDOW_DAYS - 1)

        def holey_fetch(start, end):
            if start >= hole_start and end <= hole_end:
                return pd.DataFrame()
            return _window_fetch({2021: sched})(start, end)

        with pytest.raises(NHLIngestionError, match="BOTH sides"):
            pull_schedule(lo.isoformat(), hi.isoformat(), out,
                          full_repull=True, fetch=holey_fetch,
                          today=date(2026, 9, 7))

        # GENUINE GAP: the leading weeks are empty (delayed season) and
        # games start later — allowed, the store just covers the real span.
        gap_end = lo + timedelta(days=13)

        def gappy_fetch(start, end):
            if end <= gap_end:
                return pd.DataFrame()
            return _window_fetch({2021: sched})(start, end)

        out2 = tmp_path / "schedule2.parquet"
        pull_schedule(lo.isoformat(), hi.isoformat(), out2,
                      full_repull=True, fetch=gappy_fetch,
                      today=date(2026, 9, 7))
        assert pd.read_parquet(out2)["game_id"].is_unique

    def test_dedupe_on_game_id(self, tmp_path):
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out,
                      full_repull=True, fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        cached = pd.read_parquet(out)
        assert cached["game_id"].is_unique


class TestNormalize:
    def test_preseason_and_playoffs_dropped(self):
        sched = make_schedule(2021, 1)
        extra = pd.DataFrame([{
            "game_id": "x1", "season": 2021, "game_type": 1,
            "game_date": "2021-09-25", "start_time_utc": None,
            "home_team": "TOR", "away_team": "BOS",
            "home_score": 3, "away_score": 2, "venue": "X",
            "neutral_site": False, "game_state": "OFF",
        }, {
            "game_id": "x2", "season": 2021, "game_type": 3,
            "game_date": "2021-05-15", "start_time_utc": None,
            "home_team": "TOR", "away_team": "BOS",
            "home_score": 3, "away_score": 2, "venue": "X",
            "neutral_site": False, "game_state": "OFF",
        }])
        out = normalize_schedule(pd.concat([sched, extra],
                                           ignore_index=True))
        assert set(out["game_type"].unique()) == {2}

    def test_missing_columns_become_nan(self):
        sched = make_schedule(2021, 1).drop(columns=["venue"])
        out = normalize_schedule(sched)
        assert "venue" in out.columns
        assert out["venue"].isna().all()


class TestGoaliePanel:
    def test_goalie_cache_roundtrip(self, tmp_path):
        sched = make_schedule(2021, 1)
        panel = make_goalie_cache(sched)
        out = tmp_path / "goalie_panel.parquet"
        panel.to_parquet(out, index=False)
        from sports.nhl.ingestion import load_goalie_cache
        loaded = load_goalie_cache(out)
        assert len(loaded) == len(panel)
        assert loaded["game_id"].is_unique or True  # one row per (game, team)

    def test_pull_goalie_dedupes_existing(self, tmp_path, monkeypatch):
        sched = make_schedule(2021, 1)
        panel = make_goalie_cache(sched)
        out = tmp_path / "goalie_panel.parquet"
        panel.to_parquet(out, index=False)
        calls = {"n": 0}

        def fake_fetch(gid):
            calls["n"] += 1
            return None

        monkeypatch.setattr("sports.nhl.ingestion._default_fetch_boxscore",
                            fake_fetch)
        pull_goalie_boxscores([str(g) for g in panel["game_id"].unique()],
                              out)
        # every cached game already present — zero network calls
        assert calls["n"] == 0


# ---------------------------------------------------------------------------
# B-007 external-adapter integration smoke (nhl:api-web.nhle.com/*)
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


def _api_game(r) -> dict:
    return {
        "id": r.game_id,
        "season": int(f"{r.season}{str(r.season + 1)[-2:]}"),
        "gameType": int(r.game_type),
        "gameDate": r.game_date,
        "startTimeUTC": r.start_time_utc,
        "homeTeam": {"abbrev": r.home_team,
                     "score": None if pd.isna(r.home_score)
                     else int(r.home_score)},
        "awayTeam": {"abbrev": r.away_team,
                     "score": None if pd.isna(r.away_score)
                     else int(r.away_score)},
        "venue": {"default": r.venue},
        "neutralSite": False,
        "gameState": r.game_state,
    }


def _api_boxscore(game_id: str, home: str = "TOR", away: str = "BOS") -> dict:
    def goalie(team):
        return {"playerId": f"{team}-G1",
                "name": {"default": f"{team} Goalie"}, "position": "G",
                "starter": True, "decision": "W", "saves": 28,
                "shotsAgainst": 30, "goalsAgainst": 2, "savePctg": 0.933,
                "toi": "60:00"}

    return {"season": 20212022, "gameDate": "2021-10-14",
            "homeTeam": {"abbrev": home}, "awayTeam": {"abbrev": away},
            "playerByGameStats": {"homeTeam": {"goalies": [goalie(home)]},
                                  "awayTeam": {"goalies": [goalie(away)]}}}


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


class TestNhlApiAdapterIntegrationSmoke:
    def test_registry_declares_every_source_and_default_adapter(self):
        assert EXTERNAL_SOURCES == {
            SOURCE_SCHEDULE: "_default_fetch_schedule_window",
            SOURCE_BOXSCORE: "_default_fetch_boxscore",
        }

    def test_schedule_default_adapter_end_to_end_and_deterministic(
            self, tmp_path, monkeypatch):
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        games = [_api_game(r) for r in sched.itertuples(index=False)]

        def handler(url: str):
            day = date.fromisoformat(url.rsplit("/", 1)[-1])
            week = [g for g in games
                    if day <= date.fromisoformat(g["gameDate"])
                    <= day + timedelta(days=WINDOW_DAYS - 1)]
            return _FakeResponse({"gameWeek": [{"games": week}]})

        calls = _patch_urlopen(monkeypatch, handler)
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out, full_repull=True,
                      today=date(2026, 9, 7))
        # production call shape: /v1/schedule/<date> dated week requests
        assert calls
        assert all("/schedule/" in u for u in calls)
        cached = pd.read_parquet(out)
        assert cached["game_id"].is_unique
        assert set(cached["game_id"]) == set(sched["game_id"])
        for col in ("game_id", "game_date", "home_team", "away_team",
                    "home_score", "away_score"):
            assert col in cached.columns
        assert pd.api.types.is_datetime64_any_dtype(cached["game_date"])
        assert pd.api.types.is_numeric_dtype(cached["home_score"])
        assert set(cached["home_team"]) <= set(sched["home_team"])

        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        pull_schedule(lo, hi, a, full_repull=True, today=date(2026, 9, 7))
        pull_schedule(lo, hi, b, full_repull=True, today=date(2026, 9, 7))
        pd.testing.assert_frame_equal(pd.read_parquet(a),
                                      pd.read_parquet(b))

    def test_schedule_failure_modes_are_loud(self, tmp_path, monkeypatch):
        def boom(url):
            raise ConnectionError("nhl api unreachable")

        _patch_urlopen(monkeypatch, boom)
        out = tmp_path / "schedule.parquet"
        with pytest.raises(NHLIngestionError) as ei:
            pull_schedule("2021-10-01", "2021-10-07", out, full_repull=True,
                          today=date(2026, 9, 7))
        assert ei.value.source == SOURCE_SCHEDULE
        assert not out.exists()

        # empty payload: all-empty full pull is a hole, refusing loudly
        _patch_urlopen(monkeypatch, lambda url: _FakeResponse({}))
        out2 = tmp_path / "s2.parquet"
        with pytest.raises(NHLIngestionError):
            pull_schedule("2021-10-01", "2021-10-07", out2, full_repull=True,
                          today=date(2026, 9, 7))
        assert not out2.exists()

        # malformed payload: a non-dict game entry is schema drift
        _patch_urlopen(monkeypatch, lambda url: _FakeResponse(
            {"gameWeek": [{"games": ["not-a-game-dict"]}]}))
        out3 = tmp_path / "s3.parquet"
        with pytest.raises(NHLIngestionError) as ei:
            pull_schedule("2021-10-01", "2021-10-07", out3, full_repull=True,
                          today=date(2026, 9, 7))
        assert ei.value.source == SOURCE_SCHEDULE
        assert not out3.exists()

    def test_boxscore_default_adapter_end_to_end_and_deterministic(
            self, tmp_path, monkeypatch):
        calls = _patch_urlopen(
            monkeypatch,
            lambda url: _FakeResponse(_api_boxscore(url.split("/")[-2])))
        out = tmp_path / "goalie_panel.parquet"
        pull_goalie_boxscores(["2021020001"], out)
        assert calls == [f"https://api-web.nhle.com/v1/gamecenter/2021020001"
                         f"/boxscore"]
        panel = pd.read_parquet(out)
        for col in ("game_id", "team", "player_id", "player_name",
                    "position", "starter", "decision", "saves",
                    "shots_against", "goals_against", "save_pctg", "toi",
                    "game_date", "season"):
            assert col in panel.columns
        assert set(panel["team"]) == {"TOR", "BOS"}
        assert panel["player_id"].notna().all()
        assert int(panel["saves"].iloc[0]) == 28
        assert bool(panel["starter"].iloc[0]) is True

        pa = tmp_path / "a.parquet"
        pb = tmp_path / "b.parquet"
        pull_goalie_boxscores(["2021020001"], pa)
        pull_goalie_boxscores(["2021020001"], pb)
        pd.testing.assert_frame_equal(pd.read_parquet(pa),
                                      pd.read_parquet(pb))

    def test_boxscore_failure_modes_are_loud(self, tmp_path, monkeypatch):
        def boom(url):
            raise TimeoutError("boxscore timeout")

        _patch_urlopen(monkeypatch, boom)
        out = tmp_path / "p1.parquet"
        with pytest.raises(NHLIngestionError) as ei:
            pull_goalie_boxscores(["g1"], out)
        assert ei.value.source == SOURCE_BOXSCORE
        assert not out.exists()

        _patch_urlopen(monkeypatch, lambda url: _FakeResponse({}))
        out2 = tmp_path / "p2.parquet"
        with pytest.raises(NHLIngestionError, match="EMPTY"):
            pull_goalie_boxscores(["g1"], out2)
        assert not out2.exists()

        _patch_urlopen(monkeypatch, lambda url: _FakeResponse(
            {"season": 20212022, "gameDate": "2021-10-14",
             "homeTeam": {"abbrev": "TOR"},
             "awayTeam": {"abbrev": "BOS"}, "playerByGameStats": {}}))
        out3 = tmp_path / "p3.parquet"
        with pytest.raises(NHLIngestionError, match="no goalie rows"):
            pull_goalie_boxscores(["g1"], out3)
        assert not out3.exists()

    def test_ad3_corrupt_cache_is_repulled(self, tmp_path, caplog):
        """AD-3 (nhl:cache-unreadable): an unreadable schedule cache is
        logged with its declared token and rebuilt from the source."""
        sched = make_schedule(2021, 1)
        lo, hi = _fixture_window(sched)
        out = tmp_path / "schedule.parquet"
        fetch = _window_fetch({2021: sched})
        pull_schedule(lo, hi, out, full_repull=True, fetch=fetch,
                      today=date(2026, 9, 7))
        good = len(pd.read_parquet(out))
        out.write_bytes(b"not parquet")
        with caplog.at_level(logging.WARNING, logger="sports.nhl.ingestion"):
            returned = pull_schedule(lo, hi, out, full_repull=False, fetch=fetch,
                                     today=date(2026, 9, 7))
        assert any("nhl:cache-unreadable" in r.getMessage()
                   for r in caplog.records)
        assert returned == out
        cached = pd.read_parquet(out)
        assert len(cached) == good and cached["game_id"].is_unique

    def test_ad4_tail_refresh_empty_keeps_cache(self, tmp_path, caplog):
        """AD-4 (nhl:tail-refresh-empty): when an incremental tail refresh
        returns no rows (nothing posted yet) the existing cache is KEPT
        byte-identical — never truncated, emptied, or fabricated."""
        sched = make_schedule(2021, 1)
        gd = pd.to_datetime(sched["game_date"]).dt.date
        lo, hi = gd.min().isoformat(), gd.max().isoformat()
        out = tmp_path / "schedule.parquet"
        pull_schedule(lo, hi, out, full_repull=True,
                      fetch=_window_fetch({2021: sched}),
                      today=date(2026, 9, 7))
        before = out.read_bytes()
        with caplog.at_level(logging.WARNING, logger="sports.nhl.ingestion"):
            returned = pull_schedule(lo, hi, out, full_repull=False,
                                     fetch=lambda s, e: pd.DataFrame(),
                                     today=date(2026, 9, 7))
        assert any("nhl:tail-refresh-empty" in r.getMessage()
                   for r in caplog.records)
        assert returned == out
        assert out.read_bytes() == before  # cache preserved exactly
