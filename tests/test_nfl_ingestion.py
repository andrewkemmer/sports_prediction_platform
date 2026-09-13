"""NFL ingestion tests: caching, incremental vs full rebuild, equivalence,
and loud failure behavior."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from sports.nfl.ingestion import (
    EXTERNAL_SOURCES,
    SOURCE_PBP,
    SOURCE_PLAYER_STATS,
    SOURCE_SCHEDULES,
    SOURCE_TEAMS,
    NFLIngestionError,
    compose_nfl_start_utc,
    eligible_games,
    load_pbp,
    load_player_stats,
    load_schedule,
    load_team_names,
)
from tests.nfl_fixtures import make_pbp, make_player_stats, make_schedule


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


# ---------------------------------------------------------------------------
# B-007 external-adapter integration smoke (nfl:nflreadpy.*)
#
# Exercises the REAL DEFAULT adapters (`_default_load_*`, no injected
# load=) with the transport patched to a deterministic fake: production
# call shape, normalization, determinism, loud failure modes, and the
# no-artifact-after-failure contract.
# ---------------------------------------------------------------------------


class TestNflverseAdapterIntegrationSmoke:
    def test_registry_declares_every_source_and_default_adapter(self):
        assert EXTERNAL_SOURCES == {
            SOURCE_SCHEDULES: "_default_load_schedules",
            SOURCE_PBP: "_default_load_pbp",
            SOURCE_PLAYER_STATS: "_default_load_player_stats",
            SOURCE_TEAMS: "_default_load_teams",
        }

    # -- schedules ---------------------------------------------------------- #

    def test_schedules_default_adapter_end_to_end(self, tmp_path, monkeypatch):
        seen: list = []
        fixture = make_schedule(2020, 1)

        def fake(seasons):
            seen.append(seasons)
            return fixture

        monkeypatch.setattr("nflreadpy.load_schedules", fake)
        out = load_schedule([2020], tmp_path, full_repull=True)
        assert seen == [[2020]]  # production call shape: load_schedules([season])
        for col in ("game_id", "gameday", "home_team", "away_team",
                    "home_score", "away_score"):
            assert col in out.columns
        assert out["game_id"].is_unique
        assert len(out) == len(fixture)
        assert (tmp_path / "schedules_2020.parquet").exists()

        a = load_schedule([2020], tmp_path / "a", full_repull=True,
                          load=lambda s: fake(s))
        b = load_schedule([2020], tmp_path / "b", full_repull=True,
                          load=lambda s: fake(s))
        pd.testing.assert_frame_equal(a, b)

    def test_schedules_transport_failure_is_typed(self, tmp_path, monkeypatch):
        def boom(seasons):
            raise ConnectionError("nflverse unreachable")

        monkeypatch.setattr("nflreadpy.load_schedules", boom)
        with pytest.raises(NFLIngestionError) as ei:
            load_schedule([2020], tmp_path, full_repull=True)
        assert ei.value.source == SOURCE_SCHEDULES
        assert ei.value.params["season"] == 2020
        assert isinstance(ei.value.cause, ConnectionError)
        assert not (tmp_path / "schedules_2020.parquet").exists()

    def test_schedules_empty_past_season_is_loud(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nflreadpy.load_schedules",
                            lambda seasons: pd.DataFrame())
        with pytest.raises(NFLIngestionError, match="past season 2019"):
            load_schedule([2019, 2020], tmp_path, full_repull=True)
        assert not (tmp_path / "schedules_2019.parquet").exists()

    def test_schedules_schema_drift_is_loud(self, tmp_path, monkeypatch):
        drifted = make_schedule(2020, 1).drop(columns=["game_id"])
        monkeypatch.setattr("nflreadpy.load_schedules",
                            lambda seasons: drifted)
        with pytest.raises(NFLIngestionError, match="REQUIRED") as ei:
            load_schedule([2020], tmp_path, full_repull=True)
        assert ei.value.params["missing_required_columns"] == ["game_id"]
        assert not (tmp_path / "schedules_2020.parquet").exists()

    # -- pbp ---------------------------------------------------------------- #

    def test_pbp_default_adapter_end_to_end_and_deterministic(
            self, tmp_path, monkeypatch):
        seen: list = []
        fixture = make_pbp(make_schedule(2020, 1))

        def fake(season):
            seen.append(season)
            return fixture

        monkeypatch.setattr("nflreadpy.load_pbp", fake)
        out = load_pbp([2020], tmp_path, full_repull=True)
        assert seen == [2020]  # production call shape: load_pbp(season)
        assert out is not None
        for col in ("game_id", "posteam", "defteam", "epa"):
            assert col in out.columns
        assert out["game_id"].notna().all()
        assert pd.api.types.is_numeric_dtype(out["yards_gained"])
        assert (tmp_path / "pbp_2020.parquet").exists()

        a = load_pbp([2020], tmp_path / "a", full_repull=True,
                     load=lambda s: fixture)
        b = load_pbp([2020], tmp_path / "b", full_repull=True,
                     load=lambda s: fixture)
        pd.testing.assert_frame_equal(a, b)

    def test_pbp_transport_failure_is_typed(self, tmp_path, monkeypatch):
        def boom(season):
            raise TimeoutError("nflverse pbp timeout")

        monkeypatch.setattr("nflreadpy.load_pbp", boom)
        with pytest.raises(NFLIngestionError) as ei:
            load_pbp([2020], tmp_path, full_repull=True)
        assert ei.value.source == SOURCE_PBP
        assert isinstance(ei.value.cause, TimeoutError)
        assert not (tmp_path / "pbp_2020.parquet").exists()

    def test_pbp_empty_past_season_is_loud(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nflreadpy.load_pbp",
                            lambda season: pd.DataFrame())
        with pytest.raises(NFLIngestionError, match="past season 2019"):
            load_pbp([2019, 2020], tmp_path, full_repull=True)
        assert not (tmp_path / "pbp_2019.parquet").exists()

    def test_pbp_schema_drift_is_loud(self, tmp_path, monkeypatch):
        drifted = make_pbp(make_schedule(2020, 1)).drop(columns=["posteam"])
        monkeypatch.setattr("nflreadpy.load_pbp", lambda season: drifted)
        with pytest.raises(NFLIngestionError, match="REQUIRED"):
            load_pbp([2020], tmp_path, full_repull=True)
        assert not (tmp_path / "pbp_2020.parquet").exists()

    # -- player stats ------------------------------------------------------- #

    def test_player_stats_default_adapter_end_to_end(
            self, tmp_path, monkeypatch):
        seen: list = []
        fixture = make_player_stats(2020)

        def fake(seasons):
            seen.append(seasons)
            return fixture

        monkeypatch.setattr("nflreadpy.load_player_stats", fake)
        out = load_player_stats([2020], tmp_path, full_repull=True)
        assert seen == [[2020]]  # production call shape: load_player_stats([season])
        frame = out[2020]
        for col in ("player_id", "player_name", "team", "position"):
            assert col in frame.columns
        assert set(frame["position"].unique()) == {"QB"}
        assert (tmp_path / "player_stats_2020.parquet").exists()

        a = load_player_stats([2020], tmp_path / "a", full_repull=True,
                              load=lambda s: fixture)
        b = load_player_stats([2020], tmp_path / "b", full_repull=True,
                              load=lambda s: fixture)
        pd.testing.assert_frame_equal(a[2020], b[2020])

    def test_player_stats_transport_and_empty_and_drift(
            self, tmp_path, monkeypatch):
        def boom(seasons):
            raise ConnectionError("stats unreachable")

        monkeypatch.setattr("nflreadpy.load_player_stats", boom)
        with pytest.raises(NFLIngestionError) as ei:
            load_player_stats([2020], tmp_path, full_repull=True)
        assert ei.value.source == SOURCE_PLAYER_STATS
        assert not (tmp_path / "player_stats_2020.parquet").exists()

        monkeypatch.setattr("nflreadpy.load_player_stats",
                            lambda seasons: pd.DataFrame())
        with pytest.raises(NFLIngestionError, match="past season 2019"):
            load_player_stats([2019, 2020], tmp_path / "e", full_repull=True)
        assert not (tmp_path / "e" / "player_stats_2019.parquet").exists()

        drifted = make_player_stats(2020).drop(columns=["player_id"])
        monkeypatch.setattr("nflreadpy.load_player_stats",
                            lambda seasons: drifted)
        with pytest.raises(NFLIngestionError, match="REQUIRED"):
            load_player_stats([2020], tmp_path / "d", full_repull=True)
        assert not (tmp_path / "d" / "player_stats_2020.parquet").exists()

    # -- teams -------------------------------------------------------------- #

    def test_team_names_default_adapter_end_to_end(
            self, tmp_path, monkeypatch):
        calls = {"n": 0}
        teams = pd.DataFrame({
            "team_abbr": ["ARI", "BUF"],
            "team_name": ["Arizona Cardinals", "Buffalo Bills"],
        })

        def fake():
            calls["n"] += 1
            return teams

        monkeypatch.setattr("nflreadpy.load_teams", fake)
        out = load_team_names(tmp_path)
        assert calls["n"] == 1  # production call shape: load_teams()
        assert out == {"ARI": "Arizona Cardinals", "BUF": "Buffalo Bills"}
        assert (tmp_path / "teams.parquet").exists()
        # cache hit: the transport is not called again
        assert load_team_names(tmp_path) == out
        assert calls["n"] == 1

    def test_ad3_corrupt_cache_is_repulled(self, tmp_path, caplog):
        """AD-3 (nfl:cache-unreadable): an unreadable per-season cache is
        logged with its declared token and the season is transparently
        re-pulled, leaving a valid cache behind."""
        frames = {2019: make_schedule(2019, 1), 2020: make_schedule(2020, 1)}
        load = _fake_loader(frames)
        load_schedule([2019, 2020], tmp_path, full_repull=True, load=load)
        (tmp_path / "schedules_2019.parquet").write_bytes(b"not parquet")
        with caplog.at_level(logging.WARNING, logger="sports.nfl.ingestion"):
            df = load_schedule([2019, 2020], tmp_path, full_repull=False,
                               load=load)
        assert any("nfl:cache-unreadable" in r.getMessage()
                   for r in caplog.records)
        assert not df.empty and df["game_id"].is_unique
        repaired = pd.read_parquet(tmp_path / "schedules_2019.parquet")
        assert repaired["game_id"].is_unique
        assert len(repaired) == len(frames[2019])

    def test_team_names_failure_modes_are_loud(self, tmp_path, monkeypatch):
        def boom():
            raise ConnectionError("teams unreachable")

        monkeypatch.setattr("nflreadpy.load_teams", boom)
        with pytest.raises(NFLIngestionError) as ei:
            load_team_names(tmp_path)
        assert ei.value.source == SOURCE_TEAMS
        assert not (tmp_path / "teams.parquet").exists()

        monkeypatch.setattr("nflreadpy.load_teams", lambda: pd.DataFrame())
        with pytest.raises(NFLIngestionError, match="EMPTY") as ei:
            load_team_names(tmp_path)
        assert ei.value.source == SOURCE_TEAMS
        assert not (tmp_path / "teams.parquet").exists()

        drifted = pd.DataFrame({"team_abbr": ["ARI"]})  # no team_name
        monkeypatch.setattr("nflreadpy.load_teams", lambda: drifted)
        with pytest.raises(NFLIngestionError, match="REQUIRED"):
            load_team_names(tmp_path)
        assert not (tmp_path / "teams.parquet").exists()


class TestStartInstantComposition:
    """r9a — the derived ``start_time_utc`` (Phase 7.6 start-timestamp
    normalization, closed): gameday (ET date) + gametime (ET wall clock)
    composed into a DST-correct UTC instant, failing CLOSED on anything
    unreliable. The composition is deterministic, so cached pre-r9a
    schedules heal on the next load."""

    def test_composes_et_instants_dst_correct(self):
        df = pd.DataFrame({
            "gameday": ["2026-09-13", "2026-09-13", "2026-11-01",
                        "2026-11-01"],
            "gametime": ["13:00", "20:20", "01:30", "05:00"],
        })
        s = compose_nfl_start_utc(df)
        # September = EDT (UTC-4); 20:20 ET crosses to the next UTC day.
        assert s.iloc[0] == pd.Timestamp("2026-09-13T17:00:00Z")
        assert s.iloc[1] == pd.Timestamp("2026-09-14T00:20:00Z")
        # November 1, 2026 = fall-back day: 01:30 ET is AMBIGUOUS and
        # must fail closed, not silently pick an offset.
        assert pd.isna(s.iloc[2])
        # 05:00 ET is unambiguous EST (UTC-5).
        assert s.iloc[3] == pd.Timestamp("2026-11-01T10:00:00Z")

    def test_fail_closed_on_missing_or_invalid(self):
        df = pd.DataFrame({
            "gameday": ["2026-09-13", "2026-09-13", "not-a-date",
                        "2026-03-08"],
            "gametime": ["", None, "13:00", "02:30"],
        })
        s = compose_nfl_start_utc(df)
        # empty gametime, missing gametime, bad date: all NaN (no
        # fabricated instants)
        assert s.isna().tolist()[:3] == [True, True, True]
        # 02:30 on the spring-forward day is NONEXISTENT in ET -> NaN
        assert s.isna().iloc[3]

    def test_missing_required_column_fails_loud(self):
        with pytest.raises(NFLIngestionError, match="gametime"):
            compose_nfl_start_utc(pd.DataFrame({"gameday": ["2026-09-13"]}))

    def test_load_schedule_emits_start_time_utc(self, tmp_path):
        schedule = make_schedule()
        assert "gametime" in schedule.columns
        out = load_schedule([2019, 2020], tmp_path,
                            load=lambda seasons: schedule.copy())
        assert "start_time_utc" in out.columns
        # every fixture row has a reliable gameday+gametime -> every
        # instant composes
        assert out["start_time_utc"].notna().all()
        assert str(out["start_time_utc"].dtype) == "datetime64[ns, UTC]"
        # deterministic: a second load from CACHE composes identical
        # instants (cached frames are recomposed, healing pre-r9a caches)
        out2 = load_schedule([2019, 2020], tmp_path,
                             load=lambda seasons: schedule.copy())
        assert out2["start_time_utc"].equals(out["start_time_utc"])
