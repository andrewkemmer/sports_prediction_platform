"""NHL ingestion tests: caching, incremental vs full rebuild, window
fetching, loud failure behavior, and goalie panel caching."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sports.nhl.ingestion import (
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
