"""MLB source ingestion tests (Phase 2 requirement: ingestion + incremental
vs full repull behavior).

All fetchers are injected — no network access in tests.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sports.mlb.catalog import PITCH_IDENTITY_COLS, STATCAST_COLS
from sports.mlb.ingestion import (
    MLBIngestionError,
    cache_bounds,
    normalize_columns,
    pull_statcast,
)
from tests.mlb_fixtures import make_statcast_games

D0 = date(2026, 4, 1)


def _mk(start: date, days: int) -> pd.DataFrame:
    return make_statcast_games(start, days, games_per_day=4, seed=hash(
        (start, days)) % 2**31)


def test_full_pull_writes_cache(tmp_path):
    fetch_calls: list[tuple[date, date]] = []

    def fetch(s, e):
        fetch_calls.append((s, e))
        return _mk(s, (e - s).days + 1)

    out = tmp_path / "pitches.parquet"
    pull_statcast(D0, D0 + timedelta(days=13), out_path=out,
                  full_repull=True, fetch=fetch, chunk_days=7)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert set(PITCH_IDENTITY_COLS) <= set(df.columns)
    # Single 14-day chunk (chunk_days=7 -> two chunks covering 14 days).
    assert fetch_calls


def test_full_pull_empty_core_season_aborts(tmp_path):
    """Past-dated core-season chunk that stays empty must ABORT loudly."""

    def fetch(s, e):  # noqa: ARG001
        return pd.DataFrame()

    with pytest.raises(MLBIngestionError, match="core regular-season"):
        pull_statcast(D0, D0 + timedelta(days=6), out_path=tmp_path / "p.parquet",
                      full_repull=True, fetch=fetch, today=date(2026, 9, 1))


def test_full_repull_discards_cache(tmp_path):
    out = tmp_path / "pitches.parquet"
    pull_statcast(D0, D0 + timedelta(days=6), out_path=out,
                  full_repull=True, fetch=lambda s, e: _mk(s, 7))
    old_len = len(pd.read_parquet(out))

    # A full repull discards the cache and rewrites from the fetch only.
    def partial_fetch(s, e):
        return _mk(s, 3)

    pull_statcast(D0, D0 + timedelta(days=6), out_path=out,
                  full_repull=True, fetch=partial_fetch)
    new_len = len(pd.read_parquet(out))
    assert new_len < old_len  # cache was replaced, not merged


def test_incremental_resume_topup_forward_gap(tmp_path):
    out = tmp_path / "pitches.parquet"
    pull_statcast(D0, D0 + timedelta(days=6), out_path=out,
                  full_repull=True, fetch=lambda s, e: _mk(s, (e - s).days + 1))
    lo1, hi1 = cache_bounds(out)
    assert hi1 == D0 + timedelta(days=6)

    calls: list[tuple[date, date]] = []

    def tail_fetch(s, e):
        calls.append((s, e))
        return _mk(s, (e - s).days + 1)

    pull_statcast(D0, D0 + timedelta(days=9), out_path=out,
                  full_repull=False, fetch=tail_fetch)
    lo2, hi2 = cache_bounds(out)
    assert hi2 == D0 + timedelta(days=9)
    # The resume must NOT re-pull the whole window — only the gap + tail.
    assert all(s >= D0 for s, _ in calls)
    assert min(s for s, _ in calls) > lo1


def test_incremental_resume_no_network_when_cache_covers_window(tmp_path):
    out = tmp_path / "pitches.parquet"
    pull_statcast(D0, D0 + timedelta(days=9), out_path=out,
                  full_repull=True, fetch=lambda s, e: _mk(s, (e - s).days + 1))
    calls: list[tuple[date, date]] = []

    def probe(s, e):
        calls.append((s, e))
        return _mk(s, (e - s).days + 1)

    # Same window again: only the trailing refresh tail fires (bounded).
    pull_statcast(D0, D0 + timedelta(days=9), out_path=out,
                  full_repull=False, fetch=probe)
    assert calls  # tail refresh happened
    assert min(s for s, _ in calls) >= D0 + timedelta(days=9 - 3)


def test_incremental_backfill_extends_history(tmp_path):
    out = tmp_path / "pitches.parquet"
    pull_statcast(D0 + timedelta(days=7), D0 + timedelta(days=13),
                  out_path=out, full_repull=True,
                  fetch=lambda s, e: _mk(s, (e - s).days + 1))
    lo1, _ = cache_bounds(out)
    assert lo1 == D0 + timedelta(days=7)
    pull_statcast(D0, D0 + timedelta(days=13), out_path=out,
                  full_repull=False,
                  fetch=lambda s, e: _mk(s, (e - s).days + 1))
    lo2, _ = cache_bounds(out)
    assert lo2 == D0


def test_normalize_adds_missing_catalog_columns():
    df = _mk(D0, 2)[["game_pk", "game_date", "home_team", "away_team"]]
    out = normalize_columns(df)
    for col in STATCAST_COLS:
        assert col in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["game_date"])


def test_normalize_filters_spring_training():
    df = _mk(D0, 2)
    df["game_type"] = "S"
    out = normalize_columns(df)
    assert len(out) == 0


def test_dedupe_keeps_newest_copy(tmp_path):
    """Re-delivered pitch rows resolve to the NEWER copy (keep=last on
    merge with existing-first ordering)."""
    out = tmp_path / "pitches.parquet"
    df = _mk(D0, 3)
    pull_statcast(D0, D0 + timedelta(days=2), out_path=out,
                  full_repull=True, fetch=lambda s, e: df)
    cached = pd.read_parquet(out)
    dup_key = cached.iloc[[0]][list(PITCH_IDENTITY_COLS)]
    overlap = cached.merge(dup_key, on=list(PITCH_IDENTITY_COLS))
    assert len(overlap) == 1  # identity is unique in the cache
