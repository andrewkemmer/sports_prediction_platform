"""MLB source ingestion tests (Phase 2 requirement: ingestion + incremental
vs full repull behavior).

All fetchers are injected — no network access in tests.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from sports.mlb.catalog import PITCH_IDENTITY_COLS, STATCAST_COLS
from sports.mlb.ingestion import (
    EXTERNAL_SOURCES,
    SOURCE_STATCAST,
    MLBIngestionError,
    _default_fetch,
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
    df = _mk(D0, 2)
    # Drop optional columns to verify they get added back as NaN
    df = df.drop(columns=["release_speed", "events", "description"])
    out = normalize_columns(df)
    for col in STATCAST_COLS:
        assert col in out.columns
    assert pd.isna(out["release_speed"]).all()
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


# ---------------------------------------------------------------------------
# B-007 external-adapter integration smoke (mlb:pybaseball.statcast)
#
# Exercises the REAL DEFAULT adapter path (`_default_fetch`, no injected
# fetch=) with the transport patched to a deterministic fake: the
# production call site, argument shape, normalization, determinism,
# loud failure modes, and the no-artifact-after-failure contract.
# ---------------------------------------------------------------------------

_SMOKE_WINDOW = (D0, D0 + timedelta(days=6))


def _patch_statcast(monkeypatch, fake):
    """Patch the real transport the default adapter imports lazily.

    String-target setattr: pytest performs the import internally, so this
    test module itself never imports a transport (manifest hygiene)."""
    monkeypatch.setattr("pybaseball.statcast", fake)


def _smoke_pull(out, *, full_repull=True):
    return pull_statcast(_SMOKE_WINDOW[0], _SMOKE_WINDOW[1], out_path=out,
                         full_repull=full_repull)


class TestStatcastAdapterIntegrationSmoke:
    def test_registry_declares_the_source_and_default_adapter(self):
        assert EXTERNAL_SOURCES[SOURCE_STATCAST] == "_default_fetch"
        assert callable(_default_fetch)

    def test_default_adapter_end_to_end_schema_and_values(
            self, tmp_path, monkeypatch):
        calls: list[tuple] = []

        def fake(start, end, *args, **kwargs):
            calls.append((start, end))
            return _mk(D0, 7)

        _patch_statcast(monkeypatch, fake)
        out = tmp_path / "pitches.parquet"
        _smoke_pull(out)

        # The REAL production call shape: statcast(start_dt, end_dt) as
        # ISO date strings — never (year, month) and never a team token.
        assert calls == [(str(_SMOKE_WINDOW[0]), str(_SMOKE_WINDOW[1]))]
        df = pd.read_parquet(out)
        for col in STATCAST_COLS:
            assert col in df.columns
        assert set(PITCH_IDENTITY_COLS) <= set(df.columns)
        assert df["game_pk"].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
        assert set(df["game_type"].unique()) <= {"R"}
        assert pd.api.types.is_numeric_dtype(df["home_score"])
        assert len(df) == len(_mk(D0, 7))

    def test_default_adapter_is_deterministic(
            self, tmp_path, monkeypatch):
        _patch_statcast(monkeypatch, lambda s, e, *a, **k: _mk(D0, 7))
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        _smoke_pull(a)
        _smoke_pull(b)
        pd.testing.assert_frame_equal(pd.read_parquet(a),
                                      pd.read_parquet(b))

    def test_transport_failure_raises_typed_error(
            self, tmp_path, monkeypatch):
        def boom(s, e, *a, **k):
            raise ConnectionError("statcast unreachable")

        _patch_statcast(monkeypatch, boom)
        out = tmp_path / "p.parquet"
        with pytest.raises(MLBIngestionError) as ei:
            _smoke_pull(out)
        assert ei.value.source == SOURCE_STATCAST
        assert ei.value.params["chunk_start"] == str(D0)
        # the typed chain preserves the originating transport failure
        cause = ei.value.cause
        while isinstance(cause, MLBIngestionError):
            cause = cause.cause
        assert isinstance(cause, ConnectionError)
        assert "statcast unreachable" in str(cause)
        assert not out.exists()  # no artifact after a transport failure

    def test_empty_past_window_raises_and_writes_nothing(
            self, tmp_path, monkeypatch):
        _patch_statcast(monkeypatch, lambda s, e, *a, **k: pd.DataFrame())
        out = tmp_path / "p.parquet"
        with pytest.raises(MLBIngestionError, match="core regular-season"):
            _smoke_pull(out)
        assert not out.exists()

    def test_schema_drift_raises_and_writes_nothing(
            self, tmp_path, monkeypatch):
        drifted = _mk(D0, 7).drop(columns=["game_pk"])
        _patch_statcast(monkeypatch, lambda s, e, *a, **k: drifted)
        out = tmp_path / "p.parquet"
        with pytest.raises(MLBIngestionError, match="missing required") as ei:
            _smoke_pull(out)
        assert ei.value.params["missing_columns"] == ["game_pk"]
        assert not out.exists()

    def test_malformed_payload_raises_and_writes_nothing(
            self, tmp_path, monkeypatch):
        _patch_statcast(monkeypatch, lambda s, e, *a, **k: None)
        out = tmp_path / "p.parquet"
        with pytest.raises(MLBIngestionError):
            _smoke_pull(out)
        assert not out.exists()


def test_ad3_corrupt_cache_is_repulled(tmp_path, caplog):
    """AD-3 (mlb:cache-unreadable): an unreadable cache is logged with its
    declared token and the run RECOVERS by re-pulling the source — never a
    partial/merged write onto corrupt bytes."""
    out = tmp_path / "pitches.parquet"
    fetch = lambda s, e: _mk(s, (e - s).days + 1)
    pull_statcast(D0, D0 + timedelta(days=6), out_path=out, full_repull=True,
                  fetch=fetch)
    good = len(pd.read_parquet(out))
    out.write_bytes(b"not parquet")
    with caplog.at_level(logging.WARNING, logger="sports.mlb.ingestion"):
        returned = pull_statcast(D0, D0 + timedelta(days=6), out_path=out,
                                 full_repull=False, fetch=fetch)
    assert any("mlb:cache-unreadable" in r.getMessage()
               for r in caplog.records)
    assert returned == out
    assert len(pd.read_parquet(out)) == good  # recovered, not truncated
