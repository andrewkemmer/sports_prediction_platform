"""Tests for core.prediction_window."""

from datetime import datetime, timezone

import pytest

from core.config import default_config
from core.config import (
    PredictionWindowError,
    game_in_window,
    partition_slate,
    resolve_prediction_window,
)


def test_resolve_window_default_lookahead():
    cfg = default_config()
    w = resolve_prediction_window(
        "20260907", cfg.sport("mlb"),
        now_utc=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc))
    assert w.slate_dates == ("20260907", "20260908")
    assert w.primary_date() == "20260907"
    assert w.lookahead_days == 1
    assert "20260907" in w


def test_resolve_window_rejects_bad_date():
    cfg = default_config()
    with pytest.raises(PredictionWindowError):
        resolve_prediction_window("2026-09-07", cfg.sport("mlb"))


def test_game_in_window_same_day():
    cfg = default_config()
    w = resolve_prediction_window(
        "20260907", cfg.sport("mlb"),
        now_utc=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc))
    # 7:05 PM ET = 23:05 UTC same day
    assert game_in_window("2026-09-07T23:05:00Z", w)
    # Next day is outside the primary date but inside lookahead dates
    assert game_in_window("2026-09-08T23:05:00Z", w)
    # Two days ahead is fully outside
    assert not game_in_window("2026-09-09T23:05:00Z", w)


def test_game_in_window_empty_and_invalid_excluded():
    cfg = default_config()
    w = resolve_prediction_window(
        "20260907", cfg.sport("mlb"),
        now_utc=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc))
    assert not game_in_window("", w)
    assert not game_in_window("not-a-time", w)


def test_game_in_window_slate_cutoff():
    cfg = default_config()
    w = resolve_prediction_window(
        "20260907", cfg.sport("mlb"),
        now_utc=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc))
    # 8 PM ET start = 00:00 UTC next day; with a 19h ET cutoff it rolls out
    assert not game_in_window("2026-09-08T00:00:00Z", w,
                              slate_cutoff_hour_local=19)
    # 6 PM ET start = 22:00 UTC; inside a 19h cutoff? No: 18 <= 19 -> inside
    assert game_in_window("2026-09-07T22:00:00Z", w,
                          slate_cutoff_hour_local=19)


def test_partition_slate_reports_outside_rows():
    cfg = default_config()
    w = resolve_prediction_window(
        "20260907", cfg.sport("mlb"),
        now_utc=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc))
    starts = ["2026-09-07T23:05:00Z", "", "2026-09-09T23:05:00Z"]
    inside, outside = partition_slate(starts, w)
    assert inside == ["2026-09-07T23:05:00Z"]
    assert outside == ["", "2026-09-09T23:05:00Z"]
