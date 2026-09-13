"""Tests for core.dates."""

from datetime import date, datetime

import pytest

from core.study import (
    DateError,
    artifact_date_from_filename,
    compact_date_range,
    format_compact_date,
    is_evening_start_et,
    iso_to_eastern_display,
    is_valid_compact_date,
    nearest_valid_date,
    parse_compact_date,
    shift_compact_date,
)


def test_parse_valid_compact_date():
    assert parse_compact_date("20260907") == date(2026, 9, 7)


def test_parse_rejects_bad_dates():
    for bad in (None, "", "2026090", "202609071", "2026-09-07", "abcdefgh",
                "20261301", True):
        with pytest.raises(DateError):
            parse_compact_date(bad)


def test_parse_accepts_int():
    assert parse_compact_date(20260907) == date(2026, 9, 7)
    assert parse_compact_date(20260907.0) == date(2026, 9, 7)


def test_format_roundtrip():
    d = date(2026, 9, 7)
    assert format_compact_date(d) == "20260907"
    assert parse_compact_date(format_compact_date(d)) == d


def test_is_valid_compact_date():
    assert is_valid_compact_date("20260907")
    assert not is_valid_compact_date("20269999")
    assert not is_valid_compact_date(None)


def test_compact_date_range():
    assert compact_date_range("20260901", "20260903") == [
        "20260901", "20260902", "20260903"]
    assert compact_date_range("20260901", "20260903",
                              inclusive=False) == ["20260901", "20260902"]


def test_compact_date_range_rejects_reversed():
    with pytest.raises(DateError):
        compact_date_range("20260903", "20260901")


def test_shift_compact_date():
    assert shift_compact_date("20260901", 9) == "20260910"
    assert shift_compact_date("20260910", -9) == "20260901"
    assert shift_compact_date("20260901", 0) == "20260901"


def test_nearest_valid_date_prefers_latest_on_tie():
    valid = ["20260901", "20260905", "20260907"]
    # 20260906 is 1 day from both 0905 and 0907 -> newest wins
    assert nearest_valid_date(valid, "20260906") == "20260907"
    assert nearest_valid_date(valid, "20260904") == "20260905"


def test_nearest_valid_date_defaults_to_newest():
    assert nearest_valid_date(["20260901", "20260905"]) == "20260905"
    assert nearest_valid_date([]) is None


def test_nearest_valid_date_dedupes_and_sorts_input():
    assert nearest_valid_date(["20260905", "20260901", "20260905"],
                              "20260902") == "20260901"


def test_iso_to_eastern_display():
    out = iso_to_eastern_display("2026-09-07T23:05:00Z")
    assert out.endswith("ET") and "Sep 7" in out and "7:05 PM" in out
    assert iso_to_eastern_display("") == ""
    assert iso_to_eastern_display("garbage") == ""


def test_is_evening_start_et():
    # 01:00 UTC = 9 PM ET previous day (evening)
    assert is_evening_start_et("2026-09-08T01:00:00Z")
    # 20:00 UTC = 4 PM ET (day game)
    assert not is_evening_start_et("2026-09-07T20:00:00Z")
    assert not is_evening_start_et("")


def test_artifact_date_from_filename():
    assert artifact_date_from_filename(
        "todays_games_20260907.csv", "todays_games_", ".csv") == "20260907"
    assert artifact_date_from_filename(
        "shap_game_20260907_WSH@SD.csv", "shap_game_", ".csv") == "20260907"
    assert artifact_date_from_filename(
        "model_history.json", "model_history", ".json") is None
    assert artifact_date_from_filename(
        "todays_games_20269999.csv", "todays_games_", ".csv") is None
    assert artifact_date_from_filename(
        "power_rankings_20260907.csv", "nfl_power_rankings_", ".csv") is None
