"""Tests for mandatory production run validation (core.runconfig).

Phase 1 review decision 5: START_DATE, END_DATE, FULL_REPULL are
required; no silent defaults; omission and invalid values fail loudly;
training history is never derived from these values.
"""

import pytest

from core.config import (
    ENV_END_DATE,
    ENV_FULL_REPULL,
    ENV_START_DATE,
    RunConfigError,
    RunWindow,
    resolve_run_window,
)


def test_all_three_required():
    with pytest.raises(RunConfigError, match="START_DATE is required"):
        resolve_run_window(start_date=None, end_date="2026-09-07",
                           full_repull="0")
    with pytest.raises(RunConfigError, match="END_DATE is required"):
        resolve_run_window(start_date="2026-09-01", end_date=None,
                           full_repull="0")
    with pytest.raises(RunConfigError, match="FULL_REPULL is required"):
        resolve_run_window(start_date="2026-09-01", end_date="2026-09-07",
                           full_repull=None)


def test_no_silent_today_default():
    """END_DATE must not silently fall back to today (reference-repo bug)."""
    with pytest.raises(RunConfigError):
        resolve_run_window(start_date="2026-09-01", end_date=None,
                           full_repull="0")


def test_strict_date_format():
    # Compact YYYYMMDD is rejected — only YYYY-MM-DD.
    with pytest.raises(RunConfigError, match="YYYY-MM-DD"):
        resolve_run_window("20260901", "2026-09-07", "0")
    with pytest.raises(RunConfigError, match="YYYY-MM-DD"):
        resolve_run_window("2026-9-1", "2026-09-07", "0")
    with pytest.raises(RunConfigError, match="YYYY-MM-DD"):
        resolve_run_window("2026-09-01T00:00:00", "2026-09-07", "0")
    with pytest.raises(RunConfigError, match="invalid date"):
        resolve_run_window("2026-02-30", "2026-09-07", "0")


def test_end_must_be_ge_start():
    with pytest.raises(RunConfigError, match="END_DATE"):
        resolve_run_window("2026-09-07", "2026-09-01", "0")
    # Equal is fine.
    w = resolve_run_window("2026-09-07", "2026-09-07", "0")
    assert w.n_days == 1


def test_full_repull_strict_domain():
    for bad in ("true", "yes", "2", "", "1.0"):
        with pytest.raises(RunConfigError, match="FULL_REPULL"):
            resolve_run_window("2026-09-01", "2026-09-07", bad)
    assert resolve_run_window("2026-09-01", "2026-09-07", "0").full_repull \
        is False
    assert resolve_run_window("2026-09-01", "2026-09-07", "1").full_repull \
        is True
    assert resolve_run_window("2026-09-01", "2026-09-07", 1).full_repull \
        is True


def test_env_resolution():
    env = {ENV_START_DATE: "2026-09-01", ENV_END_DATE: "2026-09-05",
           ENV_FULL_REPULL: "0"}
    w = resolve_run_window(env=env)
    assert (w.start_date, w.end_date, w.full_repull) == (
        "2026-09-01", "2026-09-05", False)


def test_env_missing_value_fails_loudly():
    with pytest.raises(RunConfigError, match="START_DATE is required"):
        resolve_run_window(env={ENV_END_DATE: "2026-09-05",
                                ENV_FULL_REPULL: "0"})


def test_whitespace_tolerated_but_not_silently_defaulted():
    env = {ENV_START_DATE: " 2026-09-01 ", ENV_END_DATE: "2026-09-05",
           ENV_FULL_REPULL: " 1 "}
    w = resolve_run_window(env=env)
    assert w.start_date == "2026-09-01"
    assert w.full_repull is True
    # Whitespace-only is NOT a value.
    with pytest.raises(RunConfigError):
        resolve_run_window(env={ENV_START_DATE: "   ",
                                ENV_END_DATE: "2026-09-05",
                                ENV_FULL_REPULL: "0"})


def test_run_window_properties():
    w = resolve_run_window("2026-09-01", "2026-09-08", "1")
    assert isinstance(w, RunWindow)
    assert w.n_days == 8
    assert w.start.isoformat() == "2026-09-01"


def test_training_history_never_derived_from_run_window():
    """The training window comes from the study config, not the run window.

    Two runs with different windows but the same study training_days
    produce the same-length training window; the run window never leaks
    into training history.
    """
    study_days = 400
    w1 = resolve_run_window("2026-09-01", "2026-09-07", "0")
    w2 = resolve_run_window("2025-03-10", "2025-03-11", "1")
    t1 = w1.training_window_days(study_days)
    t2 = w2.training_window_days(study_days)
    from datetime import date
    assert (date.fromisoformat(t1[1]) - date.fromisoformat(t1[0])).days \
        == (date.fromisoformat(t2[1]) - date.fromisoformat(t2[0])).days \
        == study_days - 1
    # The run window start plays no role in the training window.
    assert t1[1] == w1.end_date
    assert t2[1] == w2.end_date
    with pytest.raises(RunConfigError):
        w1.training_window_days(0)
