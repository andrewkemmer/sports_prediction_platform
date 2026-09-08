"""Tests for core.folds."""

import pytest

from core.config import WalkForwardConfig
from core.folds import Fold, FoldError, build_folds, train_test_boundary


def test_basic_daily_folds():
    dates = ["20260901", "20260902", "20260903", "20260904"]
    folds = build_folds(dates, WalkForwardConfig(min_train_games=1))
    # The first date has no training games -> skipped by the guard.
    assert [f.test_date for f in folds] == dates[1:]
    assert folds[0].n_train_games == 1
    assert folds[2].n_train_games == 3
    assert [f.fold_id for f in folds] == [0, 1, 2]


def test_min_train_guard_skips_early_dates():
    dates = ["20260901", "20260902", "20260903"]
    folds = build_folds(dates, WalkForwardConfig(min_train_games=2))
    # 0901 has 0 train games, 0902 has 1 -> only 0903 qualifies
    assert [f.test_date for f in folds] == ["20260903"]
    assert folds[0].n_train_games == 2


def test_no_valid_fold_raises():
    with pytest.raises(FoldError, match="min_train_games"):
        build_folds(["20260901"], WalkForwardConfig(min_train_games=5))


def test_empty_dates_raise():
    with pytest.raises(FoldError, match="no game dates"):
        build_folds([], WalkForwardConfig())


def test_multiple_games_per_date_counted():
    dates = ["20260901"] * 3 + ["20260902"] * 2
    folds = build_folds(dates, WalkForwardConfig(min_train_games=3))
    assert len(folds) == 1
    assert folds[0].test_date == "20260902"
    assert folds[0].n_train_games == 3


def test_step_days_stride():
    dates = [f"2026090{i}" for i in range(1, 6)]
    folds = build_folds(dates, WalkForwardConfig(min_train_games=1,
                                                 step_days=2))
    # 0901 has 0 training games -> skipped; stride then hits 0903/0905.
    assert [f.test_date for f in folds] == ["20260903", "20260905"]


def test_embargo_excludes_recent_games():
    dates = ["20260901", "20260902", "20260903"]
    folds = build_folds(dates, WalkForwardConfig(min_train_games=1,
                                                 embargo_days=1))
    # For test 0903, embargo excludes 0902 -> only 0901 trains
    last = folds[-1]
    assert last.test_date == "20260903"
    assert last.n_train_games == 1


def test_start_end_window():
    dates = ["20260901", "20260902", "20260903", "20260904"]
    folds = build_folds(dates, WalkForwardConfig(min_train_games=1),
                        start_date="20260902", end_date="20260903")
    assert [f.test_date for f in folds] == ["20260902", "20260903"]


def test_dates_without_games_skipped():
    dates = ["20260901", "20260903"]  # 0902 has no games
    folds = build_folds(dates, WalkForwardConfig(min_train_games=1),
                        start_date="20260901", end_date="20260903")
    # 0901 has no training games -> skipped by the min-train guard.
    assert [f.test_date for f in folds] == ["20260903"]


def test_reversed_window_raises():
    with pytest.raises(FoldError, match="after end_date"):
        build_folds(["20260901"], WalkForwardConfig(),
                    start_date="20260902", end_date="20260901")


def test_fold_validation_and_boundary():
    f = Fold(fold_id=0, train_end="20260902", test_date="20260902",
             n_train_games=5)
    assert train_test_boundary(f) == ("20260902", "20260902")
    with pytest.raises(FoldError):
        Fold(fold_id=0, train_end="bad", test_date="20260902",
             n_train_games=1)
    with pytest.raises(FoldError):
        Fold(fold_id=0, train_end="20260902", test_date="20260902",
             n_train_games=-1)
