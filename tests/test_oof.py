"""Tests for core.oof."""

import math

import pytest

from core.folds import (
    OOFError,
    brier_score,
    log_loss,
    oof_metrics,
    roc_auc,
    validate_oof_rows,
)


def _row(**overrides):
    base = dict(game_id="20260907_NYY@BOS", game_date="20260907",
                home_team="BOS", away_team="NYY",
                home_win_prob_model=0.6, home_win=1.0)
    base.update(overrides)
    return base


def test_valid_rows_pass():
    report = validate_oof_rows([_row(), _row(game_id="g2",
                                             home_win_prob_model=0.4,
                                             home_win=0.0)])
    assert report.ok
    assert report.n_rows == 2
    assert report.warnings == ()


def test_undecided_rows_warn_not_error():
    report = validate_oof_rows([_row(home_win=None)])
    assert report.ok
    assert any("undecided" in w for w in report.warnings)


def test_missing_columns_error():
    report = validate_oof_rows([{"game_id": "g1"}])
    assert not report.ok
    assert any("missing columns" in e for e in report.errors)


def test_duplicate_game_id_error():
    report = validate_oof_rows([_row(), _row()])
    assert not report.ok
    assert any("duplicate game_id" in e for e in report.errors)


def test_probability_bounds():
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan"), "x"):
        report = validate_oof_rows([_row(home_win_prob_model=bad)])
        assert not report.ok, bad


def test_home_win_must_be_binary():
    report = validate_oof_rows([_row(home_win=0.5)])
    assert not report.ok
    assert any("0/1" in e for e in report.errors)


def test_same_teams_error():
    report = validate_oof_rows([_row(away_team="BOS")])
    assert not report.ok
    assert any("home == away" in e for e in report.errors)


def test_empty_frame_is_error():
    report = validate_oof_rows([])
    assert not report.ok
    assert report.errors == ("empty OOF frame",)


def test_brier_score():
    assert brier_score([0.7], [1.0]) == pytest.approx(0.09)
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
    assert brier_score([1.0 - 1e-9], [1.0]) == pytest.approx(0.0, abs=1e-9)


def test_brier_length_mismatch():
    with pytest.raises(OOFError):
        brier_score([0.5], [1.0, 0.0])


def test_log_loss_perfect_and_clipped():
    assert log_loss([1.0 - 1e-15], [1.0]) == pytest.approx(0.0, abs=1e-9)
    assert log_loss([0.5], [1.0]) == pytest.approx(math.log(2))


def test_roc_auc_perfect_and_degenerate():
    assert roc_auc([0.9, 0.2, 0.8, 0.1], [1.0, 0.0, 1.0, 0.0]) == 1.0
    assert roc_auc([0.2, 0.9, 0.1, 0.8], [1.0, 0.0, 1.0, 0.0]) == 0.0
    assert roc_auc([0.5, 0.5], [1.0, 1.0]) is None


def test_roc_auc_handles_ties():
    # tied scores -> 0.5 AUC
    assert roc_auc([0.5, 0.5], [1.0, 0.0]) == 0.5


def test_oof_metrics_shape():
    m = oof_metrics([0.8, 0.3], [1.0, 0.0])
    assert m["n"] == 2
    assert m["auc"] == 1.0
    assert 0 <= m["brier"] <= 1
    assert m["logloss"] >= 0


def test_metrics_on_empty_raise():
    with pytest.raises(OOFError):
        oof_metrics([], [])
