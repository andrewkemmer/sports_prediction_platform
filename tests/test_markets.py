"""Tests for core.markets."""

import pytest

from core.markets import (
    Outcome,
    ProbTriple,
    SettlementError,
    normalize_push_probs,
    normalize_tie_probs,
    push_display_bucket,
    settle_moneyline,
    settle_spread,
    settle_total,
    sport_allows_ties,
    triple_from_line_grid,
)


def test_settle_total_over_under_push():
    s = settle_total(8.5, 10)
    assert s.outcome is Outcome.HOME  # over side
    s = settle_total(8.5, 6)
    assert s.outcome is Outcome.AWAY
    s = settle_total(8.0, 8)
    assert s.is_push and s.is_decided
    s = settle_total(8.5, None)
    assert not s.is_decided


def test_half_point_line_cannot_push():
    # Integer total vs half-point line settles cleanly (never ties)
    assert settle_total(8.5, 8).outcome is Outcome.AWAY
    assert settle_total(8.5, 9).outcome is Outcome.HOME
    # A fractional total against a half-point line is a data error
    with pytest.raises(SettlementError, match="cannot push"):
        settle_total(8.5, 8.5)


def test_settle_spread_favorite_and_dog():
    # home -1.5 wins by 3 -> home covers
    s = settle_spread(-1.5, 3, home_side=True)
    assert s.outcome is Outcome.HOME
    # home -1.5 wins by 1 -> home fails to cover
    s = settle_spread(-1.5, 1, home_side=True)
    assert s.outcome is Outcome.AWAY
    # away +1.5 loses by 1 -> away covers
    s = settle_spread(1.5, -1, home_side=False)
    assert s.outcome is Outcome.HOME
    # integer line exact margin -> push
    s = settle_spread(-3.0, 3, home_side=True)
    assert s.is_push
    with pytest.raises(SettlementError):
        settle_spread(-1.5, 1.5, home_side=True)


def test_settle_moneyline_undecided_and_tie():
    assert settle_moneyline(5, 3).outcome is Outcome.HOME
    assert settle_moneyline(2, 4).outcome is Outcome.AWAY
    assert not settle_moneyline(None, 3).is_decided
    assert not settle_moneyline(3, 3).is_decided  # ties are undecided by default


def test_probTriple_validation():
    t = ProbTriple(0.45, 0.10, 0.45)
    assert t.decided_mass() == pytest.approx(0.90)
    with pytest.raises(SettlementError):
        ProbTriple(0.5, 0.6, 0.5)
    with pytest.raises(SettlementError):
        ProbTriple(-0.1, 0.2, 0.9)
    with pytest.raises(SettlementError):
        ProbTriple(0.5, 0.2, 0.5)  # sums to 1.2


def test_renormalized_pair_with_and_without_push():
    t = ProbTriple(0.45, 0.10, 0.45)
    a, b = t.renormalized_pair()
    assert (a, b) == pytest.approx((0.5, 0.5))
    # all-push edge -> honest coin flip
    t0 = ProbTriple(0.0, 1.0, 0.0)
    assert t0.renormalized_pair() == (0.5, 0.5)


def test_normalize_push_probs_explicit():
    a, b, p = normalize_push_probs(0.4, 0.5, 0.1)
    assert (a, b, p) == (0.4, 0.5, 0.1)


def test_normalize_push_probs_derived_push():
    a, b, p = normalize_push_probs(0.45, 0.45)
    assert p == pytest.approx(0.1)
    assert a + b + p == pytest.approx(1.0)


def test_normalize_push_probs_rejects_overshoot():
    with pytest.raises(SettlementError, match="exceeds 1"):
        normalize_push_probs(0.6, 0.6)
    with pytest.raises(SettlementError):
        normalize_push_probs(-0.1, 0.5)


def test_normalize_tie_probs_alias():
    h, a, t = normalize_tie_probs(0.45, 0.45, 0.10)
    assert (h, a, t) == (0.45, 0.45, 0.10)


def test_triple_from_line_grid():
    t = triple_from_line_grid(0.5, 0.4)
    assert t.p_a == pytest.approx(0.5)
    assert t.p_b == pytest.approx(0.4)
    assert t.p_push == pytest.approx(0.1)


def test_push_display_buckets():
    assert push_display_bucket(0.0) == "none"
    assert push_display_bucket(0.004) == "none"
    assert push_display_bucket(0.01) == "small"
    assert push_display_bucket(0.10) == "moderate"
    assert push_display_bucket(0.20) == "high"
    with pytest.raises(SettlementError):
        push_display_bucket(-0.01)


def test_sport_allows_ties():
    assert sport_allows_ties("nfl")
    assert not sport_allows_ties("mlb")
    assert not sport_allows_ties("nhl")
    assert not sport_allows_ties("nba")
