"""Tests for core.markets."""

import pytest

from core.markets import (
    FULL_GAME_NO_TIE,
    FULL_GAME_TIE_ALLOWED,
    REGULATION_TIE_ALLOWED,
    Outcome,
    ProbTriple,
    SettlementConfig,
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


# ---------------------------------------------------------------------------
# Core settlement primitives
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TIE/PUSH distinction and configurable settlement (merged from
# tests/test_markets_tie_push.py — Phase 1 review decision 1/2: TIE
# (game has no winner) is distinct from PUSH (wager refunded on an
# integer line), and integer settlement is universally configurable —
# not totals-only.
# ---------------------------------------------------------------------------


def test_settlement_config_scope_validation():
    with pytest.raises(SettlementError, match="settlement_scope"):
        SettlementConfig(settlement_scope="overtime")


def test_prebuilt_configs():
    assert REGULATION_TIE_ALLOWED.settlement_scope == "regulation"
    assert REGULATION_TIE_ALLOWED.tie_allowed
    assert FULL_GAME_TIE_ALLOWED.tie_allowed
    assert not FULL_GAME_NO_TIE.tie_allowed
    assert FULL_GAME_NO_TIE.push_allowed


def test_nfl_full_game_tie_settles_tie():
    # 24-24 final (rare NFL tie): TIE, not push, not undecided.
    s = settle_moneyline(24, 24, FULL_GAME_TIE_ALLOWED)
    assert s.outcome is Outcome.TIE
    assert s.is_tie and not s.is_push
    assert s.is_decided  # the GAME is decided-as-tie; the wager is a tie
    # A game-level tie does NOT refund a spread wager (standard book
    # rule): 24-24 against -3 is a failure to cover for the favorite.
    sp = settle_spread(-3.0, 0, home_side=True, cfg=FULL_GAME_TIE_ALLOWED)
    assert sp.outcome is Outcome.AWAY
    assert not sp.is_push and not sp.is_tie


def test_nfl_regulation_scope_carries_through():
    s = settle_moneyline(17, 17, REGULATION_TIE_ALLOWED)
    assert s.outcome is Outcome.TIE
    assert s.scope == "regulation"


def test_mlb_nhl_nba_full_game_tie_is_undecided_not_tie():
    # MLB/NBA/NHL full-game: OT/extras always resolve; a level score in
    # the data is an undecided game, never a fabricated TIE.
    s = settle_moneyline(3, 3, FULL_GAME_NO_TIE)
    assert not s.is_decided
    assert s.outcome is Outcome.PUSH  # placeholder outcome, undecided
    assert not s.is_push  # NOT a wager push — nothing was refunded


def test_moneyline_normal_decided_results():
    assert settle_moneyline(5, 3).outcome is Outcome.HOME
    assert settle_moneyline(2, 4).outcome is Outcome.AWAY
    assert settle_moneyline(None, 3).is_decided is False


def test_nhl_regulation_tie_vs_full_game_resolution():
    # Regulation ends 2-2; the full-game moneyline is resolved by OT/SO.
    reg = settle_moneyline(2, 2, REGULATION_TIE_ALLOWED)
    assert reg.outcome is Outcome.TIE
    assert reg.scope == "regulation"
    # The same game at full game (after OT/SO a winner exists) is
    # settled from the FULL score — a 3-2 final is a home win.
    full = settle_moneyline(3, 2, FULL_GAME_NO_TIE)
    assert full.outcome is Outcome.HOME
    assert full.scope == "full_game"


def test_nhl_regulation_spread_tie_goes_to_push_not_tie():
    # A regulation tie (0-0) with a -0.5 spread offer: the margin lands
    # exactly on the half-point... which cannot happen (half lines never
    # match an integer margin). The +0.5 side covers instead.
    s = settle_spread(-0.5, 0, home_side=True, cfg=REGULATION_TIE_ALLOWED)
    assert s.outcome is Outcome.AWAY  # home -0.5 fails to cover a 0-0 tie
    s2 = settle_spread(0.5, 0, home_side=False, cfg=REGULATION_TIE_ALLOWED)
    assert s2.outcome is Outcome.HOME  # away +0.5 covers


def test_totals_integer_push_requires_push_allowed():
    assert settle_total(8.0, 8).outcome is Outcome.PUSH
    # A config that disallows push makes an exact landing a loud error.
    no_push = SettlementConfig(push_allowed=False, tie_allowed=False)
    with pytest.raises(SettlementError, match="push is not allowed"):
        settle_total(8.0, 8, no_push)


def test_totals_half_point_line_cannot_push():
    with pytest.raises(SettlementError, match="cannot push"):
        settle_total(8.5, 8.5)
    assert settle_total(8.5, 8).outcome is Outcome.AWAY
    assert settle_total(8.5, 9).outcome is Outcome.HOME


def test_totals_over_under_convention():
    assert settle_total(9.0, 12).outcome is Outcome.HOME  # over side
    assert settle_total(9.0, 5).outcome is Outcome.AWAY  # under side
    assert settle_total(9.0, None).is_decided is False


def test_spread_integer_line_push():
    assert settle_spread(-3.0, 3, home_side=True).is_push


def test_spread_half_point_never_pushes():
    with pytest.raises(SettlementError):
        settle_spread(-1.5, 1.5, home_side=True)
    assert settle_spread(-1.5, 2, home_side=True).outcome is Outcome.HOME
    assert settle_spread(-1.5, 1, home_side=True).outcome is Outcome.AWAY


def test_spread_half_point_tie_goes_to_plus_half_side():
    # +/-0.5 tie handling: the margin never equals a half-point line, so
    # the +0.5 side always covers a level margin.
    s = settle_spread(0.5, 0, home_side=True)  # home +0.5, level game
    assert s.outcome is Outcome.HOME  # home side (the +0.5 side) covers
    s = settle_spread(0.5, 0, home_side=False)  # away +0.5, level game
    assert s.outcome is Outcome.HOME  # the offered away +0.5 side covers
    # The -0.5 side of a level game fails to cover.
    s = settle_spread(-0.5, 0, home_side=True)
    assert s.outcome is Outcome.AWAY


def test_spread_push_disallowed_is_loud_error():
    no_push = SettlementConfig(push_allowed=False, tie_allowed=False)
    with pytest.raises(SettlementError, match="push is not allowed"):
        settle_spread(-3.0, 3, home_side=True, cfg=no_push)


def test_normalize_tie_probs_keeps_tie_mass():
    h, a, t = normalize_tie_probs(0.45, 0.50, 0.05)
    assert (h, a, t) == (0.45, 0.50, 0.05)


def test_normalize_tie_probs_derived():
    h, a, t = normalize_tie_probs(0.45, 0.45)
    assert t == pytest.approx(0.10)


def test_three_way_regulation_market_never_forces_null_tie():
    """A three-way regulation market keeps its explicit tie probability."""
    # NFL-style distribution output: p_home_win + p_tie + p_away_win = 1.
    h, a, t = normalize_tie_probs(0.472, 0.510, 0.018)
    assert t == pytest.approx(0.018)
    assert h + a + t == pytest.approx(1.0)


def test_tie_triple_renormalized_pair():
    from core.markets import TieTriple
    t = TieTriple(p_home=0.45, p_tie=0.10, p_away=0.45)
    assert t.renormalized_pair() == pytest.approx((0.5, 0.5))
    assert t.decided_mass() == pytest.approx(0.90)
    with pytest.raises(SettlementError):
        TieTriple(0.6, 0.6, 0.0)


def test_sport_allows_ties_full_game_only():
    assert sport_allows_ties("nfl")
    assert not sport_allows_ties("mlb")
    assert not sport_allows_ties("nhl")  # full-game: OT/SO resolves
    assert not sport_allows_ties("nba")
