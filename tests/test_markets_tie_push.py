"""Tests for TIE/PUSH distinction and configurable settlement (core.markets).

Phase 1 review decision 1/2: TIE (game has no winner) is distinct from
PUSH (wager refunded on an integer line), and integer settlement is
universally configurable — not totals-only.
"""

import pytest

from core.markets import (
    FULL_GAME_NO_TIE,
    FULL_GAME_TIE_ALLOWED,
    REGULATION_TIE_ALLOWED,
    Outcome,
    SettlementConfig,
    SettlementError,
    normalize_tie_probs,
    settle_moneyline,
    settle_spread,
    settle_total,
    sport_allows_ties,
)


# ---------------------------------------------------------------------------
# SettlementConfig validation
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


# ---------------------------------------------------------------------------
# Moneyline: NFL full-game tie vs NHL/MLB no-tie
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NHL regulation tie vs full-game moneyline resolved by OT/SO
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Totals: over/push/under (integer push, half-point error)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Spreads: integer push / half-point cover / tie handling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Three-way probability normalization: p_tie is first-class
# ---------------------------------------------------------------------------


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
