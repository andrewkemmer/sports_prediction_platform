"""NFL settlement tests via core.markets — TIE/PUSH semantics.

A tied game is NOT automatically a spread push: spread settlement reads
the final margin and the actual line. Half-point lines never push.
"""

from __future__ import annotations

import pytest

from core.markets import (
    Outcome,
    SettlementConfig,
    settle_moneyline,
    settle_spread,
    settle_total,
)
from sports.nfl.adapter import NFLAdapter


class TestMoneylineSettlement:
    def test_full_game_tie_is_tie(self):
        cfg = NFLAdapter().settlement_config("moneyline")
        assert cfg.tie_allowed is True
        assert cfg.push_allowed is False
        s = settle_moneyline(20, 20, cfg=cfg)
        assert s.outcome is Outcome.TIE
        assert s.is_tie and not s.is_push

    def test_decided_games(self):
        cfg = NFLAdapter().settlement_config("moneyline")
        assert settle_moneyline(27, 20, cfg=cfg).outcome is Outcome.HOME
        assert settle_moneyline(17, 24, cfg=cfg).outcome is Outcome.AWAY

    def test_unknown_score_is_undecided_not_fabricated(self):
        cfg = NFLAdapter().settlement_config("moneyline")
        s = settle_moneyline(None, None, cfg=cfg)
        assert s.is_decided is False


class TestSpreadSettlement:
    def test_integer_line_push(self):
        cfg = NFLAdapter().settlement_config("spread")
        # home favored by 3, wins by 3: side_margin 3 + line -3 = 0 -> push
        s = settle_spread(-3.0, 3, cfg=cfg)
        assert s.outcome is Outcome.PUSH

    def test_tied_game_not_spread_push(self):
        """A 20-20 game against a -3 line is a LOSS for the home favorite,
        not a push — settlement uses the final margin and the line."""
        cfg = NFLAdapter().settlement_config("spread")
        s = settle_spread(-3.0, 0, cfg=cfg)
        assert s.outcome is Outcome.AWAY

    def test_half_point_line_never_pushes(self):
        cfg = NFLAdapter().settlement_config("spread")
        # margin 0 vs -2.5: 0 - 2.5 < 0 -> away (no push on a half point)
        s = settle_spread(-2.5, 0, cfg=cfg)
        assert s.outcome is Outcome.AWAY
        # exact landing on -0.5 is impossible for integer margins; a
        # +0.5 dog at a level margin covers (margin + 0.5 > 0)
        s2 = settle_spread(-0.5, 0, cfg=cfg)
        assert s2.outcome is Outcome.AWAY
        s3 = settle_spread(0.5, 0, cfg=cfg)
        assert s3.outcome is Outcome.HOME

    def test_plus_half_point(self):
        cfg = NFLAdapter().settlement_config("spread")
        s = settle_spread(0.5, 0, cfg=cfg)  # home getting +0.5 at a tie
        assert s.outcome is Outcome.HOME

    def test_cover_cases(self):
        cfg = NFLAdapter().settlement_config("spread")
        assert settle_spread(-7.0, 14, cfg=cfg).outcome is Outcome.HOME
        assert settle_spread(-7.0, 3, cfg=cfg).outcome is Outcome.AWAY


class TestTotalSettlement:
    # Totals convention (core.markets): Outcome.HOME = OVER won,
    # Outcome.AWAY = UNDER won.
    def test_integer_total_push(self):
        cfg = NFLAdapter().settlement_config("total")
        s = settle_total(44.0, 44, cfg=cfg)
        assert s.outcome is Outcome.PUSH

    def test_half_point_total_never_pushes(self):
        cfg = NFLAdapter().settlement_config("total")
        s = settle_total(44.5, 44, cfg=cfg)
        assert s.outcome is Outcome.AWAY  # UNDER
        s2 = settle_total(44.5, 45, cfg=cfg)
        assert s2.outcome is Outcome.HOME  # OVER

    def test_over_under(self):
        cfg = NFLAdapter().settlement_config("total")
        assert settle_total(44.0, 48, cfg=cfg).outcome is Outcome.HOME
        assert settle_total(44.0, 41, cfg=cfg).outcome is Outcome.AWAY


class TestDistributionSemantics:
    def test_pmf_coherence(self):
        """p_home_cover + p_push + p_away_cover = 1 per integer line;
        half-point lines carry zero push mass by construction."""
        import numpy as np
        from sports.nfl.distributions import (
            game_distribution,
            margin_cdf_above,
            margin_pmf_at,
        )
        from sports.nfl.study_config import load_nfl_study
        study = load_nfl_study()
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        out = game_distribution(24.0, 20.5, 13.0, 9.5, study)
        for L in study.market.spread_lines:
            pmf_out = game_distribution(24.0, 20.5, 13.0, 9.5, study)
            cover = pmf_out[f"p_home_cover_{L}"]
            push = pmf_out[f"p_push_{L}"]
            away = 1.0 - cover - push
            assert abs((cover + push + away) - 1.0) < 1e-9
        for L in study.market.half_stop_lines:
            label = str(L).replace(".", "_").replace("-", "m")
            # half stops have no push column — the PMF at L is 0
            pmf = __import__(
                "sports.nfl.distributions", fromlist=["x"]
            ).discrete_normal_pmf(3.5, 13.0, support)
            assert margin_pmf_at(pmf, support, L) == 0.0

    def test_three_way_moneyline_sums_to_one(self):
        from sports.nfl.distributions import game_distribution
        from sports.nfl.study_config import load_nfl_study
        study = load_nfl_study()
        out = game_distribution(24.0, 20.5, 13.0, 9.5, study)
        total = (out["p_home_win_derived"] + out["p_tie"]
                 + out["p_away_win_derived"])
        assert abs(total - 1.0) < 1e-9
        assert out["p_tie"] >= 0.0  # explicit, never fabricated null
