"""NHL settlement and distribution tests.

Covers: full-game two-way moneyline (OT/SO resolution), regulation
three-way moneyline with explicit TIE, integer puckline/total pushes,
half-point lines with no push, and TIE vs PUSH as separate concepts —
all through core.markets plus the NHL distribution engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.contracts import (
    FULL_GAME_NO_TIE,
    SettlementConfig,
    Outcome,
    settle_moneyline,
    settle_spread,
    settle_total,
)
from sports.nhl.adapter import NHLAdapter
from sports.nhl.models.market.distributions import (
    discrete_normal_pmf,
    game_distribution,
    margin_cdf_above,
    margin_pmf_at,
    total_probabilities,
)
from sports.nhl.config.study_config import load_nhl_study


def _regulation() -> SettlementConfig:
    return SettlementConfig(settlement_scope="regulation",
                            tie_allowed=True, push_allowed=False)


class TestFullGameMoneyline:
    """The full-game NHL moneyline is two-way: OT/SO resolves EVERY game."""

    def test_full_game_ml_has_no_tie(self):
        cfg = NHLAdapter().settlement_config("moneyline")
        assert cfg.tie_allowed is False
        assert cfg.push_allowed is False
        assert cfg is not None

    def test_full_game_settlement_home_win(self):
        res = settle_moneyline(4, 3, cfg=FULL_GAME_NO_TIE)
        assert res.outcome == Outcome.HOME
        assert res.is_decided

    def test_full_game_level_score_never_ties(self):
        # a level score with tie_allowed=False is UNDECIDED (a data
        # anomaly for the NHL full-game view) — never a fabricated TIE
        res = settle_moneyline(3, 3, cfg=FULL_GAME_NO_TIE)
        assert res.outcome != Outcome.TIE
        assert res.is_decided is False


class TestRegulationThreeWay:
    """The regulation moneyline is three-way with explicit TIE."""

    def test_regulation_ml_allows_tie(self):
        cfg = NHLAdapter().settlement_config("moneyline_regulation")
        assert cfg.tie_allowed is True
        assert cfg.settlement_scope == "regulation"

    def test_regulation_tie_is_distinct_from_full_game(self):
        # the same 3-3 raw score: the regulation market TIES (a real
        # outcome), the full-game market is undecided — TIE and the
        # full-game two-way resolution are never conflated
        reg = settle_moneyline(3, 3, cfg=_regulation())
        assert reg.outcome == Outcome.TIE
        assert reg.is_tie
        full = settle_moneyline(3, 3, cfg=FULL_GAME_NO_TIE)
        assert full.outcome != Outcome.TIE

    def test_regulation_home_and_away(self):
        assert settle_moneyline(4, 2, cfg=_regulation()).outcome \
            == Outcome.HOME
        assert settle_moneyline(2, 4, cfg=_regulation()).outcome \
            == Outcome.AWAY


class TestPucklineAndTotals:
    def test_puckline_config_push_allowed(self):
        cfg = NHLAdapter().settlement_config("puckline")
        assert cfg.push_allowed is True
        assert cfg.tie_allowed is False

    def test_integer_puckline_push(self):
        # home gives 2, wins by exactly 2 -> push
        res = settle_spread(-2.0, 2, home_side=True,
                            cfg=SettlementConfig(
                                settlement_scope="full_game",
                                tie_allowed=False, push_allowed=True))
        assert res.outcome == Outcome.PUSH
        assert res.is_push

    def test_integer_puckline_cover_and_lose(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        assert settle_spread(-2.0, 3, home_side=True, cfg=cfg).outcome \
            == Outcome.HOME
        assert settle_spread(-2.0, 1, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY

    def test_half_point_puckline_no_push(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        # favorite gives 1.5 and wins by 1 -> fails to cover
        assert settle_spread(-1.5, 1, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY
        # wins by 2 -> covers
        assert settle_spread(-1.5, 2, home_side=True, cfg=cfg).outcome \
            == Outcome.HOME

    def test_regulation_tie_is_not_a_puckline_push(self):
        # a level FINAL margin against -1.5: the side loses the line
        # (0 + (-1.5) < 0); TIE is a moneyline concept only
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        assert settle_spread(-1.5, 0, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY
        # an integer line pushes only on an EXACT landing: home giving 2
        # and the home margin landing exactly on +2 refunds
        assert settle_spread(-2.0, 2, home_side=True, cfg=cfg).outcome \
            == Outcome.PUSH
        # ...but a level final margin against -2 is a plain loss
        assert settle_spread(-2.0, 0, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY

    def test_plus_half_puckline_dog(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        # away takes +1.5, loses by 1 -> covers (side_margin -1 + 1.5 > 0)
        assert settle_spread(1.5, -1, home_side=False, cfg=cfg).outcome \
            == Outcome.HOME

    def test_integer_total_push(self):
        res = settle_total(6.0, 6, cfg=SettlementConfig(
            settlement_scope="full_game", tie_allowed=False,
            push_allowed=True))
        assert res.outcome == Outcome.PUSH
        assert res.is_push

    def test_half_point_total_no_push(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        assert settle_total(5.5, 5, cfg=cfg).outcome == Outcome.AWAY
        assert settle_total(5.5, 6, cfg=cfg).outcome == Outcome.HOME


class TestDistributionEngine:
    @pytest.fixture(scope="class")
    def study(self):
        return load_nhl_study()

    def test_pmf_sums_to_one(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(0.3, 2.5, support)
        assert np.isclose(pmf.sum(), 1.0)

    def test_three_way_sums_to_one(self, study):
        d = game_distribution(3.1, 2.6, 2.5, 1.9, study)
        total = (d["p_home_regulation"] + d["p_tie"]
                 + d["p_away_regulation"])
        assert np.isclose(total, 1.0, atol=1e-9)

    def test_full_game_two_way_sums_to_one(self, study):
        d = game_distribution(3.1, 2.6, 2.5, 1.9, study)
        assert np.isclose(d["p_home_win_derived"]
                          + d["p_away_win_derived"], 1.0, atol=1e-9)

    def test_full_game_tie_never_persists(self, study):
        # even at mu_margin == 0, the full-game view must resolve the tie
        d = game_distribution(3.0, 3.0, 2.5, 1.9, study)
        assert np.isclose(d["p_home_win_derived"]
                          + d["p_away_win_derived"], 1.0, atol=1e-9)
        # the regulation view DOES carry the tie mass
        assert d["p_tie"] > 0

    def test_integer_line_push_mass(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(0.0, 2.5, support)
        assert margin_pmf_at(pmf, support, 0.0) > 0
        # half-point line: no push possible
        assert margin_pmf_at(pmf, support, -1.5) == 0.0

    def test_cover_probability_contract(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(0.5, 2.5, support)
        # P(m > -1.5) + P(m <= -2) = 1 for integer margins
        p_cover = margin_cdf_above(pmf, support, -1.5)
        p_not = pmf[support <= -2].sum()
        assert np.isclose(p_cover + p_not, 1.0, atol=1e-9)

    def test_total_probs_sum_to_one(self, study):
        support = np.arange(0, study.market.total_pmf_max + 1)
        pmf = discrete_normal_pmf(5.8, 2.0, support)
        over, push, under = total_probabilities(pmf, support, 6.0)
        assert np.isclose(over + push + under, 1.0, atol=1e-9)
        assert push > 0  # integer total can push
        over2, push2, under2 = total_probabilities(pmf, support, 5.5)
        assert push2 == 0.0  # half total cannot push
        assert np.isclose(over2 + under2, 1.0, atol=1e-9)

    def test_grid_keys_present(self, study):
        d = game_distribution(3.0, 2.5, 2.5, 2.0, study)
        for L in study.market.spread_lines:
            assert f"p_home_cover_{L}" in d
            assert f"p_push_{L}" in d
        for L in study.market.half_stop_lines:
            label = str(L).replace(".", "_").replace("-", "m")
            assert f"p_home_cover_{label}" in d
        for U in study.market.total_lines:
            assert f"p_over_{U}" in d
