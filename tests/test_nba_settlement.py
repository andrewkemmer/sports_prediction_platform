"""NBA settlement and distribution tests.

Covers: full-game two-way moneyline (TIE structurally impossible — OT
resolves every game), integer spread/total pushes, half-point lines with
no push, and TIE vs PUSH as separate concepts — all through core.markets
plus the NBA distribution engine.
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
from sports.nba.adapter import NBAAdapter
from sports.nba.models.market.distributions import (
    discrete_normal_pmf,
    game_distribution,
    margin_cdf_above,
    margin_pmf_at,
    total_probabilities,
)
from sports.nba.config.study_config import load_nba_study


class TestFullGameMoneyline:
    """The NBA moneyline is two-way: TIE is structurally impossible."""

    def test_full_game_ml_has_no_tie(self):
        cfg = NBAAdapter().settlement_config("moneyline")
        assert cfg.tie_allowed is False
        assert cfg.push_allowed is False

    def test_full_game_settlement_home_win(self):
        res = settle_moneyline(112, 105, cfg=FULL_GAME_NO_TIE)
        assert res.outcome == Outcome.HOME
        assert res.is_decided

    def test_full_game_level_score_never_ties(self):
        # a level score with tie_allowed=False is UNDECIDED (a data
        # anomaly for the NBA — the sport cannot produce one) — never a
        # fabricated TIE
        res = settle_moneyline(105, 105, cfg=FULL_GAME_NO_TIE)
        assert res.outcome != Outcome.TIE
        assert res.is_decided is False

    def test_unknown_market_kind_rejected(self):
        with pytest.raises(KeyError):
            NBAAdapter().settlement_config("nonsense")


class TestSpreadAndTotals:
    def test_spread_config_push_allowed(self):
        cfg = NBAAdapter().settlement_config("spread")
        assert cfg.push_allowed is True
        assert cfg.tie_allowed is False
        assert cfg.settlement_scope == "full_game"

    def test_integer_spread_push(self):
        # home gives 5, wins by exactly 5 -> push
        res = settle_spread(-5.0, 5, home_side=True,
                            cfg=SettlementConfig(
                                settlement_scope="full_game",
                                tie_allowed=False, push_allowed=True))
        assert res.outcome == Outcome.PUSH
        assert res.is_push

    def test_integer_spread_cover_and_lose(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        assert settle_spread(-5.0, 8, home_side=True, cfg=cfg).outcome \
            == Outcome.HOME
        assert settle_spread(-5.0, 2, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY

    def test_half_point_spread_no_push(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        # favorite gives 4.5 and wins by 4 -> fails to cover
        assert settle_spread(-4.5, 4, home_side=True, cfg=cfg).outcome \
            == Outcome.AWAY
        # wins by 5 -> covers
        assert settle_spread(-4.5, 5, home_side=True, cfg=cfg).outcome \
            == Outcome.HOME

    def test_plus_half_spread_dog(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        # away takes +6.5, loses by 6 -> covers
        assert settle_spread(6.5, -6, home_side=False, cfg=cfg).outcome \
            == Outcome.HOME

    def test_integer_total_push(self):
        res = settle_total(222.0, 222, cfg=SettlementConfig(
            settlement_scope="full_game", tie_allowed=False,
            push_allowed=True))
        assert res.outcome == Outcome.PUSH
        assert res.is_push

    def test_half_point_total_no_push(self):
        cfg = SettlementConfig(settlement_scope="full_game",
                               tie_allowed=False, push_allowed=True)
        assert settle_total(221.5, 221, cfg=cfg).outcome == Outcome.AWAY
        assert settle_total(221.5, 222, cfg=cfg).outcome == Outcome.HOME


class TestDistributionEngine:
    @pytest.fixture(scope="class")
    def study(self):
        return load_nba_study()

    def test_pmf_sums_to_one(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(2.5, 11.0, support)
        assert np.isclose(pmf.sum(), 1.0)

    def test_full_game_two_way_sums_to_one(self, study):
        d = game_distribution(112.0, 108.0, 11.0, 18.0, study)
        assert np.isclose(d["p_home_win_derived"]
                          + d["p_away_win_derived"], 1.0, atol=1e-9)

    def test_tie_structurally_zero(self, study):
        # even at mu_margin == 0 the NBA full-game view must resolve:
        # TIE is structurally impossible and p_tie is never emitted
        d = game_distribution(110.0, 110.0, 11.0, 18.0, study)
        assert d["p_tie"] == 0.0
        assert np.isclose(d["p_home_win_derived"]
                          + d["p_away_win_derived"], 1.0, atol=1e-9)

    def test_integer_line_push_mass(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(0.0, 11.0, support)
        assert margin_pmf_at(pmf, support, 0.0) > 0
        # half-point line: no push possible
        assert margin_pmf_at(pmf, support, -4.5) == 0.0

    def test_cover_probability_contract(self, study):
        support = np.arange(-study.market.margin_pmf_max,
                            study.market.margin_pmf_max + 1)
        pmf = discrete_normal_pmf(2.5, 11.0, support)
        # half-point line: P(m > -4.5) + P(m <= -5) = 1 exactly (no push)
        p_cover = margin_cdf_above(pmf, support, -4.5)
        p_not = pmf[support <= -5].sum()
        assert np.isclose(p_cover + p_not, 1.0, atol=1e-9)
        # integer line: cover + push + lose = 1 exactly
        p_cover5 = margin_cdf_above(pmf, support, -5.0)
        p_push5 = margin_pmf_at(pmf, support, -5.0)
        p_lose5 = pmf[support <= -6].sum()
        assert np.isclose(p_cover5 + p_push5 + p_lose5, 1.0, atol=1e-9)

    def test_total_probs_sum_to_one(self, study):
        support = np.arange(0, study.market.total_pmf_max + 1)
        pmf = discrete_normal_pmf(225.0, 18.0, support)
        over, push, under = total_probabilities(pmf, support, 222.0)
        assert np.isclose(over + push + under, 1.0, atol=1e-9)
        assert push > 0  # integer total can push
        over2, push2, under2 = total_probabilities(pmf, support, 221.5)
        assert push2 == 0.0  # half total cannot push
        assert np.isclose(over2 + under2, 1.0, atol=1e-9)

    def test_grid_keys_present(self, study):
        d = game_distribution(112.0, 107.0, 11.0, 18.0, study)
        for L in study.market.spread_lines:
            assert f"p_home_cover_{L}" in d
            assert f"p_push_{L}" in d
        for U in study.market.total_lines:
            assert f"p_over_{U}" in d

    def test_market_sized_to_sport(self, study):
        # NBA grids are basketball-sized, never copied from NHL/NFL
        assert study.market.total_grid[0] > 100
        assert study.market.margin_pmf_max >= 40
        assert study.market.p_tie_max == 0.0
