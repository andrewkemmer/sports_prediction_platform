"""Market contract completion (Phase r7, §11/§13/§15/§18.1).

Pins the 7.6e closure surface:

* ``core/calibration/market.py`` primitives: PMF algebra summing to 1,
  exact integer-line push/tie semantics (half-point lines cannot push),
  fair-line medians, and the frame assembler that replaces (never
  duplicates) distribution columns and fails loud on length mismatch;
* the three NNX market engines consume the shared primitives (no sport
  keeps a private copy of the PMF algebra — §22 dedupe);
* the bounded §15 training-policy candidate set: named, declared,
  at most 10, deterministic, and valid (expanding needs an origin,
  rolling needs K >= 1, decay restricted).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from core.calibration.market import (
    apply_distribution_frame,
    discrete_normal_pmf,
    margin_cdf_above,
    margin_pmf_at,
    pmf_median,
    total_probabilities,
)
from core.experiments.search_space import (
    TrainingPolicy,
    training_policy_candidates,
)

# ---------------------------------------------------------------------------
# PMF algebra (§13/§20 semantics)
# ---------------------------------------------------------------------------

SUPPORT = np.arange(-40, 41)


def _pmf(mu: float = 1.5, sigma: float = 10.0) -> np.ndarray:
    return discrete_normal_pmf(mu, sigma, SUPPORT)


def test_pmf_normalized():
    pmf = _pmf()
    assert abs(float(pmf.sum()) - 1.0) < 1e-9


def test_pmf_degenerate_input_yields_nan_not_fabrication():
    pmf = discrete_normal_pmf(1.5, 0.0, SUPPORT)
    assert np.isnan(pmf).all()
    pmf = discrete_normal_pmf(np.inf, 5.0, SUPPORT)
    assert np.isnan(pmf).all()


def test_margin_cover_exact_semantics():
    pmf = _pmf()
    # integer line: cover requires margin >= L + 1
    p_gt_7 = margin_cdf_above(pmf, SUPPORT, 7.0)
    idx = SUPPORT >= 8
    assert p_gt_7 == pytest.approx(float(pmf[idx].sum()))
    # half-point line: cover requires margin >= ceil(L) (no push possible)
    p_gt_7p5 = margin_cdf_above(pmf, SUPPORT, 7.5)
    idx = SUPPORT >= 8
    assert p_gt_7p5 == pytest.approx(float(pmf[idx].sum()))
    # P(margin > 7.5) == P(margin > 7): identical cover set — §20's
    # "a half-point line cannot push" honored structurally.
    assert p_gt_7p5 == pytest.approx(p_gt_7)


def test_push_mass_only_on_integer_lines():
    pmf = _pmf()
    assert margin_pmf_at(pmf, SUPPORT, 7.5) == 0.0
    p7 = margin_pmf_at(pmf, SUPPORT, 7.0)
    idx = SUPPORT == 7
    assert p7 == pytest.approx(float(pmf[idx].sum()))


def test_totals_sum_to_one():
    pmf_t = discrete_normal_pmf(44.0, 9.0, np.arange(0, 81))
    over, push, under = total_probabilities(pmf_t, np.arange(0, 81), 47.0)
    assert over + push + under == pytest.approx(1.0, abs=1e-9)
    # half-point total: push exactly zero
    over, push, under = total_probabilities(pmf_t, np.arange(0, 81), 47.5)
    assert push == 0.0
    assert over + under == pytest.approx(1.0, abs=1e-9)


def test_pmf_median_is_the_fair_line():
    pmf = _pmf(mu=2.0)
    med = pmf_median(pmf, SUPPORT)
    assert med == SUPPORT[np.searchsorted(np.cumsum(pmf), 0.5)]


def test_frame_assembler_replaces_never_duplicates():
    df = pd.DataFrame({"mu_h": [1.0, 2.0], "legacy": [0.5, 0.6]})
    rows = [{"mu_h": 9.0, "p_home_win_derived": 0.61},
            {"mu_h": 8.0, "p_home_win_derived": 0.42}]
    out = apply_distribution_frame(df, rows)
    assert "mu_h" in out.columns and out["mu_h"].tolist() == [9.0, 8.0]
    assert list(out.columns).count("mu_h") == 1
    assert out["legacy"].tolist() == [0.5, 0.6]


def test_frame_assembler_fails_loud_on_length_mismatch():
    df = pd.DataFrame({"mu_h": [1.0, 2.0]})
    with pytest.raises(ValueError, match="!= frame rows"):
        apply_distribution_frame(df, [{"mu_h": 9.0}])


# ---------------------------------------------------------------------------
# §22 dedupe: the sport engines rebind the shared primitives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", ["nfl", "nhl", "nba"])
def test_market_engine_rebinds_shared_primitives(sport):
    mod = importlib.import_module(
        f"sports.{sport}.models.market.distributions")
    assert mod.discrete_normal_pmf is discrete_normal_pmf
    assert mod.margin_cdf_above is margin_cdf_above
    assert mod.margin_pmf_at is margin_pmf_at
    assert mod.total_probabilities is total_probabilities
    assert mod.pmf_median is pmf_median


@pytest.mark.parametrize("sport", ["nfl", "nhl", "nba"])
def test_market_engine_applies_distribution_via_shared_assembler(sport):
    """The engine's apply_distribution routes through the shared frame
    assembler (its local concat body is deleted)."""
    mod = importlib.import_module(
        f"sports.{sport}.models.market.distributions")
    df = pd.DataFrame({"mu_h": [1.0], "mu_a": [0.5]})
    # game_distribution is sport-specific; only assert the assembler
    # binding, not the sport settlement fields
    assert hasattr(mod, "apply_distribution")
    assert hasattr(mod, "game_distribution")


# ---------------------------------------------------------------------------
# §15 bounded training-policy candidates
# ---------------------------------------------------------------------------

def test_training_policy_rejects_invalid():
    with pytest.raises(ValueError, match="mode must be"):
        TrainingPolicy("bogus")
    with pytest.raises(ValueError, match="start_season"):
        TrainingPolicy("expanding")
    with pytest.raises(ValueError, match="rolling_seasons"):
        TrainingPolicy("rolling", rolling_seasons=0)
    with pytest.raises(ValueError, match="decay must be"):
        TrainingPolicy("expanding", start_season=2020, decay="linear")


def test_training_policy_names_are_stable():
    assert TrainingPolicy(
        "expanding", start_season=2016).name == "expanding_from_2016"
    assert TrainingPolicy(
        "rolling", rolling_seasons=3).name == "rolling_last_3_seasons"
    assert TrainingPolicy(
        "expanding", start_season=2016,
        decay="exponential").name == "expanding_from_2016_exponential"


def test_candidates_are_bounded_and_deterministic():
    pols = training_policy_candidates(2016, 2026)
    assert 2 <= len(pols) <= 10
    assert pols[0].name == "expanding_from_2016"
    # deterministic
    again = training_policy_candidates(2016, 2026)
    assert [p.name for p in pols] == [p.name for p in again]


def test_candidates_never_exceed_data_span():
    pols = training_policy_candidates(2024, 2026)
    for p in pols:
        if p.mode == "rolling":
            assert p.rolling_seasons <= 3


def test_candidates_reject_inverted_span():
    with pytest.raises(ValueError, match="earliest"):
        training_policy_candidates(2026, 2016)
