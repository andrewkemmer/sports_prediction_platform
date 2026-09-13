"""Market calibration primitives (§2 layout; implemented Phase r7).

Shared, sport-agnostic PMF algebra for the joint score-distribution
market engines. Extracted in r7 from the three NNX distribution modules
(``sports/{nfl,nhl,nba}/models/market/distributions.py``) after
AST-level code-identity verification — the functions below are
byte-for-byte the implementations that shipped in 7.5, with sport
names removed from the docstrings (the sports rebind to these helpers;
their ``game_distribution`` composition — the §20 settlement-specific
tie/push conventions — stays sport-side by design).

All functions operate on an integer-support PMF pair ``(pmf, support)``
of equal length. NaN propagation is deliberate: a degenerate input
(non-finite parameters, non-positive sigma) yields NaN mass rather
than a fabricated distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def discrete_normal_pmf(mu: float, sigma: float,
                        support: np.ndarray) -> np.ndarray:
    """P(X = k) for integer support k, X ~ Normal(mu, sigma), normalized."""
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return np.full(len(support), np.nan)
    z = (support - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    pmf = pdf / pdf.sum()
    return pmf


def margin_cdf_above(pmf: np.ndarray, support: np.ndarray, L: float) -> float:
    """P(margin > L) for a real-valued threshold L over integer support.

    A margin m covers L iff m > L, i.e. m >= floor(L) + 1 for non-integer
    L, and m >= L + 1 for integer L. (Margins are integers in every
    supported sport; the contract P(m > L) is honored exactly.)"""
    if L == int(L):
        thresh = int(L) + 1
    else:
        thresh = int(np.floor(L)) + 1
    idx = support >= thresh
    return float(pmf[idx].sum()) if np.isfinite(pmf).all() else np.nan


def margin_pmf_at(pmf: np.ndarray, support: np.ndarray, L: float) -> float:
    """P(margin == L) — zero for non-integer L, PMF mass at int(L) else."""
    if L != int(L):
        return 0.0
    hit = support == int(L)
    return float(pmf[hit].sum()) if np.isfinite(pmf).all() else np.nan


def total_probabilities(pmf: np.ndarray, support: np.ndarray,
                        U: float) -> tuple[float, float, float]:
    """(P(total > U), P(total = U), P(total < U)) — sums to 1 exactly."""
    if not np.isfinite(pmf).all():
        return (np.nan, np.nan, np.nan)
    p_eq = float(pmf[support == int(U)].sum()) if U == int(U) else 0.0
    p_gt = float(pmf[support > U].sum())
    p_lt = float(pmf[support < U].sum())
    return (p_gt, p_eq, p_lt)


def pmf_median(pmf: np.ndarray, support: np.ndarray) -> float:
    """Smallest integer k with cumulative PMF >= 0.5 (the fair line)."""
    if not np.isfinite(pmf).all():
        return np.nan
    c = np.cumsum(pmf)
    idx = int(np.searchsorted(c, 0.5))
    return float(support[min(idx, len(support) - 1)])


def apply_distribution_frame(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Concatenate per-game distribution dicts onto a frame, replacing
    (never duplicating) any distribution column already present — the
    engine is the single authority for these values. ``rows`` must be
    aligned 1:1 with ``df`` (same order); the caller composes the rows
    via its sport-specific :func:`game_distribution`."""
    if len(rows) != len(df):
        raise ValueError(
            f"distribution rows ({len(rows)}) != frame rows ({len(df)})")
    dist = pd.DataFrame(rows, index=df.index)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns])
    return pd.concat([base.reset_index(drop=True),
                      dist.reset_index(drop=True)], axis=1)
