"""Shared ensemble blending (Phase r1 §2 dedupe extraction).

Behavior-identical to the per-sport ``_blend`` and ``_adaptive_weights``
implementations in ``sports/{nfl,nhl,nba}/training.py`` (verified
identical by digest before extraction; the sport moneyline trainers
rebind to these helpers and the sport-local definitions are removed).

Weight selection is derived ONLY from OOF rows; blending skips NaN
members per row; the result is clipped to ``[CLIP, 1 - CLIP]`` so
calibrated outputs never saturate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Probability clipping bound shared by blending and calibration
#: (the NNX training modules' value; MLB's own path clips at 1e-6).
CLIP = 1e-7

#: Reference ensemble prior used when a study carries no explicit weights
#: (the Phase-2 five-member convention).
DEFAULT_ENSEMBLE_PRIOR = {
    "xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.20,
    "randomforest": 0.15, "mlp": 0.15,
}


def adaptive_weights(oof: pd.DataFrame, study) -> dict[str, float]:
    """Softmax over pooled OOF AUC edges (study metric). Deterministic;
    derived ONLY from OOF rows. Falls back to study priors when OOF is
    unusable."""
    prior = study.training.ensemble_weights or dict(DEFAULT_ENSEMBLE_PRIOR)
    if oof is None or not len(oof):
        return prior
    y = oof["home_win"].astype(int).to_numpy()
    if len(y) < 2 or len(np.unique(y)) < 2:
        return prior
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return prior
    edges: dict[str, float] = {}
    for name in study.training.ensemble_members:
        col = f"p_{name}"
        if col not in oof.columns:
            return prior
        p = oof[col].to_numpy(dtype=float)
        ok = np.isfinite(p)
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
            return prior
        edges[name] = roc_auc_score(y[ok], p[ok])
    base_auc = max(edges.values())
    T = study.training.adaptive_weight_temperature
    exp = {n: np.exp((a - base_auc) / T) for n, a in edges.items()}
    raw = {n: max(study.training.adaptive_weight_floor,
                  min(study.training.adaptive_weight_cap,
                      e / sum(exp.values())))
           for n, e in exp.items()}
    total = sum(raw.values())
    return {n: w / total for n, w in raw.items()}


def blend_probabilities(oof: pd.DataFrame, weights: dict[str, float],
                        study) -> np.ndarray:
    """Weighted ensemble probability over OOF rows; NaN members skipped
    per row. The former per-sport ``_blend``."""
    cols = [f"p_{n}" for n in study.training.ensemble_members
            if f"p_{n}" in oof.columns]
    P = oof[cols].to_numpy(dtype=float)
    w = np.array([weights.get(c[2:], 0.0) for c in cols])
    mask = np.isfinite(P)
    wv = np.where(mask, w[None, :], 0.0)
    wsum = wv.sum(axis=1)
    out = np.divide((P * wv).sum(axis=1), wsum,
                    out=np.full(len(P), np.nan), where=wsum > 0)
    return np.clip(out, CLIP, 1.0 - CLIP)


def blend_member_predictions(member_p: dict[str, np.ndarray | None],
                             weights: dict[str, float],
                             study) -> np.ndarray:
    """Weighted ensemble probability over the served slate from per-member
    prediction arrays (``None`` = member unavailable, skipped). The
    per-row weighting logic is identical to :func:`blend_probabilities`;
    the former per-sport ``predict_slate`` blend block."""
    cols = [n for n in study.training.ensemble_members
            if member_p.get(n) is not None]
    if not cols:
        return np.full(0, np.nan)
    P = np.column_stack([member_p[n] for n in cols])
    w = np.array([weights.get(n, 0.0) for n in cols])
    mask = np.isfinite(P)
    wv = np.where(mask, w[None, :], 0.0)
    wsum = wv.sum(axis=1)
    out = np.divide((P * wv).sum(axis=1), wsum,
                    out=np.full(len(P), np.nan), where=wsum > 0)
    return np.clip(out, CLIP, 1.0 - CLIP)
