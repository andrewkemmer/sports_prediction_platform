"""OOF evaluation for the production NBA moneyline.

Every metric is computed on out-of-fold predictions only. No full-history
correlations, no in-sample importance, no future outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return np.nan
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        err += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(err)


def binary_metrics(p, y) -> dict:
    """AUC / logloss / Brier / ECE over valid finite pairs."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 2 or len(np.unique(y)) < 2:
        return {"n": int(len(p)), "auc": np.nan, "logloss": np.nan,
                "brier": np.nan, "ece": np.nan}
    return {
        "n": int(len(p)),
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(p, y),
    }


def calibration_buckets(p, y, n_bins: int = 10) -> list[dict]:
    """Reliability buckets (shared favored-team presentation contract).

    Each game contributes ONE point at ``max(p, 1-p)`` in [0.5, 1], labeled
    by whether the favorite won — information-equivalent to the home-side
    view and matching how the model is consumed. Bucket labels use the
    en-dash convention. ``gap`` = mean_predicted − mean_actual (>0 =
    overconfident).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    fav_prob = np.maximum(p, 1.0 - p)
    fav_won = np.where(p >= 0.5, y, 1.0 - y)
    half = max(n_bins // 2, 1)
    bin_edges = np.linspace(0.5, 1.0, half + 1)
    rows = []
    for i in range(len(bin_edges) - 1):
        m = (fav_prob >= bin_edges[i]) & (fav_prob < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            m |= fav_prob == bin_edges[i + 1]
        if not m.any():
            continue
        mean_pred, mean_actual = (float(fav_prob[m].mean()),
                                  float(fav_won[m].mean()))
        rows.append({
            "bucket": f"{bin_edges[i] * 100:.0f}–"
                      f"{bin_edges[i + 1] * 100:.0f}%",
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_actual, 4),
            "count": int(m.sum()),
            "gap": round(mean_pred - mean_actual, 4),
        })
    return rows
