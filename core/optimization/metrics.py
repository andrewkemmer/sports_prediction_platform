"""Pooled OOF metrics and period-stability checks.

All metrics evaluate on out-of-sample (OOF / sealed) rows ONLY — never
training rows. Uses the same metric definitions as production
(``core.oof``) so optimizer numbers are directly comparable to the
incumbent's.

Binary scopes (moneyline) use probability metrics (logloss/AUC/Brier/
ECE). Market scopes optimize the score-distribution regressand (margin)
and report regression metrics (RMSE/MAE) — the incumbent comparison is
within-scope only, never across scopes.
"""

from __future__ import annotations

import numpy as np

from core.oof import brier_score, log_loss, oof_metrics, roc_auc


def pooled_metrics(probs, outcomes) -> dict:
    """Pooled OOF metrics: logloss (primary objective), AUC (secondary),
    Brier, ECE (10 equal-count bins). Expects 1-D float arrays of equal
    length; rows with non-finite probs are excluded (counted in n)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    base = oof_metrics(p.tolist(), y.tolist())
    # oof_metrics provides auc/brier/logloss; add ECE (calibration error).
    ece = _ece(p, y)
    return {"n": int(len(p)), "logloss": base["logloss"],
            "auc": base["auc"], "brier": base["brier"], "ece": ece}


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def period_stability(fold_logloss: list[float],
                     n_periods: int = 3) -> dict:
    """Chronological fold stability: compares the mean logloss of the
    LAST third of folds against the FIRST third. A material late-period
    degradation (late − early > tolerance) flags an unstable candidate.
    Never uses training rows."""
    vals = [v for v in fold_logloss if np.isfinite(v)]
    if len(vals) < n_periods * 2:
        return {"ok": True, "early": None, "late": None, "delta": None,
                "note": f"insufficient folds ({len(vals)}) — not assessed"}
    k = max(1, len(vals) // n_periods)
    early = float(np.mean(vals[:k]))
    late = float(np.mean(vals[-k:]))
    delta = late - early
    return {"ok": True, "early": early, "late": late, "delta": delta,
            "note": "delta = late_mean − early_mean (sign: degradation if > 0)"}


def brier(probs, outcomes) -> float:
    return float(brier_score(list(probs), list(outcomes)))


def auc(probs, outcomes) -> float | None:
    return roc_auc(list(probs), list(outcomes))


def logloss(probs, outcomes) -> float:
    return float(log_loss(list(probs), list(outcomes)))


# ---------------------------------------------------------------------------
# Regression metrics (market / score-distribution scopes)
# ---------------------------------------------------------------------------
def regression_metrics(preds, outcomes) -> dict:
    """Pooled OOF regression metrics for the market scope: RMSE
    (primary objective), MAE, and bias. NaN rows are excluded and
    counted. Never used for the binary moneyline scope."""
    p = np.asarray(preds, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan")}
    err = p - y
    return {
        "n": int(len(p)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def regression_period_stability(fold_rmse: list[float],
                                n_periods: int = 3) -> dict:
    """Chronological fold RMSE stability for the market scope: the LAST
    third of folds against the FIRST third. Material late-period
    degradation (late − early > tolerance) flags an unstable candidate."""
    vals = [v for v in fold_rmse if np.isfinite(v)]
    if len(vals) < n_periods * 2:
        return {"ok": True, "early": None, "late": None, "delta": None,
                "note": f"insufficient folds ({len(vals)}) — not assessed"}
    k = max(1, len(vals) // n_periods)
    early = float(np.mean(vals[:k]))
    late = float(np.mean(vals[-k:]))
    delta = late - early
    return {"ok": True, "early": early, "late": late, "delta": delta,
            "note": "delta = late_mean − early_mean (sign: degradation if > 0)"}


# ---------------------------------------------------------------------------
# Regression metrics (market / score-distribution scopes)
# ---------------------------------------------------------------------------
def regression_metrics(preds, outcomes) -> dict:
    """Pooled OOF regression metrics for the market scope: RMSE
    (primary objective), MAE, and bias. NaN rows are excluded and
    counted. Never used for the binary moneyline scope."""
    p = np.asarray(preds, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan")}
    err = p - y
    return {
        "n": int(len(p)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def regression_period_stability(fold_rmse: list[float],
                                n_periods: int = 3) -> dict:
    """Chronological fold RMSE stability for the market scope: the LAST
    third of folds against the FIRST third. Material late-period
    degradation (late − early > tolerance) flags an unstable candidate."""
    vals = [v for v in fold_rmse if np.isfinite(v)]
    if len(vals) < n_periods * 2:
        return {"ok": True, "early": None, "late": None, "delta": None,
                "note": f"insufficient folds ({len(vals)}) — not assessed"}
    k = max(1, len(vals) // n_periods)
    early = float(np.mean(vals[:k]))
    late = float(np.mean(vals[-k:]))
    delta = late - early
    return {"ok": True, "early": early, "late": late, "delta": delta,
            "note": "delta = late_mean − early_mean (sign: degradation if > 0)"}
