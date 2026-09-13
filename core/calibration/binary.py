"""Binary probability calibration (Phase r1 §2 dedupe extraction).

The 2-parameter Platt map fit on valid OOF predictions, shared by the
NFL/NHL/NBA training modules (verified identical by digest before
extraction) and by the harness's linear-member path. Deterministic
(LBFGS, no randomness). The map is persisted at 6-decimal precision
(MLB presentation parity).

Prequential per-fold Platt fitting (fit on PRIOR folds only, applied to
the current fold) remains in the sport training modules in r1 — it is
coupled to their fold loops; the shared map here covers the serve path.
"""

from __future__ import annotations

import numpy as np

from core.training.blending import CLIP


def fit_platt(oof_p: np.ndarray, y: np.ndarray) -> dict:
    """2-parameter logistic map p -> sigmoid(a*z + b) where z = logit(p).
    Deterministic (LBFGS, no randomness). Persisted at 6-decimal precision
    (MLB presentation parity)."""
    from sklearn.linear_model import LogisticRegression
    oof_p = np.asarray(oof_p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(oof_p) & np.isfinite(y)
    oof_p, y = oof_p[ok], y[ok]
    if len(oof_p) < 2 or len(np.unique(y)) < 2:
        return {"a": None, "b": None, "n": 0}
    z = np.log(np.clip(oof_p, CLIP, 1 - CLIP)
               / (1 - np.clip(oof_p, CLIP, 1 - CLIP)))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y.astype(int))
    return {"a": round(float(lr.coef_[0][0]), 6),
            "b": round(float(lr.intercept_[0]), 6), "n": int(len(oof_p))}


def apply_platt(p: np.ndarray, cal: dict) -> np.ndarray:
    if cal is None or cal.get("a") is None:
        return np.asarray(p, dtype=float)
    p = np.asarray(p, dtype=float)
    z = np.log(np.clip(p, CLIP, 1 - CLIP) / (1 - np.clip(p, CLIP, 1 - CLIP)))
    out = 1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))
    return np.clip(out, CLIP, 1.0 - CLIP)
