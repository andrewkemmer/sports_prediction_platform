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

from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.folds.oof import brier_score, log_loss, oof_metrics, roc_auc

#: Window constants of the rolling-Brier artifact contract (shared by all
#: four sports; the values are part of the pinned payload schema).
ROLLING_BRIER_WINDOW_DAYS = 30
ROLLING_BRIER_MIN_GAMES_PER_DAY = 1


def rolling_brier_payload(markets: pd.DataFrame | None, *,
                          window_days: int = ROLLING_BRIER_WINDOW_DAYS,
                          min_games_per_day: int =
                          ROLLING_BRIER_MIN_GAMES_PER_DAY,
                          prob_col: str = "ml_win_prob",
                          date_col: str = "game_date",
                          map_scope_note: str = "league-wide rolling brier"
                          ) -> dict:
    """League-wide rolling MONEYLINE Brier from decided OOF games.

    One implementation for all four sports (r5 §22.1): MLB's
    ``_rolling_brier_payload`` and the NNX runners' standalone
    ``rolling_brier_<date>.json`` writer all compute this payload.    The series is the artifact contract's own series: ``source_column``
    names ``home_win_prob_model`` — the moneyline probability, carried
    on each sport's markets frame as ``prob_col`` (MLB: ``ml_win_prob``;
    NFL/NHL/NBA: ``p_home_win``, the calibrated-with-fallback serving
    probability) — scored against the settled ``home_score >
    away_score`` outcome, one point per calendar day (``date_col``:
    ``game_date``; NFL's OOF frame carries ``gameday``). The per-day
    value is the canonical Brier formula (mean squared error between
    predicted probability and actual outcome —
    ``core.folds.oof.brier_score``).

    Games whose moneyline probability or final score is unavailable are
    EXCLUDED from the series (never fabricated as 0.0/0.5); days left
    with fewer than ``min_games_per_day`` usable games are counted in
    ``excluded_sparse_days``. ``history_mean_brier`` is null only when no
    day qualified — it is a contract-nullable field, not a placeholder.

    SEMANTICS — this artifact is MONEYLINE, not totals:

    * probabilities come from the moneyline model's ``home_win_prob_model``
      column (carried on the markets frame as ``ml_win_prob``);
    * outcomes come from settled results, ``home_score > away_score``;
    * scoring uses the canonical ``core.folds.oof.brier_score``
      (``mean((p - y) ** 2)``);
    * unavailable or non-finite (NaN) probability/outcome rows are EXCLUDED
      — never imputed or fabricated;
    * the payload keys, order, and dtypes match the pinned
      ``rolling_brier`` registry schema exactly.

    (MLB's totals-scoring helper
    ``compute_rolling_totals_brier`` remains DELIBERATELY unwired per
    the 7.5e-D disposition: scoring a moneyline artifact against a
    totals market would be a market-semantics regression.)
    """
    series: list[dict] = []
    excluded = 0
    if markets is not None and not markets.empty:
        required = (prob_col, "home_score", "away_score", date_col)
        if set(required) <= set(markets.columns):
            usable = markets.dropna(subset=list(required)).copy()
            usable["_day"] = pd.to_datetime(usable[date_col]).dt.date
            usable["_y_home_win"] = (
                usable["home_score"] > usable["away_score"]).astype(float)
            for day, grp in usable.groupby("_day"):
                if len(grp) < min_games_per_day:
                    excluded += 1
                    continue
                series.append({
                    "date": str(day),
                    "brier": round(float(brier_score(
                        [float(v) for v in grp[prob_col]],
                        [float(v) for v in grp["_y_home_win"]])), 5),
                    "games": int(len(grp)),
                })
            series.sort(key=lambda r: r["date"])
    n_games = int(sum(r["games"] for r in series))
    return {
        "window_days": window_days,
        "min_games_per_day": min_games_per_day,
        "source_column": "home_win_prob_model",
        "calibration": "prequential",
        "map_scope_note": map_scope_note,
        "calibrator_is_identity": False,
        "n_points": len(series),
        "n_games_total": n_games,
        "excluded_sparse_days": excluded,
        "history_mean_brier": (round(float(np.mean(
            [r["brier"] for r in series])), 5) if series else None),
        "series": series,
        "n_games_in_series": n_games,
    }


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
