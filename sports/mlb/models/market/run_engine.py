"""MLB run engine: per-side run models + NB Monte-Carlo market pricing.

Faithful Phase-2 port of the reference repo's production engine, reduced to
the frozen production view:

* per-side LightGBM Poisson models on the FROZEN 53-column λ view
  (levels + environment + restored matchup gaps) — ``RUN_LAMBDA_VIEW_FROZEN``
  from the reference 2026-08-30 keep-list restore, invariant to moneyline
  feature-list changes;
* walk-forward OOF on the moneyline pipeline's folds (sports.mlb.training);
* α(λ) dispersion curves fitted on PRE-HOLDOUT OOF only (sealed 21-day
  holdout convention);
* Monte Carlo pricing from INDEPENDENT NB(λ, α) marginals over the full
  line grid — identical seed ⇒ identical output (determinism contract);
* the structural home one-run adjustment (tie mass resolved into ±1
  home-weighted), NOT a proportional no-tie renormalization.

Settlement semantics (Phase 1 core.markets contract): MLB full-game markets
are no-tie; integer totals push; half-point totals/spreads cannot push;
the per-line run-line 3-way split sums to 1.0 exactly. p_push and p_tie
stay explicit columns — no fabricated tie mass for MLB.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from sports.mlb.features.registry import (
    MARKET_FEATURE_COLS,
)
from core.folds import walk_forward_splits

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MARKET_SEED = 42
MC_DRAWS = 10_000
MC_DRAWS_TAIL = 25_000
MC_SE_TARGET = 0.0025

TOTAL_LINE_GRID = [round(6.5 + 0.5 * i, 1) for i in range(13)]  # 6.5 … 12.5
RUN_LINE_GRID = [0.5, 1.5, 2.5, 3.5]  # home-favorite margins (−0.5 … −3.5)
RUN_LINE_GRID_FULL = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # −1 … −4 by 0.5
TOTAL_REF_LINES = [7.5, 8.5, 9.5]
RUN_REF_LINES = [1.5, 2.5]
RUN_LINE_GRID_WHOLE = [1.0, 2.0, 3.0, 4.0]

HOLDOUT_DAYS = 21
MARGIN_PLUS1_HOME_SHARE = 0.744
ALPHA_CAP = 1.2
ALPHA_MIN_BIN = 250
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 20

RUN_LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 40,
    "min_gain_to_split": 0.5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Phase 3.5b standalone ENVIRONMENT-LEVEL features appended when present.
RUN_LEVEL_ENV_FEATURES = (
    "park_wind_factor", "air_density_level", "park_factor_slug",
    "dome_is_neutral_game",
)

# Persisted Phase-1 contract: fold_idx/game_id ride along internally but
# stay out of the CSV schema.
OOF_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
               "away_expected_runs", "home_score", "away_score"]

assert len(MARKET_FEATURE_COLS) == 53


# ---------------------------------------------------------------------------
# Feature-view derivation
# ---------------------------------------------------------------------------
def split_side_view(run_features: list[str] | tuple[str, ...],
                    side: str) -> tuple[list[str], list[str]]:
    """Split run features into (side_view, env_shared) for 'home'/'away'.

    Side columns: ``*_home`` / ``home_*`` (symmetrically for away). Shared
    environment = everything belonging to NEITHER side — never the
    opponent's levels.
    """
    other = "away" if side == "home" else "home"
    side_cols = [f for f in run_features
                 if f.endswith(f"_{side}") or f.startswith(f"{side}_")]
    other_cols = {f for f in run_features
                  if f.endswith(f"_{other}") or f.startswith(f"{other}_")}
    env_cols = [f for f in run_features
                if f not in side_cols and f not in other_cols]
    return side_cols, env_cols


def build_side_frame(games: pd.DataFrame, side: str,
                     run_features: list[str] | None = None,
                     include_level_env: bool = True,
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Materialize the side's model frame (levels + environment).

    Defaults to the FROZEN λ view (not re-derived from the live moneyline
    list). Missing env-level columns log a warning; missing frozen-view
    columns surface as NaN (LightGBM routes NaN natively).
    """
    feats = list(run_features) if run_features is not None \
        else list(MARKET_FEATURE_COLS)
    if include_level_env:
        present = [c for c in RUN_LEVEL_ENV_FEATURES if c in games.columns]
        missing = [c for c in RUN_LEVEL_ENV_FEATURES if c not in games.columns]
        if missing:
            logger.warning(
                "Run engine: %d/%d env-level feature columns absent from "
                "the frame: %s", len(missing), len(RUN_LEVEL_ENV_FEATURES),
                missing)
        feats = feats + present
    side_cols, env_cols = split_side_view(feats, side)
    cols = side_cols + env_cols
    frame = games.reindex(columns=cols).astype(float)
    return frame, cols


# ---------------------------------------------------------------------------
# Model fitting + diagnostics
# ---------------------------------------------------------------------------
def _fit_side_model(params: dict, tr_frame: pd.DataFrame, y_tr: np.ndarray,
                    va_frame: pd.DataFrame, y_va: np.ndarray,
                    ) -> tuple[Any, np.ndarray, int]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(**params)
    model.fit(
        tr_frame, y_tr,
        eval_set=[(va_frame, y_va)],
        eval_metric="poisson",
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   log_evaluation(0)],
    )
    best = int(getattr(model, "best_iteration_", 0) or MAX_ROUNDS)
    lam = model.predict(va_frame, num_iteration=best)
    return model, np.clip(np.asarray(lam, float), 1e-6, None), best


def poisson_deviance(y: np.ndarray, lam: np.ndarray) -> float:
    y = np.asarray(y, float)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)
    term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return float(2.0 * np.sum(term - (y - lam)))


def dispersion_ratio(y: np.ndarray, lam: np.ndarray) -> float:
    y = np.asarray(y, float)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)
    resid = (y - lam) ** 2 / lam
    dof = max(len(y) - 1, 1)
    return float(resid.sum() / dof)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() * abs(y[m].mean() - p[m].mean())
    return round(float(total / len(y)), 5)


def prequential_calibrate(y: np.ndarray, p: np.ndarray,
                          fold_idx: np.ndarray) -> np.ndarray:
    """Fold-by-fold isotonic-ish prequential calibration (expanding pool).

    Each fold's predictions are recalibrated against ALL prior folds'
    outcomes via a simple Beta-style shrink to the prior pool's rate spread.
    Deterministic; first fold (no history) passes through raw.
    """
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = p.copy()
    folds = np.unique(fold_idx)
    for i, f in enumerate(folds):
        if i == 0:
            continue
        prior = fold_idx < f
        cur = fold_idx == f
        if not prior.any() or not cur.any():
            continue
        y_prior, p_prior = y[prior], p[prior]
        # Reliability fit on the prior pool: bucketed correction.
        edges = np.linspace(0.0, 1.0, 11)
        idx = np.clip(np.digitize(p_prior, edges[1:-1]), 0, 9)
        cur_idx = np.clip(np.digitize(p[cur], edges[1:-1]), 0, 9)
        corr = np.empty(int(cur.sum()))
        for b in range(10):
            m = idx == b
            cm = cur_idx == b
            if m.sum() >= 5 and cm.any():
                corr[cm] = y_prior[m].mean()
            elif cm.any():
                corr[cm] = y_prior.mean()
        # Shrink toward the raw prediction when the bucket correction is
        # thin — deterministic, no randomness.
        weight = 0.5
        out[cur] = weight * corr + (1 - weight) * p[cur]
    return np.clip(out, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# NB machinery + alpha(lambda) curves
# ---------------------------------------------------------------------------
def fit_alpha(y: np.ndarray, lam: np.ndarray) -> float:
    """Method-of-moments NB dispersion: var = λ + α·λ² ⇒ α = (s²−λ̄)/λ̄²."""
    y, lam = np.asarray(y, float), np.clip(np.asarray(lam, float), 1e-6, None)
    mean_y, mean_lam = y.mean(), lam.mean()
    var_y = y.var(ddof=0)
    alpha = (var_y - mean_lam) / max(mean_lam ** 2, 1e-9)
    return float(np.clip(alpha, 1e-3, ALPHA_CAP))


def _nb_size_prob(mu: np.ndarray | float,
                  alpha: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    alpha = np.clip(np.asarray(alpha, float), 1e-6, None)
    n = 1.0 / alpha          # NB size
    p = n / (n + mu)         # NB success prob
    return n, p


def nb_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    n, p = _nb_size_prob(mu, alpha)
    from scipy.stats import nbinom
    return nbinom.pmf(np.asarray(k), n, p)


def alpha_bins(y: np.ndarray, lam: np.ndarray,
               min_count: int = ALPHA_MIN_BIN) -> list[dict]:
    """Sextile bins of λ; per-bin method-of-moments α."""
    lam = np.clip(np.asarray(lam, float), 1e-6, None)
    y = np.asarray(y, float)
    edges = np.quantile(lam, [i / 6 for i in range(1, 6)])
    bins = []
    for b in range(6):
        lo = -np.inf if b == 0 else edges[b - 1]
        hi = np.inf if b == 5 else edges[b]
        m = (lam > lo) & (lam <= hi) if b > 0 else (lam <= hi)
        if m.sum() < min_count:
            continue
        ys, ls = y[m], lam[m]
        alpha = (ys.var(ddof=0) - ls.mean()) / max(ls.mean() ** 2, 1e-9)
        bins.append({
            "lo": None if b == 0 else round(float(lo), 4),
            "hi": None if b == 5 else round(float(hi), 4),
            "n": int(m.sum()),
            "lam_mean": round(float(ls.mean()), 4),
            "alpha": round(float(np.clip(alpha, 1e-3, ALPHA_CAP)), 4),
        })
    return bins


def _bin_direction(lams: list[float], alphas: list[float]) -> int:
    if len(lams) < 2:
        return 0
    if all(alphas[i + 1] >= alphas[i] for i in range(len(alphas) - 1)):
        return 1
    if all(alphas[i + 1] <= alphas[i] for i in range(len(alphas) - 1)):
        return -1
    return 0


def _fit_curve_piecewise(bins: list[dict]) -> dict:
    mids = [((b["lo"] if b["lo"] is not None else b["lam_mean"] * 0.8)
             + (b["hi"] if b["hi"] is not None else b["lam_mean"] * 1.2)) / 2
            for b in bins]
    alphas = [b["alpha"] for b in bins]
    return {"kind": "piecewise", "lams": mids, "alphas": alphas,
            "n_bins": len(bins)}


def _fit_curve_linear(bins: list[dict]) -> dict:
    mids = np.array([((b["lo"] if b["lo"] is not None else b["lam_mean"] * 0.8)
                      + (b["hi"] if b["hi"] is not None
                         else b["lam_mean"] * 1.2)) / 2 for b in bins])
    alphas = np.array([b["alpha"] for b in bins])
    slope, intercept = np.polyfit(mids, alphas, 1)
    return {"kind": "linear", "slope": float(slope),
            "intercept": float(intercept), "n_bins": len(bins)}


def _fit_curve_power(bins: list[dict]) -> dict:
    mids = np.array([((b["lo"] if b["lo"] is not None else b["lam_mean"] * 0.8)
                      + (b["hi"] if b["hi"] is not None
                         else b["lam_mean"] * 1.2)) / 2 for b in bins])
    alphas = np.array([b["alpha"] for b in bins])
    log_slope, log_intercept = np.polyfit(np.log(mids), np.log(alphas), 1)
    return {"kind": "power", "log_slope": float(log_slope),
            "log_intercept": float(log_intercept), "n_bins": len(bins)}


def alpha_of(lam: np.ndarray, curve: dict) -> np.ndarray:
    lam = np.clip(np.asarray(lam, float), 1e-6, None)
    kind = curve.get("kind", "single")
    if kind == "single":
        out = np.full(len(lam), float(curve["alpha"]))
    elif kind == "piecewise":
        lams = np.asarray(curve["lams"], float)
        alphas = np.asarray(curve["alphas"], float)
        out = np.interp(lam, lams, alphas,
                        left=alphas[0], right=alphas[-1])
    elif kind == "linear":
        out = curve["slope"] * lam + curve["intercept"]
    else:
        out = np.exp(curve["log_intercept"]) * lam ** curve["log_slope"]
    return np.clip(out, 1e-3, ALPHA_CAP)


def nb_pmf_matrix(ks: np.ndarray, mu_col: np.ndarray,
                  alpha_col: np.ndarray) -> np.ndarray:
    """Row-wise NB pmf over k grid (leak-free, no scipy loop over games)."""
    n, p = _nb_size_prob(mu_col[:, None], alpha_col[:, None])
    from scipy.stats import nbinom
    return nbinom.pmf(ks[None, :], n, p)


def eval_alpha_fit(y: np.ndarray, lam: np.ndarray, alpha: np.ndarray) -> float:
    """Poisson deviance of NB-fitted means == λ (α enters variance only) —
    so the fit check compares observed vs implied variance instead."""
    var_implied = lam + alpha * lam ** 2
    return float(np.abs(var_implied - y.var(ddof=0)).mean())


def select_alpha_curve(y: np.ndarray, lam: np.ndarray,
                       seed: int = MARKET_SEED) -> tuple[dict, dict]:
    """Fit α(λ) curves on PRE-HOLDOUT rows only; select by held-bin deviance.

    Deterministic: no randomness (seed kept for call-signature parity).
    Candidates: piecewise (always available if ≥2 bins), linear, power.
    Selection: lowest mean |implied − observed variance| on the bins.
    """
    del seed  # deterministic by construction
    bins = alpha_bins(y, lam)
    if len(bins) < 2:
        alpha = fit_alpha(y, lam)
        return ({"kind": "single", "alpha": round(alpha, 4),
                 "n_bins": len(bins)},
                {"selected": "single", "reason": "thin bins"})
    candidates = {"piecewise": _fit_curve_piecewise(bins),
                  "linear": _fit_curve_linear(bins),
                  "power": _fit_curve_power(bins)}
    scores = {}
    for name, curve in candidates.items():
        a = alpha_of(lam, curve)
        scores[name] = eval_alpha_fit(y, lam, a)
    best = min(scores, key=scores[k] if False else scores.get)  # noqa: E501
    return candidates[best], {"selected": best, "scores":
                              {k: round(v, 4) for k, v in scores.items()}}


def year_effect_check(oof_pre: pd.DataFrame, side: str) -> dict:
    """Per-year mean λ drift probe on the PRE-holdout pool only."""
    years = pd.to_datetime(oof_pre["game_date"]).dt.year
    out = {}
    for yr in sorted(years.unique()):
        m = (years == yr).to_numpy()
        lam = oof_pre.loc[m, f"{side}_expected_runs"].to_numpy(float)
        y = oof_pre.loc[m, f"{side}_score"].to_numpy(float)
        out[int(yr)] = {"n": int(m.sum()),
                        "lam_mean": round(float(lam.mean()), 4),
                        "score_mean": round(float(y.mean()), 4)}
    return out


def fit_check_table(actual: np.ndarray, lam: np.ndarray,
                    alpha: float) -> dict:
    actual, lam = np.asarray(actual, float), np.clip(
        np.asarray(lam, float), 1e-6, None)
    var_implied = float((lam + alpha * lam ** 2).mean())
    return {
        "n": int(len(actual)),
        "lam_mean": round(float(lam.mean()), 4),
        "observed_mean": round(float(actual.mean()), 4),
        "observed_var": round(float(actual.var(ddof=0)), 3),
        "implied_var": round(var_implied, 3),
        "alpha": round(float(alpha), 4),
        "poisson_deviance": round(poisson_deviance(actual, lam), 5),
        "rmse": round(float(np.sqrt(((actual - lam) ** 2).mean())), 5),
    }


def fit_check_table_curve(actual: np.ndarray, lam: np.ndarray,
                          alpha_col: np.ndarray) -> dict:
    actual = np.asarray(actual, float)
    lam = np.clip(np.asarray(lam, float), 1e-6, None)
    var_implied = float((lam + alpha_col * lam ** 2).mean())
    return {
        "n": int(len(actual)),
        "lam_mean": round(float(lam.mean()), 4),
        "observed_mean": round(float(actual.mean()), 4),
        "observed_var": round(float(actual.var(ddof=0)), 3),
        "implied_var": round(var_implied, 3),
        "alpha_mean": round(float(alpha_col.mean()), 4),
        "poisson_deviance": round(poisson_deviance(actual, lam), 5),
        "rmse": round(float(np.sqrt(((actual - lam) ** 2).mean())), 5),
    }


def _as_alpha_col(alpha: float | np.ndarray, n_games: int) -> np.ndarray:
    if np.isscalar(alpha):
        return np.full(n_games, float(alpha))
    return np.asarray(alpha, float)


# ---------------------------------------------------------------------------
# Monte Carlo market pricing (full line grid, independent NB marginals)
# ---------------------------------------------------------------------------
def derive_markets_mc(lam_home: np.ndarray, lam_away: np.ndarray,
                      alpha_home: float | np.ndarray,
                      alpha_away: float | np.ndarray,
                      n_draws: int = MC_DRAWS,
                      seed: int = MARKET_SEED) -> dict[str, np.ndarray]:
    """Monte Carlo per game from INDEPENDENT NB(λ, α) marginals.

    α may be scalar or per-game vectors from the α(λ) curve. Chunked over
    games to cap memory; identical seed ⇒ identical output. Prices the FULL
    line grid from the SAME draw matrix so grid columns are mutually
    consistent.

    Tie/push semantics (Phase 1 review decision 1/2):
    * MLB full-game moneyline: no tie — the impossible P(margin=0) mass is
      resolved into ±1 home-weighted (MARGIN_PLUS1_HOME_SHARE);
    * integer totals push (P(total == line)); half-point totals cannot;
    * per-line run-line 3-way splits sum to 1.0 exactly; half-point lines
      have push = 0.
    """
    rng = np.random.default_rng(seed)
    n_games = len(lam_home)
    p_over = np.empty(n_games)
    p_cover = np.empty(n_games)
    p_win = np.empty(n_games)
    n_lines = len(TOTAL_LINE_GRID)
    n_margins = len(RUN_LINE_GRID)
    grid_over = np.empty((n_games, n_lines))
    grid_cover = np.empty((n_games, n_margins))
    grid_push = np.empty((n_games, n_lines))
    n_rl = len(RUN_LINE_GRID_FULL)
    grid_rl_home = np.empty((n_games, n_rl))
    grid_rl_push = np.empty((n_games, n_rl))
    grid_rl_away = np.empty((n_games, n_rl))
    chunk = max(1, min(n_games, 2_000_000 // max(n_draws, 1)))
    for start in range(0, n_games, chunk):
        end = min(start + chunk, n_games)
        mu_h = np.maximum(np.asarray(lam_home[start:end], float),
                          1e-6)[:, None]
        mu_a = np.maximum(np.asarray(lam_away[start:end], float),
                          1e-6)[:, None]
        nh, ph_ = _nb_size_prob(mu_h, _as_alpha_col(alpha_home,
                                                    n_games)[start:end][:, None])
        na, pa = _nb_size_prob(mu_a, _as_alpha_col(alpha_away,
                                                   n_games)[start:end][:, None])
        h = rng.negative_binomial(nh, ph_,
                                  size=(end - start, n_draws)).astype(np.int32)
        a = rng.negative_binomial(na, pa,
                                  size=(end - start, n_draws)).astype(np.int32)
        total = h + a
        diff = h - a
        # Structural home one-run adjustment: the impossible tie mass P(0)
        # resolves into ±1 home-weighted: P(+1)' = P(+1)+α·P(0), P(−1)' =
        # P(−1)+(1−α)·P(0), P(0)'=0; every other margin stays RAW.
        p0 = (diff == 0).mean(axis=1)
        p1 = (diff == 1).mean(axis=1)
        ge2 = (diff >= 2).mean(axis=1)
        ge3 = (diff >= 3).mean(axis=1)
        ge4 = (diff >= 4).mean(axis=1)
        ge5 = (diff >= 5).mean(axis=1)
        eq2 = (diff == 2).mean(axis=1)
        eq3 = (diff == 3).mean(axis=1)
        eq4 = (diff == 4).mean(axis=1)
        push1 = p1 + MARGIN_PLUS1_HOME_SHARE * p0
        # Strict over: total must EXCEED the line (total >= line + 0.5).
        p_over[start:end] = (total >= 8.5 + 0.5).mean(axis=1)
        p_cover[start:end] = ge2
        p_win[start:end] = ge2 + push1
        for j, line in enumerate(TOTAL_LINE_GRID):
            grid_over[start:end, j] = (total >= line + 0.5).mean(axis=1)
        for j, m in enumerate(RUN_LINE_GRID):
            if m == 0.5:
                grid_cover[start:end, j] = ge2 + push1
            elif m == 1.5:
                grid_cover[start:end, j] = ge2
            elif m == 2.5:
                grid_cover[start:end, j] = ge3
            else:
                grid_cover[start:end, j] = ge4
        for j, line in enumerate(TOTAL_LINE_GRID):
            grid_push[start:end, j] = (total == line).mean(axis=1)
        for j, m in enumerate(RUN_LINE_GRID_FULL):
            # Strict cover: margin > L. Push is the resolved +1 band at
            # m=1.0 only (half-lines can never push); away is the residual.
            if m <= 1.5:
                grid_rl_home[start:end, j] = ge2
                grid_rl_push[start:end, j] = push1 if m == 1.0 else 0.0
            elif m <= 2.5:
                grid_rl_home[start:end, j] = ge3
                grid_rl_push[start:end, j] = eq2 if m == 2.0 else 0.0
            elif m <= 3.5:
                grid_rl_home[start:end, j] = ge4
                grid_rl_push[start:end, j] = eq3 if m == 3.0 else 0.0
            else:
                grid_rl_home[start:end, j] = ge5
                grid_rl_push[start:end, j] = eq4
            grid_rl_away[start:end, j] = (1.0 - grid_rl_home[start:end, j]
                                          - grid_rl_push[start:end, j])
    mc_se = np.sqrt(p_over * (1 - p_over) / n_draws)
    return {"p_over_8_5": p_over, "p_home_cover_1_5": p_cover,
            "p_home_win_derived": p_win, "mc_se_totals": mc_se,
            "p_over_grid": grid_over, "p_cover_grid": grid_cover,
            "p_push_grid": grid_push,
            "p_rl_home_grid": grid_rl_home, "p_rl_push_grid": grid_rl_push,
            "p_rl_away_grid": grid_rl_away}


# ---------------------------------------------------------------------------
# Walk-forward OOF for both side models
# ---------------------------------------------------------------------------
def run_oof(games: pd.DataFrame,
            retrain_cadence_days: int = 7,
            min_val_games: int = 40,
            min_train_games: int = 200,
            run_features: list[str] | None = None,
            decided_snapshot: pd.DataFrame | None = None,
            ) -> dict[str, Any]:
    """Walk-forward OOF for both side models on the moneyline pipeline folds.

    Deterministic: fold geometry from the study config, LightGBM seeded,
    chronological row order preserved.
    """
    games = (decided_snapshot.copy() if decided_snapshot is not None
             else games.copy())
    folds = walk_forward_splits(games, retrain_cadence_days=retrain_cadence_days)
    folds = [s for s in folds if len(s["val_games"]) >= min_val_games
             and len(s["train_games"]) >= min_train_games]
    logger.info("Run engine: %d walk-forward folds, %d decided games",
                len(folds), len(games))
    if not folds:
        raise ValueError("run_oof: no usable walk-forward folds (check "
                         "warmup/minimum-games rules and frame size)")

    frames = {side: build_side_frame(games, side, run_features=run_features)
              for side in ("home", "away")}
    params = dict(RUN_LGBM_PARAMS)

    out_rows: list[dict] = []
    best_iters: dict[str, list[int]] = {s: [] for s in ("home", "away")}
    metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}
    base_metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}

    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        _d = pd.to_datetime(va["game_date"]).dt.strftime("%Y%m%d")
        rec_base = {"game_pk": va["game_pk"].to_numpy(),
                    "game_date": pd.to_datetime(va["game_date"]).dt.strftime(
                        "%Y-%m-%d"),
                    "fold_idx": split["fold_idx"],
                    "game_id": (_d + "_" + va["away_team"].astype(str) + "@"
                                + va["home_team"].astype(str))}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            _, cols_all = frames[side]
            tr_frame = tr.reindex(columns=cols_all).astype(float)
            va_frame = va.reindex(columns=cols_all).astype(float)
            y_tr = tr[target].to_numpy(dtype=float)
            y_va = va[target].to_numpy(dtype=float)
            _, lam, best = _fit_side_model(params, tr_frame, y_tr,
                                           va_frame, y_va)
            best_iters[side].append(best)
            key = f"{side}_expected_runs"
            rec_base[key] = np.round(lam, 4)
            rec_base[target] = y_va.astype(int)
            m = metrics[side]
            m["deviance"].append(poisson_deviance(y_va, lam))
            m["rmse"].append(float(np.sqrt(((y_va - lam) ** 2).mean())))
            m["mae"].append(float(np.abs(y_va - lam).mean()))
            mu = float(y_tr.mean())  # TRAIN mean — no val leakage
            bm = base_metrics[side]
            bm["deviance"].append(poisson_deviance(
                y_va, np.full(len(y_va), mu)))
            bm["rmse"].append(float(np.sqrt(((y_va - mu) ** 2).mean())))
            bm["mae"].append(float(np.abs(y_va - mu).mean()))
        out_rows.extend(pd.DataFrame(rec_base).to_dict(orient="records"))

    oof = pd.DataFrame(out_rows).sort_values(["game_date", "game_pk"])
    oof = oof.drop_duplicates(subset=["game_pk"], keep="first")
    summary: dict[str, Any] = {"n_folds": len(folds), "n_games": len(oof)}
    summary["final_fit_rounds"] = {
        s: int(np.median(best_iters[s])) if best_iters[s] else MAX_ROUNDS
        for s in ("home", "away")}
    for side in ("home", "away"):
        summary[f"{side}_model"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in metrics[side].items()}
        summary[f"{side}_baseline"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in base_metrics[side].items()}
    for side in ("home", "away"):
        y = oof[f"{side}_score"].to_numpy(float)
        lam = oof[f"{side}_expected_runs"].to_numpy(float)
        summary[f"{side}_pooled"] = {
            "poisson_deviance": round(poisson_deviance(y, lam), 5),
            "rmse": round(float(np.sqrt(((y - lam) ** 2).mean())), 5),
            "mae": round(float(np.abs(y - lam).mean()), 5)}
        summary[f"{side}_dispersion_ratio"] = round(
            dispersion_ratio(y, lam), 4)
    return {"oof": oof, "summary": summary}


# ---------------------------------------------------------------------------
# Market construction (Phase 3: α(λ) curves + sealed holdout gate)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# k-edge (C2 level-preserving edge expansion — reference-exact)
# ---------------------------------------------------------------------------
def fit_k_edge(lam_h: np.ndarray, lam_a: np.ndarray, margin: np.ndarray,
               mask: np.ndarray) -> float:
    """Fit the C2 edge multiplier k on the MASKED (strictly-prior) games only:
    k = OLS slope of actual margin on the λ edge (λ_H − λ_A). The masked set
    is the pre-holdout OOF — sealed games never see k."""
    d = np.asarray(lam_h, float)[mask] - np.asarray(lam_a, float)[mask]
    m = np.asarray(margin, float)[mask]
    if len(d) < 100 or np.std(d) < 1e-9:
        logger.warning("fit_k_edge: insufficient edge variance "
                       "(n=%d, sd=%.4f) — returning 1.0 (no expansion)",
                       len(d), float(np.std(d)))
        return 1.0
    return float(np.polyfit(d, m, 1)[0])


def apply_k_edge(lam_h: np.ndarray, lam_a: np.ndarray, k: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """C2 linear edge expansion (level-preserving). k=1.0 is the identity:
    λ'_H = μ + k(λ_H − μ), λ'_A = μ + k(λ_A − μ), μ = (λ_H + λ_A)/2."""
    lam_h = np.asarray(lam_h, float)
    lam_a = np.asarray(lam_a, float)
    mu = (lam_h + lam_a) / 2.0
    return mu + k * (lam_h - mu), mu + k * (lam_a - mu)


def k_edge_holdout_mask(oof: pd.DataFrame, holdout_days: int) -> np.ndarray:
    """Pre-holdout mask mirroring derive_markets_v3's own discipline."""
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    return (dates < cutoff).to_numpy()


def k_edge_meta(k: float) -> dict:
    from sports.mlb.market_config import K_EDGE_BAND, K_EDGE_REF
    return {
        "k": round(float(k), 4),
        "fit": "run-oof-refit (pre-holdout)",
        "reference_k": K_EDGE_REF,
        "drift_band": [round(K_EDGE_REF - K_EDGE_BAND, 3),
                       round(K_EDGE_REF + K_EDGE_BAND, 3)],
        "drift_alert": bool(abs(k - K_EDGE_REF) > K_EDGE_BAND),
    }


def agreement_stats(p_run: np.ndarray, p_ml: np.ndarray,
                    delta: float | None = None) -> dict:
    """Moneyline-vs-derived divergence stats (reference-exact).

    Reference semantics (run_engine.agreement_stats): delta_primary is the
    CONFIGURED agreement filter delta (AGREEMENT_FILTER_DELTA = 0.08), not
    a data-driven quantile; the 0.08/0.10 shares are the standard secondary
    read. Flagged counts are absolute game counts, not rates.
    """
    from sports.mlb.market_config import AGREEMENT_FILTER_DELTA
    delta = AGREEMENT_FILTER_DELTA if delta is None else delta
    diff = np.abs(np.asarray(p_run, float) - np.asarray(p_ml, float))
    stats = {
        "delta_primary": delta,
        "n": int(len(diff)),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "share_gt_primary": round(float((diff > delta).mean()), 4),
        "share_gt_0_08": round(float((diff > 0.08).mean()), 4),
        "share_gt_0_10": round(float((diff > 0.10).mean()), 4),
    }
    stats["n_flagged_primary"] = int((diff > delta).sum())
    stats["n_flagged_0_08"] = int((diff > 0.08).sum())
    stats["n_flagged_0_10"] = int((diff > 0.10).sum())
    return stats


def _safe_auc(p: np.ndarray, y: np.ndarray) -> float | None:
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        return None
    return auc if np.isfinite(auc) else None


def rl_col(m: float, side: str) -> str:
    return f"p_rl_{str(m).replace('.', '_')}_{side}"


def derive_markets_v3(oof: pd.DataFrame,
                      moneyline_probs: pd.DataFrame | None = None,
                      n_draws: int = MC_DRAWS,
                      seed: int = MARKET_SEED,
                      holdout_days: int = HOLDOUT_DAYS,
                      k_edge: float | None = None,
                      ) -> dict[str, Any]:
    """α(λ) dispersion curves + full-line-grid MC markets + sealed holdout.

    The last ``holdout_days`` of OOF games are SEALED: no α fitting, binning,
    or form selection ever sees them. Everything is scored pooled OOF and
    again holdout-only, each against constant-base-rate baselines.

    k_edge (reference-exact C2 expansion): when None, k is REFIT on this
    run's OOF pre-holdout games (per-run refit policy — OLS slope of actual
    margin on the λ edge). When given, the supplied k is used (k_edge=1.0
    disables the expansion explicitly). The λ pair is expanded BEFORE α-
    curve fitting + NB MC (level preserved); the fitted/supplied k + drift
    band ALWAYS land in ``summary['k_edge']``. Sealed games never see k.
    """
    _identity = oof.get("game_pk")
    if _identity is not None:
        _bad = _identity.isna()
        if _bad.any():
            logger.warning("derive_markets_v3: dropping %d identity-less "
                           "OOF row(s)", int(_bad.sum()))
            oof = oof.loc[~_bad].reset_index(drop=True)
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    pre_mask = (dates < cutoff).to_numpy()

    # --- k-edge: fit/apply BEFORE α-curve fitting + MC (reference seam) ---
    if k_edge is None:
        k_edge = fit_k_edge(
            oof["home_expected_runs"].to_numpy(float),
            oof["away_expected_runs"].to_numpy(float),
            (oof["home_score"] - oof["away_score"]).to_numpy(float),
            pre_mask)
        logger.warning("Run engine (k-edge): fitted k=%.4f "
                       "(pre-holdout OOF, n=%d)", k_edge, int(pre_mask.sum()))
    if abs(k_edge - 1.0) > 1e-9:
        lh, la = apply_k_edge(oof["home_expected_runs"].to_numpy(float),
                              oof["away_expected_runs"].to_numpy(float),
                              k_edge)
        oof = oof.copy()
        oof["home_expected_runs"] = np.round(lh, 4)
        oof["away_expected_runs"] = np.round(la, 4)

    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    total_runs = hs + as_
    fold_idx = oof["fold_idx"].to_numpy()

    summary: dict[str, Any] = {
        "n_draws": n_draws, "seed": seed,
        "holdout_cutoff": str(cutoff.date()),
        "n_pre": int(pre_mask.sum()), "n_holdout": int((~pre_mask).sum()),
        "line_grid": {"totals": TOTAL_LINE_GRID,
                      "run_lines": [-m for m in RUN_LINE_GRID],
                      "run_lines_full": [-m for m in RUN_LINE_GRID_FULL],
                      "run_lines_whole": [-m for m in RUN_LINE_GRID_WHOLE],
                      "total_ref_lines": list(TOTAL_REF_LINES),
                      "run_ref_lines": [-m for m in RUN_REF_LINES]},
    }

    curves, alpha_cols, single_alpha = {}, {}, {}
    phase2_fc, phase3_fc, variance_check = {}, {}, {}
    for side, target in (("home", "home_score"), ("away", "away_score")):
        y_all = oof[target].to_numpy(float)
        lam_all = oof[f"{side}_expected_runs"].to_numpy(float)
        y_pre, lam_pre = y_all[pre_mask], lam_all[pre_mask]
        curve, diag = select_alpha_curve(y_pre, lam_pre)
        curves[side] = curve
        summary[f"alpha_{side}"] = {
            **curve, "selection": diag,
            "fitted_on": "pre-holdout OOF only",
            "cap": ALPHA_CAP, "min_bin_count": ALPHA_MIN_BIN}
        alpha_cols[side] = alpha_of(lam_all, curve)
        single_alpha[side] = fit_alpha(y_pre, lam_pre)
        phase2_fc[side] = fit_check_table(y_all, lam_all, single_alpha[side])
        phase3_fc[side] = fit_check_table_curve(y_all, lam_all,
                                                alpha_cols[side])
        var_implied = float((lam_all + alpha_cols[side] * lam_all ** 2).mean())
        variance_check[side] = {
            "implied_var": round(var_implied, 3),
            "observed_var": round(float(y_all.var(ddof=0)), 3),
            "phase2_implied_var": round(float(
                lam_pre.mean() + single_alpha[side] * lam_pre.mean() ** 2), 3)}
        summary[f"year_effect_{side}"] = year_effect_check(
            oof.iloc[np.where(pre_mask)[0]], side)
    summary["phase2_single_alpha"] = single_alpha
    summary["fit_check_single_alpha"] = phase2_fc
    summary["fit_check_alpha_lambda"] = phase3_fc
    summary["variance_check"] = variance_check

    mc = derive_markets_mc(oof["home_expected_runs"].to_numpy(float),
                           oof["away_expected_runs"].to_numpy(float),
                           alpha_cols["home"], alpha_cols["away"],
                           n_draws=n_draws, seed=seed)
    se = mc["mc_se_totals"]
    used_draws, draw_reason = n_draws, "default"
    if se.max() > MC_SE_TARGET and n_draws < MC_DRAWS_TAIL:
        used_draws, draw_reason = MC_DRAWS_TAIL, (
            f"SE {se.max():.4f} > {MC_SE_TARGET} at N={n_draws} — bumped")
        mc = derive_markets_mc(oof["home_expected_runs"].to_numpy(float),
                               oof["away_expected_runs"].to_numpy(float),
                               alpha_cols["home"], alpha_cols["away"],
                               n_draws=MC_DRAWS_TAIL, seed=seed)
        se = mc["mc_se_totals"]
    summary["mc_meta"] = {
        "n_draws": used_draws, "requested_draws": n_draws,
        "reason": draw_reason,
        "mc_se_totals_max": round(float(se.max()), 6)}

    def score_at(name: str, p: np.ndarray, yv: np.ndarray,
                 mask: np.ndarray | None = None) -> None:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        yv = np.asarray(yv, float)
        base = float(yv.mean())
        row = {
            "engine_logloss": round(log_loss(yv, p, labels=[0.0, 1.0]), 5),
            "engine_brier": round(brier_score(yv, p), 5),
            "engine_ece_raw": ece_score(yv, p),
            "baseline_rate": round(base, 4),
            "baseline_logloss": round(log_loss(
                yv, np.full(len(yv), base), labels=[0.0, 1.0]), 5),
            "baseline_brier": round(brier_score(
                yv, np.full(len(yv), base)), 5),
        }
        row["auc"] = _safe_auc(p, yv)
        p_cal = prequential_calibrate(yv, p, fold_idx)
        row["engine_logloss_calibrated"] = round(
            log_loss(yv, p_cal, labels=[0.0, 1.0]), 5)
        row["engine_ece_calibrated"] = ece_score(yv, p_cal)
        row["predicted_mean"] = round(float(np.mean(p_cal)), 4)
        row["n"] = int(len(yv))
        if mask is not None and mask.any():
            ph, yh = p[mask], yv[mask]
            base_h = float(yh.mean())
            h_ll = round(log_loss(yh, ph, labels=[0.0, 1.0]), 5)
            h_bll = round(log_loss(
                yh, np.full(len(yh), base_h), labels=[0.0, 1.0]), 5)
            row["holdout"] = {
                "n": int(mask.sum()),
                "engine_logloss": h_ll,
                "engine_brier": round(brier_score(yh, ph), 5),
                "engine_ece_raw": ece_score(yh, ph),
                "baseline_rate": round(base_h, 4),
                "baseline_logloss": h_bll,
                "baseline_brier": round(brier_score(
                    yh, np.full(len(yh), base_h)), 5),
                "beats_baseline_logloss": bool(h_ll < h_bll)}
        row["beats_baseline_logloss"] = bool(
            row["engine_logloss"] < row["baseline_logloss"])
        summary[f"market_{name}"] = row

    y_over = (total_runs >= 9).astype(float)
    y_win = (hs > as_).astype(float)
    score_at("derived_moneyline", mc["p_home_win_derived"], y_win)
    score_at("derived_moneyline_holdout", mc["p_home_win_derived"], y_win,
             mask=~pre_mask)
    for line in TOTAL_REF_LINES:
        col = TOTAL_LINE_GRID.index(line)
        key = f"over_{str(line).replace('.', '_')}"
        score_at(key, mc["p_over_grid"][:, col],
                 (total_runs >= line + 0.5).astype(float))
        score_at(key + "_holdout", mc["p_over_grid"][:, col],
                 (total_runs >= line + 0.5).astype(float), mask=~pre_mask)
    for m in RUN_REF_LINES:
        col = RUN_LINE_GRID.index(m)
        key = f"home_cover_{str(m).replace('.', '_')}"
        score_at(key, mc["p_cover_grid"][:, col],
                 ((hs - as_) >= m + 0.5).astype(float))
        score_at(key + "_holdout", mc["p_cover_grid"][:, col],
                 ((hs - as_) >= m + 0.5).astype(float), mask=~pre_mask)

    markets = oof[["game_pk", "game_date"]].copy()
    for tc in ("home_team", "away_team"):
        if tc in oof.columns:
            markets[tc] = oof[tc].to_numpy()
    markets["kind"] = "oof"
    markets["home_expected_runs"] = np.round(
        oof["home_expected_runs"].to_numpy(float), 4)
    markets["away_expected_runs"] = np.round(
        oof["away_expected_runs"].to_numpy(float), 4)
    markets["alpha_home"] = np.round(alpha_cols["home"], 4)
    markets["alpha_away"] = np.round(alpha_cols["away"], 4)
    for j, line in enumerate(TOTAL_LINE_GRID):
        key = f"p_over_{str(line).replace('.', '_')}"
        markets[key] = np.round(mc["p_over_grid"][:, j], 5)
        markets[key.replace("p_over_", "p_push_")] = np.round(
            mc["p_push_grid"][:, j], 5)
        markets[key.replace("p_over_", "p_under_")] = np.round(
            1 - mc["p_over_grid"][:, j] - mc["p_push_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID):
        markets[f"p_home_cover_{str(m).replace('.', '_')}"] = np.round(
            mc["p_cover_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID_FULL):
        markets[rl_col(m, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        markets[rl_col(m, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        markets[rl_col(m, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)
    markets["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    markets["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)
    markets["home_score"] = hs.astype(int)
    markets["away_score"] = as_.astype(int)
    markets["total_runs"] = total_runs.astype(int)

    if moneyline_probs is not None and len(moneyline_probs):
        key = "game_pk" if "game_pk" in moneyline_probs.columns else "game_id"
        merged = pd.DataFrame({
            key: oof[key].to_numpy() if key in oof.columns
            else markets["game_pk"].to_numpy(),
            "p_run": mc["p_home_win_derived"],
        }).merge(moneyline_probs[[key, "home_win_prob_model"]],
                 on=key, how="inner")
        if len(merged) > 30:
            stats = agreement_stats(merged["p_run"],
                                    merged["home_win_prob_model"])
            summary["agreement_vs_moneyline"] = stats
            diff_series = ((merged["p_run"]
                            - merged["home_win_prob_model"]).abs()
                           > stats["delta_primary"])
            conflict_map = dict(zip(merged[key], diff_series))
            prob_map = dict(zip(merged[key], merged["home_win_prob_model"]))
            join_key = markets["game_pk"]
            markets["agreement_conflict"] = [
                bool(conflict_map.get(k, False)) for k in join_key]
            markets["ml_win_prob"] = [prob_map.get(k) for k in join_key]
    if "agreement_conflict" not in markets.columns:
        logger.warning("Run engine: moneyline probabilities unavailable — "
                       "agreement conflicts NOT computed for this run")
        markets["agreement_conflict"] = False
        markets["ml_win_prob"] = np.nan

    # k_edge meta AFTER the agreement block so the persisted meta key order
    # matches the reference production artifact exactly:
    # ..., agreement_vs_moneyline, k_edge, agreement_slate.
    summary["k_edge"] = k_edge_meta(k_edge)

    return {"markets": markets, "summary": summary,
            "_curves": curves, "_alpha_cols": alpha_cols}


# ---------------------------------------------------------------------------
# Slate prediction
# ---------------------------------------------------------------------------
def _resolve_slate_key(slate: pd.DataFrame) -> str:
    if "game_pk" in slate.columns and slate["game_pk"].notna().any():
        return "game_pk"
    if "game_id" in slate.columns:
        return "game_id"
    raise KeyError("slate frame has neither game_pk nor game_id")


def predict_slate_runs(decided_games: pd.DataFrame,
                       slate_games: pd.DataFrame,
                       final_fit_rounds: dict[str, int],
                       curves: dict[str, dict],
                       n_draws: int = MC_DRAWS,
                       seed: int = MARKET_SEED) -> pd.DataFrame:
    """λ + full market grid for the prediction window's slate.

    Side models refit on ALL decided games at fixed rounds (median fold
    early-stopping iteration — no early stopping against the future), then
    priced through the SAME α(λ) curves and MC machinery as OOF rows.
    """
    if slate_games.empty:
        return pd.DataFrame()
    from lightgbm import LGBMRegressor

    _missing = [c for c in ("game_date",) if c not in slate_games.columns]
    if _missing:
        raise KeyError(f"Slate frame missing required column(s): {_missing}")

    _slate_key = _resolve_slate_key(slate_games)
    out = slate_games[[_slate_key, "game_date"]].copy()
    if _slate_key == "game_id":
        out = out.rename(columns={"game_id": "game_pk"})
    _pks: list[object] = []
    from pandas import isna as _pd_isna
    for _i in out.index:
        _v = out.at[_i, "game_pk"]
        if _pd_isna(_v) and "game_id" in slate_games.columns:
            _v = slate_games.at[_i, "game_id"]
        if _v is None or _pd_isna(_v):
            _pks.append(None)
            continue
        if isinstance(_v, float) and _v == int(_v):
            _v = int(_v)
        _pks.append(str(_v))
    out["game_pk"] = _pks
    for tc in ("home_team", "away_team"):
        if tc in slate_games.columns:
            out[tc] = slate_games[tc].to_numpy()
    for side in ("home", "away"):
        _, cols = build_side_frame(decided_games, side)
        tr = decided_games.reindex(columns=cols).astype(float)
        model = LGBMRegressor(**RUN_LGBM_PARAMS)
        model.set_params(n_estimators=int(final_fit_rounds[side]))
        model.fit(tr, decided_games[f"{side}_score"].to_numpy(dtype=float))
        va = slate_games.reindex(columns=cols).astype(float)
        out[f"{side}_expected_runs"] = np.round(
            np.clip(model.predict(va), 1e-6, None), 4)
    alpha_h = alpha_of(out["home_expected_runs"].to_numpy(float),
                       curves["home"])
    alpha_a = alpha_of(out["away_expected_runs"].to_numpy(float),
                       curves["away"])
    out["alpha_home"] = np.round(alpha_h, 4)
    out["alpha_away"] = np.round(alpha_a, 4)
    mc = derive_markets_mc(out["home_expected_runs"].to_numpy(float),
                           out["away_expected_runs"].to_numpy(float),
                           alpha_h, alpha_a, n_draws=n_draws, seed=seed)
    out["kind"] = "slate"
    for j, line in enumerate(TOTAL_LINE_GRID):
        key = f"p_over_{str(line).replace('.', '_')}"
        out[key] = np.round(mc["p_over_grid"][:, j], 5)
        out[key.replace("p_over_", "p_push_")] = np.round(
            mc["p_push_grid"][:, j], 5)
        out[key.replace("p_over_", "p_under_")] = np.round(
            1 - mc["p_over_grid"][:, j] - mc["p_push_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID):
        out[f"p_home_cover_{str(m).replace('.', '_')}"] = np.round(
            mc["p_cover_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID_FULL):
        out[rl_col(m, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        out[rl_col(m, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        out[rl_col(m, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)
    out["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    out["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)
    # Undecided by definition.
    out["home_score"] = pd.NA
    out["away_score"] = pd.NA
    out["total_runs"] = pd.NA
    out["agreement_conflict"] = False
    out["ml_win_prob"] = np.nan
    unresolved = out["game_pk"].isna()
    if unresolved.any():
        logger.warning("predict_slate_runs: dropping %d unresolvable slate "
                       "row(s) (no game_pk/game_id)", int(unresolved.sum()))
        out = out[~unresolved].reset_index(drop=True)
    return out


def compute_rolling_totals_brier(markets: pd.DataFrame,
                                 window_days: int = 30,
                                 min_games_per_day: int = 1) -> dict:
    """Rolling over-9.0 Brier series from persisted OOF markets rows."""
    oof = markets[markets["kind"] == "oof"].copy() \
        if "kind" in markets.columns else markets.copy()
    if oof.empty or "total_runs" not in oof.columns:
        return {"window_days": window_days,
                "min_games_per_day": min_games_per_day,
                "series": [], "n_points": 0, "n_games_total": 0,
                "excluded_sparse_days": 0, "history_mean_brier": None,
                "n_games_in_series": 0}
    oof["game_date"] = pd.to_datetime(oof["game_date"])
    y_over = (oof["total_runs"] >= 9.5).astype(float)
    p_over = oof["p_over_9_0"].to_numpy(float)
    brier_by_day = ((p_over - y_over.to_numpy()) ** 2)
    series = []
    excluded = 0
    for day, grp in oof.groupby(oof["game_date"].dt.date):
        if len(grp) < min_games_per_day:
            excluded += 1
            continue
        idx = grp.index
        series.append({"date": str(day),
                       "brier": round(float(brier_by_day.loc[idx].mean()), 5),
                       "games": int(len(grp))})
    series.sort(key=lambda r: r["date"])
    return {"window_days": window_days,
            "min_games_per_day": min_games_per_day,
            "series": series, "n_points": len(series),
            "n_games_total": int(sum(r["games"] for r in series)),
            "excluded_sparse_days": excluded,
            "history_mean_brier": (round(float(np.mean(
                [r["brier"] for r in series])), 5) if series else None),
            "n_games_in_series": int(sum(r["games"] for r in series))}
