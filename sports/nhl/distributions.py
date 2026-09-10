"""Production NHL puck-line / margin and totals distributional model.

Architecture (platform parity): a JOINT score distribution.

  mu_h, mu_a  <- gradient-boosted regressions on point-in-time features
  margin      ~ discrete normal on mu_margin = mu_h - mu_a,
                 sigma_margin estimated from OOF residuals
  total       ~ discrete normal on mu_total = mu_h + mu_a, sigma_total
                 estimated from OOF residuals

Everything derives from ONE (mu_h, mu_a) pair per game, so the margin and
total distributions are mathematically coherent: moneyline-derived, fair
puckline, fair total, every cover/over/push probability, and the mu
quartet all come from the same fitted score distribution.

TIE/PUSH semantics (sport-specific): the derived FULL-GAME moneyline is
two-way (OT/SO resolves every game — the margin is never 0 in the
settled outcome, so the tie mass is excluded from full-game markets);
the derived REGULATION moneyline is THREE-way with the explicit
margin==0 mass as p_tie. Integer pucklines and totals carry explicit
push mass; half-point lines (the NHL standard ±1.5, 5.5, 6.5) cannot
push (margins/totals are integers).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports.nhl.features import tree_view
from core.folds import make_folds
from sports.nhl.study_config import load_nhl_study

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrete normal PMF helpers (integer support, hockey goal mechanics)
# ---------------------------------------------------------------------------


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
    L, and m >= L + 1 for integer L. (NHL margins are integers; the
    contract P(m > L) is honored exactly.)"""
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


def _pmf_median(pmf: np.ndarray, support: np.ndarray) -> float:
    """Smallest integer k with cumulative PMF >= 0.5 (the fair line)."""
    if not np.isfinite(pmf).all():
        return np.nan
    c = np.cumsum(pmf)
    idx = int(np.searchsorted(c, 0.5))
    return float(support[min(idx, len(support) - 1)])


# ---------------------------------------------------------------------------
# Regression members (per-side score regressions on point-in-time features)
# ---------------------------------------------------------------------------


def _make_reg(name: str, seed: int):
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=3,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed,
                            verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, max_depth=4, num_leaves=12,
                             min_child_samples=30, learning_rate=0.05,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed,
                             verbose=-1)
    raise KeyError(f"unknown regression member {name!r}")


class ScoreRegressor:
    """mu_h / mu_a from a boosted-tree regression pair (tree view).

    Representation contract (mirrors moneyline.member_fit_input): both
    regressors are FIT and PREDICTED on the NAMED tree-view DataFrame.
    """

    def __init__(self, seed: int = 42) -> None:
        self.home_model = _make_reg("xgboost", seed)
        self.away_model = _make_reg("lightgbm", seed)

    def _matrix(self, df: pd.DataFrame, study) -> pd.DataFrame:
        Xv = tree_view(df, study=study)
        X = Xv.to_numpy(dtype=np.float64)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        return pd.DataFrame(X, columns=Xv.columns, index=Xv.index)

    def fit(self, df: pd.DataFrame, study) -> "ScoreRegressor":
        X = self._matrix(df, study)
        self.home_model.fit(X, df["home_score"].astype(float).to_numpy())
        self.away_model.fit(X, df["away_score"].astype(float).to_numpy())
        return self

    def predict(self, df: pd.DataFrame, study) -> tuple[np.ndarray,
                                                        np.ndarray]:
        X = self._matrix(df, study)
        return self.home_model.predict(X), self.away_model.predict(X)


# ---------------------------------------------------------------------------
# sigma estimation / calibration — pooled OOF residuals, one scalar pair
# ---------------------------------------------------------------------------


def calibrate_sigma(resid_margin: np.ndarray, resid_total: np.ndarray,
                    study) -> dict:
    """Robust sigma estimates from pooled OOF residuals (1.4826 * MAD = sigma
    of a normal). One scalar pair keeps the PMF coherent."""
    mc = study.market
    rm = resid_margin[np.isfinite(resid_margin)]
    rt = resid_total[np.isfinite(resid_total)]
    s_m = (float(1.4826 * np.median(np.abs(rm - np.median(rm))))
           if len(rm) else mc.margin_sigma)
    s_t = (float(1.4826 * np.median(np.abs(rt - np.median(rt))))
           if len(rt) else mc.total_sigma)
    s_m = float(np.clip(s_m, mc.sigma_floor_margin, mc.sigma_cap_margin))
    s_t = float(np.clip(s_t, mc.sigma_floor_total, mc.sigma_cap_total))
    return {"sigma_margin": s_m, "sigma_total": s_t}


# ---------------------------------------------------------------------------
# Distribution engine — everything derives from (mu_h, mu_a, sigma_m, sigma_t)
# ---------------------------------------------------------------------------


def game_distribution(mu_h: float, mu_a: float, sigma_margin: float,
                      sigma_total: float, study) -> dict:
    """All distributional outputs for one game from the joint score model.

    The full-game moneyline is TWO-way (p_home_win + p_away_win = 1; OT/SO
    resolves every game). The regulation moneyline is THREE-way with
    ``p_tie_regulation`` carrying the margin==0 mass explicitly. The two
    views are never conflated."""
    mc = study.market
    support_m = np.arange(-mc.margin_pmf_max, mc.margin_pmf_max + 1)
    support_t = np.arange(0, mc.total_pmf_max + 1)
    mu_margin = mu_h - mu_a
    mu_total = mu_h + mu_a
    pmf_m = discrete_normal_pmf(mu_margin, sigma_margin, support_m)
    pmf_t = discrete_normal_pmf(mu_total, sigma_total, support_t)

    # REGULATION three-way moneyline: P(regulation home win), P(reg tie),
    # P(regulation away win) — explicit p_tie, never fabricated away.
    p_tie_reg = margin_pmf_at(pmf_m, support_m, 0.0)
    p_home_reg = margin_cdf_above(pmf_m, support_m, 0.0)
    p_away_reg = 1.0 - p_home_reg - p_tie_reg

    # FULL-GAME two-way moneyline: the settled outcome always has a
    # winner. The model's continuous tie mass splits proportionally to
    # the regulation win shares (a documented, deterministic convention —
    # not a fabricated 50/50).
    safe_den = p_home_reg + p_away_reg
    if np.isfinite(safe_den) and safe_den > 0:
        share_h = p_home_reg / safe_den
    else:
        share_h = 0.5
    p_home_win = np.clip(p_home_reg + p_tie_reg * share_h, 0.0, 1.0)
    p_away_win = 1.0 - p_home_win

    out = {
        "mu_h": float(mu_h), "mu_a": float(mu_a),
        "mu_margin": float(mu_margin), "mu_total": float(mu_total),
        "p_home_win_derived": float(p_home_win),
        "p_away_win_derived": float(p_away_win),
        "p_tie": float(p_tie_reg),          # regulation tie mass (3-way view)
        "p_home_regulation": float(p_home_reg),
        "p_away_regulation": float(p_away_reg),
    }
    out["fair_spread"] = _pmf_median(pmf_m, support_m)
    out["fair_total"] = _pmf_median(pmf_t, support_t)
    out["p_over_fair"] = total_probabilities(
        pmf_t, support_t, float(out["fair_total"]))[0]
    out["p_cover_fair"] = margin_cdf_above(
        pmf_m, support_m, float(out["fair_spread"]))
    # puckline grid: p_home_cover_L / p_push_L for integer L
    for L in mc.spread_lines:
        out[f"p_home_cover_{L}"] = margin_cdf_above(pmf_m, support_m, float(L))
        out[f"p_push_{L}"] = margin_pmf_at(pmf_m, support_m, float(L))
    # half-stop lines (the NHL standard ±1.5, 5.5, 6.5): no push band
    for L in mc.half_stop_lines:
        label = str(L).replace(".", "_").replace("-", "m")
        out[f"p_home_cover_{label}"] = margin_cdf_above(pmf_m, support_m, L)
    # totals grid: over/push/under for integer U
    for U in mc.total_lines:
        p_over, p_push, p_under = total_probabilities(pmf_t, support_t,
                                                      float(U))
        out[f"p_over_{U}"] = p_over
        out[f"p_push_{U}"] = p_push
        out[f"p_under_{U}"] = p_under
    return out


def apply_distribution(df: pd.DataFrame, sigma_margin: float,
                       sigma_total: float, study) -> pd.DataFrame:
    """Expand a frame with mu_h/mu_a into the full distributional columns.
    Distribution columns already present on the input are replaced (never
    duplicated) — the engine is the single authority for these values."""
    rows = [game_distribution(r.mu_h, r.mu_a, sigma_margin, sigma_total,
                              study)
            for r in df.itertuples(index=False)]
    dist = pd.DataFrame(rows, index=df.index)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns])
    return pd.concat([base.reset_index(drop=True),
                      dist.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Walk-forward OOF for the distribution model
# ---------------------------------------------------------------------------


def walk_forward_oof(game_df: pd.DataFrame, study,
                     date_col: str = "game_date") -> dict:
    """Expanding walk-forward OOF: per fold, fit mu regressions on strictly
    prior training games, predict validation, collect residuals for the
    pooled sigma calibration. Returns {oof, fold_table}."""
    df = game_df.sort_values(date_col).reset_index(drop=True)
    fold_list = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col=date_col,
                           min_val_games=study.min_val_games)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        if len(train) < study.oof.min_training_rows:
            continue
        try:
            reg = ScoreRegressor(study.training.random_seed).fit(train, study)
            mu_h, mu_a = reg.predict(val, study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dist OOF fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
        part = pd.DataFrame({
            "game_id": val["game_id"].to_numpy(),
            "game_date": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val["season"].to_numpy(),
            "fold_id": fold.fold_id,
            "mu_h": mu_h, "mu_a": mu_a,
            "home_score": val["home_score"].astype(float).to_numpy(),
            "away_score": val["away_score"].astype(float).to_numpy(),
        })
        part["margin"] = part["home_score"] - part["away_score"]
        part["total"] = part["home_score"] + part["away_score"]
        part["resid_margin"] = part["margin"] - (part["mu_h"] - part["mu_a"])
        part["resid_total"] = part["total"] - (part["mu_h"] + part["mu_a"])
        parts.append(part)
        fold_rows.append({
            "fold_id": fold.fold_id,
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "n_train": int(len(train)), "n_val": int(len(val)),
        })
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows)}


def fit_final(game_df: pd.DataFrame, study) -> ScoreRegressor:
    """Final full-history refit of the mu regressions."""
    return ScoreRegressor(study.training.random_seed).fit(game_df, study)
