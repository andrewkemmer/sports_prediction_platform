"""Production NBA spread / margin and totals distributional model.

Architecture (platform parity): a JOINT score distribution.

  mu_h, mu_a  <- gradient-boosted regressions on point-in-time features
  margin      ~ discrete normal on mu_margin = mu_h - mu_a,
                 sigma_margin estimated from OOF residuals
  total       ~ discrete normal on mu_total = mu_h + mu_a, sigma_total
                 estimated from OOF residuals

Everything derives from ONE (mu_h, mu_a) pair per game, so the margin and
total distributions are mathematically coherent: moneyline-derived, fair
spread, fair total, every cover/over/push probability, and the mu
quartet all come from the same fitted score distribution.

TIE/PUSH semantics (sport-specific): an NBA game CANNOT end in a tie â€”
overtime repeats until a winner exists, so the derived FULL-GAME
moneyline is strictly TWO-way and the model's continuous margin==0 mass
splits deterministically to the sides (documented convention, NOT a
fabricated 50/50 tie outcome â€” no tie outcome exists to fabricate).
Integer spreads and totals carry explicit push mass; there are no
half-point lines in the Phase 5 grid (core.markets still refuses a push
on one if ever configured).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports.nba.features.build_frame import tree_view
from core.folds import make_folds
from sports.nba.config.study_config import load_nba_study

from core.calibration.market import (
    apply_distribution_frame,
    discrete_normal_pmf,
    margin_cdf_above,
    margin_pmf_at,
    pmf_median,
    total_probabilities,
)

# The fair-line helper keeps its historical intra-module name.
_pmf_median = pmf_median


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrete normal PMF helpers (integer support, basketball point mechanics)
# ---------------------------------------------------------------------------



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
        # MARKET basis (§11, r7): the market model selects its own
        # contract — never the moneyline's feature list.
        Xv = tree_view(df, study=study, basis="market")
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
# sigma estimation / calibration â€” pooled OOF residuals, one scalar pair
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
# Distribution engine â€” everything derives from (mu_h, mu_a, sigma_m, sigma_t)
# ---------------------------------------------------------------------------


def game_distribution(mu_h: float, mu_a: float, sigma_margin: float,
                      sigma_total: float, study) -> dict:
    """All distributional outputs for one game from the joint score model.

    The full-game moneyline is strictly TWO-way (p_home_win +
    p_away_win = 1): an NBA game cannot tie, so ``p_tie`` is STRUCTURALLY
    ZERO â€” the margin==0 continuous mass splits proportionally to the
    side shares (a documented, deterministic convention â€” not a
    fabricated 50/50 tie outcome; no tie outcome exists in the sport)."""
    mc = study.market
    support_m = np.arange(-mc.margin_pmf_max, mc.margin_pmf_max + 1)
    support_t = np.arange(0, mc.total_pmf_max + 1)
    mu_margin = mu_h - mu_a
    mu_total = mu_h + mu_a
    pmf_m = discrete_normal_pmf(mu_margin, sigma_margin, support_m)
    pmf_t = discrete_normal_pmf(mu_total, sigma_total, support_t)

    # Continuous tie mass (margin == 0): split deterministically by the
    # side shares of the strictly-positive/strictly-negative masses.
    p_tie_mass = margin_pmf_at(pmf_m, support_m, 0.0)
    p_home_pos = margin_cdf_above(pmf_m, support_m, 0.0)
    p_away_neg = float(pmf_m[support_m < 0].sum())
    safe_den = p_home_pos + p_away_neg
    if np.isfinite(safe_den) and safe_den > 0:
        share_h = p_home_pos / safe_den
    else:
        share_h = 0.5
    p_home_win = np.clip(p_home_pos + p_tie_mass * share_h, 0.0, 1.0)
    p_away_win = 1.0 - p_home_win

    out = {
        "mu_h": float(mu_h), "mu_a": float(mu_a),
        "mu_margin": float(mu_margin), "mu_total": float(mu_total),
        "p_home_win_derived": float(p_home_win),
        "p_away_win_derived": float(p_away_win),
        "p_tie": 0.0,                       # STRUCTURALLY zero in the NBA
        "p_home_regulation": float(p_home_win),
        "p_away_regulation": float(p_away_win),
    }
    out["fair_spread"] = _pmf_median(pmf_m, support_m)
    out["fair_total"] = _pmf_median(pmf_t, support_t)
    out["p_over_fair"] = total_probabilities(
        pmf_t, support_t, float(out["fair_total"]))[0]
    out["p_cover_fair"] = margin_cdf_above(
        pmf_m, support_m, float(out["fair_spread"]))
    # spread grid: p_home_cover_L / p_push_L for integer L
    for L in mc.spread_lines:
        out[f"p_home_cover_{L}"] = margin_cdf_above(pmf_m, support_m, float(L))
        out[f"p_push_{L}"] = margin_pmf_at(pmf_m, support_m, float(L))
    # half-stop lines (none in the Phase 5 grid): no push band
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
    duplicated) â€” the engine is the single authority for these values."""
    rows = [game_distribution(r.mu_h, r.mu_a, sigma_margin, sigma_total,
                              study)
            for r in df.itertuples(index=False)]
    return apply_distribution_frame(df, rows)


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
