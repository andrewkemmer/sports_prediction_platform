"""Production NFL moneyline: five-member ensemble (xgboost / lightgbm /
logistic / randomforest / mlp) with expanding walk-forward OOF, adaptive
OOF-derived ensemble weights, and Platt calibration fit ONLY on valid OOF
predictions.

Determinism: explicit study-configured seeds on every member;
preprocessing (train-only median imputation + scaling for linear/MLP) is
fit strictly on the training fold and reused at apply time. Trees consume
NaN natively and get the raw matrix (no imputation, no scaling).

Missing-data policy (Phase 2.1 parity): entirely-unavailable columns are
routed explicitly and surfaced as warnings — never silently dropped or
fabricated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports.nfl.feature_registry import (
    apply_impute,
    ensure_feature_columns,
    unavailable_columns,
)
from core.folds import make_folds
from sports.nfl.study_config import load_nfl_study

logger = logging.getLogger(__name__)

CLIP = 1e-7

LINEAR_MEMBERS = {"logistic", "mlp"}
TREE_MEMBERS = ("xgboost", "lightgbm", "randomforest")


# ---------------------------------------------------------------------------
# Preprocessing — fit on training data only
# ---------------------------------------------------------------------------


class TrainFoldPreprocessor:
    """Train-median imputation + standard scaling for linear/MLP members.

    Fit strictly on the training fold; the same fitted transform is applied
    to validation and serving frames. A column with NO training
    observations imputes to the documented neutral 0.0 deterministically —
    never a fabricated signal, and the column is listed as unavailable.
    """

    def __init__(self) -> None:
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.unavailable: list[str] = []

    def fit(self, X: pd.DataFrame) -> "TrainFoldPreprocessor":
        self.medians = X.median(numeric_only=True)
        self.means = X.mean(numeric_only=True)
        self.stds = X.std(numeric_only=True).replace(0.0, 1.0)
        self.unavailable = [c for c in X.columns
                            if X[c].notna().sum() == 0]
        if self.unavailable:
            logger.warning(
                "TrainFoldPreprocessor: %d entirely-unavailable columns "
                "(neutral 0.0 for linear members; explicitly tracked): %s",
                len(self.unavailable), self.unavailable)
        self.medians = self.medians.fillna(0.0)
        self.means = self.means.fillna(0.0)
        self.stds = self.stds.fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X.astype(float)
        Xf = Xf.fillna(self.medians)
        Xf = (Xf - self.means) / self.stds
        return Xf.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


def make_member(name: str, seed: int = 42):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3,
                             min_child_weight=5, gamma=1.0, subsample=0.8,
                             colsample_bytree=0.8, learning_rate=0.05,
                             random_state=seed, eval_metric="logloss",
                             verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=12,
                              min_child_samples=30, learning_rate=0.05,
                              subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed,
                              verbose=-1)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, max_depth=8,
                                      min_samples_leaf=20,
                                      min_samples_split=10,
                                      max_features="sqrt", bootstrap=True,
                                      random_state=seed, n_jobs=-1)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01,
                             early_stopping=True, validation_fraction=0.15,
                             max_iter=400, learning_rate_init=0.001,
                             n_iter_no_change=12, activation="relu",
                             random_state=seed)
    raise KeyError(f"unknown ensemble member {name!r}")


def member_matrix(name: str, df: pd.DataFrame, study=None) -> pd.DataFrame:
    """The feature matrix a member consumes (model-family representation)."""
    if name in LINEAR_MEMBERS:
        cols = list(study.moneyline_feature_cols) + ["is_home"] \
            if study else None
    else:
        cols = None
    if name in LINEAR_MEMBERS:
        study = study or load_nfl_study()
        keep = [c for c in list(study.moneyline_feature_cols) + ["is_home"]
                if c in df.columns]
        return df.reindex(columns=keep).astype(float)
    from sports.nfl.features import tree_view
    return tree_view(df, study=study)


def member_fit_input(name: str, X_raw: pd.DataFrame,
                     pre: TrainFoldPreprocessor | None):
    """The ONE authoritative representation handed to member.fit().

    Linear members -> the fitted preprocessor's ndarray (imputed+scaled);
    tree members -> the NAMED tree-view DataFrame, feature names preserved
    (pinned at BOTH fit and predict — never a raw ndarray at one site).
    """
    if name in LINEAR_MEMBERS:
        return pre.transform(X_raw)
    return X_raw


def _member_predict_proba(model, name: str, X_raw: pd.DataFrame,
                          pre: TrainFoldPreprocessor | None) -> np.ndarray:
    X = member_fit_input(name, X_raw, pre)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, CLIP, 1.0 - CLIP)


# ---------------------------------------------------------------------------
# Walk-forward OOF
# ---------------------------------------------------------------------------


def walk_forward_oof(game_df: pd.DataFrame, study,
                     date_col: str = "gameday") -> dict:
    """Expanding walk-forward OOF for every ensemble member + the ensemble.

    Returns a dict with: ``oof`` (game_id, gameday, fold_id, per-member
    p_home, ensemble), ``member_weights`` (adaptive OOF weights),
    ``fold_table`` (per-fold diagnostics). OOF rows are strictly
    out-of-sample: training sets are strictly prior to each window.
    """
    df = game_df.sort_values(date_col).reset_index(drop=True)
    fold_list = make_folds(df, study.oof_first_season,
                           cadence_days=study.walk_forward.step_days,
                           date_col=date_col,
                           val_scope="oof_season", end_of_day=True)

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        if len(train) < study.oof.min_training_rows:
            logger.info("fold %d skipped: %d training rows < minimum %d",
                        fold.fold_id, len(train),
                        study.oof.min_training_rows)
            continue
        y_train = train["home_win"].astype(int).to_numpy()

        member_p: dict[str, np.ndarray | None] = {}
        for name in study.training.ensemble_members:
            X_tr_raw = member_matrix(name, train, study)
            X_va_raw = member_matrix(name, val, study)
            pre = None
            if name in LINEAR_MEMBERS:
                pre = TrainFoldPreprocessor().fit(X_tr_raw)
            try:
                model = make_member(name, study.training.random_seed)
                model.fit(member_fit_input(name, X_tr_raw, pre), y_train)
                member_p[name] = _member_predict_proba(model, name,
                                                       X_va_raw, pre)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fold %s member %s failed: %s",
                               fold.fold_id, name, exc)
                member_p[name] = None

        rows = pd.DataFrame({
            "game_id": val["game_id"].to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val["season"].to_numpy(),
            "week": val["week"].to_numpy() if "week" in val.columns else None,
            "home_team": val["home_team"].to_numpy(),
            "away_team": val["away_team"].to_numpy(),
            "fold_id": fold.fold_id,
            "home_win": val["home_win"].astype(float).to_numpy(),
        })
        for name in study.training.ensemble_members:
            rows[f"p_{name}"] = member_p.get(name)

        fold_rows.append({
            "fold_id": fold.fold_id,
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
        })
        oof_parts.append(rows)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts \
        else pd.DataFrame()

    weights = _adaptive_weights(oof, study)
    ensemble_p = _blend(oof, weights, study)
    if len(oof):
        oof["p_ensemble"] = ensemble_p
    return {"oof": oof, "member_weights": weights,
            "fold_table": pd.DataFrame(fold_rows)}


def _adaptive_weights(oof: pd.DataFrame, study) -> dict[str, float]:
    """Softmax over pooled OOF AUC edges (study metric). Deterministic;
    derived ONLY from OOF rows. Falls back to study priors when OOF is
    unusable."""
    prior = (study.training.ensemble_weights
             or {"xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.20,
                 "randomforest": 0.15, "mlp": 0.15})
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


def _blend(oof: pd.DataFrame, weights: dict[str, float],
           study) -> np.ndarray:
    """Weighted ensemble probability; NaN members skipped per row."""
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


# ---------------------------------------------------------------------------
# Final full-history refit (production serving models)
# ---------------------------------------------------------------------------


def fit_final_models(game_df: pd.DataFrame, study) -> tuple[dict,
                                                            TrainFoldPreprocessor]:
    """Fit every member on ALL eligible settled history. Returns
    ({name: {'model', 'pre'}}, fitted_preprocessor)."""
    models: dict = {}
    y = game_df["home_win"].astype(int).to_numpy()
    for name in study.training.ensemble_members:
        X_raw = member_matrix(name, game_df, study)
        if name in LINEAR_MEMBERS:
            member_pre = TrainFoldPreprocessor().fit(X_raw)
            model = make_member(name, study.training.random_seed)
            model.fit(member_fit_input(name, X_raw, member_pre), y)
            models[name] = {"model": model, "pre": member_pre}
        else:
            model = make_member(name, study.training.random_seed)
            model.fit(member_fit_input(name, X_raw, None), y)
            models[name] = {"model": model, "pre": None}
    return models, TrainFoldPreprocessor()


def predict_slate(models: dict, slate_df: pd.DataFrame,
                  weights: dict[str, float], study) -> np.ndarray:
    """Production serving: per-member predictions on the slate, blended with
    the SAME weights the OOF derived (identical ensemble logic at serve)."""
    member_p: dict[str, np.ndarray | None] = {}
    for name in study.training.ensemble_members:
        entry = models.get(name)
        if entry is None:
            member_p[name] = None
            continue
        X_raw = member_matrix(name, slate_df, study)
        member_p[name] = _member_predict_proba(entry["model"], name,
                                               X_raw, entry["pre"])
    cols = [n for n in study.training.ensemble_members
            if member_p.get(n) is not None]
    if not cols:
        return np.full(len(slate_df), np.nan)
    P = np.column_stack([member_p[n] for n in cols])
    w = np.array([weights.get(n, 0.0) for n in cols])
    mask = np.isfinite(P)
    wv = np.where(mask, w[None, :], 0.0)
    wsum = wv.sum(axis=1)
    out = np.divide((P * wv).sum(axis=1), wsum,
                    out=np.full(len(P), np.nan), where=wsum > 0)
    return np.clip(out, CLIP, 1.0 - CLIP)


# ---------------------------------------------------------------------------
# Platt calibration — fit ONLY on valid OOF predictions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Board (slate) moneyline prediction — reference predict_games path
# ---------------------------------------------------------------------------


def predict_slate_moneyline(decided: pd.DataFrame, slate: pd.DataFrame,
                            study) -> pd.DataFrame:
    """Apply the final-fit ensemble to pre-game slate rows and attach
    ``home_win_prob_model`` / ``model_pick`` (the board contract).
    Deterministic; no leakage (the slate is pre-game rows only)."""
    if slate.empty:
        return slate
    models, _ = fit_final_models(decided, study)
    weights = study.training.ensemble_weights or {
        "xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.20,
        "randomforest": 0.15, "mlp": 0.15}
    p_home = predict_slate(models, slate, weights, study)
    out = slate.copy()
    out["home_win_prob_model"] = p_home
    out["away_win_prob_model"] = (1.0 - p_home) if np.isfinite(p_home).all() \
        else 1.0 - out["home_win_prob_model"]
    out["model_pick"] = np.where(
        p_home >= 0.5, out["home_team"], out["away_team"])
    return out
