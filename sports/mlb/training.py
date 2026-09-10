"""MLB walk-forward training geometry.

Implements the reference repo's expanding-window walk-forward splits on the
canonical decided frame: each validation window is ``retrain_cadence_days``
wide, the training set is all games strictly before the validation window
start, and windows are non-overlapping and chronological. A final fold
whose full cadence window would overrun the frame's tail runs PARTIALLY
into the tail (leakage-free: train stays strictly < val_start).

Phase 2 freezes the ensemble stack (no optimization); the fold geometry
comes from the study config, never from the run window.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fold geometry lives in core.folds (Phase 7.5 Task 1 consolidation).
from core.folds import walk_forward_splits  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen ensemble member configs (Phase 2: no optimization)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "max_depth": 2, "min_child_weight": 8, "gamma": 2.13,
    "subsample": 0.60, "colsample_bytree": 0.56, "learning_rate": 0.058,
    "random_state": RANDOM_SEED, "eval_metric": "logloss",
    "enable_categorical": True,
}
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20

LIGHTGBM_PARAMS = {
    "n_estimators": 50, "max_depth": 5, "num_leaves": 6,
    "min_child_samples": 59, "min_gain_to_split": 1.745,
    "bagging_fraction": 0.556, "bagging_freq": 1, "feature_fraction": 0.749,
    "learning_rate": 0.053, "random_state": RANDOM_SEED, "verbose": -1,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16), "alpha": 0.01, "early_stopping": True,
    "validation_fraction": 0.15, "max_iter": 300,
    "learning_rate": "constant", "learning_rate_init": 0.001,
    "batch_size": "auto", "n_iter_no_change": 10, "activation": "relu",
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 800, "max_depth": 6, "min_samples_leaf": 17,
    "min_samples_split": 6, "max_features": "log2", "bootstrap": True,
    "random_state": RANDOM_SEED, "n_jobs": -1,
}

LOGISTIC_PARAMS = {
    "C": 0.5, "max_iter": 1000, "random_state": RANDOM_SEED,
}

COIN_FLIP_THRESHOLD = 0.02


def _ece(probs: list[float], outcomes: list[float],
         n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    p = np.clip(np.asarray(probs, float), 0.0, 1.0)
    y = np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(y[m].mean() - p[m].mean())
    return round(float(total / max(len(y), 1)), 6)


ENSEMBLE_WEIGHTS = {
    "xgboost": 0.25,
    "lightgbm": 0.25,
    "logistic": 0.30,
    "randomforest": 0.10,
    "mlp": 0.10,
}


# ---------------------------------------------------------------------------
# Moneyline ensemble training (Phase 2 frozen stack)
# ---------------------------------------------------------------------------
def _feature_matrix(games: pd.DataFrame,
                    feature_cols: tuple[str, ...]) -> np.ndarray:
    """Feature matrix preserving NaN (reference semantics).

    Missing observations stay NULL: tree members (xgboost, lightgbm,
    randomforest) route NaN natively; only the logistic/mlp members —
    which cannot consume NaN — get TRAIN-fold-median imputation applied
    inside ``_fit_member`` (fit on X_tr only, never on validation/slate
    rows). Columns absent from the frame come back as all-NaN (never
    silently dropped) with one loud warning: WIDTH IS AN INVARIANT.
    """
    missing = [c for c in feature_cols if c not in games.columns]
    if missing:
        logger.warning(
            "Feature matrix: %d/%d expected columns absent (%s%s) — filled "
            "as NULL (tree members route NaN; logistic/mlp impute from "
            "train medians); investigate the source frame",
            len(missing), len(feature_cols), ", ".join(missing[:6]),
            " …" if len(missing) > 6 else "")
    return games.reindex(columns=list(feature_cols)).to_numpy(dtype=float)


def _impute_train_median(X_tr: np.ndarray,
                         X_apply: np.ndarray) -> np.ndarray:
    """Reference-exact train-only median imputation (``_impute_median``).

    Medians are computed on TRAIN ROWS ONLY and reused at apply time
    (validation, slate). A column that is ALL-NaN in training has no
    median: it is EXPLICITLY UNAVAILABLE — filled with 0.0 (the
    reference's documented fallback so the logistic/mlp members stay
    usable) and NEVER with a fabricated statistic. Missingness semantics
    are preserved for tree members (they never see this function's
    output); the fill is a routing device, not invented signal.
    """
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_tr, axis=0) if len(X_tr) \
            else np.zeros(X_tr.shape[1])
    unavailable = np.isnan(medians)
    if unavailable.any():
        logger.warning(
            "Imputation: %d/%d feature columns are entirely unavailable in "
            "training (all-NaN) — routed as explicitly unavailable (0.0) "
            "for the linear members only: %s",
            int(unavailable.sum()), X_tr.shape[1],
            ", ".join(str(i) for i in np.where(unavailable)[0][:8]))
    medians = np.where(unavailable, 0.0, medians)
    X = np.asarray(X_apply, dtype=float).copy()
    idx = np.isnan(X)
    X[idx] = np.take(np.asarray(medians, dtype=float), idx.nonzero()[1])
    return X


def _fit_member(name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray) -> Any:
    """Fit one ensemble member with its frozen params (deterministic)."""
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(**XGBOOST_PARAMS,
                                  n_estimators=XGBOOST_FOLD_ROUNDS,
                                  early_stopping_rounds=XGBOOST_EARLY_STOP)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            return model
        except ImportError:
            logger.warning("xgboost not available — member skipped")
            return None
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation
            model = LGBMClassifier(**LIGHTGBM_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      eval_metric="logloss",
                      callbacks=[early_stopping(50, verbose=False),
                                 log_evaluation(0)])
            return model
        except ImportError:
            logger.warning("lightgbm not available — member skipped")
            return None
    if name == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # Missing-data policy: medians fit on TRAIN ONLY (never val/slate);
        # all-NaN training columns are routed as explicitly unavailable.
        X_tr_i = _impute_train_median(X_tr, X_tr)
        X_va_i = _impute_train_median(X_tr, X_va)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
        ])
        pipe.fit(X_tr_i, y_tr)
        pipe._mlb_apply_impute_ = lambda Xa: _impute_train_median(X_tr, Xa)
        return pipe
    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**RF_PARAMS)
        # RF via train-fold medians; medians fit on train rows only.
        X_tr_i = _impute_train_median(X_tr, X_tr)
        model.fit(X_tr_i, y_tr)
        # carried for predict-time routing (same train medians)
        model._mlb_apply_impute_ = lambda Xa: _impute_train_median(X_tr, Xa)
        return model
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X_tr_i = _impute_train_median(X_tr, X_tr)
        X_va_i = _impute_train_median(X_tr, X_va)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(**MLP_PARAMS)),
        ])
        pipe.fit(X_tr_i, y_tr)
        pipe._mlb_apply_impute_ = lambda Xa: _impute_train_median(X_tr, Xa)
        return pipe
    raise ValueError(f"unknown ensemble member: {name}")


def _member_predict(name: str, model: Any, X_va: np.ndarray) -> np.ndarray:
    apply_impute = getattr(model, "_mlb_apply_impute_", None)
    if apply_impute is not None:
        X_va = apply_impute(X_va)
    return model.predict_proba(X_va)[:, 1]


def train_moneyline_ensemble(
    games: pd.DataFrame,
    feature_cols: tuple[str, ...],
    study,
    *,
    min_val_games: int | None = None,
    min_train_games: int | None = None,
    max_eval_folds: int = 0,
) -> dict[str, Any]:
    """Walk-forward moneyline ensemble on the frozen candidate frame.

    Fold geometry, minimum rows, and OOF rules come from the study config —
    never from the run window. Returns pooled OOF predictions with
    prequential-calibrated twins, pooled metrics, adaptive (softmax-on-
    logloss) member weights, and the ``predictions_history`` frame the
    run engine merges for the agreement columns.
    """
    from core.oof import brier_score, log_loss, oof_metrics, roc_auc

    wf = study.walk_forward
    min_val = int(min_val_games if min_val_games is not None
                  else wf.min_val_games)
    min_train = int(min_train_games if min_train_games is not None
                    else wf.min_train_games)
    members = tuple(study.training.ensemble_members)

    folds = walk_forward_splits(
        games, retrain_cadence_days=wf.step_days,
        max_eval_folds=max_eval_folds,
        min_train_days=study.training.min_train_days_warmup)
    folds = [s for s in folds
             if len(s["val_games"]) >= min_val
             and len(s["train_games"]) >= min_train]
    if not folds:
        raise ValueError(
            "train_moneyline_ensemble: no usable walk-forward folds "
            "(check study warmup/min-games rules and frame size)")

    all_rows: list[dict] = []
    member_p: dict[str, list[float]] = {m: [] for m in members}
    member_cal: dict[str, list[float]] = {m: [] for m in members}
    blend_p: list[float] = []
    blend_cal: list[float] = []
    y_all: list[float] = []
    fold_logloss: dict[str, list[float]] = {m: [] for m in members}
    fold_frames: list[pd.DataFrame] = []

    weights = dict(ENSEMBLE_WEIGHTS)
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        X_tr = _feature_matrix(tr, feature_cols)
        X_va = _feature_matrix(va, feature_cols)
        y_tr = tr["home_win"].to_numpy(dtype=float)
        y_va = va["home_win"].to_numpy(dtype=float)
        probs: dict[str, np.ndarray] = {}
        for name in members:
            try:
                model = _fit_member(name, X_tr, y_tr, X_va, y_va)
                if model is None:
                    continue
                p = _member_predict(name, model, X_va)
                probs[name] = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
                fold_logloss[name].append(log_loss(
                    y_va.tolist(), probs[name].tolist()))
            except Exception as exc:  # member failure must not kill a fold
                logger.warning("Member %s fold %d failed: %s",
                               name, split["fold_idx"], exc)
        if not probs:
            continue
        # Adaptive softmax weights from PRIOR+current fold logloss
        # (temperature 1.0, floor 0.02, cap 0.60 — frozen Phase 2 rules).
        live = [n for n in probs if fold_logloss[n]]
        if live:
            lls = {n: float(np.mean(fold_logloss[n])) for n in live}
            best = min(lls.values())
            raw = {n: ENSEMBLE_WEIGHTS.get(n, 0.1)
                   * math.exp(-(lls[n] - best)) for n in live}
            total = sum(raw.values())
            weights = {n: float(min(max(raw[n] / total, 0.02), 0.60))
                       for n in live}
            wsum = sum(weights.values())
            weights = {n: w / wsum for n, w in weights.items()}
        blend = np.zeros(len(va))
        for name, p in probs.items():
            blend += weights.get(name, 0.0) * p
        den = sum(weights.get(n, 0.0) for n in probs) or 1.0
        blend /= den
        blend = np.clip(blend, 1e-6, 1 - 1e-6)

        _d = pd.to_datetime(va["game_date"]).dt.strftime("%Y%m%d")
        fold_frame = pd.DataFrame({
            "game_pk": va["game_pk"].to_numpy(),
            "game_date": pd.to_datetime(va["game_date"]).dt.strftime(
                "%Y-%m-%d"),
            "game_id": _d + "_" + va["away_team"].astype(str) + "@"
                       + va["home_team"].astype(str),
            "home_team": va["home_team"].to_numpy(),
            "away_team": va["away_team"].to_numpy(),
            "home_win": y_va.astype(int),
            "home_win_prob_model": np.round(blend, 5),
            "fold_idx": split["fold_idx"],
        })
        fold_frames.append(fold_frame)
        all_rows.append(fold_frame)
        for name in probs:
            member_p[name].extend(probs[name].tolist())
        blend_p.extend(blend.tolist())
        y_all.extend(y_va.tolist())

    oof = pd.concat(all_rows, ignore_index=True) if all_rows \
        else pd.DataFrame()
    if oof.empty:
        raise ValueError("train_moneyline_ensemble: every fold failed")

    # Prequential calibration: fold k's Platt map fits PRIOR folds only.
    from sklearn.linear_model import LogisticRegression
    fold_idx = oof["fold_idx"].to_numpy()
    y_arr = oof["home_win"].to_numpy(float)
    p_arr = oof["home_win_prob_model"].to_numpy(float)
    cal = np.clip(p_arr, 1e-6, 1 - 1e-6).copy()
    for f in sorted(set(fold_idx.tolist())):
        prior = fold_idx < f
        cur = fold_idx == f
        if prior.sum() >= 50 and cur.any():
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(np.log(p_arr[prior] / (1 - p_arr[prior])).reshape(-1, 1),
                   y_arr[prior])
            z = np.log(p_arr[cur] / (1 - p_arr[cur])).reshape(-1, 1)
            cal[cur] = np.clip(lr.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)
    oof["home_win_prob_model_calibrated"] = np.round(cal, 5)
    blend_cal = cal.tolist()

    pooled = oof_metrics(p_arr.tolist(), y_arr.tolist())
    pooled_cal = oof_metrics(cal.tolist(), y_arr.tolist())

    # predictions_history frame (fixture schema: 12 columns).
    hist = oof[["game_id", "game_date", "home_team", "away_team",
                "home_win_prob_model", "home_win_prob_model_calibrated"]].copy()
    hist["home_score"] = pd.NA
    hist["away_score"] = pd.NA
    hist["home_win"] = oof["home_win"].astype(float)
    hist["model_pick"] = np.where(oof["home_win_prob_model"] >= 0.5,
                                  oof["home_team"], oof["away_team"])
    hist["actual_winner"] = np.where(oof["home_win"] == 1.0,
                                     oof["home_team"], oof["away_team"])
    hist["correct"] = (hist["model_pick"] == hist["actual_winner"]).astype(int)
    hist = hist[["game_id", "game_date", "home_team", "away_team",
                 "home_score", "away_score", "home_win",
                 "home_win_prob_model", "home_win_prob_model_calibrated",
                 "model_pick", "actual_winner", "correct"]]

    return {
        "oof": oof,
        "predictions_history": oof[["game_pk", "home_win_prob_model"]],
        "predictions_history_frame": hist,
        "weights": weights,
        "metrics": {**pooled,
                    "brier_calibrated": pooled_cal["brier"],
                    "logloss_calibrated": pooled_cal["logloss"],
                    "ece": _ece(p_arr.tolist(), y_arr.tolist()),
                    "ece_calibrated": _ece(cal.tolist(), y_arr.tolist())},
        "n_games": int(len(oof)),
        "n_folds": int(oof["fold_idx"].nunique()),
    }


def predict_slate_moneyline(
    decided: pd.DataFrame,
    slate: pd.DataFrame,
    feature_cols: tuple[str, ...],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply the frozen ensemble to the prediction window's slate rows.

    Mirrors the run engine's slate pattern (``predict_slate_runs``): each
    live member is refit on ALL decided games with its frozen params and
    predicts the slate rows; the blend uses the final walk-forward weights
    when given. Adds ``home_win_prob_model`` (rounded, in (0,1) via the
    same 1e-6 clip as OOF), ``away_win_prob_model``, ``model_pick``, and
    ``edge_home``/``edge_away`` = 0.0 (no market odds input in Phase 2 —
    the reference's no-odds path, never a fabricated edge).
    """
    if slate.empty:
        return slate
    # Width invariant: guarantee every feature column exists on the slate
    # (missing -> true NaN, routed per the missing-data policy — never a
    # silent column drop or a fabricated value).
    from sports.mlb.feature_registry import ensure_feature_columns
    slate = ensure_feature_columns(slate, feature_cols)
    X_slate = _feature_matrix(slate, feature_cols)
    X_tr = _feature_matrix(decided, feature_cols)
    y_tr = decided["home_win"].to_numpy(dtype=float)
    w = dict(weights or ENSEMBLE_WEIGHTS)
    live = [n for n in w if n in set(ENSEMBLE_WEIGHTS)]
    blend = np.zeros(len(slate))
    wsum = 0.0
    for name in live:
        try:
            model = _fit_member(name, X_tr, y_tr, X_tr, y_tr)
            if model is None:
                continue
            blend += w[name] * _member_predict(name, model, X_slate)
            wsum += w[name]
        except Exception as exc:  # member failure must not kill the board
            logger.warning("Slate member %s failed: %s", name, exc)
    if wsum <= 0:
        logger.warning("predict_slate_moneyline: no live members — "
                       "board probabilities stay NA")
        return slate
    p = np.clip(blend / wsum, 1e-6, 1 - 1e-6)
    out = slate.copy()
    out["home_win_prob_model"] = np.round(p, 4)
    out["away_win_prob_model"] = np.round(1 - p, 4)
    out["model_pick"] = np.where(p >= 0.5, out["home_team"],
                                 out["away_team"])
    out["edge_home"] = 0.0
    out["edge_away"] = 0.0
    return out
