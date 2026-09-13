"""Sport-agnostic adapter harness for the optimization runner.

This module owns ONLY the shared, sport-independent machinery the
runner's scope bindings need: the durable-store root, the derivation
matrix builder, the production-policy member factory, the raw-column
selector, and the slate-availability check. No ingestion, derivation, or
model code is introduced here.

Per-sport adapter bindings live OUTSIDE core, in each sport's own
``optimization`` module:

* ``sports/mlb/optimization.py`` -> ``mlb_adapters``
* ``sports/nfl/optimization.py`` -> ``nfl_adapters``
* ``sports/nhl/optimization.py`` -> ``nhl_adapters``
* ``sports/nba/optimization.py`` -> ``nba_adapters``

and are composed by the thin registry ``sports/optimization_adapters.py``
(``SPORT_ADAPTER_BUILDERS`` / ``all_adapters``), also outside core.
Policy section 18: core never imports ``sports.*`` — at module level or
function-local — so this module imports nothing from ``sports`` at all.

Scope kinds (unchanged):
* moneyline — binary home_win probability; the linear member path uses
  the production train-only imputation + standard-scaling policy.
* market — continuous margin regressand (the distributional engine's
  target); evaluated through the same walk-forward OOF machinery with a
  ridge member on the same preprocessing. Bounds and gates are identical
  and independent per model.

A sport without a real store returns {} — the runner reports the scope
as DEFERRED (never fabricated, never silently skipped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.experiments.search_space import spec_from_name

#: Durable store root. Read as a MODULE-LEVEL name so per-sport modules
#: can bind it (``from core.experiments.adapters import STORE``) and
#: tests can retarget their own sport's binding with
#: ``monkeypatch.setattr(<sport>.optimization, "STORE", tmp_path)``.
STORE = Path("store")

# Identity/audit columns never eligible as candidate raw fields.
_META_COLS = frozenset({
    "game_id", "game_date", "gameday", "season", "week", "home_team",
    "away_team", "start_time_utc", "venue", "game_state", "game_type",
    "neutral_site", "home_score", "away_score", "home_win", "margin",
    "total", "home_record", "away_record", "game_pk",
})


# ---------------------------------------------------------------------------
# Shared matrix / member machinery (production policy, per scope kind)
# ---------------------------------------------------------------------------
def _fit_linear_preprocessor(X: pd.DataFrame):
    """Production TrainFoldPreprocessor policy: train-median imputation
    + standard scaling, all-NaN columns impute to neutral 0.0."""
    medians = X.median(numeric_only=True)
    means = X.mean(numeric_only=True)
    stds = X.std(numeric_only=True).replace(0.0, 1.0)
    return (medians.fillna(0.0), means.fillna(0.0), stds.fillna(1.0))


def _apply_preprocessor(X: pd.DataFrame, med, mean, std) -> np.ndarray:
    Xf = X.astype(float).fillna(med)
    return ((Xf - mean) / std).to_numpy(dtype=np.float64)


def matrix_builder(raw_fields: tuple[str, ...]):
    """Matrix builder: incumbent features come from the frame's served
    columns; generated candidates resolve through the shared derivation
    evaluator via their deterministic names (identical evaluator in both
    places). Unknown columns ship true NaN with explicit routing."""

    def matrix(df: pd.DataFrame, feats) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for f in feats:
            if f in df.columns:
                cols[f] = df[f].astype(float)
                continue
            spec = spec_from_name(f, raw_fields)
            if spec is None:
                cols[f] = pd.Series(np.nan, index=df.index)
                continue
            from core.experiments.search_space import derive_column

            cols[f] = derive_column(
                df, spec, team_cols=("home_team", "away_team"),
                date_col="game_date").astype(float)
        return pd.DataFrame(cols, index=df.index)

    return matrix


def raw_columns_fn(exclude: tuple[str, ...] = ()):
    def raw_columns(df: pd.DataFrame) -> tuple[str, ...]:
        return tuple(c for c in df.columns
                     if c not in _META_COLS and c not in exclude
                     and pd.api.types.is_numeric_dtype(df[c]))
    return raw_columns


def _member_factory(kind: str, seed: int):
    """Production-policy member factory for one fold fit.

    moneyline: logistic regression on train-imputed + scaled values
    (same policy class as the production linear members; trees route NaN
    natively and the harness's linear proxy is the bounded-sweep stand-in
    across all candidates identically). predict returns PROBABILITIES.
    market: ridge regression on the same preprocessing (the distribution
    engine's linear regressand path); predict returns the regressand.
    """
    state: dict = {}

    def fit(X: pd.DataFrame, y: np.ndarray) -> None:
        med, mean, std = _fit_linear_preprocessor(X)
        state["pre"] = (med, mean, std)
        if kind == "market":
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=1.0, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression

            m = LogisticRegression(max_iter=1000, random_state=seed)
        m.fit(_apply_preprocessor(X, med, mean, std), y)
        state["model"] = m
        state["kind"] = kind

    def predict(X: pd.DataFrame) -> np.ndarray:
        med, mean, std = state["pre"]
        Xp = _apply_preprocessor(X, med, mean, std)
        if state["kind"] == "market":
            return state["model"].predict(Xp)
        return state["model"].predict_proba(Xp)[:, 1]

    return {"fit": fit, "predict": predict}


def member_factory_for(kind: str):
    def factory(seed: int):
        return _member_factory(kind, seed)
    return factory


def slate_check(slate: pd.DataFrame, feats) -> bool:
    return all(f in slate.columns for f in feats)
