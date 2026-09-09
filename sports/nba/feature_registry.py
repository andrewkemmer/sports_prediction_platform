"""NBA feature registry and controlled derivation factory.

Declares the served feature contract (``nba-prod-v1``, frozen column
order) and the controlled derivation paths every model family consumes.
Missing-data policy (Phase 2.1 parity):

* matrix width is an invariant — absent columns ship as true NaN with a
  loud warning, never silently dropped;
* tree members route NaN natively — missingness fully preserved;
* linear members (logistic, mlp) use medians fit on TRAIN rows only,
  reused at apply time (validation, slate) — never refit on the data
  being predicted;
* entirely-unavailable columns (no training observations) are routed
  explicitly as unavailable and surfaced in features_metadata warnings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.contracts import FeatureContract, FeatureSpec
from sports.nba.study_config import FEATURE_COLUMNS, load_nba_study

logger = logging.getLogger(__name__)


class NBARegistryError(ValueError):
    """Raised when a feature-registry derivation would violate policy."""


# ---------------------------------------------------------------------------
# Feature contract (shared semantic interface)
# ---------------------------------------------------------------------------

_SPEC_DEFS: dict[str, dict[str, str]] = {
    "elo_diff": {
        "summary": "Pre-game Elo difference (home − away)",
        "definition": ("elo_home_entering − elo_away_entering; Elo update "
                       "r += K*(actual − expected), expected = "
                       "1/(1+10**((r_opp − r_self)/400))"),
        "source": "NBA Stats API schedules (decided REG games)",
        "window": "full history (iterative)",
        "units": "rating points",
        "direction": "positive favors home",
    },
    "win_pct_diff": {
        "summary": "Trailing win% difference",
        "definition": "mean(team_win) over the team's prior 12 games",
        "source": "decided game outcomes",
        "window": "12 games",
        "units": "win rate",
        "direction": "positive favors home",
    },
    "rest_days_diff": {
        "summary": "Days since previous game, home − away",
        "definition": "(game_date_t − game_date_{t−1}).days per team",
        "source": "NBA Stats API schedules",
        "window": "1 game",
        "units": "days",
        "direction": "positive favors home",
    },
    "b2b_home": {
        "summary": "Home team on a back-to-back",
        "definition": "1.0 when the home team played the previous calendar day",
        "source": "NBA Stats API schedules",
        "window": "1 game",
        "units": "flag",
        "direction": "positive = home fatigued",
    },
    "b2b_away": {
        "summary": "Away team on a back-to-back",
        "definition": "1.0 when the away team played the previous calendar day",
        "source": "NBA Stats API schedules",
        "window": "1 game",
        "units": "flag",
        "direction": "positive = away fatigued",
    },
    "ewm_net_pts_diff": {
        "summary": "EWM net points/game difference",
        "definition": ("ewm(halflife=3).mean() of team net points over "
                       "strictly-prior games"),
        "source": "decided game scores",
        "window": "decaying (halflife=3 games)",
        "units": "points",
        "direction": "positive favors home",
    },
    "travel_miles_diff": {
        "summary": "Travel distance difference",
        "definition": ("haversine(away_arena, venue) − "
                       "haversine(home_arena, venue)"),
        "source": "committed NBA arena table + schedules",
        "window": "static pre-game fact",
        "units": "miles",
        "direction": "positive = away travels farther",
    },
    "conf_game": {
        "summary": "Intra-conference matchup",
        "definition": "1.0 when both teams share a conference",
        "source": "committed team-conference mapping",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
    "prime_time": {
        "summary": "Evening start flag",
        "definition": "1.0 when UTC start hour ≥ 19",
        "source": "NBA Stats API schedules",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
}


def build_feature_contract() -> FeatureContract:
    """The NBA feature contract from the frozen registry."""
    specs = []
    for name in FEATURE_COLUMNS:
        d = _SPEC_DEFS[name]
        specs.append(FeatureSpec(
            name=name,
            summary=d["summary"],
            definition=d["definition"],
            formula=d["definition"],
            source=d["source"],
            window=d["window"],
            units=d["units"],
            direction=d["direction"],
            members=("xgboost", "lightgbm", "logistic", "randomforest",
                     "mlp"),
            tooltip=d["summary"],
        ))
    return FeatureContract(sport="nba", version="nba-prod-v1",
                           features=tuple(specs))


# ---------------------------------------------------------------------------
# Controlled derivation factory: model-family matrices
# ---------------------------------------------------------------------------


def ensure_feature_columns(df: pd.DataFrame,
                           feature_cols: tuple[str, ...] | None = None,
                           *, warn: bool = True) -> pd.DataFrame:
    """Matrix-width invariant: every declared column exists (missing → NaN,
    loud warning) and column order matches the registry."""
    cols = list(feature_cols or FEATURE_COLUMNS)
    out = df.reindex(columns=cols)
    if warn:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(
                "feature matrix missing %d declared columns (shipped as "
                "true NaN — never silently dropped): %s", len(missing),
                missing)
    return out


def unavailable_columns(df: pd.DataFrame,
                        feature_cols: tuple[str, ...] | None = None
                        ) -> list[str]:
    """Declared columns with ZERO observations in the frame — routed
    explicitly as unavailable (surfaced in features_metadata warnings)."""
    cols = list(feature_cols or FEATURE_COLUMNS)
    out = []
    for c in cols:
        col = df[c] if c in df.columns else None
        if col is None or col.notna().sum() == 0:
            out.append(c)
    return out


def unavailable_warnings(df: pd.DataFrame,
                         feature_cols: tuple[str, ...] | None = None
                         ) -> list[str]:
    return [f"unavailable: {c} (no observations in source data — routed "
            f"as explicitly unavailable per missing-data policy; not "
            f"fabricated)" for c in unavailable_columns(df, feature_cols)]


def _impute_train_median(train: pd.DataFrame,
                         feature_cols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Fit train-only medians; identify entirely-unavailable columns.

    A column with no training observations ships as NaN to tree members
    and imputes to the documented neutral 0.0 for linear members — never a
    fabricated signal; the column is listed in ``unavailable`` for the
    explicit features_metadata markers.
    """
    X = train.reindex(columns=list(feature_cols)).astype(float)
    medians = X.median(numeric_only=True)
    unavailable = [c for c in feature_cols
                   if c not in X.columns or X[c].notna().sum() == 0]
    medians = medians.fillna(0.0)
    return X, medians, unavailable


def apply_impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Apply fitted train medians (validation/slate rows) — never refit."""
    return X.astype(float).fillna(medians)
