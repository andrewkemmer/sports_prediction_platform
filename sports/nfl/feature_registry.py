"""NFL feature registry and controlled derivation factory.

Declares the served feature contract (``nfl-prod-v1``, reference-frozen
column order) and the controlled derivation paths every model family
consumes. Missing-data policy (Phase 2.1 parity):

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
from sports.nfl.study_config import FEATURE_COLUMNS, load_nfl_study

logger = logging.getLogger(__name__)


class NFLRegistryError(ValueError):
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
        "source": "nflverse schedules (decided REG games)",
        "window": "full history (iterative)",
        "units": "rating points",
        "direction": "positive favors home",
    },
    "win_pct_diff": {
        "summary": "Trailing win% difference",
        "definition": "mean(team_win) over the team's prior 12 games (ties 0.5)",
        "source": "decided game outcomes",
        "window": "12 games",
        "units": "win rate",
        "direction": "positive favors home",
    },
    "rest_days_diff": {
        "summary": "Days since previous game, home − away",
        "definition": "(gameday_t − gameday_{t−1}).days per team",
        "source": "nflverse schedules",
        "window": "1 game",
        "units": "days",
        "direction": "positive favors home",
    },
    "is_dome_home": {
        "summary": "Home venue is a dome",
        "definition": "1.0 roof ∈ {dome, closed}; 0.0 outdoors; NaN unknown",
        "source": "nflverse schedule roof field",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
    "ewm_net_pts_diff": {
        "summary": "EWM net points/game difference",
        "definition": ("ewm(halflife=2).mean() of team net points over "
                       "strictly-prior games"),
        "source": "decided game scores",
        "window": "decaying (halflife=2 games)",
        "units": "points",
        "direction": "positive favors home",
    },
    "ewm_ypp_diff": {
        "summary": "EWM yards/play difference",
        "definition": ("ewm(halflife=2).mean() of game-level yards/play "
                       "over strictly-prior games"),
        "source": "nflverse play-by-play",
        "window": "decaying (halflife=2 games)",
        "units": "yards/play",
        "direction": "positive favors home",
    },
    "pace_plays_min_diff": {
        "summary": "Trailing plays/minute difference",
        "definition": "mean(n_plays / elapsed_min) over prior 4 games",
        "source": "nflverse play-by-play",
        "window": "4 games",
        "units": "plays/minute",
        "direction": "context",
    },
    "rest_short_diff": {
        "summary": "Short-rest difference",
        "definition": "1.0 when rest_days < 7 else 0.0, home − away",
        "source": "nflverse schedules",
        "window": "1 game",
        "units": "flag",
        "direction": "positive = home on shorter rest",
    },
    "div_game": {
        "summary": "Divisional matchup",
        "definition": "nflverse div_game flag",
        "source": "nflverse schedules",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
    "travel_miles_diff": {
        "summary": "Travel distance difference",
        "definition": ("haversine(home_team_stadium, venue) − "
                       "haversine(away_team_stadium, venue)"),
        "source": "committed stadiums table + nflverse schedules",
        "window": "static pre-game fact",
        "units": "miles",
        "direction": "positive = away travels farther",
    },
    "altitude_home": {
        "summary": "Venue altitude",
        "definition": "stadium altitude in feet",
        "source": "committed stadiums table",
        "window": "static pre-game fact",
        "units": "feet",
        "direction": "context",
    },
    "prime_time": {
        "summary": "Evening kickoff flag",
        "definition": "1.0 when ET gametime hour ≥ 17",
        "source": "nflverse schedules",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
}


def build_feature_contract() -> FeatureContract:
    """The NFL feature contract from the frozen registry."""
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
    return FeatureContract(sport="nfl", version="nfl-prod-v1",
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


# ---------------------------------------------------------------------------
# Candidate frame assembly (decided + slate), mirroring the MLB pattern
# ---------------------------------------------------------------------------


def build_candidate_frame(decided: pd.DataFrame,
                          feature_cols: tuple[str, ...] | None = None
                          ) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Wide candidate frame for training: declared features + targets.

    Returns (candidate_frame, feature_cols). The frame keeps game keys and
    targets beside features for OOF assembly; targets are never model
    inputs (enforced by the training module's matrix routing).
    """
    cols = tuple(feature_cols or FEATURE_COLUMNS)
    out = ensure_feature_columns(decided, cols, warn=False)
    keep = ["game_id", "gameday", "season", "week", "home_team",
            "away_team", "home_score", "away_score", "margin", "total",
            "home_win", "home_record", "away_record", "stadium", "gametime"]
    for c in keep:
        if c in decided.columns and c not in out.columns:
            out[c] = decided[c]
    return out, cols
