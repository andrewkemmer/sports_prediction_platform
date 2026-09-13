"""NHL feature registry and controlled derivation factory.

Declares the served feature contract (``nhl-prod-v1``, frozen column
order) and the controlled derivation paths every model family consumes.
Missing-data policy (Phase 2.1 parity):

* matrix width is an invariant â€” absent columns ship as true NaN with a
  loud warning, never silently dropped;
* tree members route NaN natively â€” missingness fully preserved;
* linear members (logistic, mlp) use medians fit on TRAIN rows only,
  reused at apply time (validation, slate) â€” never refit on the data
  being predicted;
* entirely-unavailable columns (no training observations) are routed
  explicitly as unavailable and surfaced in features_metadata warnings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.contracts import (
    FeatureContract,
    FeatureSpec,
    validate_columns_have_metadata,
    validate_metadata_is_reachable,
)
from core.features import require_classes_for

logger = logging.getLogger(__name__)


class NHLRegistryError(ValueError):
    """Raised when a feature-registry derivation would violate policy."""


# ---------------------------------------------------------------------------
# Versioned per-model contracts (Phase 7.5b canonical registry ownership)
# ---------------------------------------------------------------------------
# The registry OWNS both contracts: explicit, independent tuple([...])
# constructions (no aliases, no shared objects). Content is byte-identical
# to the historic single pool (prior version: nhl-prod-v1) â€” behavior-neutral.

MONEYLINE_FEATURE_COLS: tuple[str, ...] = tuple([
    "elo_diff", "win_pct_diff", "rest_days_diff", "b2b_home", "b2b_away",
    "ewm_net_goals_diff", "travel_miles_diff", "div_game", "prime_time",
])
MONEYLINE_CONTRACT_VERSION = "nhl-moneyline-v75"

MARKET_FEATURE_COLS: tuple[str, ...] = tuple([
    "elo_diff", "win_pct_diff", "rest_days_diff", "b2b_home", "b2b_away",
    "ewm_net_goals_diff", "travel_miles_diff", "div_game", "prime_time",
])
MARKET_CONTRACT_VERSION = "nhl-market-v75"

#: Prior provenance version (single shared pool before Phase 7.5 Task 2).
PRIOR_CONTRACT_VERSION = "nhl-prod-v1"


# ---------------------------------------------------------------------------
# Feature contract (shared semantic interface)
# ---------------------------------------------------------------------------

_SPEC_DEFS: dict[str, dict[str, str]] = {
    "elo_diff": {
        "summary": "Pre-game Elo difference (home âˆ’ away)",
        "definition": ("elo_home_entering âˆ’ elo_away_entering; Elo update "
                       "r += K*(actual âˆ’ expected), expected = "
                       "1/(1+10**((r_opp âˆ’ r_self)/400))"),
        "source": "NHL Stats API schedules (decided REG games)",
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
        "summary": "Days since previous game, home âˆ’ away",
        "definition": "(game_date_t âˆ’ game_date_{tâˆ’1}).days per team",
        "source": "NHL Stats API schedules",
        "window": "1 game",
        "units": "days",
        "direction": "positive favors home",
    },
    "b2b_home": {
        "summary": "Home team on a back-to-back",
        "definition": "1.0 when the home team played the previous calendar day",
        "source": "NHL Stats API schedules",
        "window": "1 game",
        "units": "flag",
        "direction": "positive = home fatigued",
    },
    "b2b_away": {
        "summary": "Away team on a back-to-back",
        "definition": "1.0 when the away team played the previous calendar day",
        "source": "NHL Stats API schedules",
        "window": "1 game",
        "units": "flag",
        "direction": "positive = away fatigued",
    },
    "ewm_net_goals_diff": {
        "summary": "EWM net goals/game difference",
        "definition": ("ewm(halflife=2).mean() of team net goals over "
                       "strictly-prior games"),
        "source": "decided game scores",
        "window": "decaying (halflife=2 games)",
        "units": "goals",
        "direction": "positive favors home",
    },
    "travel_miles_diff": {
        "summary": "Travel distance difference",
        "definition": ("haversine(away_arena, venue) âˆ’ "
                       "haversine(home_arena, venue)"),
        "source": "committed NHL arena table + schedules",
        "window": "static pre-game fact",
        "units": "miles",
        "direction": "positive = away travels farther",
    },
    "div_game": {
        "summary": "Divisional matchup",
        "definition": "1.0 when both teams share a division",
        "source": "committed team-division mapping",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
    "prime_time": {
        "summary": "Evening start flag",
        "definition": "1.0 when UTC start hour â‰¥ 19",
        "source": "NHL Stats API schedules",
        "window": "static pre-game fact",
        "units": "flag",
        "direction": "context",
    },
}


def validate_registry() -> None:
    """Validate both declared contract tuples against the registry metadata.

    NHL's metadata model is the ``_SPEC_DEFS`` mapping; the shared
    ``core.contracts`` interface compares names only, so this sport keeps
    its existing model â€” no parallel registry is created. Raises on drift:
    a declared column with no spec, or a spec no declared tuple serves.
    """
    metadata = frozenset(_SPEC_DEFS)
    validate_columns_have_metadata(
        "nhl", "moneyline", MONEYLINE_FEATURE_COLS, metadata)
    validate_columns_have_metadata(
        "nhl", "market", MARKET_FEATURE_COLS, metadata)
    validate_metadata_is_reachable(
        "nhl", metadata,
        set(MONEYLINE_FEATURE_COLS) | set(MARKET_FEATURE_COLS))
    # Â§7.1 B-001-RESIDUAL: every contract field carries an availability
    # class â€” the PIT metadata layer is closed over the contracts.
    require_classes_for(
        "nhl", set(MONEYLINE_FEATURE_COLS) | set(MARKET_FEATURE_COLS))


def build_market_feature_contract(
        version: str = MARKET_CONTRACT_VERSION) -> FeatureContract:
    """The NHL market view as a versioned contract (§11: the
    market model documents its own basis, never inherits the moneyline's).
    Members mirror the market margin/totals regressor."""
    validate_registry()
    specs = []
    for name in MARKET_FEATURE_COLS:
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
            members=("score_regressor",),
            tooltip=d["summary"],
        ))
    return FeatureContract(sport="nhl", version=version,
                           features=tuple(specs))


def build_feature_contract(
        version: str = MONEYLINE_CONTRACT_VERSION) -> FeatureContract:
    """The NHL feature contract from the frozen registry (moneyline-only).

    ``version`` defaults to the active registry version; callers may pass
    the active study contract's moneyline version so emitted feature
    metadata is stamped from the live contract, never a hardcoded literal.
    """
    validate_registry()
    specs = []
    for name in MONEYLINE_FEATURE_COLS:
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
    return FeatureContract(sport="nhl", version=version,
                           features=tuple(specs))


# ---------------------------------------------------------------------------
# Controlled derivation factory: model-family matrices
# ---------------------------------------------------------------------------


def ensure_feature_columns(df: pd.DataFrame,
                           feature_cols: tuple[str, ...] | None = None,
                           *, warn: bool = True) -> pd.DataFrame:
    """Matrix-width invariant: every declared column exists (missing â†’ NaN,
    loud warning) and column order matches the registry."""
    cols = list(feature_cols or MONEYLINE_FEATURE_COLS)
    out = df.reindex(columns=cols)
    if warn:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(
                "feature matrix missing %d declared columns (shipped as "
                "true NaN â€” never silently dropped): %s", len(missing),
                missing)
    return out


def unavailable_columns(df: pd.DataFrame,
                        feature_cols: tuple[str, ...] | None = None
                        ) -> list[str]:
    """Declared columns with ZERO observations in the frame â€” routed
    explicitly as unavailable (surfaced in features_metadata warnings)."""
    cols = list(feature_cols or MONEYLINE_FEATURE_COLS)
    out = []
    for c in cols:
        col = df[c] if c in df.columns else None
        if col is None or col.notna().sum() == 0:
            out.append(c)
    return out


def unavailable_warnings(df: pd.DataFrame,
                         feature_cols: tuple[str, ...] | None = None
                         ) -> list[str]:
    return [f"unavailable: {c} (no observations in source data â€” routed "
            f"as explicitly unavailable per missing-data policy; not "
            f"fabricated)" for c in unavailable_columns(df, feature_cols)]


def _impute_train_median(train: pd.DataFrame,
                         feature_cols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Fit train-only medians; identify entirely-unavailable columns.

    A column with no training observations ships as NaN to tree members
    and imputes to the documented neutral 0.0 for linear members â€” never a
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
    """Apply fitted train medians (validation/slate rows) â€” never refit."""
    return X.astype(float).fillna(medians)
