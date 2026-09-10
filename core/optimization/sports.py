"""Per-sport candidate pools and model scopes.

Declares, for each sport, the raw fields available in that sport's
durable catalog/store and binds them to each model's INDEPENDENT
optimization scope (binary moneyline vs market/run-engine distribution
model). A model's feature list is never inferred from the other's.

Raw-field policy: only fields already present in the sport's durable
store or produced by the sport's existing catalog/feature engine (all
schedule-derivable, PIT-safe, published before game start) are eligible.
No pbp/event value enters before its public release time; no
fabricated/unavailable fields are listed.
"""

from __future__ import annotations

from core.optimization.models import ModelScope

# ---------------------------------------------------------------------------
# Raw field pools — durable-store / catalog fields per sport
# ---------------------------------------------------------------------------
# Each entry is a per-side column the sport's feature engine ALREADY
# computes per game (shift(1)-prior ladder values); candidates compose
# strictly-prior derivations over these (diffs, per-team rolling
# windows, pairwise products). No fabricated or unavailable fields.
NBA_RAW_FIELDS = (
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "rest_days_home", "rest_days_away", "ewm_net_pts_home",
    "ewm_net_pts_away", "b2b_home", "b2b_away",
)
NFL_RAW_FIELDS = (
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "rest_days_home", "rest_days_away", "ewm_net_pts_home",
    "ewm_net_pts_away", "ewm_ypp_home", "ewm_ypp_away",
)
NHL_RAW_FIELDS = (
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "rest_days_home", "rest_days_away", "ewm_net_goals_home",
    "ewm_net_goals_away", "b2b_home", "b2b_away",
)
MLB_RAW_FIELDS = (
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "rest_days_home", "rest_days_away", "sp_era_home", "sp_era_away",
    "sp_k9_home", "sp_k9_away",
)


def _per_side(raw: tuple[str, ...]) -> tuple[str, ...]:
    """Per-side expansions the derivations operate on (home/away)."""
    out: list[str] = []
    for f in raw:
        out.extend((f"{f}__home", f"{f}__away"))
    return tuple(out)


SPORT_RAW_FIELDS = {
    "nba": NBA_RAW_FIELDS,
    "nfl": NFL_RAW_FIELDS,
    "nhl": NHL_RAW_FIELDS,
    "mlb": MLB_RAW_FIELDS,
}

# Market scopes share the moneyline side-column pool: the run-engine
# distribution raws (mu_total/mu_margin) are produced by the run-engine
# ladder, not build_game_features, and are NOT present in the decided
# frame — listing them would fabricate unavailable candidates. The
# market scope's independent sweep therefore runs the same documented
# derivations against its own regressand (margin) with regression
# metrics. Kept as a mapping so future sports may add frame-present
# distribution raws without changing the scope builder.
MARKET_EXTRA_FIELDS: dict[str, tuple[str, ...]] = {}


def model_scope(sport: str, model: str, prod_features: tuple[str, ...],
                target: str, horizon_field: str = "") -> ModelScope:
    """Build one sport/model scope with that sport's raw pool."""
    if sport not in SPORT_RAW_FIELDS:
        raise ValueError(f"unknown sport '{sport}'")
    raw = SPORT_RAW_FIELDS[sport]
    if model == "market":
        raw = raw + MARKET_EXTRA_FIELDS.get(sport, ())
    return ModelScope(
        sport=sport, model=model, raw_fields=raw,
        prod_features=prod_features, target=target,
        horizon_field=horizon_field,
    )


def default_scopes(sport: str, study_features: tuple[str, ...],
                   ) -> tuple[ModelScope, ...]:
    """The sport's two independent scopes sharing the incumbent served
    feature list and the same side-column raw pool: binary moneyline
    (target home_win, probability metrics) and the market model (target
    margin — the distributional engine's regressand, regression
    metrics). Each model's feature list is never inferred from the
    other's; only the target/metric/member differ."""
    return (
        model_scope(sport, "moneyline", study_features, "home_win"),
        model_scope(sport, "market", study_features, "margin",
                    horizon_field="game_date"),
    )
