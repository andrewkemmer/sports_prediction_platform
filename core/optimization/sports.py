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
MLB_RAW_FIELDS = ()  # unused — MLB names its sides ``*_home``/``*_away``
# MLB per-side pool for the moneyline scope: the per-side members of the
# frozen 64-column FEATURE_COLS moneyline view (the MLB convention:
# ``home_elo`` not ``elo_home``). The 13 pairs generated here are exactly
# the derivation bases the served frame carries.
MLB_SIDE_FIELDS = (
    "home_elo", "away_elo", "home_win_pct", "away_win_pct",
    "rest_days_home", "rest_days_away", "sp_era_home", "sp_era_away",
    "sp_k9_home", "sp_k9_away", "sp_xwoba_home", "sp_xwoba_away",
    "lineup_woba_mean_home", "lineup_woba_mean_away",
    "lineup_woba_top3_home", "lineup_woba_top3_away",
    "woba_30g_home", "woba_30g_away",
    "bullpen_whip_10g_home", "bullpen_whip_10g_away",
    "bullpen_whip_3g_home", "bullpen_whip_3g_away",
    "team_barrel_15g_home", "team_barrel_15g_away",
    "team_exitvelo_15g_home", "team_exitvelo_15g_away",
)
# MLB per-side pool for the market (run-engine) scope: the per-side
# members of the frozen 53-column RUN_FEATURE_COLS lambda view — the
# distribution model's own regressand geometry, independent of the
# moneyline's.
MLB_RUN_SIDE_FIELDS = MLB_SIDE_FIELDS


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


def mlb_scopes(ml_incumbent: tuple[str, ...],
               run_incumbent: tuple[str, ...]) -> tuple[ModelScope, ...]:
    """MLB's two independent scopes (each with its OWN frame contract).

    moneyline: the frozen 64-column FEATURE_COLS view (incumbent minus
    its per-side members) over the moneyline candidate frame; target
    home_win (binary, probability metrics).

    market: the frozen 53-column RUN_FEATURE_COLS run-engine lambda view
    (incumbent minus its per-side members) over the run-engine frame;
    target margin (the distribution engine's regressand, regression
    metrics). Each scope's feature list is NEVER inferred from the
    other's — the geometry mirrors production exactly.
    """
    return (
        ModelScope(sport="mlb", model="moneyline", raw_fields=MLB_SIDE_FIELDS,
                   prod_features=tuple(ml_incumbent), target="home_win"),
        ModelScope(sport="mlb", model="market",
                   raw_fields=MLB_RUN_SIDE_FIELDS,
                   prod_features=tuple(run_incumbent), target="margin",
                   horizon_field="game_date"),
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
