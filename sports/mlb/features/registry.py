"""Feature registry and controlled derivation factory.

Features are DECLARED in a registry (name, summary, formula, source, window,
units, direction) and derived by a controlled factory â€” no ad-hoc column
creation in the pipeline. The registry doubles as the ``features_metadata``
artifact source (dashboard tooltips).

The registry owns BOTH versioned per-model contracts (Phase 7.5b canonical
ownership):

* ``MONEYLINE_FEATURE_COLS`` â€” the frozen 64-column moneyline view
  (reference repo's training feature list as of 2026-09-07; the Experiment
  #2 E/F replacement included: the 6 S-family baselines removed, the 8
  exp2 candidates shipped), version ``mlb-moneyline-v75``;
* ``MARKET_FEATURE_COLS`` â€” the frozen 53-column run-engine lambda view
  (RUN_LAMBDA_VIEW_FROZEN from the reference 2026-08-30 keep-list restore;
  53 kept / 14 dropped), version ``mlb-market-v75``.

Both are explicit, independent ``tuple([...])`` constructions â€” no aliases,
no legacy bare frozen-view symbols. The lists are carried over
byte-identical to the pre-7.5b state (behavior-neutral).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts import (
    FeatureContract,
    FeatureSpec,
    validate_columns_have_metadata,
    validate_metadata_is_reachable,
)
from core.features import require_classes_for
from core.validation import ValidationError, require_columns


# ---------------------------------------------------------------------------
# Versioned moneyline feature contract (explicit, independent declaration)
# ---------------------------------------------------------------------------

MONEYLINE_FEATURE_COLS: tuple[str, ...] = tuple([
    # 1. Baseline
    "is_home",
    # 2-4. Core pre-game diffs
    "win_pct_diff", "elo_diff", "rest_days_diff",
    # 5. Starting pitcher diffs (season-to-date + last-5-start)
    "sp_era_diff", "sp_era_5g_diff",
    # 9-11. SP stuff diffs (trailing 3-start)
    "sp_fbvelo_diff", "sp_fbpct_diff", "sp_whiff_diff",
    # SP xwOBA diffs
    "sp_xwoba_vs_l_diff",
    # 12-14. Lineup wOBA diffs
    "lineup_woba_mean_diff", "lineup_woba_top3_diff", "lineup_woba_std_diff",
    # 15. Team rolling wOBA diff
    "woba_30g_diff",
    # 16-18. Bullpen diffs
    "bullpen_whip_diff", "bullpen_whip_3g_diff", "bullpen_pitches_diff",
    # 19-21. Team contact form diffs (trailing 15g)
    "team_barrel_diff", "team_hardhit_diff", "team_exitvelo_diff",
    # 22. Lineup handedness matchup advantage
    "lineup_handedness_matchup_advantage",
    # 23. Travel fatigue
    "travel_fatigue_diff",
    # 24. Closer availability
    "closer_availability_diff",
    # 25. Dome neutral flag
    "dome_is_neutral",
    # 26. Park factor
    "park_factor_slug_diff",
    # 27-28. Weather-driven interactions (real observations only)
    "wind_advantage_flyball_factor", "air_density_velocity_boost",
    # 29-32. Derived interaction features
    "bullpen_meltdown_risk", "pitcher_regression_indicator",
    "lineup_depth_multiplier", "ace_efficiency_factor",
    # 33-56. Raw per-side inputs
    "home_elo", "away_elo",
    "home_win_pct", "away_win_pct",
    "sp_era_home", "sp_era_away",
    "sp_k9_home", "sp_k9_away",
    "sp_xwoba_home", "sp_xwoba_away",
    "lineup_woba_mean_home", "lineup_woba_mean_away",
    "lineup_woba_top3_home", "lineup_woba_top3_away",
    "woba_30g_home", "woba_30g_away",
    "bullpen_whip_10g_home", "bullpen_whip_10g_away",
    "bullpen_whip_3g_home", "bullpen_whip_3g_away",
    "team_barrel_15g_home", "team_barrel_15g_away",
    "team_exitvelo_15g_home", "team_exitvelo_15g_away",
    # 63. Run-engine expected-run margin (OOF on the moneyline's own folds)
    "run_margin_diff",
    # 60-67. Experiment #2 matchup candidates (shipped 2026-09-07)
    "exp2_centered_k_diff", "exp2_cat_k_fastball_diff",
    "exp2_cat_k_breaking_diff", "exp2_cat_k_offspeed_diff",
    "exp2_cat_xwoba_fastball_diff", "exp2_cat_xwoba_breaking_diff",
    "exp2_cat_xwoba_offspeed_diff", "exp2_cat_platoon_k_fastball_diff",
])
MONEYLINE_CONTRACT_VERSION = "mlb-moneyline-v75"

# ---------------------------------------------------------------------------
# Versioned market feature contract (explicit, independent declaration)
# ---------------------------------------------------------------------------
# The FROZEN run-engine lambda view: byte-identical to the reference repo's
# RUN_LAMBDA_VIEW_FROZEN (53 kept / 14 dropped). The moneyline list may
# change without run-engine sign-off; this view is pinned.
MARKET_FEATURE_COLS: tuple[str, ...] = tuple([
    "is_home", "win_pct_diff", "elo_diff", "rest_days_diff",
    "sp_era_diff", "sp_era_5g_diff", "sp_k9_diff", "sp_k9_5g_diff",
    "sp_fbvelo_diff", "sp_fbpct_diff", "sp_whiff_diff", "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff", "lineup_woba_mean_diff", "lineup_woba_top3_diff",
    "lineup_woba_std_diff", "woba_30g_diff", "bullpen_whip_diff",
    "bullpen_whip_3g_diff", "bullpen_pitches_diff", "team_barrel_diff",
    "team_hardhit_diff", "team_exitvelo_diff", "travel_fatigue_diff",
    "closer_availability_diff", "dome_is_neutral", "park_factor_slug_diff",
    "wind_advantage_flyball_factor", "air_density_velocity_boost",
    "home_elo", "away_elo", "home_win_pct", "away_win_pct",
    "sp_era_home", "sp_era_away", "sp_k9_home", "sp_k9_away",
    "sp_xwoba_home", "sp_xwoba_away",
    "lineup_woba_mean_home", "lineup_woba_mean_away",
    "lineup_woba_top3_home", "lineup_woba_top3_away",
    "woba_30g_home", "woba_30g_away",
    "bullpen_whip_10g_home", "bullpen_whip_10g_away",
    "bullpen_whip_3g_home", "bullpen_whip_3g_away",
    "team_barrel_15g_home", "team_barrel_15g_away",
    "team_exitvelo_15g_home", "team_exitvelo_15g_away",
])
MARKET_CONTRACT_VERSION = "mlb-market-v75"
#: Prior provenance versions (frozen 2026-09-07 reference state).
PRIOR_MONEYLINE_CONTRACT_VERSION = "phase2"
PRIOR_MARKET_CONTRACT_VERSION = "phase2"

assert len(MONEYLINE_FEATURE_COLS) == 64 and len(MARKET_FEATURE_COLS) == 53


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureEntry:
    """One declared feature in the MLB registry."""

    name: str
    summary: str
    definition: str
    formula: str
    source: str
    window: str
    units: str
    direction: str
    members: tuple[str, ...] = ()
    tooltip: str = ""
    #: Contract tuples this entry belongs to. Defaults to the moneyline
    #: view; the run-engine-only diffs are marked ``{"market"}`` and are
    #: excluded from ``build_feature_contract`` (which stays moneyline-only).
    contracts: frozenset[str] = frozenset({"moneyline"})


def _spec(e: FeatureEntry) -> FeatureSpec:
    return FeatureSpec(
        name=e.name, summary=e.summary, definition=e.definition,
        formula=e.formula, source=e.source, window=e.window, units=e.units,
        direction=e.direction, members=e.members, tooltip=e.tooltip)


# The declared registry. Every moneyline-contract member must appear here (checked
# by build_feature_contract at import/validation time).
_REGISTRY: tuple[FeatureEntry, ...] = (
    FeatureEntry(
        "is_home", "Home-field anchor",
        "1 when the row's home team is batting at home", "1",
        "static", "n/a", "binary", "positive"),
    FeatureEntry(
        "win_pct_diff", "Win% gap (smoothed)",
        "Smoothed home win% minus away win%", "home_win_pct - away_win_pct",
        "derived", "season", "pct", "positive"),
    FeatureEntry(
        "elo_diff", "Elo gap",
        "Home Elo minus away Elo (pre-game, season-reverted)",
        "home_elo - away_elo", "derived", "season", "points", "positive"),
    FeatureEntry(
        "rest_days_diff", "Rest-day gap",
        "Home rest days minus away rest days (capped at 6)",
        "rest_days_home - rest_days_away", "schedule", "game-to-game",
        "days", "positive"),
    FeatureEntry(
        "sp_era_diff", "SP season ERA gap",
        "Home starter season-to-date ERA minus away",
        "sp_era_home - sp_era_away", "statcast", "season-to-date",
        "era", "negative"),
    FeatureEntry(
        "sp_era_5g_diff", "SP last-5-start ERA gap",
        "Home starter last-5-start ERA minus away",
        "sp_era_5g_home - sp_era_5g_away", "statcast", "5 starts",
        "era", "negative"),
    FeatureEntry(
        "sp_fbvelo_diff", "SP fastball velo gap (3-start)",
        "Home starter last-3-start fastball velocity minus away",
        "sp_fbvelo_3g_home - sp_fbvelo_3g_away", "statcast", "3 starts",
        "mph", "positive"),
    FeatureEntry(
        "sp_fbpct_diff", "SP fastball rate gap (3-start)",
        "Home starter last-3-start fastball share minus away",
        "sp_fbpct_3g_home - sp_fbpct_3g_away", "statcast", "3 starts",
        "share", "positive"),
    FeatureEntry(
        "sp_whiff_diff", "SP whiff rate gap (3-start)",
        "Home starter last-3-start whiff rate minus away",
        "sp_whiff_3g_home - sp_whiff_3g_away", "statcast", "3 starts",
        "rate", "positive"),
    FeatureEntry(
        "sp_xwoba_vs_l_diff", "SP xwOBA-vs-L gap",
        "Home starter season xwOBA allowed vs lefties minus away",
        "sp_xwoba_vs_l_home - sp_xwoba_vs_l_away", "statcast",
        "season-to-date", "wOBA", "negative"),
    FeatureEntry(
        "lineup_woba_mean_diff", "Lineup wOBA gap (top-9 mean)",
        "Home expected-lineup wOBA mean minus away",
        "lineup_woba_mean_home - lineup_woba_mean_away", "statcast",
        "30 games", "wOBA", "positive"),
    FeatureEntry(
        "lineup_woba_top3_diff", "Lineup wOBA gap (top-3)",
        "Home expected top-3 wOBA minus away",
        "lineup_woba_top3_home - lineup_woba_top3_away", "statcast",
        "30 games", "wOBA", "positive"),
    FeatureEntry(
        "lineup_woba_std_diff", "Lineup wOBA spread gap",
        "Home lineup wOBA stddev minus away",
        "lineup_woba_std_home - lineup_woba_std_away", "statcast",
        "30 games", "wOBA", "n/a"),
    FeatureEntry(
        "woba_30g_diff", "Team wOBA gap (30g)",
        "Home team 30-game wOBA minus away",
        "woba_30g_home - woba_30g_away", "statcast", "30 games",
        "wOBA", "positive"),
    FeatureEntry(
        "bullpen_whip_diff", "Bullpen WHIP gap (10g)",
        "Home bullpen 10-game WHIP minus away",
        "bullpen_whip_10g_home - bullpen_whip_10g_away", "statcast",
        "10 games", "whip", "negative"),
    FeatureEntry(
        "bullpen_whip_3g_diff", "Bullpen WHIP gap (3g)",
        "Home bullpen 3-game WHIP minus away",
        "bullpen_whip_3g_home - bullpen_whip_3g_away", "statcast",
        "3 games", "whip", "negative"),
    FeatureEntry(
        "bullpen_pitches_diff", "Bullpen workload gap (3d)",
        "Home bullpen 3-day pitch count minus away",
        "bullpen_pitches_3d_home - bullpen_pitches_3d_away", "statcast",
        "3 days", "pitches", "negative"),
    FeatureEntry(
        "team_barrel_diff", "Barrel% gap (15g)",
        "Home team 15-game barrel rate minus away",
        "team_barrel_15g_home - team_barrel_15g_away", "statcast",
        "15 games", "rate", "positive"),
    FeatureEntry(
        "team_hardhit_diff", "Hard-hit% gap (15g)",
        "Home team 15-game hard-hit rate minus away",
        "team_hardhit_15g_home - team_hardhit_15g_away", "statcast",
        "15 games", "rate", "positive"),
    FeatureEntry(
        "team_exitvelo_diff", "Exit velo gap (15g)",
        "Home team 15-game avg exit velocity minus away",
        "team_exitvelo_15g_home - team_exitvelo_15g_away", "statcast",
        "15 games", "mph", "positive"),
    FeatureEntry(
        "lineup_handedness_matchup_advantage",
        "Handedness matchup advantage",
        "Each lineup's OPS vs tonight's opposing starter hand, differenced",
        "lineup_ops_vs_starter_hand_home - ..._away", "statcast",
        "30 games", "OPS", "positive",
        members=("moneyline",)),
    FeatureEntry(
        "travel_fatigue_diff", "Travel fatigue gap",
        "Time zones crossed in the last 3 days, home minus away",
        "time_zones_crossed_last_3d_home - ..._away", "schedule",
        "3 days", "zones", "negative"),
    FeatureEntry(
        "closer_availability_diff", "Closer availability gap",
        "Home closer availability minus away (1 = available)",
        "closer_available_home - closer_available_away", "statcast",
        "1 day", "binary", "positive"),
    FeatureEntry(
        "dome_is_neutral", "Dome neutral flag",
        "1 when the venue roof is closed/dome (weather neutralized)",
        "roof_state in ('dome','closed')", "statsapi", "game", "binary",
        "n/a"),
    FeatureEntry(
        "park_factor_slug_diff", "Park factor gap",
        "Slug-based park factor, home minus away context",
        "park_factor_slug_home - park_factor_slug_away", "static",
        "n/a", "multiplier", "positive"),
    FeatureEntry(
        "wind_advantage_flyball_factor",
        "Wind advantage (fly-ball)",
        "Outward wind boosts fly-ball carry at the venue (real "
        "Open-Meteo observation; NULL when unobserved)",
        "f(wind vector, venue orientation)", "open-meteo", "game",
        "multiplier", "positive"),
    FeatureEntry(
        "air_density_velocity_boost",
        "Air-density boost",
        "Thin-air/high-temp density boost to carry (real observation; "
        "NULL when unobserved)",
        "f(temp, elevation, humidity)", "open-meteo", "game",
        "multiplier", "positive"),
    FeatureEntry(
        "bullpen_meltdown_risk", "Bullpen meltdown risk",
        "Workload x effectiveness interaction",
        "bullpen_pitches_diff * bullpen_whip_diff", "derived", "3 days",
        "interaction", "negative", members=("run_engine_excluded",)),
    FeatureEntry(
        "pitcher_regression_indicator", "Pitcher regression indicator",
        "Velo x ERA regression interaction",
        "sp_fbvelo_diff * sp_era_diff", "derived", "3 starts",
        "interaction", "n/a", members=("run_engine_excluded",)),
    FeatureEntry(
        "lineup_depth_multiplier", "Lineup depth multiplier",
        "Lineup depth interaction",
        "lineup_woba_mean_diff * lineup_woba_top3_diff", "derived",
        "30 games", "interaction", "positive",
        members=("run_engine_excluded",)),
    FeatureEntry(
        "ace_efficiency_factor", "Ace efficiency factor",
        "Strikeout x whiff interaction",
        "sp_k9_diff * sp_whiff_diff", "derived", "season",
        "interaction", "positive", members=("run_engine_excluded",)),
    FeatureEntry(
        "home_elo", "Home Elo", "Home team pre-game Elo (season-reverted)",
        "elo", "derived", "season", "points", "positive"),
    FeatureEntry(
        "away_elo", "Away Elo", "Away team pre-game Elo (season-reverted)",
        "elo", "derived", "season", "points", "negative"),
    FeatureEntry(
        "home_win_pct", "Home win%", "Home smoothed season win%",
        "wins/(wins+losses) shrunk to .500", "derived", "season",
        "pct", "positive"),
    FeatureEntry(
        "away_win_pct", "Away win%", "Away smoothed season win%",
        "wins/(wins+losses) shrunk to .500", "derived", "season",
        "pct", "negative"),
    FeatureEntry(
        "sp_era_home", "Home SP ERA", "Home starter season-to-date ERA",
        "9*ER/IP", "statcast", "season-to-date", "era", "negative"),
    FeatureEntry(
        "sp_era_away", "Away SP ERA", "Away starter season-to-date ERA",
        "9*ER/IP", "statcast", "season-to-date", "era", "negative"),
    FeatureEntry(
        "sp_k9_home", "Home SP K/9", "Home starter season-to-date K/9",
        "9*K/IP", "statcast", "season-to-date", "k9", "positive"),
    FeatureEntry(
        "sp_k9_away", "Away SP K/9", "Away starter season-to-date K/9",
        "9*K/IP", "statcast", "season-to-date", "k9", "positive"),
    FeatureEntry(
        "sp_xwoba_home", "Home SP xwOBA allowed",
        "Home starter trailing xwOBA allowed",
        "avg(xwOBA)", "statcast", "30 games", "wOBA", "negative"),
    FeatureEntry(
        "sp_xwoba_away", "Away SP xwOBA allowed",
        "Away starter trailing xwOBA allowed",
        "avg(xwOBA)", "statcast", "30 games", "wOBA", "negative"),
    FeatureEntry(
        "lineup_woba_mean_home", "Home lineup wOBA",
        "Home expected top-9 wOBA mean", "mean(shrunk wOBA)",
        "statcast", "30 games", "wOBA", "positive"),
    FeatureEntry(
        "lineup_woba_mean_away", "Away lineup wOBA",
        "Away expected top-9 wOBA mean", "mean(shrunk wOBA)",
        "statcast", "30 games", "wOBA", "negative"),
    FeatureEntry(
        "lineup_woba_top3_home", "Home top-3 wOBA",
        "Home expected top-3 wOBA", "mean(shrunk wOBA | top3)",
        "statcast", "30 games", "wOBA", "positive"),
    FeatureEntry(
        "lineup_woba_top3_away", "Away top-3 wOBA",
        "Away expected top-3 wOBA", "mean(shrunk wOBA | top3)",
        "statcast", "30 games", "wOBA", "negative"),
    FeatureEntry(
        "woba_30g_home", "Home team wOBA (30g)",
        "Home team trailing 30-game wOBA", "wOBA per PA",
        "statcast", "30 games", "wOBA", "positive"),
    FeatureEntry(
        "woba_30g_away", "Away team wOBA (30g)",
        "Away team trailing 30-game wOBA", "wOBA per PA",
        "statcast", "30 games", "wOBA", "negative"),
    FeatureEntry(
        "bullpen_whip_10g_home", "Home bullpen WHIP (10g)",
        "Home bullpen trailing 10-game WHIP", "(BB+H)/IP",
        "statcast", "10 games", "whip", "negative"),
    FeatureEntry(
        "bullpen_whip_10g_away", "Away bullpen WHIP (10g)",
        "Away bullpen trailing 10-game WHIP", "(BB+H)/IP",
        "statcast", "10 games", "whip", "negative"),
    FeatureEntry(
        "bullpen_whip_3g_home", "Home bullpen WHIP (3g)",
        "Home bullpen trailing 3-game WHIP", "(BB+H)/IP",
        "statcast", "3 games", "whip", "negative"),
    FeatureEntry(
        "bullpen_whip_3g_away", "Away bullpen WHIP (3g)",
        "Away bullpen trailing 3-game WHIP", "(BB+H)/IP",
        "statcast", "3 games", "whip", "negative"),
    FeatureEntry(
        "team_barrel_15g_home", "Home barrel% (15g)",
        "Home team trailing 15-game barrel rate", "barrels/BBE",
        "statcast", "15 games", "rate", "positive"),
    FeatureEntry(
        "team_barrel_15g_away", "Away barrel% (15g)",
        "Away team trailing 15-game barrel rate", "barrels/BBE",
        "statcast", "15 games", "rate", "negative"),
    FeatureEntry(
        "team_exitvelo_15g_home", "Home exit velo (15g)",
        "Home team trailing 15-game avg exit velocity", "avg(launch_speed)",
        "statcast", "15 games", "mph", "positive"),
    FeatureEntry(
        "team_exitvelo_15g_away", "Away exit velo (15g)",
        "Away team trailing 15-game avg exit velocity", "avg(launch_speed)",
        "statcast", "15 games", "mph", "negative"),
    FeatureEntry(
        "run_margin_diff", "Run-engine margin (OOF)",
        "lambda_home - lambda_away from the run engine's per-side Poisson "
        "models, computed OUT-OF-FOLD on the moneyline's own fold split",
        "oof margin", "run_engine", "per fold", "runs", "positive",
        members=("moneyline",)),
    FeatureEntry(
        "exp2_centered_k_diff", "Exp2 centered K gap",
        "League-centered strikeout-rate matchup gap",
        "candidate C (centered K)", "statcast", "season-to-date",
        "rate", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_k_fastball_diff", "Exp2 fastball K gap",
        "Pitch-category strikeout-rate gap (fastball)",
        "candidate C (cat K, fastball)", "statcast", "season-to-date",
        "rate", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_k_breaking_diff", "Exp2 breaking K gap",
        "Pitch-category strikeout-rate gap (breaking)",
        "candidate C (cat K, breaking)", "statcast", "season-to-date",
        "rate", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_k_offspeed_diff", "Exp2 offspeed K gap",
        "Pitch-category strikeout-rate gap (offspeed)",
        "candidate C (cat K, offspeed)", "statcast", "season-to-date",
        "rate", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_xwoba_fastball_diff", "Exp2 fastball xwOBA gap",
        "Pitch-category xwOBA gap (fastball)",
        "candidate D (cat xwOBA, fastball)", "statcast", "season-to-date",
        "wOBA", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_xwoba_breaking_diff", "Exp2 breaking xwOBA gap",
        "Pitch-category xwOBA gap (breaking)",
        "candidate D (cat xwOBA, breaking)", "statcast", "season-to-date",
        "wOBA", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_xwoba_offspeed_diff", "Exp2 offspeed xwOBA gap",
        "Pitch-category xwOBA gap (offspeed)",
        "candidate D (cat xwOBA, offspeed)", "statcast", "season-to-date",
        "wOBA", "n/a", members=("exp2",)),
    FeatureEntry(
        "exp2_cat_platoon_k_fastball_diff",
        "Exp2 platoon fastball K gap",
        "Platoon-adjusted fastball strikeout-rate gap",
        "candidate E (platoon K, fastball)", "statcast", "season-to-date",
        "rate", "n/a", members=("exp2",)),
    # ------------------------------------------------------------------
    # Run-engine-only diffs: declared in MARKET_FEATURE_COLS but NOT in the
    # moneyline contract. Metadata mirrors the production derivations in
    # ``_DIFF_INPUTS`` (raw inputs) and the sibling ``sp_k9_*`` /
    # ``sp_xwoba_*`` registry entries above â€” never an invented placeholder.
    # ------------------------------------------------------------------
    FeatureEntry(
        "sp_k9_diff", "SP season K/9 gap",
        "Home starter season-to-date K/9 minus away",
        "sp_k9_home - sp_k9_away", "statcast", "season-to-date",
        "k9", "positive", contracts=frozenset({"market"})),
    FeatureEntry(
        "sp_k9_5g_diff", "SP last-5-start K/9 gap",
        "Home starter last-5-start K/9 minus away",
        "sp_k9_5g_home - sp_k9_5g_away", "statcast", "5 starts",
        "k9", "positive", contracts=frozenset({"market"})),
    FeatureEntry(
        "sp_xwoba_diff", "SP xwOBA-allowed gap (30g)",
        "Home starter trailing-30-game xwOBA allowed minus away",
        "sp_xwoba_home - sp_xwoba_away", "statcast", "30 games",
        "wOBA", "negative", contracts=frozenset({"market"})),
)


class FeatureRegistryError(ValueError):
    """Raised when the feature registry is inconsistent with the frozen views."""


def registry_entries() -> tuple[FeatureEntry, ...]:
    """All declared registry entries."""
    return _REGISTRY


def build_market_feature_contract(
        version: str = MARKET_CONTRACT_VERSION) -> FeatureContract:
    """The MLB market (run-engine lambda) view as a versioned contract
    (§11: the market model documents its own basis, never inherits the
    moneyline's). Members mirror the run-engine's model family."""
    validate_registry()
    specs = []
    for name in MARKET_FEATURE_COLS:
        entry = next(e for e in _REGISTRY if e.name == name)
        spec = _spec(entry)
        specs.append(FeatureSpec(
            name=spec.name, summary=spec.summary, definition=spec.definition,
            formula=spec.formula, source=spec.source, window=spec.window,
            units=spec.units, direction=spec.direction,
            members=("run_engine",), tooltip=spec.tooltip))
    return FeatureContract(sport="mlb", version=version,
                           features=tuple(specs))


def build_feature_contract(
        version: str = MONEYLINE_CONTRACT_VERSION) -> FeatureContract:
    """Build the shared FeatureContract from the MLB registry.

    Defaults to the ACTIVE v75 moneyline contract version â€” no active
    builder defaults to a legacy version string. Moneyline-only: entries
    marked ``contracts={"market"}`` are excluded, so the contract stays the
    frozen 64-feature moneyline view."""
    validate_registry()
    return FeatureContract(
        sport="mlb", version=version,
        features=tuple(_spec(e) for e in _REGISTRY
                       if "moneyline" in e.contracts))


def validate_registry() -> None:
    """Validate both declared contract tuples against the registry metadata.

    Uses the shared, name-only interface from ``core.contracts`` for the
    coverage (every declared column has metadata) and anti-orphan (every
    metadata entry is claimed by a declared tuple) checks; the
    moneyline-view identity and the market-only marking are MLB-specific.
    Raises on drift; never silently accepts a missing or parallel entry."""
    declared = {e.name for e in _REGISTRY}
    moneyline_marked = {e.name for e in _REGISTRY
                        if "moneyline" in e.contracts}
    # (1) every declared contract column has registry metadata.
    validate_columns_have_metadata(
        "mlb", "moneyline", MONEYLINE_FEATURE_COLS, declared)
    validate_columns_have_metadata(
        "mlb", "market", MARKET_FEATURE_COLS, declared)
    # (2) no orphan metadata: every entry is claimed by a declared tuple.
    validate_metadata_is_reachable(
        "mlb", declared,
        set(MONEYLINE_FEATURE_COLS) | set(MARKET_FEATURE_COLS))
    # Â§7.1 B-001-RESIDUAL: every contract field carries an availability
    # class â€” the PIT metadata layer is closed over the contracts.
    require_classes_for(
        "mlb", set(MONEYLINE_FEATURE_COLS) | set(MARKET_FEATURE_COLS))
    # (3) the moneyline-marked entries are EXACTLY the moneyline view.
    if moneyline_marked != set(MONEYLINE_FEATURE_COLS):
        raise FeatureRegistryError(
            "moneyline-contract entries != MONEYLINE_FEATURE_COLS: "
            f"missing={sorted(set(MONEYLINE_FEATURE_COLS) - moneyline_marked)} "
            f"extra={sorted(moneyline_marked - set(MONEYLINE_FEATURE_COLS))}")
    # (4) every non-moneyline entry is a market view member, never served
    # in the moneyline contract.
    for e in _REGISTRY:
        if "moneyline" in e.contracts:
            continue
        if e.name in MONEYLINE_FEATURE_COLS or e.name not in MARKET_FEATURE_COLS:
            raise FeatureRegistryError(
                f"non-moneyline registry entry {e.name!r} must be in "
                f"MARKET_FEATURE_COLS and absent from MONEYLINE_FEATURE_COLS")


# ---------------------------------------------------------------------------
# Controlled derivation factory
# ---------------------------------------------------------------------------

# Raw home/away inputs each diff is computed from. The factory recomputes
# every diff from the RAW columns â€” presence of a diff column is never
# evidence its values are current.
_DIFF_INPUTS: dict[str, tuple[str, str]] = {
    "win_pct_diff": ("home_win_pct", "away_win_pct"),
    "elo_diff": ("home_elo", "away_elo"),
    "rest_days_diff": ("rest_days_home", "rest_days_away"),
    "sp_era_diff": ("sp_era_home", "sp_era_away"),
    "sp_era_5g_diff": ("sp_era_5g_home", "sp_era_5g_away"),
    "sp_k9_diff": ("sp_k9_home", "sp_k9_away"),
    "sp_k9_5g_diff": ("sp_k9_5g_home", "sp_k9_5g_away"),
    "sp_fbvelo_diff": ("sp_fbvelo_3g_home", "sp_fbvelo_3g_away"),
    "sp_fbpct_diff": ("sp_fbpct_3g_home", "sp_fbpct_3g_away"),
    "sp_whiff_diff": ("sp_whiff_3g_home", "sp_whiff_3g_away"),
    # sp_xwoba_30g_* does not exist in the Phase-2 factory output; the
    # factory's sp_xwoba_* IS the trailing-30-game pitcher xwOBA (its 30g
    # rolling table). Map to the produced name rather than shipping NaN.
    "sp_xwoba_diff": ("sp_xwoba_home", "sp_xwoba_away"),
    "sp_xwoba_vs_l_diff": ("sp_xwoba_vs_l_home", "sp_xwoba_vs_l_away"),
    "lineup_woba_mean_diff": ("lineup_woba_mean_home", "lineup_woba_mean_away"),
    "lineup_woba_top3_diff": ("lineup_woba_top3_home", "lineup_woba_top3_away"),
    "lineup_woba_std_diff": ("lineup_woba_std_home", "lineup_woba_std_away"),
    "woba_30g_diff": ("woba_30g_home", "woba_30g_away"),
    "bullpen_whip_diff": ("bullpen_whip_10g_home", "bullpen_whip_10g_away"),
    "bullpen_whip_3g_diff": ("bullpen_whip_3g_home", "bullpen_whip_3g_away"),
    "bullpen_pitches_diff": ("bullpen_pitches_3d_home", "bullpen_pitches_3d_away"),
    "team_barrel_diff": ("team_barrel_15g_home", "team_barrel_15g_away"),
    "team_hardhit_diff": ("team_hardhit_15g_home", "team_hardhit_15g_away"),
    "team_exitvelo_diff": ("team_exitvelo_15g_home", "team_exitvelo_15g_away"),
    "travel_fatigue_diff": ("time_zones_crossed_last_3d_home",
                            "time_zones_crossed_last_3d_away"),
    "closer_availability_diff": ("closer_available_home", "closer_available_away"),
    "park_factor_slug_diff": ("park_factor_slug_home", "park_factor_slug_away"),
}

_COMPOSITES: dict[str, tuple[str, str]] = {
    "bullpen_meltdown_risk": ("bullpen_pitches_diff", "bullpen_whip_diff"),
    "pitcher_regression_indicator": ("sp_fbvelo_diff", "sp_era_diff"),
    "lineup_depth_multiplier": ("lineup_woba_mean_diff", "lineup_woba_top3_diff"),
    "ace_efficiency_factor": ("sp_k9_diff", "sp_whiff_diff"),
}


def derive_diff_features(df, *, require_records: bool = False):
    """Recompute every diff feature from the raw home/away columns.

    Idempotent: diffs are always recomputed from the RAW inputs, never
    trusted from a pre-built export. With ``require_records=True`` a missing
    win_pct input raises (official results should already be applied).
    """
    import numpy as np
    import pandas as pd
    if require_records and not {"home_wins", "home_losses"} <= set(df.columns):
        raise ValidationError(
            "require_records=True but record columns are missing â€” apply "
            "official results before deriving diffs")
    out = df.copy()
    for diff, (h, a) in _DIFF_INPUTS.items():
        if h in out.columns and a in out.columns:
            out[diff] = pd.to_numeric(out[h], errors="coerce") \
                - pd.to_numeric(out[a], errors="coerce")
        else:
            out[diff] = np.nan
    # Handedness matchup advantage: each lineup's OPS vs the opposing
    # starter's hand, differenced.
    if {"lineup_ops_vs_starter_hand_home", "lineup_ops_vs_starter_hand_away"} \
            <= set(out.columns):
        out["lineup_handedness_matchup_advantage"] = (
            pd.to_numeric(out["lineup_ops_vs_starter_hand_home"],
                          errors="coerce")
            - pd.to_numeric(out["lineup_ops_vs_starter_hand_away"],
                            errors="coerce"))
    else:
        out["lineup_handedness_matchup_advantage"] = np.nan
    # Composites derive from their (already recomputed) diff inputs.
    for comp, (d1, d2) in _COMPOSITES.items():
        if d1 in out.columns and d2 in out.columns:
            out[comp] = pd.to_numeric(out[d1], errors="coerce") \
                * pd.to_numeric(out[d2], errors="coerce")
        else:
            out[comp] = np.nan
    # is_home anchor + dome flag defaults (environment context).
    if "is_home" not in out.columns:
        out["is_home"] = 1.0
    if "dome_is_neutral" not in out.columns:
        out["dome_is_neutral"] = 0.0
    return out


def ensure_feature_columns(df, feature_cols: tuple[str, ...] | None = None):
    """Guarantee every requested feature column exists (missing -> NaN).

    Never fabricates values â€” missing observations ship as true NULLs (tree
    models route NaN natively; zero/median fills fabricated signal).
    """
    import numpy as np
    out = df.copy()
    for c in (feature_cols if feature_cols is not None
              else MONEYLINE_FEATURE_COLS):
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_candidate_frame(game_df, *, include_run_view: bool = True):
    """The wide point-in-time candidate frame.

    Composes: Elo/records enrichment (PIT), diff recomputation, feature
    column guarantee. Returns (frame, feature_cols) where feature_cols is
    the frozen moneyline view.
    """
    from sports.mlb.frames import enrich_elo_and_records
    df = enrich_elo_and_records(game_df, rename_team_woba=True)
    df = derive_diff_features(df)
    df = ensure_feature_columns(df, MONEYLINE_FEATURE_COLS)
    if include_run_view:
        df = ensure_feature_columns(df, MARKET_FEATURE_COLS)
    validate_registry()
    return df, MONEYLINE_FEATURE_COLS
