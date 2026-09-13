"""NFL raw-field catalog — the declared nflverse source-column contract.

Mirrors ``sports.mlb.features.raw.catalog``: the ingestion layer narrows every pull to
these columns (never fabricating ones the source lacks) and the feature
engine is the only place that interprets them. Schedules carry the nflverse
column names verbatim; play-by-play is narrowed to the rollup needs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schedules (nflverse `load_schedules`, narrowed)
# ---------------------------------------------------------------------------
SCHEDULE_COLS = [
    "game_id", "season", "week", "game_type", "gameday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "roof", "surface",
    "div_game", "stadium", "location", "referee",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
]

#: Derived start instant: ``start_time_utc`` is NOT an nflverse column —
#: it is composed at ingestion from ``gameday`` (ET calendar date) +
#: ``gametime`` (ET wall clock) via :func:`compose_nfl_start_utc` and
#: fails closed on missing/invalid inputs (no fabricated instants, ever).
#: Declared here so the derived field is a documented catalog member.
DERIVED_SCHEDULE_COLS = ["start_time_utc"]

#: Columns REQUIRED to compose the start instant. ``gametime`` is often
#: absent for far-future weeks; composition is applied per row and rows
#: without a reliable instant carry a NaN ``start_time_utc`` — the §7.1
#: gates fail closed on any served row lacking one.
START_COMPOSITION_REQUIRED = ("gameday", "gametime")

#: Regular season only — no pre/post season in the decided universe.
KEEP_GAME_TYPES = {"REG"}

GAME_IDENTITY_COLS = ["game_id"]

# ---------------------------------------------------------------------------
# Play-by-play (nflverse `load_pbp`, narrowed to the per-game rollup)
# ---------------------------------------------------------------------------
PBP_COLS = [
    "game_id", "posteam", "defteam", "yards_gained", "epa", "qb_epa",
    "game_seconds_remaining", "interception", "fumble_lost",
    "passing_yards", "pass_attempt", "sack", "penalty", "penalty_yards",
    "penalty_team", "third_down_converted", "third_down_failed",
    "yardline_100", "touchdown", "field_goal_result", "drive",
]

# ---------------------------------------------------------------------------
# Player stats (nflverse `load_player_stats`, QB display fields only —
# display information; never a model feature without leakage-safe admission)
# ---------------------------------------------------------------------------
PLAYER_STATS_COLS = [
    "player_id", "player_name", "position", "team", "season", "week",
    "season_type", "attempts", "completions", "passing_yards",
    "passing_tds", "passing_interceptions",
]

# ---------------------------------------------------------------------------
# Teams (abbr -> full name for frontend display)
# ---------------------------------------------------------------------------
TEAMS_COLS = ["team_abbr", "team_name"]

# ---------------------------------------------------------------------------
# Franchise-rebrand aliases: the nflverse SCHEDULES carry the abbr in use
# at game time (2018-2019 Raiders = OAK) while modern nflverse PBP carries
# the CURRENT franchise abbr (LV) for every historical season. Without this
# map the team ladder's trailing history chains break at the rebrand
# boundary (a 2019 OAK game joins no LV chain and its per-team state goes
# NaN). Alias direction: source (schedule) abbr -> canonical (pbp/teams)
# abbr. Additions require a reviewed catalog commit (never silent drift).
# ---------------------------------------------------------------------------
TEAM_ABBR_ALIASES = {
    "OAK": "LV",   # Raiders: Oakland -> Las Vegas (2020)
    "SD": "LAC",   # Chargers: San Diego -> Los Angeles (2017)
    "STL": "LA",   # Rams: St. Louis -> Los Angeles (2016)
}


def normalize_team_abbr(abbr) -> str:
    """Canonical franchise abbr for a source abbr (identity otherwise)."""
    return TEAM_ABBR_ALIASES.get(str(abbr), str(abbr))
