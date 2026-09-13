"""MLB raw-field catalog.

The declared Statcast source schema the MLB ingestion normalizes to. This is
the raw-field catalog required by Phase 2 item 2: every column the platform
reads from raw Statcast payloads, its aliases, the columns intentionally
dropped, and the game-type keep-set. Feature derivations may only consume
columns declared here.
"""

from __future__ import annotations

# Canonical Statcast columns kept in the raw store (pitches).
STATCAST_COLS: tuple[str, ...] = (
    "game_date", "game_pk", "game_type", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "at_bat_number", "pitch_number", "pitcher", "batter",
    "p_throws", "stand",
    "pitch_type", "release_speed", "release_pos_x", "release_pos_z",
    "player_name",
    "description", "events",
    "spin_rate", "spin_axis",
    "release_spin_rate", "release_extension",
    "plate_x", "plate_z",
    "zone", "pfx_x", "pfx_z",
    "hit_distance_sc", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "babip_value", "iso_value",
    "barrel", "hard_contact",
    "home_score", "away_score",
    "delta_home_win_exp", "delta_run_exp",
)

# Vendor schema drift aliases (pybaseball renames across versions).
COLUMN_ALIASES: dict[str, str] = {
    "barrel_pct": "barrel", "is_barrel": "barrel",
    "hardhit": "hard_contact", "hard_hit": "hard_contact",
    "exit_velocity": "launch_speed", "exit_velo": "launch_speed",
    "la": "launch_angle",
    "xwoba": "estimated_woba_using_speedangle",
    "xwOBA": "estimated_woba_using_speedangle",
    "xba": "estimated_ba_using_speedangle",
    "pitcher_name": "player_name",
    "event": "events",
}

# Columns dropped after normalization (never consumed by any derivation).
UNUSED_COLS: tuple[str, ...] = (
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "if_fielding_alignment", "of_fielding_alignment",
    "post_home_score", "post_away_score",
    "event", "type", "launch_speed_angle",
)

# Statcast game_type codes to KEEP: R regular season, F/D/L/W postseason.
# 'S' (spring training) and 'E' (exhibition) are excluded — Savant posts
# pitch data for scrimmages whose stats are not representative.
KEEP_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})

# Pitch identity: dedupe key for merges and boundary guards.
PITCH_IDENTITY_COLS: tuple[str, ...] = (
    "game_pk", "at_bat_number", "pitch_number")

# String-typed source columns (never numeric).
STRING_COLS: frozenset[str] = frozenset({
    "pitch_type", "events", "description", "player_name",
    "home_team", "away_team", "pitcher", "batter",
    "stand", "p_throws", "game_type", "inning_topbot",
})


def catalog_columns() -> tuple[str, ...]:
    """All canonical columns the raw store guarantees (declared or aliased)."""
    return STATCAST_COLS


def resolve_alias(name: str) -> str:
    """Map a vendor column name to its canonical form (identity when none)."""
    return COLUMN_ALIASES.get(name, name)
