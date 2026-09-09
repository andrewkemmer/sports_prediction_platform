"""NHL raw-field catalog — the declared NHL Stats API source contract.

Data source (selected and verified Phase 4): the OFFICIAL NHL Stats API,
``https://api-web.nhle.com/v1/`` (no key required, public, documented in
the community as the api-web endpoint family behind NHL GameCenter).

Proven availability (probed live, Phase 4):

* schedules, historical + current: ``/v1/schedule/<YYYY-MM-DD>`` returns
  the requested date's week; deep coverage verified back to 2010 (gameType
  2 = regular season; 1 preseason, 3 playoffs);
* final scores, home/away teams: ``homeTeam.score`` / ``awayTeam.score``
  on schedule rows (present once the game is OFF/FINAL; null before);
* venues: ``venue.default`` (name). Indoor/outdoor status: the NHL plays
  indoors except listed neutral-site special games — the API exposes
  ``neutralSite`` but NOT an explicit indoor/outdoor flag; the catalog
  treats every non-neutral NHL venue as indoor (documented convention,
  not fabrication: outdoor stadium series games carry neutralSite=true);
* player identities and positions: boxscore ``playerByGameStats`` rows
  carry ``playerId``, ``position`` (C/LW/RW/D/G), ``name.default``;
* goalie starts and appearances: boxscore goalie rows carry
  ``starter`` (bool), ``decision`` (W/L/OTL/N), ``toi`` — an appearance
  is any goalie row; a start is ``starter: true``;
* shots against and goals allowed: boxscore goalie rows carry
  ``shotsAgainst``, ``saves``, ``goalsAgainst``, ``savePctg``;
* play-by-play: ``/v1/gamecenter/<id>/play-by-play`` returns event-level
  plays (328 events in the probed 2023 game) with period, type, situation
  codes — sufficient for the NHL feature engine's needs;
* history depth: schedule + boxscore verified for 2010+; the study
  window (2021+ by default) is comfortably inside availability.

Practical limits (documented, not assumed):

* no official bulk download — season backfills require per-day schedule
  requests (~210 dated calls/season at the weekly-window granularity);
  rate limits are informal but real: the ingestion layer spaces requests
  and caches per (season, date-window) so a season pulls once;
* boxscore/play-by-play are per-game endpoints (~1,300 games/season):
  Phase 4 pulls boxscores ONLY for the games whose goalie-panel fields it
  needs (the current slate + trailing window), never the full history —
  the team-level feature engine needs only schedule rows;
* no NHL API field distinguishes goalies announced for a FUTURE game;
  starter identity for undecided games is genuinely unavailable and
  renders TBD (never fabricated).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schedule rows (narrowed from /v1/schedule/<date>.gameWeek[].games[])
# ---------------------------------------------------------------------------
SCHEDULE_COLS = [
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score", "venue",
    "neutral_site", "game_state",
]

#: Regular season only (gameType 2) — the decided universe, matching the
#: reference platform's regular-season convention. gameType 1 = preseason,
#: 3 = playoffs: excluded from the feature ladder and OOF population.
KEEP_GAME_TYPES = {2}

GAME_IDENTITY_COLS = ["game_id"]

# ---------------------------------------------------------------------------
# Boxscore goalie rows (from /v1/gamecenter/<id>/boxscore
# .playerByGameStats.{home,away}Team.goalies[]) — display/panel fields only.
# ---------------------------------------------------------------------------
GOALIE_COLS = [
    "game_id", "team", "player_id", "player_name", "position", "starter",
    "decision", "saves", "shots_against", "goals_against", "save_pctg",
    "toi", "season", "week",
]

GOALIE_STARTS_COLS = [
    "game_id", "team", "player_id", "player_name", "starter", "decision",
]

# ---------------------------------------------------------------------------
# Teams (32 franchises; abbreviations from the API's team objects)
# ---------------------------------------------------------------------------
TEAMS_EXPECTED = 32


def normalize_season_code(value) -> int | float:
    """Convert an API season code to the study's end-year convention.

    The schedule endpoint emits 8-digit codes (``20212022``); study.yaml
    speaks end-year plain integers (``2022``). A code >= 1,000,000 maps
    to its last four digits; plain years pass through unchanged. NaN
    stays NaN (never fabricated as a season).
    """
    try:
        s = int(value)
    except (TypeError, ValueError):
        return float("nan")
    if s >= 1_000_000:
        return s % 10_000
    return s
