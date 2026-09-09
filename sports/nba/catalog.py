"""NBA raw-field catalog — the declared NBA Stats API source contract.

Data source (selected and verified Phase 5): the OFFICIAL NBA Stats API,
``https://stats.nba.com/stats/`` (no key required, public; the endpoint
family behind nba.com/stats). Requires browser-like headers
(User-Agent + Referer) — documented, not secret.

Proven availability (probed live, Phase 5):

* schedules, historical + current:
  ``/stats/scheduleleaguev2?LeagueID=00&Season=<YYYY-YY>`` returns the
  FULL season schedule in ONE call (225 gameDates / 1,230 regular-season
  games for 2023-24; verified back through 2005-06 and forward to the
  current 2025-26 season). Rows carry home/away tricodes, final scores
  (``homeTeam.score`` / ``awayTeam.score`` once ``gameStatus == 3``),
  ``gameStatusText`` (``Final`` / ``Final/OT`` / ``Final/OT2`` /
  ``Scheduled``-style pre-game text), arena name/city, ``isNeutral``,
  and ``gameDateTimeUTC`` (the start instant; the bare ``gameTimeUTC``
  is a 1900 time-of-day placeholder and is never used);
* final scores and home/away teams: on the same schedule rows (no
  separate results feed needed — one call per season covers schedule +
  finals);
* venues / neutral-site flags: ``arenaName`` / ``arenaCity`` /
  ``isNeutral`` (0 neutral-site games in the 2023-24 probe);
* player identities and positions: per-game
  ``/stats/boxscoretraditionalv2?GameID=<id>`` PlayerStats rows carry
  ``PLAYER_ID``, ``PLAYER_NAME``, ``START_POSITION`` (F/C/G or empty);
  verified back through a 1996-97 game;
* team scoring leaders per game: the schedule rows embed
  ``pointsLeaders`` (top scorer per game with name + points), and the
  boxscore rows carry full PTS/AST — both verified live;
* box-score/event data sufficient for team and player rolling features:
  the team GameLog (``/stats/leaguegamelog?PlayerOrTeam=T``) carries
  PTS/REB/AST/WL per team-game back to 1946-47 (verified: 662 rows for
  1946-47), one call per season;
* history depth: gamelog verified for 1946-47 onward; schedule +
  boxscores verified for 2005-06 / 1996-97 onward respectively. The
  study window (2015+ by default) is comfortably inside availability.

Practical limits (documented, not assumed):

* NO official bulk download — history backfills require ONE schedule
  call + ONE gamelog call per season (~2 calls/season, ~60 calls for a
  30-season backfill; the cheapest of the three sports so far). Informal
  rate limits are real: the ingestion layer spaces requests (>= 0.5 s
  pause) and caches per (season) Parquet so each season pulls once;
* boxscores are PER-GAME (~1,230 games/season): Phase 5 pulls boxscores
  ONLY for the games the participant panel needs (the current slate +
  its trailing window), never the full history — the team-level feature
  engine needs only schedule/gamelog rows;
* the schedule's ``gameTimeUTC`` is a time-of-day placeholder
  (1900-01-01); the real start instant is ``gameDateTimeUTC`` — the
  catalog normalizes to that field and never fabricates start times;
* no NBA API field names the expected top scorer of a FUTURE game;
  participant identity for undecided games is genuinely unavailable and
  renders TBD (never fabricated).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schedule rows (narrowed from scheduleleaguev2.leagueSchedule.gameDates[])
# ---------------------------------------------------------------------------
SCHEDULE_COLS = [
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score", "venue",
    "neutral_site", "game_state",
]

#: Regular season only. The schedule gameId's 3rd digit encodes the
#: game type: 1 = preseason, 2 = regular season, 4 = playoffs, 5 =
#: play-in. Phase 5 keeps the regular-season universe (matching the
#: platform convention); the digit filter is the authoritative check.
KEEP_GAME_TYPES = {2}

GAME_IDENTITY_COLS = ["game_id"]

# ---------------------------------------------------------------------------
# Boxscore player rows (from boxscoretraditionalv2 PlayerStats) —
# display/panel fields only (team-level features come from the schedule
# scores; the boxscore serves the scoring-leader panel).
# ---------------------------------------------------------------------------
PLAYER_COLS = [
    "game_id", "team", "player_id", "player_name", "position", "minutes",
    "points", "assists", "rebounds", "season",
]

# ---------------------------------------------------------------------------
# Teams (30 franchises; tricodes from the API's team objects)
# ---------------------------------------------------------------------------
TEAMS_EXPECTED = 30


def normalize_season_code(value) -> int | float:
    """Convert an NBA season label to the study's end-year convention.

    The schedule/gamelog endpoints emit ``YYYY-YY`` labels (``2023-24``)
    or 8-digit codes; study.yaml speaks end-year plain integers
    (``2024``). A ``YYYY-YY`` string maps to ``2000 + YY`` (and
    ``1999-00`` -> ``2000``); plain years pass through unchanged. NaN
    stays NaN (never fabricated as a season).
    """
    if value is None:
        return float("nan")
    s = str(value).strip()
    if "-" in s:
        try:
            return int(s.split("-")[1]) + 2000
        except (TypeError, ValueError):
            return float("nan")
    try:
        return int(s)
    except (TypeError, ValueError):
        return float("nan")


def game_type_from_id(game_id: str) -> int | None:
    """The game type digit embedded in the NBA gameId (positions 2:3).

    ``0022300061`` -> 2 (regular season). Returns None when the id is
    not parseable (never guessed).
    """
    s = str(game_id)
    if len(s) < 3 or not s[:3].isdigit():
        return None
    return int(s[2])
