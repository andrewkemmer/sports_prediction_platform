"""Synthetic NBA test fixtures.

Shared builders for the NBA test modules. Frames mirror the NBA Stats API
schedule column contract (sports/nba/catalog.py); undecided slate rows
carry NaN scores (no fabricated outcomes). Scores are deterministic and
guarantee both classes in every validation window (needed by the
five-member stack's training gate). Seasons use the study's END-YEAR
convention (2023 = the 2023-24 season).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["BOS", "DEN", "MIL", "PHX", "NYK", "MIA"]


def make_schedule(start_season: int = 2015, n_seasons: int = 2,
                  weeks_per_season: int = 8) -> pd.DataFrame:
    """Synthetic REG schedule (gameId type digit 2): every team plays once
    per week; scores are deterministic (home wins ~60%) with both classes
    in every weekly window; the final week of the latest season is
    undecided."""
    rows = []
    rng = np.random.default_rng(42)
    for season in range(start_season, start_season + n_seasons):
        for week in range(1, weeks_per_season + 1):
            undecided = (season == start_season + n_seasons - 1
                         and week == weeks_per_season)
            # Circle-method round robin: 6 teams -> 3 distinct pairs per
            # week, each pair meets once across the rotation.
            fixed = TEAMS[0]
            others = TEAMS[1:]
            shifted = others[(week - 1) % len(others):] \
                + others[:(week - 1) % len(others)]
            rot = [fixed] + shifted
            for i in range(0, len(TEAMS), 2):
                home, away = rot[i], rot[i + 1]
                if home > away:
                    home, away = away, home
                gd = pd.Timestamp(f"{season - 1}-10-01") + pd.Timedelta(
                    weeks=week - 1, days=(TEAMS.index(home) % 3))
                # Realistic distinct instants (23:30Z / 00:30Z stagger)
                # so the ladder's instant-ordered leakage gate and the
                # prime_time feature are exercised exactly as in real
                # stats.nba.com data (gameDateTimeUTC pre-game fact).
                st_utc = (gd + pd.Timedelta(
                    hours=23 if (TEAMS.index(home) % 2) else 0,
                    minutes=30 if (TEAMS.index(home) % 2) else 30)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                if undecided:
                    hs, as_ = np.nan, np.nan
                else:
                    home_wins = rng.random() < 0.6
                    if home_wins:
                        hs = int(rng.integers(100, 130))
                        as_ = int(rng.integers(85, hs - 2))
                    else:
                        as_ = int(rng.integers(100, 130))
                        hs = int(rng.integers(85, as_ - 2))
                rows.append({
                    "game_id": f"002{season % 100:02d}"
                               f"{week:02d}{TEAMS.index(home):02d}"
                               f"{TEAMS.index(away):02d}",
                    "season": season, "game_type": 2,
                    "game_date": gd.date().isoformat(),
                    "start_time_utc": st_utc,
                    "home_team": home, "away_team": away,
                    "home_score": hs, "away_score": as_,
                    "venue": f"{home} Arena", "neutral_site": False,
                    "game_state": "Final" if not undecided else "7:30 pm ET",
                })
    out = pd.DataFrame(rows).drop_duplicates("game_id")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permanent evidence fixture (Phase 7.5e-A rebaseline)
# ---------------------------------------------------------------------------
#: Deterministic bounded raw dataset backing the committed
#: ``tests/fixtures/evidence/nba.parquet`` consumed by the permanent
#: fold/matrix evidence tests.
EVIDENCE_START_SEASON = 2015
EVIDENCE_SEASONS = 3
EVIDENCE_WEEKS = 12


def evidence_frame() -> pd.DataFrame:
    """Regenerate the committed NBA evidence fixture deterministically.

    Tests LOAD ``tests/fixtures/evidence/nba.parquet`` and never call this
    to overwrite it; it exists for reviewed rebaselines only.
    """
    return make_schedule(EVIDENCE_START_SEASON, EVIDENCE_SEASONS,
                         EVIDENCE_WEEKS)


def make_player_cache(schedule: pd.DataFrame) -> pd.DataFrame:
    """Synthetic player rows: a stable high-scorer per team per decided
    game with realistic points/assists."""
    rows = []
    for r in schedule.itertuples(index=False):
        if pd.isna(r.home_score):
            continue
        for team in (r.home_team, r.away_team):
            rows.append({
                "game_id": r.game_id, "team": team,
                "player_id": f"{team}-P1",
                "player_name": f"{team} Star", "position": "F",
                "minutes": "34:00", "points": 27, "assists": 7,
                "rebounds": 9,
                "game_date": r.game_date, "season": r.season,
            })
    return pd.DataFrame(rows)
