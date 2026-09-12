"""Synthetic NHL test fixtures.

Shared builders for the NHL test modules. Frames mirror the NHL Stats API
schedule column contract (sports/nhl/catalog.py); undecided slate rows
carry NaN scores (no fabricated outcomes). Scores are deterministic and
guarantee both classes in every validation window (needed by the
five-member stack's training gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["TOR", "BOS", "MTL", "NYR", "TBL", "VGK"]


def make_schedule(start_season: int = 2021, n_seasons: int = 2,
                  weeks_per_season: int = 8) -> pd.DataFrame:
    """Synthetic REG schedule (gameType 2): every team plays once per
    week; scores are deterministic (home wins ~60%) with both classes in
    every weekly window; the final week of the latest season is
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
                gd = pd.Timestamp(f"{season}-10-01") + pd.Timedelta(
                    weeks=week - 1, days=(TEAMS.index(home) % 3))
                # Realistic distinct instants (19:00Z / 00:00Z stagger)
                # so the ladder's instant-ordered leakage gate and the
                # prime_time feature are exercised exactly as in real
                # api-web.nhle.com data (start_time_utc pre-game fact).
                st_utc = (gd + pd.Timedelta(
                    hours=19 if (TEAMS.index(home) % 2) else 0)
                ).strftime("%Y-%m-%dT%H:00:00Z")
                if undecided:
                    hs, as_ = np.nan, np.nan
                else:
                    home_wins = rng.random() < 0.6
                    if home_wins:
                        hs = int(rng.integers(2, 7))
                        as_ = int(rng.integers(0, hs))
                    else:
                        as_ = int(rng.integers(2, 7))
                        hs = int(rng.integers(0, as_))
                rows.append({
                    "game_id": f"{season}{week:02d}{away}{home}",
                    "season": season, "game_type": 2,
                    "game_date": gd.date().isoformat(),
                    "start_time_utc": st_utc,
                    "home_team": home, "away_team": away,
                    "home_score": hs, "away_score": as_,
                    "venue": f"{home} Arena", "neutral_site": False,
                    "game_state": "OFF" if not undecided else "FUT",
                })
    out = pd.DataFrame(rows).drop_duplicates("game_id")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permanent evidence fixture (Phase 7.5e-A rebaseline)
# ---------------------------------------------------------------------------
#: Deterministic bounded raw dataset backing the committed
#: ``tests/fixtures/evidence/nhl.parquet`` consumed by the permanent
#: fold/matrix evidence tests.
EVIDENCE_START_SEASON = 2021
EVIDENCE_SEASONS = 3
EVIDENCE_WEEKS = 12


def evidence_frame() -> pd.DataFrame:
    """Regenerate the committed NHL evidence fixture deterministically.

    Tests LOAD ``tests/fixtures/evidence/nhl.parquet`` and never call this
    to overwrite it; it exists for reviewed rebaselines only.
    """
    return make_schedule(EVIDENCE_START_SEASON, EVIDENCE_SEASONS,
                         EVIDENCE_WEEKS)


def make_goalie_cache(schedule: pd.DataFrame) -> pd.DataFrame:
    """Synthetic goalie rows: one starter per team per decided game with
    realistic shot/goal counts."""
    rows = []
    for r in schedule.itertuples(index=False):
        if pd.isna(r.home_score):
            continue
        for team in (r.home_team, r.away_team):
            rows.append({
                "game_id": r.game_id, "team": team,
                "player_id": f"{team}-G1",
                "player_name": f"{team} Goalie1", "position": "G",
                "starter": True, "decision": "W",
                "saves": 28, "shots_against": 30,
                "goals_against": 2, "save_pctg": 0.933, "toi": "60:00",
                "game_date": r.game_date, "season": r.season,
            })
    return pd.DataFrame(rows)
