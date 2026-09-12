"""Synthetic NFL test fixtures.

Shared builders for the NFL test modules. Frames mirror the nflverse
column contract; undecided slate rows carry NaN scores (no fabricated
outcomes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI"]


def make_schedule(start_season: int = 2019, n_seasons: int = 2,
                  weeks_per_season: int = 6) -> pd.DataFrame:
    """Synthetic REG schedule: every team plays once per week; scores are
    deterministic (home wins ~60%) with a mid-slate tie for settlement
    tests; the final week of the latest season is undecided."""
    rows = []
    rng = np.random.default_rng(42)
    for season in range(start_season, start_season + n_seasons):
        for week in range(1, weeks_per_season + 1):
            undecided = (season == start_season + n_seasons - 1
                         and week == weeks_per_season)
            # Circle-method round robin: 6 teams -> 3 distinct pairs per
            # week, each pair meets once across the 5-week rotation.
            fixed = TEAMS[0]
            others = TEAMS[1:]
            shifted = others[(week - 1) % len(others):] \
                + others[:(week - 1) % len(others)]
            rot = [fixed] + shifted
            for i in range(0, len(TEAMS), 2):
                home, away = rot[i], rot[i + 1]
                if home > away:
                    home, away = away, home
                gd = pd.Timestamp(f"{season}-09-01") + pd.Timedelta(
                    weeks=week - 1, days=(TEAMS.index(home) % 3))
                if undecided:
                    hs, as_ = np.nan, np.nan
                elif week == 3 and home == TEAMS[0]:
                    hs, as_ = 20, 20  # an exact tie
                else:
                    # home wins ~60%: deterministic pseudo-randomness with
                    # BOTH classes present in every week's window
                    home_wins = rng.random() < 0.6
                    if home_wins:
                        hs = int(rng.integers(14, 38))
                        as_ = int(rng.integers(3, hs))
                    else:
                        as_ = int(rng.integers(14, 38))
                        hs = int(rng.integers(3, as_))
                rows.append({
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season, "week": week, "game_type": "REG",
                    "gameday": gd.date().isoformat(),
                    "gametime": "13:00",
                    "home_team": home, "away_team": away,
                    "home_score": hs, "away_score": as_,
                    "roof": "outdoors", "div_game": 0,
                    "stadium": f"{home} Stadium", "location": "USA",
                    "home_qb_id": None, "away_qb_id": None,
                    "home_qb_name": None, "away_qb_name": None,
                })
    out = pd.DataFrame(rows).drop_duplicates("game_id")
    return out.reset_index(drop=True)


def make_pbp(schedule: pd.DataFrame) -> pd.DataFrame:
    """Synthetic PBP: ~20 plays per team per game with positive yards."""
    rows = []
    rng = np.random.default_rng(7)
    for r in schedule.itertuples(index=False):
        if pd.isna(r.home_score):
            continue
        for team, score in ((r.home_team, r.home_score),
                            (r.away_team, r.away_score)):
            for play in range(20):
                rows.append({
                    "game_id": r.game_id, "posteam": team,
                    "defteam": r.away_team if team == r.home_team
                    else r.home_team,
                    "yards_gained": float(rng.integers(0, 12)),
                    "epa": float(rng.normal(0, 0.5)),
                    "qb_epa": float(rng.normal(0, 0.4)),
                    "game_seconds_remaining": 3600 - play * 150,
                    "interception": 0, "fumble_lost": 0,
                    "passing_yards": float(rng.integers(0, 15)),
                    "pass_attempt": 1, "sack": 0, "penalty": 0,
                    "penalty_yards": 0, "penalty_team": None,
                    "third_down_converted": 0, "third_down_failed": 0,
                    "yardline_100": 50, "touchdown": 0,
                    "field_goal_result": None, "drive": play // 5,
                })
    return pd.DataFrame(rows)


def make_player_stats(season: int) -> pd.DataFrame:
    """Synthetic QB player stats: two QBs per team over 10 weeks."""
    rows = []
    for team in TEAMS:
        for name_i, qb in enumerate((f"{team} QB1", f"{team} QB2")):
            for week in range(1, 11):
                rows.append({
                    "player_id": f"{team}-{name_i}",
                    "player_name": qb, "position": "QB", "team": team,
                    "season": season, "week": week, "season_type": "REG",
                    "attempts": 30, "completions": 19,
                    "passing_yards": 240.0, "passing_tds": 1.5,
                    "passing_interceptions": 0.5,
                })
    return pd.DataFrame(rows)


def cache_only_loader(cache_dir, kind: str):
    """Deterministic, network-free stand-in for the nflverse transport.

    Returns the cached season frame when the store has it, else an empty
    frame (a season the store has not published yet). Lets hash/fold
    pinning tests exercise the real store without ever touching the
    network; the production default transport is exercised by the
    ingestion integration smokes with an injected adapter instead.
    """
    from pathlib import Path as _Path

    base = _Path(cache_dir)

    def load(seasons):
        season = seasons[0] if isinstance(seasons, (list, tuple)) else seasons
        path = base / f"{kind}_{season}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    return load


# ---------------------------------------------------------------------------
# Permanent evidence fixture (Phase 7.5e-A rebaseline)
# ---------------------------------------------------------------------------
#: Deterministic bounded raw dataset backing the committed
#: ``tests/fixtures/evidence/nfl.parquet`` (schedules + pbp in one frame,
#: discriminated by the ``__kind`` column) consumed by the permanent
#: fold/matrix evidence tests.
EVIDENCE_START_SEASON = 2018
EVIDENCE_SEASONS = 10
EVIDENCE_WEEKS = 6


def evidence_frame() -> pd.DataFrame:
    """Regenerate the committed NFL evidence fixture deterministically.

    One compact frame carries both raw families (``__kind`` in
    {``schedules``, ``pbp``}); the evidence-store materializer splits it
    back into the per-season store files. Tests LOAD the committed parquet
    and never call this to overwrite it.
    """
    sched = make_schedule(EVIDENCE_START_SEASON, EVIDENCE_SEASONS,
                          EVIDENCE_WEEKS).assign(__kind="schedules")
    pbp = make_pbp(sched).assign(__kind="pbp")
    pbp["season"] = pbp["game_id"].map(
        dict(zip(sched["game_id"], sched["season"])))
    return pd.concat([sched, pbp], ignore_index=True)


def make_venue_csv(tmp_path) -> str:
    """Minimal stadiums table covering the fixture teams."""
    rows = []
    for team in TEAMS:
        rows.append({
            "stadium": f"{team} Stadium", "facility": team,
            "teams": team, "lat": 40.0 + TEAMS.index(team),
            "lon": -80.0 + TEAMS.index(team),
            "altitude_ft": 100.0, "tz": "America/New_York",
            "source": "test",
        })
    p = tmp_path / "nfl_stadiums.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)
