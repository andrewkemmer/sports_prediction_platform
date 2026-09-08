"""Synthetic Statcast fixtures for MLB tests.

``make_statcast_games`` emits a minimal-but-valid Statcast pitch frame:
one pitch row per plate appearance (real Statcast is one row per pitch;
the feature SQL reads PA-ending rows for offense/pitcher aggregates and
first-inning rows for starter identification), with enough history for the
walk-forward geometry plus one undecided slate day.

Running scores accumulate per game from per-PA run attributions; a decided
game's final row carries the complete totals (what the ``game_winners``
CTE reads). Slate-day games end without a final row and keep mid-game
running scores, so ``home_win`` stays NaN — never fabricated.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL",
         "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYY",
         "NYM", "OAK", "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB",
         "TEX", "TOR", "WSH"]

PA_END_EVENTS = [
    "single", "double", "triple", "home_run", "strikeout", "walk",
    "hit_by_pitch", "field_out", "field_error", "fielders_choice",
    "grounded_into_double_play", "double_play", "sac_fly",
]


def make_statcast_games(
    start: date,
    n_days: int,
    games_per_day: int = 4,
    seed: int = 42,
    include_slate_day: bool = True,
) -> pd.DataFrame:
    """Build a synthetic Statcast frame spanning ``n_days`` from ``start``.

    The final day's games are left UNDECIDED (no final-score row) when
    ``include_slate_day`` is True, so tests exercise the slate path.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    pk = 700_000
    ab = 0

    def _emit_game(gd: date, home: str, away: str, *, decided: bool) -> None:
        nonlocal pk, ab
        pk += 1
        h_i, a_i = TEAMS.index(home), TEAMS.index(away)
        h_runs = int(rng.integers(1, 8)) + (h_i - a_i) % 3
        a_runs = int(rng.integers(0, 7)) + (a_i - h_i) % 2
        n_pa = max(60, (h_runs + a_runs) * 8)
        home_pitcher, away_pitcher = 9000 + h_i, 9000 + a_i

        rows: list[dict] = []
        scored = {"home": 0, "away": 0}
        for half, (topbot, batting) in enumerate(
                (("Top", "away"), ("Bot", "home"))):
            pitcher = away_pitcher if half == 0 else home_pitcher
            for pa in range(n_pa // 2):
                ab += 1
                event = PA_END_EVENTS[int(rng.integers(0, len(PA_END_EVENTS)))]
                hit = event in ("single", "double", "triple", "home_run",
                                "field_out", "field_error", "fielders_choice")
                runs_here = 0
                if hit and rng.random() < 0.18:
                    runs_here = int(rng.integers(1, 3))
                scored[batting] += runs_here
                rows.append({
                    "game_pk": pk,
                    "game_date": pd.Timestamp(gd),
                    "game_type": "R",
                    "home_team": home,
                    "away_team": away,
                    "inning": min(9, 1 + pa // 6),
                    "inning_topbot": topbot,
                    "at_bat_number": ab,
                    "pitch_number": 1,
                    "pitcher": pitcher,
                    "batter": 100 + (pa % 9) + (10 if half else 0),
                    "player_name": f"Pitcher {pitcher}",
                    "p_throws": "R",
                    "stand": "R",
                    "pitch_type": "FF",
                    "release_speed": float(92 + rng.integers(0, 6)),
                    "description": ("hit_into_play" if hit
                                    else "called_strike"),
                    "events": event,
                    "launch_speed": float(88 + rng.integers(0, 25)),
                    "launch_angle": float(rng.integers(-20, 45)),
                    "estimated_woba_using_speedangle":
                        float(np.round(rng.random() * 0.9 + 0.05, 3)),
                    "barrel": 1.0 if rng.random() < 0.05 else 0.0,
                    "hard_contact": 1.0 if rng.random() < 0.3 else 0.0,
                    "home_score": float(scored["home"]),
                    "away_score": float(scored["away"]),
                    "delta_home_win_exp": float(np.round(rng.random() - 0.5, 3)),
                    "delta_run_exp": float(np.round(rng.random() - 0.5, 3)),
                })
        if decided:
            # Cap the running score at the intended final so the last row's
            # totals ARE the final score (deterministic, no re-attribution).
            last = rows[-1]
            last["home_score"] = float(max(h_runs, scored["home"]))
            last["away_score"] = float(max(a_runs, scored["away"]))
            last["inning"] = 9
            last["inning_topbot"] = "Bot"
            last["description"] = "final"
            last["events"] = "field_out"
        else:
            # Raw Statcast carries no game-completion flag, so the feature
            # factory can only leave a game UNDECIDED when the last observed
            # scores are equal (no winner determinable). End slate-day games
            # in a tied state so they exercise the undecided path
            # deterministically — never a fabricated winner.
            last = rows[-1]
            tied = float(max(scored["home"], scored["away"]))
            last["home_score"] = tied
            last["away_score"] = tied
        frames.append(pd.DataFrame(rows))

    n_slate_days = 1 if include_slate_day else 0
    for day in range(n_days):
        gd = start + timedelta(days=day)
        decided = day < n_days - n_slate_days
        pair_seed = rng.permutation(len(TEAMS))
        for g in range(games_per_day):
            _emit_game(gd, TEAMS[pair_seed[2 * g]], TEAMS[pair_seed[2 * g + 1]],
                       decided=decided)
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
