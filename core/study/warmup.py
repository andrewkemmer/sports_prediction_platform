"""Warmup rules and minimum-games-per-team gates.

A sport may not publish model-derived displays until its history satisfies
the warmup contract. These gates are shared by the pipeline (which skips
publishing) and the frontend (which renders an honest "warming up" state
instead of fabricated numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from core.config.platform import WarmupConfig
from core.study.date_resolution import parse_compact_date, shift_compact_date


@dataclass(frozen=True)
class WarmupStatus:
    """Result of evaluating the warmup contract for one sport/date."""

    ready: bool
    reason: str
    days_of_history: int
    min_games_per_team_met: bool
    teams_below_minimum: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ready


def evaluate_warmup(
    *,
    sport_warmup: WarmupConfig,
    first_game_date: str | None,
    run_date: str,
    team_game_counts: dict[str, int] | None = None,
) -> WarmupStatus:
    """Evaluate whether the sport has warmed up for ``run_date``.

    ``first_game_date`` is the compact date of the earliest game in the
    sport's store (None when the store is empty). ``team_game_counts`` maps
    team abbreviation -> decided games available for training. A team with
    fewer than ``min_games_per_team`` decided games fails the gate and is
    named in ``teams_below_minimum`` (the dashboard shows the honest
    shortfall rather than a number the model cannot support).
    """
    counts = dict(team_game_counts or {})

    if first_game_date is None:
        return WarmupStatus(
            ready=False,
            reason="no games ingested yet",
            days_of_history=0,
            min_games_per_team_met=False,
            teams_below_minimum=tuple(sorted(counts)),
        )

    first = parse_compact_date(first_game_date)
    run = parse_compact_date(run_date)
    days = (run - first).days
    if days < sport_warmup.min_history_days:
        return WarmupStatus(
            ready=False,
            reason=(f"history {days}d < required "
                    f"{sport_warmup.min_history_days}d"),
            days_of_history=days,
            min_games_per_team_met=False,
        )

    below = tuple(
        sorted(team for team, n in counts.items()
               if n < sport_warmup.min_games_per_team)
    )
    met = not below
    if not met:
        return WarmupStatus(
            ready=False,
            reason=(f"teams below {sport_warmup.min_games_per_team} decided "
                    f"games: {', '.join(below)}"),
            days_of_history=days,
            min_games_per_team_met=False,
            teams_below_minimum=below,
        )

    return WarmupStatus(
        ready=True,
        reason="ok",
        days_of_history=days,
        min_games_per_team_met=True,
    )


def warmup_complete_date(sport_warmup: WarmupConfig,
                         first_game_date: str) -> str:
    """The first compact date on which the history-days warmup is satisfied."""
    return shift_compact_date(
        first_game_date, sport_warmup.min_history_days)


def teams_ready_for_display(
    team_game_counts: dict[str, int],
    min_games: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split teams into (ready, below_minimum) tuples, both sorted.

    Used by the Power Rankings adapter to grey out (not hide) teams the
    model has not seen enough of — the reference convention is to show the
    row with an honest shortfall marker, never a fabricated Elo.
    """
    ready = tuple(sorted(t for t, n in team_game_counts.items()
                         if n >= min_games))
    below = tuple(sorted(t for t, n in team_game_counts.items()
                         if n < min_games))
    return ready, below
