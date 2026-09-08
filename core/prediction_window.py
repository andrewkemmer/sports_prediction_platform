"""Prediction-window resolution.

Given a sport's config and the run date, resolve which slate dates the run
may publish predictions for, and whether a given game belongs to the served
window. This is the shared semantic the daily Kaggle run, the artifact
contracts, and (later) the frontend date navigation all agree on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.config import PredictionWindowConfig, SportConfig
from core.dates import (
    ET,
    format_compact_date,
    is_valid_compact_date,
    parse_compact_date,
)


class PredictionWindowError(ValueError):
    """Raised when a prediction window cannot be resolved."""


@dataclass(frozen=True)
class PredictionWindow:
    """The resolved slate window for one run.

    ``run_date`` is the compact run date; ``slate_dates`` are the compact
    dates the run may publish (run_date + 0..lookahead_days). ``anchor`` is
    the run's UTC anchor instant.
    """

    run_date: str
    slate_dates: tuple[str, ...]
    lookahead_days: int
    anchor_utc: datetime

    def __post_init__(self) -> None:
        if not is_valid_compact_date(self.run_date):
            raise PredictionWindowError(
                f"invalid run_date: {self.run_date!r}")
        for d in self.slate_dates:
            if not is_valid_compact_date(d):
                raise PredictionWindowError(f"invalid slate date: {d!r}")

    def __contains__(self, date_str: str) -> bool:
        return date_str in self.slate_dates

    def primary_date(self) -> str:
        """The main slate date (run_date itself)."""
        return self.slate_dates[0]


def resolve_prediction_window(
    run_date: str,
    sport: SportConfig,
    *,
    now_utc: datetime | None = None,
) -> PredictionWindow:
    """Resolve the prediction window for a run.

    The window starts at the run date (ET) and spans ``lookahead_days``
    additional days. ``now_utc`` defaults to the real clock and is used only
    to stamp the anchor; the resolution itself is deterministic in
    ``run_date`` so replays produce identical windows.
    """
    if not is_valid_compact_date(run_date):
        raise PredictionWindowError(f"invalid run_date: {run_date!r}")
    window_cfg: PredictionWindowConfig = sport.prediction_window
    start = parse_compact_date(run_date)
    dates = tuple(
        format_compact_date(start + timedelta(days=i))
        for i in range(window_cfg.lookahead_days + 1)
    )
    anchor = now_utc or datetime.now(tz=timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return PredictionWindow(
        run_date=run_date,
        slate_dates=dates,
        lookahead_days=window_cfg.lookahead_days,
        anchor_utc=anchor,
    )


def game_in_window(
    start_time_utc: str,
    window: PredictionWindow,
    *,
    slate_cutoff_hour_local: int | None = None,
) -> bool:
    """True when a game start belongs to the window's primary slate date.

    A game is in the window when its ET calendar date is the run date (or
    any lookahead date) and, when a cutoff hour is supplied, its ET start
    hour is at or before the cutoff (late games roll to the next slate).
    Empty/invalid start times are excluded — never fabricated into a slate.
    """
    if not start_time_utc:
        return False
    try:
        ts = datetime.fromisoformat(str(start_time_utc).replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local = ts.astimezone(ET)
    if format_compact_date(local) not in window.slate_dates:
        return False
    if slate_cutoff_hour_local is not None and local.hour > slate_cutoff_hour_local:
        return False
    return True


def partition_slate(
    start_times_utc: list[str],
    window: PredictionWindow,
    *,
    slate_cutoff_hour_local: int | None = None,
) -> tuple[list[str], list[str]]:
    """Split game start times into (in_window, out_of_window).

    Out-of-window rows are carried (not dropped) so callers can report them
    honestly — the reference repo never silently drops games.
    """
    inside: list[str] = []
    outside: list[str] = []
    for ts in start_times_utc:
        if game_in_window(ts, window,
                          slate_cutoff_hour_local=slate_cutoff_hour_local):
            inside.append(ts)
        else:
            outside.append(ts)
    return inside, outside
