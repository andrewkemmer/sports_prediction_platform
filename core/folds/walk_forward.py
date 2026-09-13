"""Walk-forward fold definitions.

Two fold builders live here, both sport-agnostic (core NEVER imports
sports.*):

1. ``build_folds`` — the original daily-stride fold (compact-date keyed,
   single test date per fold, embargo-aware). Used by core.study and the
   daily prequential path.

2. ``make_folds`` — the calendar-window expanding walk-forward builder
   consolidated from the per-sport fold modules (Phase 7.5 Task 1).
   Validation windows are consecutive ``cadence_days``-wide calendar spans
   from the first OOF-season game date through the last decided date;
   training is everything strictly before each window (expanding, no
   embargo — windows are date-keyed and non-overlapping by construction).
   Per-sport behavior is selected by explicit parameters:

   * ``val_scope`` — ``"oof_season"`` restricts validation windows to
     OOF-season games (the NFL semantic; warmup rows may join training
     sets but never a validation window). ``"all"`` (NBA/NHL semantic)
     windows over the whole decided frame from the first OOF-season date.
   * ``min_val_games`` — windows below this many validation rows are
     skipped (NBA/NHL pass the study value; NFL passes 1).
   * ``end_of_day`` — when True (NFL), the window's upper bound includes
     the full final day (23:59:59) so intraday timestamps on the boundary
     date validate; when False (NBA/NHL), plain inclusive date comparison.

   NBA and NHL used identical implementations; the NFL variant differs
   only in the three parameters above. The consolidated builder
   reproduces each sport's previous folds EXACTLY (verified by the
   Phase 7.5 fingerprint comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from core.config.platform import WalkForwardConfig
from core.study.date_resolution import (
    DateError,
    format_compact_date,
    parse_compact_date,
    shift_compact_date,
)


class FoldError(ValueError):
    """Raised when fold construction is impossible or invalid."""


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    ``train_end`` is the exclusive cutoff: games with ``game_date <
    train_end`` (and, with embargo, ``<= train_end - embargo``) may train.
    ``test_date`` is the single date being predicted (the daily stride
    matches the reference repo's prequential daily walk-forward).
    """

    fold_id: int
    train_end: str  # exclusive compact date
    test_date: str  # compact date
    n_train_games: int

    def __post_init__(self) -> None:
        for label, value in (("train_end", self.train_end),
                             ("test_date", self.test_date)):
            try:
                parse_compact_date(value)
            except DateError as exc:
                raise FoldError(f"invalid {label}: {value!r}") from exc
        if self.n_train_games < 0:
            raise FoldError("n_train_games must be >= 0")


@dataclass
class WindowFold:
    """One calendar-window walk-forward fold (``make_folds`` output).

    Positional-compatible with the former per-sport ``Fold`` dataclasses:
    ``fold_id``, ``val_start``, ``val_end``, ``train_idx``, ``val_idx``
    (NFL field order) plus the NBA/NHL ``train_end`` alias field.
    """

    fold_id: int
    train_idx: pd.Index
    val_idx: pd.Index
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def build_folds(
    game_dates: list[str],
    config: WalkForwardConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Fold]:
    """Build the ordered list of walk-forward folds.

    ``game_dates`` is the list of compact dates with decided games
    (duplicates allowed; they are counted, not deduplicated — two games on
    one date count as two training rows).

    A fold exists for each test date in ``[start_date or first test
    candidate, end_date or last game date]`` stepped by ``step_days``, where
    the training pool (games strictly before the test date, minus the
    embargo window) has at least ``min_train_games`` rows. Dates with no
    games are skipped (no fold is emitted for an empty slate).

    Raises ``FoldError`` when no fold satisfies the minimum-training guard.
    """
    if not game_dates:
        raise FoldError("no game dates supplied — cannot build folds")

    counts: dict[str, int] = {}
    for d in game_dates:
        try:
            parse_compact_date(d)
        except DateError as exc:
            raise FoldError(f"invalid game date: {d!r}") from exc
        counts[d] = counts.get(d, 0) + 1

    all_dates = sorted(counts)
    if start_date is not None:
        try:
            parse_compact_date(start_date)
        except DateError as exc:
            raise FoldError(f"invalid start_date: {start_date!r}") from exc
    if end_date is not None:
        try:
            parse_compact_date(end_date)
        except DateError as exc:
            raise FoldError(f"invalid end_date: {end_date!r}") from exc

    # The first fold candidate must leave enough decided games BEFORE it to
    # satisfy the minimum-training guard; otherwise walk forward from the
    # first date that can qualify.
    first_test = start_date or all_dates[0]
    last_test = end_date or all_dates[-1]
    if first_test > last_test:
        raise FoldError(
            f"start_date {first_test} after end_date {last_test}")

    # Cumulative decided games strictly before each candidate date.
    sorted_unique = all_dates
    folds: list[Fold] = []
    fold_id = 0
    step = config.step_days
    embargo = config.embargo_days

    candidate = first_test
    while candidate <= last_test:
        # Strictly-before training pool, minus embargo tail. The embargo
        # excludes the ``embargo_days`` dates immediately preceding the test
        # date (embargo_days=1 for test 0903 drops 0902, keeps 0901).
        cutoff_exclusive = candidate
        embargo_floor = (shift_compact_date(candidate, -embargo)
                         if embargo else None)
        n_train = 0
        for d in sorted_unique:
            if d >= cutoff_exclusive:
                break
            if embargo_floor is not None and d >= embargo_floor:
                continue
            n_train += counts[d]
        if n_train >= config.min_train_games and candidate in counts:
            folds.append(Fold(
                fold_id=fold_id,
                train_end=cutoff_exclusive,
                test_date=candidate,
                n_train_games=n_train,
            ))
            fold_id += 1
        candidate = format_compact_date(
            parse_compact_date(candidate) + timedelta(days=step))

    if not folds:
        raise FoldError(
            f"no fold satisfies min_train_games="
            f"{config.min_train_games} over "
            f"[{first_test}, {last_test}]")
    return folds


def train_test_boundary(fold: Fold) -> tuple[str, str]:
    """The (train_max_inclusive, test_date) boundary for point-in-time joins.

    Training rows satisfy ``game_date < train_end``; test rows satisfy
    ``game_date == test_date``. Returned as compact strings.
    """
    return fold.train_end, fold.test_date


# ---------------------------------------------------------------------------
# Calendar-window expanding walk-forward (consolidated per-sport builder)
# ---------------------------------------------------------------------------

def make_folds(
    df: pd.DataFrame,
    oof_first_season: int,
    *,
    cadence_days: int,
    date_col: str = "game_date",
    min_val_games: int = 1,
    val_scope: str = "all",
    end_of_day: bool = False,
) -> list[WindowFold]:
    """Expanding calendar-window walk-forward folds over the decided frame.

    Validation windows are consecutive ``cadence_days``-wide calendar spans
    from the first OOF-season game date through the last decided date.
    Training is everything strictly before each window (expanding, no
    embargo — windows are date-keyed and non-overlapping by construction).

    Args:
        df: decided game frame with ``date_col`` and a numeric ``season``
            column.
        oof_first_season: first season eligible for validation windows.
        cadence_days: validation window width in calendar days.
        date_col: the game-date column.
        min_val_games: windows with fewer validation rows are skipped.
        val_scope: ``"oof_season"`` masks validation rows to OOF-season
            games (NFL); ``"all"`` windows every game in the frame from the
            first OOF-season date (NBA/NHL).
        end_of_day: include the full final boundary day (NFL's 23:59:59
            upper bound); False keeps the plain inclusive date comparison.

    Returns a list of :class:`WindowFold` in chronological order.
    """
    if date_col not in df.columns:
        raise KeyError(f"make_folds: missing date column {date_col!r}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    seasons = pd.to_numeric(df["season"], errors="coerce")
    oof_mask = seasons >= oof_first_season
    if not oof_mask.any():
        return []

    if val_scope == "oof_season":
        core_mask = oof_mask
        core_dates = dates[core_mask].dropna()
        if core_dates.empty:
            return []
        start = core_dates.min().normalize()
        end = core_dates.max().normalize()
        if end_of_day:
            end_upper = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
        else:
            end_upper = end
    elif val_scope == "all":
        start = dates[oof_mask].min()
        end = dates.max()
        end_upper = end
        core_mask = None
    else:
        raise ValueError(
            f"make_folds: unknown val_scope {val_scope!r} "
            "(expected 'all' or 'oof_season')")

    folds: list[WindowFold] = []
    fold_id = 0
    win_start = start
    while win_start <= end:
        win_end = win_start + pd.Timedelta(days=cadence_days - 1)
        if end_of_day:
            upper = win_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
        else:
            upper = win_end
        if core_mask is not None:
            val_mask = core_mask & (dates >= win_start) & (dates <= upper)
        else:
            val_mask = (dates >= win_start) & (dates <= upper)
        val_idx = df.index[val_mask]
        if len(val_idx) >= min_val_games:
            train_idx = df.index[dates < win_start]
            folds.append(WindowFold(
                fold_id=fold_id,
                train_idx=train_idx,
                val_idx=val_idx,
                train_end=(dates.loc[train_idx].max()
                           if len(train_idx) else pd.NaT),
                val_start=win_start,
                val_end=win_end,
            ))
            fold_id += 1
        win_start = win_end + pd.Timedelta(days=1)
    return folds


def walk_forward_splits(
    games: pd.DataFrame,
    retrain_cadence_days: int = 7,
    max_eval_folds: int = 0,
    min_train_days: int = 0,
) -> list[dict]:
    """Expanding-window walk-forward splits (the MLB training geometry).

    Consolidated from sports.mlb.training (Phase 7.5 Task 1): each
    validation window is ``retrain_cadence_days`` unique-date slots wide,
    the training set is all decided games strictly before the validation
    window start, and windows are non-overlapping and chronological. A
    final fold whose full cadence window would overrun the frame's tail
    runs PARTIALLY into the tail (leakage-free: train stays strictly
    < val_start).

    Returns a list of dicts with keys: train_games, val_games, fold_idx,
    val_start, val_end, is_partial_tail.
    """
    import logging

    logger = logging.getLogger(__name__)
    if "game_date" not in games.columns:
        raise ValueError("games must have a 'game_date' column")
    if "home_win" not in games.columns:
        logger.warning("walk_forward_splits: no 'home_win' column — cannot split")
        return []

    df = games.dropna(subset=["home_win"]).copy()
    if df.empty:
        logger.warning(
            "walk_forward_splits: all %d rows have NaN home_win — cannot split",
            len(games))
        return []
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_date"] = df["game_date"].dt.normalize()
    df = df.sort_values("game_date").reset_index(drop=True)

    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) < retrain_cadence_days + 1:
        logger.warning(
            "walk_forward_splits: only %d unique dates (need >= %d for "
            "cadence %d)", len(unique_dates), retrain_cadence_days + 1,
            retrain_cadence_days)
        return []

    splits: list[dict] = []
    fold_idx = 0
    val_start_idx = max(retrain_cadence_days, min_train_days)
    while val_start_idx < len(unique_dates):
        val_start = unique_dates[val_start_idx]
        val_end_idx = min(val_start_idx + retrain_cadence_days, len(unique_dates))
        val_end = unique_dates[val_end_idx - 1]
        is_partial_tail = val_end_idx < val_start_idx + retrain_cadence_days

        train_mask = df["game_date"] < val_start
        val_mask = (df["game_date"] >= val_start) & (df["game_date"] <= val_end)
        train_games = df[train_mask].copy()
        val_games = df[val_mask].copy()

        if not train_games.empty and not val_games.empty:
            splits.append({
                "train_games": train_games,
                "val_games": val_games,
                "fold_idx": fold_idx,
                "val_start": val_start,
                "val_end": val_end,
                "is_partial_tail": is_partial_tail,
            })
            fold_idx += 1
        val_start_idx = val_end_idx

    if max_eval_folds > 0 and len(splits) > max_eval_folds:
        splits = splits[-max_eval_folds:]
    return splits


def fold_summary(folds: list) -> dict:
    """Diagnostics for the final validation record (NFL convention)."""
    if not folds:
        return {"n_folds": 0}
    return {
        "n_folds": len(folds),
        "first_val_start": str(folds[0].val_start.date()),
        "last_val_end": str(folds[-1].val_end.date()),
        "min_train": int(min(len(f.train_idx) for f in folds)),
        "max_train": int(max(len(f.train_idx) for f in folds)),
        "total_val_games": int(sum(len(f.val_idx) for f in folds)),
    }


def fold_table(df: pd.DataFrame, folds: list,
               date_col: str = "gameday") -> pd.DataFrame:
    """Per-fold reporting table (fold_id, train_end_date, validation window,
    n_train, n_val). ``df`` must be the SAME frame (same row order)
    the folds were generated over. Column names match the immutable
    artifact contract (``fold_table``: fold_id/n_train/n_val) and every
    other fold-table producer in the platform."""
    rows = []
    dates = pd.to_datetime(df[date_col], errors="coerce")
    for f in folds:
        rows.append({
            "fold_id": f.fold_id,
            "train_end_date": (dates.loc[f.train_idx].max().date()
                               if len(f.train_idx) else pd.NaT),
            "validation_start": f.val_start.date(),
            "validation_end": f.val_end.date(),
            "n_train": int(len(f.train_idx)),
            "n_val": int(len(f.val_idx)),
        })
    return pd.DataFrame(rows)
