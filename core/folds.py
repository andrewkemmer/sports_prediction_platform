"""Walk-forward fold definitions.

Folds are deterministic given the set of decided game dates: a fold is a
train/test split at a date boundary, stepped by ``step_days``, guarded by
``min_train_games``. Every sport's pipeline and every experiment uses the
same fold builder so OOF metrics are comparable across sports and studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from core.config import WalkForwardConfig
from core.dates import (
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
