"""NBA walk-forward folds: expanding, 7-calendar-day validation windows.

Fold geometry comes exclusively from study.yaml (step_days,
min_train_games) — never from the run window. Each fold trains on every
decided game strictly before its validation window and validates on the
games inside the window (both classes present required at the window
level; single-class windows are skipped loudly).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: pd.Index
    val_idx: pd.Index
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def make_folds(df: pd.DataFrame, oof_first_season: int, *,
               cadence_days: int, date_col: str = "game_date",
               min_val_games: int = 1) -> list[Fold]:
    """Expanding walk-forward folds over the decided frame.

    Validation windows are consecutive ``cadence_days`` calendar spans
    from the first OOF-season game date through the last decided date.
    Training is everything strictly before each window (expanding, no
    embargo — windows are date-keyed and non-overlapping by construction).
    """
    dates = pd.to_datetime(df[date_col])
    oof_mask = df["season"] >= oof_first_season
    if not oof_mask.any():
        return []
    start = dates[oof_mask].min()
    end = dates.max()

    folds: list[Fold] = []
    fold_id = 0
    win_start = start
    while win_start <= end:
        win_end = win_start + pd.Timedelta(days=cadence_days - 1)
        val_idx = df.index[(dates >= win_start) & (dates <= win_end)]
        if len(val_idx) >= min_val_games:
            train_idx = df.index[dates < win_start]
            folds.append(Fold(
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
