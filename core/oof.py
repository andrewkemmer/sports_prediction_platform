"""Out-of-fold (OOF) rules.

OOF predictions are the point-in-time contract the whole platform trusts:
a prediction for a game may only use information available before that
game's start. This module defines the row schema, the validation rules, and
the small set of derived metrics (Brier, log-loss, AUC) shared by monitors
and calibration pages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

OOF_REQUIRED_COLUMNS = (
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_win_prob_model",
    "home_win",
)


class OOFError(ValueError):
    """Raised when an OOF frame violates the point-in-time contract."""


@dataclass(frozen=True)
class OOFValidationReport:
    """Result of validating an OOF frame."""

    n_rows: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_oof_rows(rows: list[dict]) -> OOFValidationReport:
    """Validate OOF rows against the contract.

    Rules (mirroring the reference repo's strict point-in-time policy):
    * required columns present on every row,
    * ``home_win_prob_model`` a finite float in (0, 1) — never exactly 0/1
      (calibrated probabilities live strictly inside the unit interval),
    * ``home_win`` exactly {0.0, 1.0} (NaN/None means undecided and is a
      warning, not an error — the row must be excluded from metrics),
    * ``game_date`` a valid compact date,
    * no duplicate ``game_id`` (duplicates silently double-count outcomes),
    * home/away teams non-empty and distinct.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    if not rows:
        return OOFValidationReport(n_rows=0, errors=("empty OOF frame",),
                                   warnings=())

    for i, row in enumerate(rows):
        missing = [c for c in OOF_REQUIRED_COLUMNS if c not in row]
        if missing:
            errors.append(f"row {i}: missing columns {missing}")
            continue

        gid = str(row["game_id"])
        if gid in seen_ids:
            errors.append(f"row {i}: duplicate game_id {gid!r}")
        seen_ids.add(gid)

        p = row["home_win_prob_model"]
        try:
            pf = float(p)
        except (TypeError, ValueError):
            errors.append(f"row {i} ({gid}): home_win_prob_model not numeric")
            continue
        if not math.isfinite(pf) or not (0.0 < pf < 1.0):
            errors.append(
                f"row {i} ({gid}): home_win_prob_model {pf!r} outside (0,1)")

        hw = row["home_win"]
        if hw is None or (isinstance(hw, float) and math.isnan(hw)):
            warnings.append(f"row {i} ({gid}): undecided (home_win null)")
        else:
            try:
                hwf = float(hw)
            except (TypeError, ValueError):
                errors.append(f"row {i} ({gid}): home_win not numeric")
                continue
            if hwf not in (0.0, 1.0):
                errors.append(
                    f"row {i} ({gid}): home_win must be 0/1, got {hwf!r}")

        gd = str(row["game_date"])
        try:
            from core.dates import is_valid_compact_date
            if not is_valid_compact_date(gd):
                raise ValueError
        except Exception:
            errors.append(f"row {i} ({gid}): invalid game_date {gd!r}")

        home, away = str(row["home_team"]), str(row["away_team"])
        if not home or not away:
            errors.append(f"row {i} ({gid}): empty team code")
        elif home == away:
            errors.append(f"row {i} ({gid}): home == away ({home!r})")

    return OOFValidationReport(
        n_rows=len(rows), errors=tuple(errors), warnings=tuple(warnings))


def brier_score(probs: list[float], outcomes: list[float]) -> float:
    """Mean Brier score over decided rows (0 = perfect, 1 = worst)."""
    if len(probs) != len(outcomes):
        raise OOFError("probs/outcomes length mismatch")
    if not probs:
        raise OOFError("no decided rows")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def log_loss(probs: list[float], outcomes: list[float],
             eps: float = 1e-15) -> float:
    """Mean log-loss with clipping away from the 0/1 boundary."""
    if len(probs) != len(outcomes):
        raise OOFError("probs/outcomes length mismatch")
    if not probs:
        raise OOFError("no decided rows")
    total = 0.0
    for p, y in zip(probs, outcomes):
        pc = min(max(p, eps), 1.0 - eps)
        total += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
    return total / len(probs)


def roc_auc(probs: list[float], outcomes: list[float]) -> float | None:
    """AUC via the rank (Mann-Whitney U) statistic. None when degenerate."""
    if len(probs) != len(outcomes):
        raise OOFError("probs/outcomes length mismatch")
    n_pos = sum(1 for y in outcomes if y == 1.0)
    n_neg = len(outcomes) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, outcomes) if y == 1.0)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def oof_metrics(probs: list[float], outcomes: list[float]) -> dict:
    """The shared metric block every monitor/calibration page reads."""
    auc = roc_auc(probs, outcomes)
    return {
        "n": len(probs),
        "brier": round(brier_score(probs, outcomes), 6),
        "logloss": round(log_loss(probs, outcomes), 6),
        "auc": round(auc, 6) if auc is not None else None,
    }
