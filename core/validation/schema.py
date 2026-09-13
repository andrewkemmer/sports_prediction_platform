"""Schema validation helpers (former core/validation.py, §2 split).

Probability and metric validation plus the shared frame-column guard.
These guards are used by pipelines, artifact writers, and tests.
"""

from __future__ import annotations

import math
from typing import Any


class ValidationError(ValueError):
    """Raised when a value violates the shared validation contract."""


def validate_probability(value: float, name: str = "probability",
                         *, strict: bool = True) -> float:
    """Validate a probability.

    With ``strict=True`` (default) the value must lie in the open interval
    (0, 1) — calibrated outputs never saturate. With ``strict=False`` the
    closed interval [0, 1] is allowed (raw model scores before clipping).
    """
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(p):
        raise ValidationError(f"{name} is not finite: {value!r}")
    if strict:
        if not 0.0 < p < 1.0:
            raise ValidationError(f"{name} {p!r} outside (0,1)")
    else:
        if not 0.0 <= p <= 1.0:
            raise ValidationError(f"{name} {p!r} outside [0,1]")
    return p


def validate_probability_pair(p_home: float, p_away: float,
                              *, tol: float = 1e-6) -> tuple[float, float]:
    """Validate a complementary home/away pair (must sum to ~1)."""
    validate_probability(p_home, "p_home", strict=False)
    validate_probability(p_away, "p_away", strict=False)
    total = p_home + p_away
    if abs(total - 1.0) > tol:
        raise ValidationError(
            f"p_home + p_away = {total}, expected 1.0 (tol {tol})")
    return p_home, p_away


def validate_metric(value: float, name: str) -> float:
    """Validate a displayed metric: finite float, no NaN/Inf."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(v):
        raise ValidationError(f"{name} is not finite: {value!r}")
    return v


def require_columns(frame: Any, columns: list[str],
                    context: str = "frame") -> None:
    """Raise ValidationError when required columns are missing."""
    cols = set(getattr(frame, "columns", []))
    missing = [c for c in columns if c not in cols]
    if missing:
        raise ValidationError(
            f"{context} missing required columns: {missing}")
