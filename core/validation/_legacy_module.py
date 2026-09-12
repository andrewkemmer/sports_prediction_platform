"""Validation and determinism utilities.

Shared guards used by pipelines, artifact writers, and tests:

* probability validation (strict unit interval, calibrated outputs never
  exactly 0/1),
* finite-float checks for every metric the monitors display,
* deterministic serialization helpers (stable JSON with sorted keys, fixed
  rounding) so replays produce byte-identical artifacts,
* a replay-check helper that runs a callable twice and compares outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
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


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def round_float(value: float, digits: int = 6) -> float:
    """Deterministic rounding used in every serialized metric."""
    v = validate_metric(value, "value")
    return round(v, digits)


def stable_value(value: Any) -> Any:
    """Convert a value into a JSON-stable representation.

    Dataclasses become dicts (sorted keys later), tuples become lists,
    floats are passed through (callers round explicitly), Paths become
    strings.
    """
    from pathlib import PurePath
    if is_dataclass(value) and not isinstance(value, type):
        return {k: stable_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): stable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable_value(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def stable_json(payload: Any, *, indent: int | None = 2) -> str:
    """Deterministic JSON: sorted keys, no locale drift, stable floats."""
    return json.dumps(stable_value(payload), indent=indent, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a UTF-8 string (artifact fingerprinting)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes (binary artifact fingerprinting)."""
    return hashlib.sha256(data).hexdigest()


def assert_deterministic(fn, *args, **kwargs) -> str:
    """Run ``fn`` twice and assert identical stable-JSON outputs.

    Returns the digest of the output. Raises ``ValidationError`` when the
    two runs differ (nondeterminism found). Used by artifact writers and
    tests to guarantee replayable runs.
    """
    out1 = fn(*args, **kwargs)
    out2 = fn(*args, **kwargs)
    d1, d2 = sha256_text(stable_json(out1)), sha256_text(stable_json(out2))
    if d1 != d2:
        raise ValidationError(
            f"nondeterministic output from {getattr(fn, '__name__', fn)!r}")
    return d1
