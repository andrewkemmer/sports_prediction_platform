"""Determinism helpers (former core/validation.py, §2 split).

Deterministic serialization (stable JSON with sorted keys, fixed
rounding) so replays produce byte-identical artifacts, SHA-256
fingerprinting, and a replay-check helper that runs a callable twice and
compares outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from core.validation.schema import ValidationError


def round_float(value: float, digits: int = 6) -> float:
    """Deterministic rounding used in every serialized metric."""
    from core.validation.schema import validate_metric

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
