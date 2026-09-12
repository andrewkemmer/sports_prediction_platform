"""Shared validation package (B-001, Phase 7.5d WS4).

Converts the former ``core/validation.py`` module into a package:
``core.validation.__init__`` re-exports the complete existing public
surface verbatim (import compatibility: zero call-site edits), and
``core.validation.future_availability`` hosts the §7.1 point-in-time
validators.
"""

from core.validation._legacy_module import (  # noqa: F401
    ValidationError,
    assert_deterministic,
    require_columns,
    round_float,
    sha256_bytes,
    sha256_text,
    stable_json,
    stable_value,
    validate_metric,
    validate_probability,
    validate_probability_pair,
)

__all__ = [
    "ValidationError",
    "assert_deterministic",
    "require_columns",
    "round_float",
    "sha256_bytes",
    "sha256_text",
    "stable_json",
    "stable_value",
    "validate_metric",
    "validate_probability",
    "validate_probability_pair",
]
