"""Shared validation package (§2 layout; legacy bridge removed in r1).

Submodules:
* ``schema``             — probability/metric/column validation guards
* ``determinism``        — stable serialization, fingerprints, replay checks
* ``leakage``            — optimization-candidate PIT/leakage audit
* ``future_availability`` — §7.1 per-field point-in-time validators

``core.validation.__init__`` re-exports the complete historical public
surface verbatim (import compatibility: zero call-site edits).
"""

from core.validation.schema import (  # noqa: F401
    ValidationError,
    require_columns,
    validate_metric,
    validate_probability,
    validate_probability_pair,
)
from core.validation.determinism import (  # noqa: F401
    assert_deterministic,
    round_float,
    sha256_bytes,
    sha256_text,
    stable_json,
    stable_value,
)

__all__ = [
    "ValidationError",
    "require_columns",
    "validate_metric",
    "validate_probability",
    "validate_probability_pair",
    "assert_deterministic",
    "round_float",
    "sha256_bytes",
    "sha256_text",
    "stable_json",
    "stable_value",
]
