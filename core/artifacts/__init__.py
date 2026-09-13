"""Artifacts package (Phase r1 §2 restructure).

* ``retention`` — the 20-day frontend-artifact retention policy (former
                  ``core/retention.py``).
"""

from core.artifacts.retention import (  # noqa: F401
    RetentionError,
    RetentionPlan,
    _family_name,
    apply_retention,
    plan_retention,
    retention_cutoff,
    retention_exempt_families,
    run_production_retention,
)

__all__ = [
    "RetentionError", "RetentionPlan", "apply_retention", "plan_retention",
    "retention_cutoff", "retention_exempt_families",
    "run_production_retention",
]
