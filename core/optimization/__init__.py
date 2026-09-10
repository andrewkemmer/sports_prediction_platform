"""Shared dynamic-optimization harness (Phase 7).

Pure, deterministic, PIT-safe candidate generation + evaluation + gate
logic for per-sport/model feature-set optimization. No artifact writes,
no production mutation, no automatic promotion — the harness only
RECOMMENDS; promotion stays a human decision.
"""

from core.optimization.candidates import (
    CandidateSpec,
    build_candidates,
    candidate_catalog_fingerprint,
    horizon_candidates,
    model_candidates,
    spec_from_name,
)
from core.optimization.gates import (
    GateResult,
    evaluate_gates,
    recommend,
)
from core.optimization.metrics import (
    period_stability,
    pooled_metrics,
    regression_metrics,
    regression_period_stability,
)
from core.optimization.models import (
    Bounds,
    OptimizationConfig,
    load_optimization_config,
)
from core.optimization.pit import (
    leakage_audit,
    pit_safe_matrix,
    slate_available_columns,
)
from core.optimization.sports import (
    SPORT_RAW_FIELDS,
    default_scopes,
    model_scope,
)
from core.optimization.runner import run_optimization

__all__ = [
    "Bounds",
    "CandidateSpec",
    "GateResult",
    "OptimizationConfig",
    "SPORT_RAW_FIELDS",
    "build_candidates",
    "candidate_catalog_fingerprint",
    "evaluate_gates",
    "horizon_candidates",
    "leakage_audit",
    "load_optimization_config",
    "model_candidates",
    "model_scope",
    "default_scopes",
    "pooled_metrics",
    "period_stability",
    "pit_safe_matrix",
    "recommend",
    "regression_metrics",
    "regression_period_stability",
    "run_optimization",
    "slate_available_columns",
    "spec_from_name",
]
