"""Experiments package (Phase r1 §2 restructure).

Search-space, optimizer, gates, and the production-binding adapter seam
(former ``core/optimization/``). Production runners bind per-sport
adapters through ``sports/<sport>/optimization.py``; this package never
imports ``sports.*`` (import direction, §18).
"""

from core.experiments.search_space import (  # noqa: F401
    DEFAULT_MAX_SECONDS,
    DEFAULT_SEED,
    DEFAULT_SWEEP_CAP,
    MAX_CANDIDATE_FEATURES,
    MAX_HORIZON_CANDIDATES,
    MAX_INTERACTION_ORDER,
    MAX_MODEL_CANDIDATES,
    MIN_HORIZON_CANDIDATES,
    MIN_MODEL_CANDIDATES,
    Bounds,
    CandidateSpec,
    ModelScope,
    OptimizationConfig,
    build_candidates,
    candidate_catalog_fingerprint,
    derive_column,
    horizon_candidates,
    interaction_candidates,
    load_optimization_config,
    model_candidates,
    spec_from_name,
    split_side,
)
from core.experiments.optimizer import (  # noqa: F401
    GATE_NAMES,
    GateResult,
    ScopeRunner,
    evaluate_gates,
    recommend,
    run_optimization,
)
from core.experiments.sports import (  # noqa: F401
    MLB_RAW_FIELDS,
    MLB_RUN_SIDE_FIELDS,
    MLB_SIDE_FIELDS,
    NBA_RAW_FIELDS,
    NFL_RAW_FIELDS,
    NHL_RAW_FIELDS,
    SPORT_RAW_FIELDS,
    default_scopes,
    mlb_scopes,
    model_scope,
)

# Re-exported for the harness's public surface (the former
# core.optimization package init re-exported these evaluation and
# leakage helpers; tests and callers bind them from here).
from core.evaluation.probabilistic_metrics import (  # noqa: F401
    period_stability,
    pooled_metrics,
    regression_metrics,
    regression_period_stability,
)
from core.validation.leakage import (  # noqa: F401
    leakage_audit,
    pit_safe_matrix,
    slate_available_columns,
)

__all__ = [
    "DEFAULT_MAX_SECONDS", "DEFAULT_SEED", "DEFAULT_SWEEP_CAP",
    "MAX_CANDIDATE_FEATURES", "MAX_HORIZON_CANDIDATES",
    "MAX_INTERACTION_ORDER", "MAX_MODEL_CANDIDATES",
    "MIN_HORIZON_CANDIDATES", "MIN_MODEL_CANDIDATES",
    "Bounds", "CandidateSpec", "ModelScope", "OptimizationConfig",
    "build_candidates", "candidate_catalog_fingerprint", "derive_column",
    "horizon_candidates", "interaction_candidates",
    "load_optimization_config", "model_candidates", "spec_from_name",
    "split_side",
    "GATE_NAMES", "GateResult", "ScopeRunner", "evaluate_gates",
    "recommend", "run_optimization",
    "period_stability", "pooled_metrics", "regression_metrics",
    "regression_period_stability", "leakage_audit", "pit_safe_matrix",
    "slate_available_columns",
    "MLB_RAW_FIELDS", "MLB_RUN_SIDE_FIELDS", "MLB_SIDE_FIELDS",
    "NBA_RAW_FIELDS", "NFL_RAW_FIELDS", "NHL_RAW_FIELDS",
    "SPORT_RAW_FIELDS", "default_scopes", "mlb_scopes", "model_scope",
]
