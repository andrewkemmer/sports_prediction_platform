"""Core shared infrastructure for the sports prediction platform.

Phase 1 modules: configuration, dates, prediction windows, study configs,
warmup rules, walk-forward folds, OOF rules, contracts, market settlement,
storage, retention, and validation/determinism utilities.
"""

from core.config import (  # noqa: F401
    DEFAULT_SPORT,
    DeliveryConfig,
    PlatformConfig,
    RetentionConfig,
    SportConfig,
    WalkForwardConfig,
    WarmupConfig,
    default_config,
    is_unknown_sport,
    load_config,
    normalize_sport_key,
)
from core.contracts import (  # noqa: F401
    ArtifactContract,
    ArtifactFamily,
    FeatureContract,
    FeatureSpec,
    ParticipantBox,
    default_artifact_contract,
    load_artifact_contract,
)
from core.study import (  # noqa: F401
    DateError,
    artifact_date_from_filename,
    format_compact_date,
    is_evening_start_et,
    iso_to_eastern_display,
    is_valid_compact_date,
    nearest_valid_date,
    parse_compact_date,
    shift_compact_date,
    today_et,
)
from core.folds import Fold, FoldError, build_folds  # noqa: F401
from core.contracts import (  # noqa: F401
    Outcome,
    ProbTriple,
    Settlement,
    SettlementError,
    normalize_push_probs,
    normalize_tie_probs,
    push_display_bucket,
    settle_moneyline,
    settle_spread,
    settle_total,
    sport_allows_ties,
    triple_from_line_grid,
)
from core.folds import (  # noqa: F401
    OOFError,
    OOFValidationReport,
    brier_score,
    log_loss,
    oof_metrics,
    roc_auc,
    validate_oof_rows,
)
from core.config import (  # noqa: F401
    PredictionWindow,
    PredictionWindowError,
    game_in_window,
    partition_slate,
    resolve_prediction_window,
)
from core.artifacts import (  # noqa: F401
    RetentionError,
    RetentionPlan,
    apply_retention,
    plan_retention,
    retention_cutoff,
)
from core.study import StudyConfig, StudyConfigError, load_study  # noqa: F401
from core.validation import (  # noqa: F401
    ValidationError,
    assert_deterministic,
    require_columns,
    stable_json,
    validate_metric,
    validate_probability,
    validate_probability_pair,
)
from core.study import (  # noqa: F401
    WarmupStatus,
    evaluate_warmup,
    teams_ready_for_display,
    warmup_complete_date,
)

__all__ = [
    "DEFAULT_SPORT", "DeliveryConfig", "PlatformConfig", "RetentionConfig",
    "SportConfig", "WalkForwardConfig", "WarmupConfig", "default_config",
    "is_unknown_sport", "load_config", "normalize_sport_key",
    "ArtifactContract", "ArtifactFamily", "FeatureContract", "FeatureSpec",
    "ParticipantBox", "default_artifact_contract", "load_artifact_contract",
    "DateError", "artifact_date_from_filename", "format_compact_date",
    "is_evening_start_et", "iso_to_eastern_display", "is_valid_compact_date",
    "nearest_valid_date", "parse_compact_date", "shift_compact_date",
    "today_et", "Fold", "FoldError", "build_folds",
    "Outcome", "ProbTriple", "Settlement", "SettlementError",
    "normalize_push_probs", "normalize_tie_probs", "push_display_bucket",
    "settle_moneyline", "settle_spread", "settle_total", "sport_allows_ties",
    "triple_from_line_grid",
    "OOFError", "OOFValidationReport", "brier_score", "log_loss",
    "oof_metrics", "roc_auc", "validate_oof_rows",
    "PredictionWindow", "PredictionWindowError", "game_in_window",
    "partition_slate", "resolve_prediction_window",
    "RetentionError", "RetentionPlan", "apply_retention", "plan_retention",
    "retention_cutoff",
    "StudyConfig", "StudyConfigError", "load_study",
    "ValidationError", "assert_deterministic", "require_columns",
    "stable_json", "validate_metric", "validate_probability",
    "validate_probability_pair",
    "WarmupStatus", "evaluate_warmup", "teams_ready_for_display",
    "warmup_complete_date",
]
