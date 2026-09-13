"""Contracts package (Phase r1 §2 restructure).

* ``contracts``         — FeatureContract/FeatureSpec, ArtifactContract,
                          SportAdapter protocol (former ``core/contracts.py``).
* ``artifact_schema``   — the 56-entry record-type registry and
                          ``validate_record`` pre-write gate (former
                          ``core/record_validation.py``).
* ``settlement_rules``  — market settlement, push/tie probability
                          normalization (former ``core/markets.py``).
"""

from core.contracts.contracts import (  # noqa: F401
    FEATURE_SPEC_FIELDS,
    ArtifactContract,
    ArtifactFamily,
    FeatureContract,
    FeatureContractError,
    FeatureSpec,
    FeaturesMetadataError,
    ParticipantBox,
    SportAdapter,
    default_artifact_contract,
    load_artifact_contract,
    require_features_metadata,
    validate_columns_have_metadata,
    validate_metadata_is_reachable,
)
from core.contracts.artifact_schema import (  # noqa: F401
    EXPECTED_REGISTRY_SIZE,
    REGISTRY,
    RecordType,
    RecordValidationError,
    validate_record,
)
from core.contracts.settlement_rules import (  # noqa: F401
    FULL_GAME_NO_TIE,
    FULL_GAME_TIE_ALLOWED,
    Outcome,
    ProbTriple,
    REGULATION_TIE_ALLOWED,
    Settlement,
    SettlementConfig,
    SettlementError,
    TieTriple,
    normalize_push_probs,
    normalize_tie_probs,
    push_display_bucket,
    settle_moneyline,
    settle_spread,
    settle_total,
    sport_allows_ties,
    triple_from_line_grid,
)

__all__ = [
    "FEATURE_SPEC_FIELDS", "ArtifactContract", "ArtifactFamily",
    "FeatureContract", "FeatureContractError", "FeatureSpec",
    "FeaturesMetadataError", "ParticipantBox", "SportAdapter",
    "default_artifact_contract", "load_artifact_contract",
    "require_features_metadata", "validate_columns_have_metadata",
    "validate_metadata_is_reachable",
    "EXPECTED_REGISTRY_SIZE", "REGISTRY", "RecordType",
    "RecordValidationError", "validate_record",
    "FULL_GAME_NO_TIE", "FULL_GAME_TIE_ALLOWED", "REGULATION_TIE_ALLOWED",
    "Outcome", "ProbTriple", "Settlement", "SettlementConfig",
    "SettlementError", "TieTriple", "normalize_push_probs",
    "normalize_tie_probs", "push_display_bucket", "settle_moneyline",
    "settle_spread", "settle_total", "sport_allows_ties",
    "triple_from_line_grid",
]
