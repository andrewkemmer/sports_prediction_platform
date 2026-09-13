"""Platform configuration package (Phase r1 §2 restructure).

Contents:
* ``platform``      — platform/sport/delivery/retention configuration
                      (former ``core/config.py``).
* ``run_window``    — mandatory START_DATE/END_DATE/FULL_REPULL run-window
                      resolution (former ``core/runconfig.py``).
* ``prediction_window`` — first/last prediction game-date window semantics
                      (former ``core/prediction_window.py``).
* ``sources``       — per-sport market settlement rules from
                      ``market_rules.yaml`` (r4).
* ``source_policy`` — per-sport data-source/schema/transport policy from
                      ``data_sources.yaml`` (r4).

Public surface is re-exported verbatim; every legacy ``core.config`` /
``core.runconfig`` / ``core.prediction_window`` import path keeps working
through the Phase-r1 compatibility façades (see ``core/config.py``).
"""

from core.config.platform import (  # noqa: F401
    DEFAULT_SPORT,
    DEFAULT_ALLOWLISTED_FAMILIES,
    DEFAULT_MIN_GAMES_PER_TEAM,
    DEFAULT_RUN_UTC_HOUR,
    DEFAULT_WARMUP_DAYS,
    FRONTEND_ARTIFACT_RETENTION_DAYS,
    MLB_MIN_GAMES_PER_TEAM,
    SPORTS,
    DeliveryConfig,
    PlatformConfig,
    PredictionWindowConfig,
    RetentionConfig,
    SportConfig,
    WalkForwardConfig,
    WarmupConfig,
    default_config,
    is_unknown_sport,
    load_config,
    normalize_sport_key,
)
from core.config.sources import (  # noqa: F401
    MarketRulesError,
    SportMarketRules,
    load_market_rules,
)
from core.config.source_policy import (  # noqa: F401
    DataSourcesError,
    SourcePolicy,
    SourcePolicyError,
    SportDataSources,
    load_data_sources,
)
from core.config.prediction_window import (  # noqa: F401
    PredictionWindow,
    PredictionWindowError,
    game_in_window,
    partition_slate,
    resolve_prediction_window,
)
from core.config.run_window import (  # noqa: F401
    ENV_END_DATE,
    ENV_FULL_REPULL,
    ENV_START_DATE,
    RunConfigError,
    RunWindow,
    resolve_run_window,
)

__all__ = [
    "DEFAULT_SPORT",
    "DEFAULT_MIN_GAMES_PER_TEAM",
    "DEFAULT_RUN_UTC_HOUR",
    "DEFAULT_WARMUP_DAYS",
    "FRONTEND_ARTIFACT_RETENTION_DAYS",
    "MLB_MIN_GAMES_PER_TEAM",
    "SPORTS",
    "DEFAULT_ALLOWLISTED_FAMILIES",
    "DeliveryConfig",
    "PlatformConfig",
    "PredictionWindowConfig",
    "RetentionConfig",
    "SportConfig",
    "WalkForwardConfig",
    "WarmupConfig",
    "default_config",
    "is_unknown_sport",
    "load_config",
    "normalize_sport_key",
    "PredictionWindow",
    "PredictionWindowError",
    "game_in_window",
    "partition_slate",
    "resolve_prediction_window",
    "ENV_END_DATE",
    "ENV_FULL_REPULL",
    "ENV_START_DATE",
    "RunConfigError",
    "RunWindow",
    "resolve_run_window",
    "MarketRulesError",
    "SportMarketRules",
    "load_market_rules",
    "DataSourcesError",
    "SourcePolicy",
    "SourcePolicyError",
    "SportDataSources",
    "load_data_sources",
]
