"""Shared platform configuration loading.

Loads platform configuration from a JSON file (or dict) with deterministic
defaults for every Phase 1 subsystem. Nothing here touches the network; the
filesystem is only read when ``load_config`` is called with a path. Every
field has an explicit default so tests can construct configs
programmatically.

Design decisions carried over from the reference repo audit:

* The frontend artifact sink is ``sports/<sport>/data_delivery/`` (never the
  legacy ``<sport>-backend/`` naming).
* Undated artifacts are forbidden going forward; every frontend artifact is
  date-stamped ``<family>_<YYYYMMDD>.<ext>`` and subject to the 20-day
  rolling retention policy (spec §23; B-003, Phase 7.5d).
* Market schemas stay sport-specific; the shared layer only defines semantic
  interfaces (see ``core.contracts``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SPORTS = ("mlb", "nfl", "nhl", "nba")

#: Number of days a frontend/dashboard artifact is retained in the sink
#: before deletion. Raw payloads, Parquet/DuckDB stores, feature/training
#: stores, model contracts, study configs, and production code are exempt.
FRONTEND_ARTIFACT_RETENTION_DAYS = 20

#: Minimum decided games per team before any model-derived display is
#: trusted on the dashboard (guarded by warmup rules, not fabricated).
DEFAULT_MIN_GAMES_PER_TEAM = 10

#: The agreed initial MLB warmup default (Phase 2): 30 completed games per
#: team before any model-derived display is trusted for MLB.
MLB_MIN_GAMES_PER_TEAM = 30

#: Days of prior history a sport requires before its first walk-forward
#: prediction is emitted.
DEFAULT_WARMUP_DAYS = 30

#: Default UTC hour a daily pipeline run is anchored to (Kaggle schedule).
DEFAULT_RUN_UTC_HOUR = 9

#: Default delivery allowlist: the frontend artifact families GitPython may
#: push to GitHub. Experiment/optimization runs never push regardless.
#: B-004 (Phase 7.5d): expanded to the EXACT emitted dated families for all
#: four sports (runner-emission <-> allowlist parity test enforces both
#: directions). The stale alias families nfl_moneyline_json / nfl_feature_json
#: / nfl_markets / nfl_markets_monitor were REMOVED: nothing emits them and a
#: sink scan found zero historical files matching them (never make historical
#: files undeletable — the C2 auto-derivation from ArtifactContract is the
#: future hardening, targeted Phase 7.6).
DEFAULT_ALLOWLISTED_FAMILIES = (
    # --- MLB (unprefixed) ---
    "todays_games",
    "power_rankings",
    "calibration",
    "model_monitor",
    "markets",
    "markets_monitor",
    "shap_game",
    "predictions_history",
    "feature_drift",
    "feature_coverage",
    "rolling_brier",
    "model_history",
    "model_version_history",
    "features_metadata",
    "run_engine_markets",
    "run_engine_monitor",
    # --- NFL (sport-prefixed, exact emitted names) ---
    "nfl_moneyline_v1",
    "nfl_calibration",
    "nfl_predictions_history",
    "nfl_power_rankings",
    "nfl_run_engine_markets",
    "nfl_run_engine_monitor",
    "nfl_qb_matchup",
    "nfl_feature_v1",
    "nfl_model_monitor",
    "nfl_shap_game",
    "nfl_feature_drift",
    "nfl_feature_coverage",
    "nfl_rolling_brier",
    "nfl_features_metadata",
    # --- NHL (sport-prefixed, exact emitted names) ---
    "nhl_moneyline_v1",
    "nhl_calibration",
    "nhl_predictions_history",
    "nhl_power_rankings",
    "nhl_run_engine_markets",
    "nhl_run_engine_monitor",
    "nhl_goalie_matchup",
    "nhl_feature_v1",
    "nhl_model_monitor",
    "nhl_shap_game",
    "nhl_rolling_brier",
    "nhl_features_metadata",
    # --- NBA (sport-prefixed, exact emitted names) ---
    "nba_moneyline_v1",
    "nba_calibration",
    "nba_predictions_history",
    "nba_power_rankings",
    "nba_player_matchup",
    "nba_feature_v1",
    "nba_model_monitor",
    "nba_run_engine_markets",
    "nba_run_engine_monitor",
    "nba_shap_game",
    "nba_rolling_brier",
    "nba_features_metadata",
)


@dataclass(frozen=True)
class PredictionWindowConfig:
    """The prediction window a daily run serves.

    ``lookahead_days`` is how far ahead the model may publish predictions
    (1 = tomorrow only). ``slate_cutoff_hour_local`` is the local-time hour
    after which a game no longer belongs to today's slate.
    """

    lookahead_days: int = 1
    slate_cutoff_hour_local: int = 23

    def __post_init__(self) -> None:
        if not 0 <= self.lookahead_days <= 7:
            raise ValueError("lookahead_days must be within 0..7")
        if not 0 <= self.slate_cutoff_hour_local <= 23:
            raise ValueError("slate_cutoff_hour_local must be within 0..23")


@dataclass(frozen=True)
class WarmupConfig:
    """Warmup rules gating when a sport may publish predictions."""

    min_history_days: int = DEFAULT_WARMUP_DAYS
    min_games_per_team: int = DEFAULT_MIN_GAMES_PER_TEAM

    def __post_init__(self) -> None:
        if self.min_history_days < 0:
            raise ValueError("min_history_days must be >= 0")
        if self.min_games_per_team < 1:
            raise ValueError("min_games_per_team must be >= 1")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward fold definitions shared by all sports.

    ``min_train_games`` is the smallest training set a fold may use (the
    reference repo's ``min fold guard``). ``step_days`` is the daily fold
    stride. ``embargo_days`` keeps games after the fold's train cutoff out
    of training to avoid leakage across the boundary.
    """

    min_train_games: int = 200
    step_days: int = 1
    embargo_days: int = 0

    def __post_init__(self) -> None:
        if self.min_train_games < 1:
            raise ValueError("min_train_games must be >= 1")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")


@dataclass(frozen=True)
class RetentionConfig:
    """Frontend-artifact retention policy (20-day rolling deletion)."""

    frontend_days: int = FRONTEND_ARTIFACT_RETENTION_DAYS

    def __post_init__(self) -> None:
        if self.frontend_days < 1:
            raise ValueError("frontend_days must be >= 1")


@dataclass(frozen=True)
class DeliveryConfig:
    """Kaggle -> GitPython artifact delivery rules.

    Only allowlisted frontend artifact families may be pushed to GitHub;
    experiment and optimization runs never push artifacts.
    """

    owner: str = ""
    repo: str = ""
    branch: str = "main"
    allowlisted_families: tuple[str, ...] = DEFAULT_ALLOWLISTED_FAMILIES

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}".strip("/")


@dataclass(frozen=True)
class SportConfig:
    """Per-sport configuration block."""

    key: str
    label: str
    emoji: str
    participant_role: str  # pitcher | qb | goalie | top_scorer
    enabled: bool = True
    has_run_engine: bool = False
    prediction_window: PredictionWindowConfig = field(
        default_factory=PredictionWindowConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    def __post_init__(self) -> None:
        if self.key not in SPORTS:
            raise ValueError(f"unknown sport key: {self.key!r}")
        if self.participant_role not in ("pitcher", "qb", "goalie",
                                         "top_scorer"):
            raise ValueError(
                f"unknown participant_role: {self.participant_role!r}")


@dataclass(frozen=True)
class PlatformConfig:
    """Top-level platform configuration."""

    sports: dict[str, SportConfig]
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    run_utc_hour: int = DEFAULT_RUN_UTC_HOUR

    def __post_init__(self) -> None:
        if not self.sports:
            raise ValueError("platform config must define at least one sport")
        for key, sport in self.sports.items():
            if key != sport.key:
                raise ValueError(
                    f"sport registry key {key!r} != sport.key {sport.key!r}")
        if not 0 <= self.run_utc_hour <= 23:
            raise ValueError("run_utc_hour must be within 0..23")

    def sport(self, key: str) -> SportConfig:
        """Resolve a sport key, falling back to MLB (never raises)."""
        norm = normalize_sport_key(key)
        return self.sports.get(norm, self.sports[DEFAULT_SPORT])

    def data_delivery_dir(self, sport_key: str, root: Path) -> Path:
        """The artifact sink for a sport: ``<root>/sports/<sport>/data_delivery``."""
        return root / "sports" / normalize_sport_key(sport_key) / "data_delivery"


DEFAULT_SPORT = "mlb"


def normalize_sport_key(sport_key: str | None) -> str:
    """Lowercase/strip a sport key; unknown or empty values fall back to MLB."""
    key = str(sport_key or "").strip().lower()
    if key not in SPORTS:
        return DEFAULT_SPORT
    return key


def is_unknown_sport(sport_key: str | None) -> bool:
    """True only for a genuinely unknown NON-EMPTY sport value."""
    key = str(sport_key or "").strip().lower()
    if not key or key == "none":
        return False
    return key not in SPORTS


def default_sport_configs() -> dict[str, SportConfig]:
    """The default per-sport registry (participant role per Phase 0 decisions)."""
    return {
        "mlb": SportConfig(key="mlb", label="MLB", emoji="\u26be",
                           participant_role="pitcher", has_run_engine=True,
                           warmup=WarmupConfig(
                               min_history_days=DEFAULT_WARMUP_DAYS,
                               min_games_per_team=MLB_MIN_GAMES_PER_TEAM)),
        "nfl": SportConfig(key="nfl", label="NFL", emoji="\U0001f3c8",
                           participant_role="qb", has_run_engine=True),
        "nhl": SportConfig(key="nhl", label="NHL", emoji="\U0001f3c8",
                           participant_role="goalie", enabled=False),
        "nba": SportConfig(key="nba", label="NBA", emoji="\U0001f3c0",
                           participant_role="top_scorer", enabled=False),
    }


def default_config() -> PlatformConfig:
    """The deterministic default platform configuration."""
    return PlatformConfig(sports=default_sport_configs())


def load_config(path: str | Path | None = None,
                raw: dict | None = None) -> PlatformConfig:
    """Load platform configuration from a JSON file, a dict, or defaults.

    Unknown keys are ignored; known keys are validated. Raises
    ``ValueError`` (or ``OSError`` for unreadable files) on bad input —
    never silently degrades to defaults when an explicit config was given.
    """
    if path is not None and raw is not None:
        raise ValueError("pass either path or raw, not both")
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    elif raw is not None:
        data = dict(raw)
    else:
        return default_config()

    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object")

    sports_raw = data.get("sports")
    sports = default_sport_configs()
    if sports_raw is not None:
        if not isinstance(sports_raw, dict):
            raise ValueError("'sports' must be an object")
        for key, block in sports_raw.items():
            if not isinstance(block, dict):
                raise ValueError(f"sport {key!r} must be an object")
            base = sports.get(key)
            if base is None:
                raise ValueError(f"unknown sport key: {key!r}")
            window = PredictionWindowConfig(
                lookahead_days=block.get(
                    "lookahead_days",
                    base.prediction_window.lookahead_days),
                slate_cutoff_hour_local=block.get(
                    "slate_cutoff_hour_local",
                    base.prediction_window.slate_cutoff_hour_local),
            )
            warmup = WarmupConfig(
                min_history_days=block.get(
                    "min_history_days", base.warmup.min_history_days),
                min_games_per_team=block.get(
                    "min_games_per_team", base.warmup.min_games_per_team),
            )
            wf = WalkForwardConfig(
                min_train_games=block.get(
                    "min_train_games", base.walk_forward.min_train_games),
                step_days=block.get("step_days", base.walk_forward.step_days),
                embargo_days=block.get(
                    "embargo_days", base.walk_forward.embargo_days),
            )
            sports[key] = SportConfig(
                key=key,
                label=str(block.get("label", base.label)),
                emoji=str(block.get("emoji", base.emoji)),
                participant_role=str(block.get(
                    "participant_role", base.participant_role)),
                enabled=bool(block.get("enabled", base.enabled)),
                has_run_engine=bool(
                    block.get("has_run_engine", base.has_run_engine)),
                prediction_window=window,
                warmup=warmup,
                walk_forward=wf,
            )

    retention_raw = data.get("retention") or {}
    retention = RetentionConfig(
        frontend_days=int(retention_raw.get(
            "frontend_days", FRONTEND_ARTIFACT_RETENTION_DAYS)))

    delivery_raw = data.get("delivery") or {}
    delivery = DeliveryConfig(
        owner=str(delivery_raw.get("owner", "")),
        repo=str(delivery_raw.get("repo", "")),
        branch=str(delivery_raw.get("branch", "main")),
        allowlisted_families=tuple(delivery_raw.get(
            "allowlisted_families", DEFAULT_ALLOWLISTED_FAMILIES)),
    )

    return PlatformConfig(
        sports=sports,
        retention=retention,
        delivery=delivery,
        run_utc_hour=int(data.get("run_utc_hour", DEFAULT_RUN_UTC_HOUR)),
    )
