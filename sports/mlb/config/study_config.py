"""MLB study configuration loader.

The study YAML is the single source of truth for warmup, fold geometry, OOF
rules, and minimum training rows. Production run-window environment variables
(``MLB_START_DATE`` / ``MLB_END_DATE`` / ``MLB_FULL_REPULL``) scope only
ingestion/serving — never training history. This module parses the YAML into
typed dataclasses and fails loudly on unknown/invalid fields (no silent
defaults drift).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from core.config import WalkForwardConfig
from core.study import is_valid_compact_date
from core.config.platform import WarmupConfig

#: Phase 7.5b canonical contract ownership: the feature contracts are
#: DECLARED in ``sports.mlb.features.registry`` and imported here — the
#: study config binds them onto the study dataclass but never declares
#: feature lists directly.
from sports.mlb.features.registry import (  # noqa: E402
    MARKET_CONTRACT_VERSION,
    MARKET_FEATURE_COLS,
    MONEYLINE_CONTRACT_VERSION,
    MONEYLINE_FEATURE_COLS,
    PRIOR_MARKET_CONTRACT_VERSION,
    PRIOR_MONEYLINE_CONTRACT_VERSION,
)


class MLBStudyError(ValueError):
    """Raised when the MLB study configuration is missing or invalid."""


STUDY_PATH = Path(__file__).resolve().parent / "study.yaml"

_ALLOWED_TOP = {
    "sport", "kind", "study_id", "data_start_date", "warmup",
    "walk_forward", "oof", "training", "retention",
    "allowlisted_families", "retention_exempt_families",
    "prediction_cutoff_buffer_minutes", "availability_enforcement",
}


@dataclass(frozen=True)
class OOFRules:
    """Fixed OOF rules (Phase 2: frozen, no dynamic optimization)."""

    require_prob_strictly_inside: bool = True
    undecided_rows_excluded: bool = True
    require_unique_game_id: bool = True
    require_decided_only: bool = True
    min_training_rows: int = 200

    def __post_init__(self) -> None:
        if self.min_training_rows < 1:
            raise MLBStudyError("oof.min_training_rows must be >= 1")


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen training geometry (Phase 2: no optimization)."""

    ensemble_members: tuple[str, ...] = (
        "xgboost", "lightgbm", "logistic", "randomforest", "mlp")
    retrain_cadence_days: int = 7
    min_train_days_warmup: int = 30

    def __post_init__(self) -> None:
        if self.retrain_cadence_days < 1:
            raise MLBStudyError("training.retrain_cadence_days must be >= 1")
        if self.min_train_days_warmup < 0:
            raise MLBStudyError("training.min_train_days_warmup must be >= 0")
        if not self.ensemble_members:
            raise MLBStudyError("training.ensemble_members must be non-empty")


@dataclass(frozen=True)
class MLBStudy:
    """The parsed MLB study configuration."""

    study_id: str
    kind: str
    sport: str
    data_start_date: str
    warmup: WarmupConfig
    walk_forward: WalkForwardConfig
    min_val_games: int
    max_eval_folds: int
    oof: OOFRules
    training: TrainingConfig
    frontend_days: int
    allowlisted_families: tuple[str, ...]
    retention_exempt_families: tuple[str, ...]
    prediction_cutoff_buffer_minutes: int
    availability_enforcement: str
    moneyline_feature_cols: tuple[str, ...] = MONEYLINE_FEATURE_COLS
    moneyline_contract_version: str = MONEYLINE_CONTRACT_VERSION
    market_feature_cols: tuple[str, ...] = MARKET_FEATURE_COLS
    market_contract_version: str = MARKET_CONTRACT_VERSION
    source_path: Path | None = None

    @property
    def warmup_start_date(self) -> str | None:
        """The first date the warmup contract can be satisfied, given the
        configured data start and min history days (None when unconstrained)."""
        from core.study import shift_compact_date
        if not self.data_start_date:
            return None
        return shift_compact_date(
            self.data_start_date, self.warmup.min_history_days)


def load_mlb_study(path: str | Path | None = None) -> MLBStudy:
    """Load and validate the MLB study configuration.

    Fails loudly (``MLBStudyError``) on a missing file, unknown top-level
    fields, missing required sections, or invalid values — never on silent
    defaults.
    """
    p = Path(path) if path is not None else STUDY_PATH
    if not p.is_file():
        raise MLBStudyError(f"MLB study config not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MLBStudyError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise MLBStudyError(f"study config {p} must be a mapping")

    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise MLBStudyError(f"unknown study fields: {sorted(unknown)}")

    for required in ("sport", "kind", "study_id", "data_start_date",
                     "warmup", "walk_forward", "oof"):
        if required not in data:
            raise MLBStudyError(f"study config missing required field: {required}")

    if str(data["sport"]) != "mlb":
        raise MLBStudyError(
            f"study config is for sport {data['sport']!r}, not mlb")
    if str(data["kind"]) != "production":
        raise MLBStudyError(
            f"study kind must be 'production', got {data['kind']!r}")

    start = str(data["data_start_date"])
    if not is_valid_compact_date(start):
        raise MLBStudyError(
            f"data_start_date must be compact YYYYMMDD, got {start!r}")

    w = data["warmup"]
    try:
        warmup = WarmupConfig(
            min_history_days=int(w["min_history_days"]),
            min_games_per_team=int(w["min_games_per_team"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MLBStudyError(f"invalid warmup block: {exc}") from exc
    if warmup.min_games_per_team != 30:
        # The agreed initial MLB default is frozen in Phase 2; changing it
        # requires a reviewed study commit (no silent drift).
        raise MLBStudyError(
            "warmup.min_games_per_team must be 30 in Phase 2 "
            f"(got {warmup.min_games_per_team}); change requires review")

    wf_raw = data["walk_forward"]
    try:
        walk_forward = WalkForwardConfig(
            min_train_games=int(wf_raw["min_train_games"]),
            step_days=int(wf_raw["step_days"]),
            embargo_days=int(wf_raw.get("embargo_days", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MLBStudyError(f"invalid walk_forward block: {exc}") from exc
    min_val_games = int(wf_raw.get("min_val_games", 40))
    max_eval_folds = int(wf_raw.get("max_eval_folds", 0))
    if min_val_games < 1:
        raise MLBStudyError("walk_forward.min_val_games must be >= 1")
    if max_eval_folds < 0:
        raise MLBStudyError("walk_forward.max_eval_folds must be >= 0")

    o = data["oof"]
    try:
        oof = OOFRules(
            require_prob_strictly_inside=bool(o.get(
                "require_prob_strictly_inside", True)),
            undecided_rows_excluded=bool(o.get(
                "undecided_rows_excluded", True)),
            require_unique_game_id=bool(o.get("require_unique_game_id", True)),
            require_decided_only=bool(o.get("require_decided_only", True)),
            min_training_rows=int(o["min_training_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MLBStudyError(f"invalid oof block: {exc}") from exc

    t_raw = data.get("training") or {}
    training = TrainingConfig(
        ensemble_members=tuple(t_raw.get(
            "ensemble_members", TrainingConfig.ensemble_members)),
        retrain_cadence_days=int(t_raw.get(
            "retrain_cadence_days", TrainingConfig.retrain_cadence_days)),
        min_train_days_warmup=int(t_raw.get(
            "min_train_days_warmup", TrainingConfig.min_train_days_warmup)),
    )

    r_raw = data.get("retention") or {}
    frontend_days = int(r_raw.get("frontend_days", 20))
    if frontend_days != 20:
        raise MLBStudyError(
            f"retention.frontend_days must be 20 (got {frontend_days})")

    # §7.1 PIT gate (B-001 WS4): buffer hard-pinned, mode validated.
    buffer = int(data.get("prediction_cutoff_buffer_minutes", 60))
    if buffer != 60:
        raise MLBStudyError(
            f"prediction_cutoff_buffer_minutes must be 60 per §7.1 (MLB), "
            f"got {buffer}")
    from core.validation.future_availability import (
        normalize_enforcement_mode)
    availability_enforcement = normalize_enforcement_mode(
        data.get("availability_enforcement", "report_only"), "mlb study")

    allow = tuple(data.get("allowlisted_families") or ())
    if not allow:
        raise MLBStudyError("allowlisted_families must be non-empty")
    exempt = tuple(data.get("retention_exempt_families") or ())
    overlap = set(allow) & set(exempt)
    if overlap:
        raise MLBStudyError(
            f"families cannot be both allowlisted and retention-exempt: "
            f"{sorted(overlap)}")

    return MLBStudy(
        study_id=str(data["study_id"]),
        kind=str(data["kind"]),
        sport=str(data["sport"]),
        data_start_date=start,
        warmup=warmup,
        walk_forward=walk_forward,
        min_val_games=min_val_games,
        max_eval_folds=max_eval_folds,
        oof=oof,
        training=training,
        frontend_days=frontend_days,
        allowlisted_families=allow,
        retention_exempt_families=exempt,
        prediction_cutoff_buffer_minutes=buffer,
        availability_enforcement=availability_enforcement,
        moneyline_feature_cols=MONEYLINE_FEATURE_COLS,
        moneyline_contract_version=MONEYLINE_CONTRACT_VERSION,
        market_feature_cols=MARKET_FEATURE_COLS,
        market_contract_version=MARKET_CONTRACT_VERSION,
        source_path=p,
    )
