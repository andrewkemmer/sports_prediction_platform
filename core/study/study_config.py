"""Study configuration model.

A *study* is one declared, reproducible experiment or optimization run:
the sport, the date range, the fold plan, the model family, and the
artifacts it is allowed to write. Studies are the unit the delivery
allowlist gates on — **experiment and optimization runs never push
production artifacts** (Phase 0 decision 6); only the daily production run
does.

Study configs are serialized to JSON so Kaggle runs can be replayed
deterministically from a committed spec.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config.platform import SPORTS, WalkForwardConfig
from core.study.date_resolution import is_valid_compact_date

STUDY_KINDS = ("experiment", "optimization", "production", "diagnostic")


class StudyConfigError(ValueError):
    """Raised when a study configuration is invalid."""


@dataclass(frozen=True)
class StudyConfig:
    """A declared, replayable study specification."""

    study_id: str
    kind: str  # experiment | optimization | production | diagnostic
    sport: str
    date_start: str  # compact YYYYMMDD
    date_end: str  # compact YYYYMMDD (inclusive)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    model_family: str = "ensemble"
    notes: str = ""
    tags: tuple[str, ...] = ()
    may_push_artifacts: bool = False

    def __post_init__(self) -> None:
        if not self.study_id or not isinstance(self.study_id, str):
            raise StudyConfigError("study_id must be a non-empty string")
        if self.kind not in STUDY_KINDS:
            raise StudyConfigError(
                f"kind must be one of {STUDY_KINDS}, got {self.kind!r}")
        if self.sport not in SPORTS:
            raise StudyConfigError(f"unknown sport: {self.sport!r}")
        if not is_valid_compact_date(self.date_start):
            raise StudyConfigError(
                f"invalid date_start: {self.date_start!r}")
        if not is_valid_compact_date(self.date_end):
            raise StudyConfigError(f"invalid date_end: {self.date_end!r}")
        if self.date_end < self.date_start:
            raise StudyConfigError("date_end precedes date_start")

    @property
    def is_production(self) -> bool:
        """Only production studies may ever push frontend artifacts."""
        return self.kind == "production"

    def validate_push_allowed(self) -> None:
        """Gate for the delivery layer: raise when this study may not push."""
        if not self.is_production or not self.may_push_artifacts:
            raise StudyConfigError(
                f"study {self.study_id!r} ({self.kind}) may not push "
                "artifacts — experiment and optimization runs never push "
                "production artifacts")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["walk_forward"] = asdict(self.walk_forward)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "StudyConfig":
        if not isinstance(data, dict):
            raise StudyConfigError("study config must be a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise StudyConfigError(f"unknown study fields: {sorted(unknown)}")
        wf_raw = data.get("walk_forward") or {}
        try:
            wf = WalkForwardConfig(
                min_train_games=int(wf_raw.get("min_train_games", 200)),
                step_days=int(wf_raw.get("step_days", 1)),
                embargo_days=int(wf_raw.get("embargo_days", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise StudyConfigError(f"invalid walk_forward: {exc}") from exc
        try:
            return cls(
                study_id=str(data["study_id"]),
                kind=str(data["kind"]),
                sport=str(data["sport"]),
                date_start=str(data["date_start"]),
                date_end=str(data["date_end"]),
                walk_forward=wf,
                model_family=str(data.get("model_family", "ensemble")),
                notes=str(data.get("notes", "")),
                tags=tuple(data.get("tags", ())),
                may_push_artifacts=bool(data.get("may_push_artifacts", False)),
            )
        except KeyError as exc:
            raise StudyConfigError(f"missing study field: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> "StudyConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise StudyConfigError(f"invalid study JSON: {exc}") from exc


def load_study(path: str | Path) -> StudyConfig:
    """Load a study config from a JSON file."""
    return StudyConfig.from_json(Path(path).read_text(encoding="utf-8"))
