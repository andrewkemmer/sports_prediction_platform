"""Frontend-artifact retention utilities.

Implements the 10-day rolling deletion policy over the frontend/dashboard
artifact families in a sport's ``data_delivery`` sink. The policy applies
ONLY to frontend artifacts (Phase 0 decision 7): raw source payloads,
historical Parquet/DuckDB data, feature/training stores, approved model
contracts, study configurations, and production code are never touched.

Deletion is dry-run by default — callers must pass ``execute=True`` after
reviewing the plan. The delivery allowlist (``core.config.DeliveryConfig``)
is a second gate: families not on it are never deleted, even if dated.

Production runners must not call ``apply_retention(execute=True)``
directly; they call :func:`run_production_retention` AFTER successful
artifact generation, passing ``generation_succeeded`` explicitly. On any
generation failure the plan stays a dry run — a failed pipeline never
shrinks the sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config import PlatformConfig, normalize_sport_key
from core.dates import (
    format_compact_date,
    is_valid_compact_date,
    parse_compact_date,
    shift_compact_date,
    today_et,
)


class RetentionError(ValueError):
    """Raised when a retention request is invalid or unsafe."""


@dataclass(frozen=True)
class RetentionPlan:
    """The result of a retention pass (dry-run or executed)."""

    sport: str
    cutoff_date: str
    candidates: tuple[Path, ...]
    deleted: tuple[Path, ...]
    protected: tuple[Path, ...]
    dry_run: bool

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def _artifact_stamp(path: Path) -> str | None:
    """The trailing YYYYMMDD stamp on an artifact filename, if any."""
    stem = path.stem
    parts = stem.split("_")
    for part in reversed(parts):
        if len(part) == 8 and part.isdigit() and is_valid_compact_date(part):
            return part
    return None


def retention_cutoff(frontend_days: int,
                     reference_date: str | None = None) -> str:
    """The compact cutoff date: artifacts older than this are candidates.

    ``reference_date`` defaults to today (ET). The cutoff is
    ``reference - frontend_days``; artifacts strictly older than the cutoff
    are deletable (the cutoff day itself is retained).
    """
    ref = (parse_compact_date(reference_date) if reference_date
           else today_et())
    return shift_compact_date(format_compact_date(ref), -frontend_days)


def plan_retention(
    data_delivery_dir: Path,
    config: PlatformConfig,
    sport: str,
    *,
    reference_date: str | None = None,
) -> RetentionPlan:
    """Build the deletion plan for a sport's frontend artifacts.

    A file is a *candidate* when: it lives directly in the sink (or one
    level deep, e.g. ``shap_game_*`` naming — never nested stores), it
    carries a date stamp, the stamp is older than the cutoff, and its
    family is on the delivery allowlist. Undated files (durable contracts,
    model binaries) and anything under ``store/`` are always protected.
    """
    s = normalize_sport_key(sport)
    sink = Path(data_delivery_dir)
    if not sink.is_dir():
        return RetentionPlan(
            sport=s, cutoff_date=retention_cutoff(
                config.retention.frontend_days, reference_date),
            candidates=(), deleted=(), protected=(), dry_run=True)

    cutoff = retention_cutoff(config.retention.frontend_days, reference_date)
    allowlist = set(config.delivery.allowlisted_families)

    candidates: list[Path] = []
    protected: list[Path] = []
    for p in sorted(sink.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(sink)
        # Never touch nested durable stores or model binaries.
        if any(part in ("store", "models") for part in rel.parts[:-1]):
            protected.append(p)
            continue
        stamp = _artifact_stamp(p)
        if stamp is None:
            protected.append(p)
            continue
        if stamp >= cutoff:
            continue
        # Family = filename up to the first _<date> segment.
        family = _family_name(p.name)
        if family not in allowlist:
            protected.append(p)
            continue
        candidates.append(p)

    return RetentionPlan(
        sport=s,
        cutoff_date=cutoff,
        candidates=tuple(candidates),
        deleted=(),
        protected=tuple(protected),
        dry_run=True,
    )


def _family_name(filename: str) -> str:
    """The artifact family: filename minus date stamp and middle segments.

    ``shap_game_20260907_WSH@SD.csv`` -> ``shap_game`` (the family is the
    leading underscore-delimited segments up to the date stamp).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    # Walk from the end: drop the team-vs segment (SHAP), then any date
    # stamp, keeping at least the first segment as the family.
    while len(parts) > 1 and (
            (len(parts[-1]) == 8 and parts[-1].isdigit())
            or "@" in parts[-1]):
        parts.pop()
    return "_".join(parts)


def apply_retention(
    data_delivery_dir: Path,
    config: PlatformConfig,
    sport: str,
    *,
    reference_date: str | None = None,
    execute: bool = False,
) -> RetentionPlan:
    """Plan (and optionally execute) the 10-day rolling deletion.

    With ``execute=False`` (default) nothing is deleted. With
    ``execute=True`` the plan's candidates are unlinked; failures on
    individual files are recorded by leaving the file in ``protected``
    (deletion never aborts the whole pass).
    """
    plan = plan_retention(data_delivery_dir, config, sport,
                          reference_date=reference_date)
    if not execute:
        return plan

    deleted: list[Path] = []
    still_protected = list(plan.protected)
    remaining: list[Path] = []
    for p in plan.candidates:
        try:
            p.unlink()
            deleted.append(p)
        except OSError:
            remaining.append(p)
    return RetentionPlan(
        sport=plan.sport,
        cutoff_date=plan.cutoff_date,
        candidates=plan.candidates,
        deleted=tuple(deleted),
        protected=tuple(still_protected + remaining),
        dry_run=False,
    )


def run_production_retention(
    data_delivery_dir: Path,
    config: PlatformConfig,
    sport: str,
    *,
    generation_succeeded: bool,
    reference_date: str | None = None,
) -> RetentionPlan:
    """The ONLY sanctioned production execution path for retention.

    Production runners call this AFTER artifact generation, passing the
    generation outcome explicitly. Retention executes (deletes) only when
    ``generation_succeeded=True``; on any generation failure the plan is
    returned as a dry run so a failed pipeline never prunes the sink
    (which would leave the dashboard with fewer artifacts than before the
    run).

    Guarantees enforced here and in ``plan_retention``:

    * only date-stamped, allowlisted, non-exempt frontend artifacts are
      candidates;
    * raw source caches, ``store/`` (DuckDB/Parquet), ``models/``,
      undated contracts, and study configs are always protected;
    * a missing sink or empty candidate set is a no-op, not an error.
    """
    if not generation_succeeded:
        return plan_retention(data_delivery_dir, config, sport,
                              reference_date=reference_date)
    return apply_retention(data_delivery_dir, config, sport,
                           reference_date=reference_date, execute=True)


def retention_exempt_families(config: PlatformConfig,
                              sport: str) -> tuple[str, ...]:
    """Families the policy must never delete for a sport (durable contracts)."""
    from core.contracts import default_artifact_contract
    contract = default_artifact_contract(sport)
    exempt = {f.family for f in contract.families if f.retention_exempt}
    return tuple(sorted(exempt))
