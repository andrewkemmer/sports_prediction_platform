"""Mandatory production run validation.

Production runners MUST receive ``START_DATE``, ``END_DATE``, and
``FULL_REPULL`` explicitly — there are no silent defaults. The reference
repo's master pipelines defaulted ``MLB_START_DATE`` to a hard-coded date
and ``MLB_END_DATE`` to *today*, which silently changed the training
window from run to run. This module makes omission a loud failure.

Guarantees:

* Missing or invalid values raise :class:`RunConfigError` immediately.
* Format is strict ISO ``YYYY-MM-DD`` (no ``YYYYMMDD``, no datetime).
* ``END_DATE >= START_DATE``.
* ``FULL_REPULL`` is ``0`` or ``1`` only (not "true"/"yes" — the reference
  accepted those, which invites ambiguity in scheduled environments).
* Training history windows are NEVER derived from these values — they come
  from the study configuration. These values scope only the *run window*
  (which days to ingest/serve). A ``training_window_days`` helper exists
  only to make that separation explicit and testable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Environment variable names (sport-agnostic; sport runners may also
#: accept ``<SPORT>_START_DATE`` style aliases, but must not default them).
ENV_START_DATE = "START_DATE"
ENV_END_DATE = "END_DATE"
ENV_FULL_REPULL = "FULL_REPULL"


class RunConfigError(ValueError):
    """Raised when mandatory production run configuration is missing/invalid."""


def _parse_iso_date(value: str, env_name: str) -> date:
    """Strict YYYY-MM-DD parsing; raises RunConfigError on anything else."""
    if not value or not value.strip():
        raise RunConfigError(f"{env_name} is required (no silent defaults)")
    v = value.strip()
    if not _ISO_DATE_RE.match(v):
        raise RunConfigError(
            f"{env_name} must be YYYY-MM-DD format, got {value!r}")
    try:
        return date.fromisoformat(v)
    except ValueError as exc:
        raise RunConfigError(f"{env_name} invalid date: {value!r}") from exc


@dataclass(frozen=True)
class RunWindow:
    """The validated production run window (explicit, never defaulted)."""

    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    full_repull: bool

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def end(self) -> date:
        return date.fromisoformat(self.end_date)

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1

    def training_window_days(self, study_training_days: int) -> tuple[str, str]:
        """The training window, derived ONLY from the study configuration.

        This is the deliberate anti-pattern guard: the run window
        (``START_DATE``/``END_DATE``) scopes ingestion/serving; the
        training window comes from the study config. This function exists
        so tests can prove the run window never leaks into training
        history.
        """
        if study_training_days < 1:
            raise RunConfigError("study_training_days must be >= 1")
        train_end = self.end
        train_start = train_end - timedelta(days=study_training_days - 1)
        return (train_start.isoformat(), train_end.isoformat())


def resolve_run_window(
    start_date: str | None = None,
    end_date: str | None = None,
    full_repull: str | int | bool | None = None,
    *,
    env: dict[str, str] | None = None,
) -> RunWindow:
    """Resolve and validate the mandatory production run window.

    All three values must be supplied (directly or via the environment).
    Missing values raise :class:`RunConfigError` — no silent defaults, no
    date.today() fallbacks, no implicit full-history inference.
    """
    source = dict(os.environ) if env is None else env

    start_raw = start_date if start_date is not None else source.get(
        ENV_START_DATE)
    end_raw = end_date if end_date is not None else source.get(ENV_END_DATE)
    repull_raw = (full_repull if full_repull is not None
                  else source.get(ENV_FULL_REPULL))

    start = _parse_iso_date(start_raw, ENV_START_DATE)  # type: ignore[arg-type]
    end = _parse_iso_date(end_raw, ENV_END_DATE)  # type: ignore[arg-type]

    if repull_raw is None or (isinstance(repull_raw, str)
                              and not repull_raw.strip()):
        raise RunConfigError(
            f"{ENV_FULL_REPULL} is required (no silent defaults)")
    if isinstance(repull_raw, bool):
        repull = repull_raw
    elif isinstance(repull_raw, int):
        if repull_raw not in (0, 1):
            raise RunConfigError(
                f"{ENV_FULL_REPULL} must be 0 or 1, got {repull_raw!r}")
        repull = bool(repull_raw)
    else:
        v = str(repull_raw).strip()
        if v not in ("0", "1"):
            raise RunConfigError(
                f"{ENV_FULL_REPULL} must be 0 or 1, got {repull_raw!r}")
        repull = v == "1"

    if end < start:
        raise RunConfigError(
            f"END_DATE ({end.isoformat()}) must be >= START_DATE "
            f"({start.isoformat()})")

    return RunWindow(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        full_repull=repull,
    )
