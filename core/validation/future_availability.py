"""§7.1 point-in-time validators (B-001, Phase 7.5d WS4 — PARTIAL).

Spec §7.1 (FULL_PROJECT_SCOPE.md lines 393-406) is authoritative:

* per-field strict check: ``available_at < prediction_cutoff``;
* ``prediction_cutoff = scheduled game start − per-sport buffer``
  (MLB/NHL 60 min, NFL/NBA 90 min — consumed from each sport's
  ``study.yaml`` via ``prediction_cutoff_buffer_minutes``);
* missing/unparseable availability metadata FAILS CLOSED — no
  fabricated timestamps, ever;
* the slate-window guard below is a SECONDARY gate: it may not define
  or close B-001 (B-001 closes only when per-field ``available_at``
  metadata exists and enforcement is real — B-001-RESIDUAL, Phase 7.6).

The settlement sub-gate (:func:`validate_settlement_record`) is a
DISTINCT semantic check (market-rule correctness: win/loss/tie/push
and regulation vs full-game scope). It is NOT part of the B-005
record-shape registry and is NOT citable as PIT evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.validation import ValidationError
from core.dates import parse_compact_date

# §7.1 buffers (hard-pinned; also pinned per sport in study_config.py).
BUFFER_MINUTES = {"mlb": 60, "nhl": 60, "nfl": 90, "nba": 90}

#: Start-timestamp resolution per sport's start column (B-001 WS4):
#: MLB carries instant UTC starts ("start_time_utc"); Phase 7.6-B (7.6c,
#: dashboard/data-delivery) rebound NHL and NBA to their ALREADY-POPULATED
#: true UTC start instants (ingestion maps startTimeUTC / gameDateTimeUTC
#: into ``start_time_utc``), so their future-start check compares at
#: instant granularity with the anchor unchanged. The NFL schedule still
#: carries only ``gameday`` (date) plus an ET wall-clock ``gametime``
#: (often empty for future weeks); constructing a reliable NFL start
#: instant — timezone handling, missing-gametime handling, validation of
#: the composed timestamp, and fail-closed behavior when no reliable
#: instant exists — is the remaining Phase 7.6 start-timestamp item, so
#: NFL stays date-granularity until that lands.
INSTANT_START_COL = "start_time_utc"

#: The per-sport comparison resolution (single source of truth; the
#: guardrail test pins each runner to exactly this value).
SPORT_RESOLUTION = {"mlb": "instant", "nfl": "date", "nhl": "instant",
                    "nba": "instant"}

#: The per-sport start column the slate gate compares on.
SPORT_START_COL = {"mlb": "start_time_utc", "nfl": "gameday",
                   "nhl": "start_time_utc", "nba": "start_time_utc"}

#: Sports whose serving slates must carry the instant start column.
#: The guardrail test re-derives this set as the instant-resolution sports.
INSTANT_SPORTS = ("mlb", "nhl", "nba")


def window_anchor(window_first_date: str) -> datetime:
    """Deterministic LOWER-BOUND anchor of the serving window (never wall
    clock — identical semantics live and replay, so live and replay runs
    agree exactly).

    ``window_first_date`` is the compact (``YYYYMMDD``) FIRST date of the
    window the run serves:

    * MLB: the first date of the prediction window
      (``PredictionWindow.primary_date()``);
    * NFL/NHL/NBA: the first date of the run window (``RunWindow.start_date``
      — the scope the slate rows are filtered to before this gate).

    The anchor is the instant the window OPENS (00:00 UTC of that date). A
    served start must be AT OR AFTER the window's lower bound, so every
    pre-window row is caught in all four sports; the comparison granularity
    (instant vs calendar date) follows the start data each sport carries.
    """
    d = parse_compact_date(window_first_date)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

_ENFORCEMENT_MODES = ("report_only", "enforce")


@dataclass(frozen=True)
class FutureAvailabilityReport:
    """The §7.1 per-field gate result (telemetry in BOTH modes)."""

    label: str
    n_rows: int
    n_violations: int   # available_at >= prediction_cutoff
    n_missing: int      # missing/unparseable metadata (fail-closed rows)
    violations: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.n_violations == 0 and self.n_missing == 0


@dataclass(frozen=True)
class SlateWindowReport:
    """The SECONDARY slate gate result (every served start at/after the
    serving window's lower-bound anchor + no decided rows in a served
    slate). This is NOT §7.1 PIT evidence."""

    label: str
    n_rows: int
    n_future_violations: int
    n_decided_rows: int
    n_missing_start: int
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class SettlementReport:
    """The settlement sub-gate result (market-rule correctness)."""

    label: str
    n_records: int
    n_violations: int
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.n_violations == 0


def prediction_cutoff(scheduled_start_utc: datetime,
                      buffer_minutes: int) -> datetime:
    """§7.1: cutoff = scheduled game start − per-sport buffer."""
    return scheduled_start_utc - timedelta(minutes=buffer_minutes)


def _parse_utc(value, field: str, label: str) -> datetime:
    """Parse a UTC timestamp; ANY failure raises (fail-closed, never
    fabricated)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(
            f"{label}: missing {field} — §7.1 fails CLOSED (no "
            "fabricated timestamps)")
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{label}: unparseable {field} {value!r} — §7.1 fails "
                "CLOSED") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def validate_available_at(records, *, buffer_minutes: int, label: str,
                          start_col: str = "scheduled_start_utc",
                          available_col: str = "available_at",
                          game_id_col: str = "game_id",
                          ) -> FutureAvailabilityReport:
    """Per-field §7.1 gate: ``available_at < prediction_cutoff`` STRICTLY.

    ``available_at == cutoff`` FAILS. Missing or unparseable metadata
    FAILS CLOSED (rows counted in ``n_missing`` with representative IDs —
    never fabricated, never silently skipped). Works over frame rows or
    mappings (``available_col``/``start_col`` keyed). """
    buffer_minutes = int(buffer_minutes)
    if buffer_minutes not in BUFFER_MINUTES.values():
        raise ValidationError(
            f"{label}: prediction_cutoff_buffer_minutes must be 60 (MLB/"
            "NHL) or 90 (NFL/NBA) per §7.1 — got "
            f"{buffer_minutes}")
    rows = list(records)
    violations: list[str] = []
    missing: list[str] = []

    def _get(row, col):
        if hasattr(row, "get"):            # mapping
            return row.get(col)
        return getattr(row, col, None)

    for i, row in enumerate(rows):
        gid = _get(row, game_id_col)
        gid = str(gid) if gid is not None else f"row[{i}]"
        try:
            start = _parse_utc(_get(row, start_col), start_col, label)
        except ValidationError as exc:
            missing.append(f"{gid}: {exc}")
            continue
        try:
            avail = _parse_utc(_get(row, available_col), available_col,
                               label)
        except ValidationError as exc:
            missing.append(f"{gid}: {exc}")
            continue
        cutoff = prediction_cutoff(start, buffer_minutes)
        if not avail < cutoff:  # STRICT: equality fails per §7.1
            violations.append(
                f"{gid}: available_at {avail.isoformat()} >= "
                f"prediction_cutoff {cutoff.isoformat()} "
                f"(start {start.isoformat()} − {buffer_minutes}m)")
    return FutureAvailabilityReport(
        label=label, n_rows=len(rows), n_violations=len(violations),
        n_missing=len(missing), violations=tuple(violations),
        missing=tuple(missing))


def validate_slate_window(frame, *, anchor_utc: datetime, start_col: str,
                          game_id_col: str = "game_id",
                          decided_cols: tuple[str, ...] = ("home_win",),
                          resolution: str = "instant",
                          label: str = "slate") -> SlateWindowReport:
    """SECONDARY guard (never §7.1 PIT evidence) — UNIFORM semantics for
    all four sports: every SERVED slate row must have a resolvable game
    start that is INSIDE the serving window (at or after the window's
    lower-bound anchor), and a served slate must carry no decided rows.
    Missing/invalid starts are violations (fail-closed), never fabricated
    or silently dropped.

    The anchor is serving-window-derived (see :func:`window_anchor`) —
    deterministic, never wall clock, so live and replay runs behave
    identically. The future-start check ALWAYS runs in all four sports;
    ``resolution`` only selects the comparison granularity each sport's
    start data supports:

    * ``"instant"`` (MLB ``start_time_utc``; NHL/NBA ``start_time_utc``
      since the Phase 7.6-B rebinding): start instant must be at or
      after the window's opening instant;
    * ``"date"`` (NFL ``gameday``): start calendar date must be at or
      after the window's first date. An instant-granularity NFL start is
      the remaining Phase 7.6 start-timestamp item (timezone +
      missing-gametime handling, tracked in docs/TRACKER.md), at which
      point NFL tightens to ``"instant"``.
    """
    if resolution not in ("instant", "date"):
        raise ValidationError(
            f"{label}: resolution must be 'instant' or 'date' (got "
            f"{resolution!r})")
    import pandas as pd

    violations: list[str] = []
    n_decided = 0
    n_missing = 0
    n_future = 0
    try:
        n_rows = int(len(frame))
    except TypeError:
        n_rows = 0
    if n_rows == 0:
        return SlateWindowReport(label=label, n_rows=0,
                                 n_future_violations=0, n_decided_rows=0,
                                 n_missing_start=0, violations=())
    if start_col not in frame.columns:
        return SlateWindowReport(
            label=label, n_rows=n_rows, n_future_violations=0,
            n_decided_rows=0, n_missing_start=n_rows,
            violations=(f"{label}: start column {start_col!r} absent — "
                        "fail-closed",))
    starts = pd.to_datetime(frame[start_col], errors="coerce", utc=True)
    for i, ts in enumerate(starts):
        gid = str(frame[game_id_col].iloc[i]) if game_id_col in frame.columns \
            else f"row[{i}]"
        if ts is None or pd.isna(ts):
            n_missing += 1
            violations.append(
                f"{gid}: missing/unparseable {start_col} — fail-closed")
            continue
        if resolution == "instant":
            in_window = ts.to_pydatetime() >= anchor_utc
        else:  # date resolution: calendar-date comparison at the window edge
            in_window = ts.date() >= anchor_utc.date()
        if not in_window:
            n_future += 1
            violations.append(
                f"{gid}: game start {ts.isoformat()} predates the serving "
                f"window (anchor {anchor_utc.isoformat()}, resolution "
                f"{resolution!r})")
    for col in decided_cols:
        if col in frame.columns and frame[col].notna().any():
            n_decided = int(frame[col].notna().sum())
            violations.append(
                f"{label}: served slate carries {n_decided} decided rows "
                f"in column {col!r}")
    return SlateWindowReport(
        label=label, n_rows=n_rows, n_future_violations=n_future,
        n_decided_rows=n_decided, n_missing_start=n_missing,
        violations=tuple(violations))


def validate_settlement_record(record: dict, *, market_kind: str,
                               settlement_cfg, label: str = "settlement"
                               ) -> SettlementReport:
    """Settlement sub-gate — market-rule correctness for ONE settled
    market record (one market type per record, per rule 8).

    Checks:
    * the record carries a win/loss/push(/tie) outcome consistent with
      the market kind's allowed masses (``tie_allowed`` from
      ``settlement_cfg(market_kind)``; push only where ``push_allowed``);
    * scope: regulation moneyline records must not settle full-game
      (overtime-inclusive) outcomes and vice versa (scope field, when
      present, must match the market kind).
    """
    violations: list[str] = []
    tie_allowed = bool(getattr(settlement_cfg, "tie_allowed", False))
    push_allowed = bool(getattr(settlement_cfg, "push_allowed", False))
    outcome = record.get("outcome")
    scope = record.get("scope")
    if outcome is None:
        violations.append(f"{label}: settled record missing 'outcome'")
    elif outcome not in ("home", "away", "tie", "push"):
        violations.append(f"{label}: unknown outcome {outcome!r}")
    else:
        if outcome == "tie" and not tie_allowed:
            violations.append(
                f"{label}: tie outcome rejected — {market_kind} has "
                "tie_allowed=False")
        if outcome == "push" and not push_allowed:
            violations.append(
                f"{label}: push outcome rejected — {market_kind} has "
                "push_allowed=False")
    if scope is not None:
        expected_scope = ("regulation" if "regulation" in market_kind
                          else "full_game")
        if scope != expected_scope:
            violations.append(
                f"{label}: scope {scope!r} does not match market kind "
                f"{market_kind!r} (expected {expected_scope!r})")
    n = 1 if record else 0
    return SettlementReport(label=label, n_records=n,
                            n_violations=len(violations),
                            violations=tuple(violations))


def normalize_enforcement_mode(value, label: str) -> str:
    """Validate the ``availability_enforcement`` study key
    (``report_only`` | ``enforce``). ``enforce`` is reserved for the
    human-approved 7.6 metadata-layer phase."""
    mode = str(value or "report_only").strip().lower()
    if mode not in _ENFORCEMENT_MODES:
        raise ValidationError(
            f"{label}: availability_enforcement must be one of "
            f"{_ENFORCEMENT_MODES} (got {value!r}); 'enforce' is "
            "reserved for the Phase 7.6 metadata-layer activation")
    return mode
