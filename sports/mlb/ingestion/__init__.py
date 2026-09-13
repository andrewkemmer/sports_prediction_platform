"""MLB Statcast ingestion with Parquet caching.

Mirrors the reference repo's proven ingestion contract:

* chunked pybaseball pulls with retry/backoff; a past-dated core-season
  chunk that stays empty after retries ABORTS the run (silent data loss is
  the failure mode that poisoned training in the reference repo);
* resume semantics: incremental top-up (forward gap + trailing tail
  refresh so mid-game frozen finals get corrected) versus full re-pull
  (``full_repull=True`` discards the cache);
* dedupe on pitch identity (game_pk, at_bat_number, pitch_number),
  keeping the NEWEST copy;
* normalization to the declared raw-field catalog
* (``sports.mlb.features.raw.catalog``).

The production runner passes ``full_repull`` from the mandatory
``MLB_FULL_REPULL`` environment variable — there is no default path that
silently decides repull behavior.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DataSourcesError, load_data_sources

from sports.mlb.features.raw.catalog import (
    COLUMN_ALIASES,
    KEEP_GAME_TYPES,
    PITCH_IDENTITY_COLS,
    STATCAST_COLS,
    STRING_COLS,
    UNUSED_COLS,
)

logger = logging.getLogger(__name__)

# A transient network/parse failure retries with backoff before it may be
# judged empty. An empty past-dated core-season chunk after exhaustion is a
# hard abort (see _abort_on_exhausted_core_season_empty_chunk). The retry,
# backoff and refresh-tail values are declared in
# ``config/data_sources.yaml`` (spec §5) and loaded with the source
# registry below (which they cross-check).

_REGULAR_SEASON_CORE_MONTHS = {4, 5, 6, 7, 8, 9}


class MLBIngestionError(RuntimeError):
    """Raised when Statcast ingestion fails or would silently degrade.

    Carries typed context fields: ``source``, ``params``, and ``cause``.
    """

    def __init__(self, message: str, *, source: str = "mlb:pybaseball.statcast",
                 params: dict | None = None, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.source = source
        self.params = params or {}
        self.cause = cause


SOURCE_STATCAST = "mlb:pybaseball.statcast"
SOURCE_SCHEDULE = "mlb:statsapi.mlb.com/schedule"

#: Declared external source registry: canonical source id -> the default
#: adapter function that owns the real transport. Enforced by the B-007
#: integration-smoke guardrail (tests/core/test_spec_guardrails.py).
EXTERNAL_SOURCES: dict[str, str] = {
    SOURCE_STATCAST: "_default_fetch",
    SOURCE_SCHEDULE: "_default_fetch_schedule",
}

#: Required columns and transport policy — declared in
#: ``config/data_sources.yaml`` (spec §5) and loaded once per process.
#: The declared ``source_id`` is cross-checked against SOURCE_STATCAST so
#: the registry and the policy file cannot drift apart silently.
_SOURCES = load_data_sources("mlb")
_REQUIRED_DECLARED_IDS = {"statcast": SOURCE_STATCAST,
                          "schedule": SOURCE_SCHEDULE}
if set(_SOURCES.required_columns) != set(_REQUIRED_DECLARED_IDS):
    raise DataSourcesError(
        "data_sources.yaml sources do not match the MLB ingestion "
        f"registry: yaml={sorted(_SOURCES.required_columns)} "
        f"code={sorted(_REQUIRED_DECLARED_IDS)}")
if _SOURCES.source_ids.get("statcast") != SOURCE_STATCAST:
    raise DataSourcesError(
        f"data_sources.yaml source_id drift for 'statcast': "
        f"yaml={_SOURCES.source_ids.get('statcast')!r} "
        f"code={SOURCE_STATCAST!r}")
if _SOURCES.source_ids.get("schedule") != SOURCE_SCHEDULE:
    raise DataSourcesError(
        f"data_sources.yaml source_id drift for 'schedule': "
        f"yaml={_SOURCES.source_ids.get('schedule')!r} "
        f"code={SOURCE_SCHEDULE!r}")
CHUNK_RETRIES = _SOURCES.policy.chunk_retries
CHUNK_RETRY_BASE_MS = _SOURCES.policy.chunk_retry_base_ms

# Re-pull this many trailing days on every resume so games captured
# mid-game (partial Statcast posts) get their real finals.
REFRESH_TAIL_DAYS = _SOURCES.policy.refresh_tail_days

#: Approved non-fatal degradations — every entry is an explicitly
#: reviewed exception to the no-silent-fallback guardrail. Any new
#: warn-and-continue path must add a key here first (guardrail-pinned:
#: every declared key must appear at a logger.* telemetry site).
APPROVED_DEGRADATIONS: dict[str, str] = {
    "mlb:offseason-or-future-empty-chunk": (
        "a chunk outside the core regular-season months or in the future "
        "may legitimately be empty (no completed games can exist there); "
        "core-season past-dated empties still abort loudly"),
    "mlb:cache-unreadable": (
        "a corrupt cache is re-pulled from the source"),
    "mlb:schedule-empty-window": (
        "a full-repull schedule window may legitimately contain no games "
        "(offseason/future days); the cache stays as-is and a later pull "
        "covers the span"),
}

REQUIRED_STATCAST_COLS: tuple[str, ...] = _SOURCES.required_columns["statcast"]
REQUIRED_SCHEDULE_COLS: tuple[str, ...] = _SOURCES.required_columns["schedule"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MLB StatsAPI schedule — the source-populated first-pitch instant
# ---------------------------------------------------------------------------


def _schedule_game_row(game: dict) -> dict:
    """Map one StatsAPI schedule game object to the normalized row
    (pure — the transport-independent half of the adapter)."""
    return {
        "game_pk": game.get("gamePk"),
        "game_date": (game.get("gameDate") or "").split("T")[0],
        "start_time_utc": game.get("gameDate"),
        "coded_game_state": (game.get("status") or {})
        .get("codedGameState"),
        "detailed_state": (game.get("status") or {}).get("detailedState"),
    }


def _default_fetch_schedule(day: date) -> list[dict]:
    """One day of the free MLB StatsAPI schedule (the real production
    transport for first-pitch instants, which Statcast does NOT carry —
    proven by the r10 live smoke run: the §7.1 instant gates fail closed
    without this source)."""
    import json
    import urllib.request

    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&date={day.isoformat()}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "sports_prediction_platform/r10"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - typed below
        raise MLBIngestionError(
            f"MLB StatsAPI schedule transport failure for {day}: {exc}",
            source=SOURCE_SCHEDULE, params={"date": day.isoformat()},
            cause=exc) from exc
    rows: list[dict] = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            rows.append(_schedule_game_row(g))
    return rows


def _schedule_rows_from_day(rows: list[dict]) -> pd.DataFrame:
    """Normalize one day's schedule rows to the declared contract."""
    import numpy as np

    if not rows:
        return pd.DataFrame(columns=list(REQUIRED_SCHEDULE_COLS))
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_SCHEDULE_COLS if c not in df.columns]
    if missing:
        raise MLBIngestionError(
            f"MLB StatsAPI schedule payload missing REQUIRED column(s): "
            f"{missing}",
            source=SOURCE_SCHEDULE,
            params={"missing_required_columns": missing})
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce")
    df = df.dropna(subset=["game_pk", "game_date", "start_time_utc"])
    for c in ("coded_game_state", "detailed_state"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def pull_schedule(start_date: str | date, end_date: str | date,
                  out_path: str | Path = "store/mlb/raw/schedule.parquet",
                  *, full_repull: bool, fetch=None,
                  today: date | None = None) -> Path:
    """Pull the StatsAPI schedule for [start, end] into the Parquet cache
    (one API call per day, dedup on game_pk, keep=last so re-delivered
    games refresh their status). Future-dated days return empty rows —
    tolerated (no games can exist there); the caller merges by game_pk.

    §5 cache-first: incremental runs fetch only days OUTSIDE the cached
    game_date span (history extension + forward gap incl. future slate
    days); days inside the span are assumed settled schedule facts.
    Makeup games scheduled into an already-covered past span are picked
    up on the next FULL_REPULL=1 — the honest, documented tradeoff.
    """
    fetch = fetch or _default_fetch_schedule
    start = start_date if isinstance(start_date, date) \
        else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) \
        else date.fromisoformat(str(end_date))
    out = Path(out_path)
    resume_ok = out.exists() and not full_repull
    days: list[date] = []
    cursor = start
    cached = pd.DataFrame()
    if resume_ok:
        try:
            cached = pd.read_parquet(out)
        except Exception:
            cached = pd.DataFrame()
        if not cached.empty:
            gd = pd.to_datetime(cached["game_date"], errors="coerce").dt.date
            lo, hi = gd.min(), gd.max()
            while cursor <= end:
                if cursor < lo or cursor > hi:
                    days.append(cursor)
                cursor += timedelta(days=1)
            logger.info("schedule resume: cache covers %s..%s; "
                        "fetching %d uncovered day(s)", lo, hi, len(days))
        else:
            resume_ok = False  # unreadable/empty cache: treat as no cache
    if not resume_ok:
        while cursor <= end:
            days.append(cursor)
            cursor += timedelta(days=1)
    frames: list[pd.DataFrame] = []
    for day in days:
        rows = fetch(day)
        if rows:
            frames.append(_schedule_rows_from_day(rows))
        time.sleep(0.25)  # rate-limit friendliness
    if not frames:
        if full_repull:
            logger.warning(
                "StatsAPI schedule: no rows for %s .. %s [%s]",
                start, end, "mlb:schedule-empty-window")
        return out
    if not cached.empty:
        frames.insert(0, cached)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    combined = combined.sort_values("game_pk").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    logger.info("MLB schedule cache: %d games -> %s", len(combined), out)
    return out


def cache_bounds(path: Path) -> tuple[date | None, date | None]:
    """(earliest, latest) game_date in a cached parquet; (None, None) if
    unreadable. Both ends are needed: the max detects stale tails, the min
    detects a request to extend history earlier."""
    try:
        gd = pd.read_parquet(path, columns=["game_date"])["game_date"]
        lo = pd.Timestamp(gd.min()).date() if pd.notna(gd.min()) else None
        hi = pd.Timestamp(gd.max()).date() if pd.notna(gd.max()) else None
        return lo, hi
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read cache date bounds from %s: %s [%s]",
                       path, exc, "mlb:cache-unreadable")
        return None, None


def _merge_and_save(out: Path, inc: pd.DataFrame,
                    existing_first: bool) -> Path:
    """Merge an increment with the existing cache, dedupe on pitch identity,
    and overwrite the file. ``existing_first`` puts cache rows before the
    increment so re-delivered rows resolve to the NEWER copy (keep=last)."""
    existing = pd.read_parquet(out)
    n_existing = len(existing)
    parts = ([existing, inc] if existing_first else [inc, existing])
    merged = pd.concat(parts, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(
        subset=[c for c in PITCH_IDENTITY_COLS if c in merged.columns],
        keep="last",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "Cache updated: %d pitches now (%d new, %d overlap rows dropped) -> %s",
        len(merged), max(len(merged) - n_existing, 0), n_before - len(merged),
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Chunked pull with retry/backoff (test-injectable fetcher)
# ---------------------------------------------------------------------------


def _default_fetch(start: date, end: date) -> pd.DataFrame:
    """The real Statcast fetch (pybaseball). Imported lazily so tests can
    inject fetchers without the dependency installed."""
    from pybaseball import statcast
    try:
        df = statcast(str(start), str(end))
    except Exception as exc:
        raise MLBIngestionError(
            f"Statcast transport failure for {start}..{end}: {exc}",
            source=SOURCE_STATCAST,
            params={"start_date": str(start), "end_date": str(end)},
            cause=exc,
        ) from exc
    return df if df is not None else pd.DataFrame()


def _is_past_dated_core_season_chunk(chunk_start: date, chunk_end: date,
                                     today: date) -> bool:
    """True when the chunk is past-dated AND its midpoint falls in the
    April-September core regular season — the dangerous empty case."""
    if chunk_start >= today:
        return False
    mid = chunk_start + (chunk_end - chunk_start) / 2
    return mid.month in _REGULAR_SEASON_CORE_MONTHS


def _chunked_statcast(start: date, end: date, chunk_days: int,
                      pause_sec: float, fetch=None,
                      today: date | None = None) -> list[pd.DataFrame]:
    """Pull Statcast in rate-limit-friendly chunks with retry/backoff.

    A past-dated core-season chunk that is STILL empty or erroring after all
    retries RAISES (``MLBIngestionError``) — it must never silently proceed
    to feature building with missing games. Future-dated and offseason
    empties are allowed (no games can exist there).
    """
    fetch = fetch or _default_fetch
    today = today or date.today()
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        logger.info("  Chunk: %s -> %s", cursor, chunk_end)
        outcome = "ok"
        last_exc: Exception | None = None
        for attempt in range(CHUNK_RETRIES):
            if attempt:
                backoff = (CHUNK_RETRY_BASE_MS * (2 ** (attempt - 1))) / 1000.0
                logger.info("    retrying %s -> %s (attempt %d/%d, %.1fs)",
                            cursor, chunk_end, attempt + 1, CHUNK_RETRIES,
                            backoff)
                time.sleep(backoff)
            try:
                df = fetch(cursor, chunk_end)
                if df is not None and not df.empty:
                    chunks.append(df)
                    logger.info("    -> %d pitches", len(df))
                    outcome = "ok"
                    break
                outcome = "empty"
                last_exc = None
            except Exception as exc:  # noqa: BLE001
                outcome = "error"
                last_exc = exc
                logger.warning("    -> chunk %s -> %s attempt %d/%d failed: %s",
                               cursor, chunk_end, attempt + 1, CHUNK_RETRIES, exc)
        if outcome != "ok":
            reason = (f"error: {last_exc}" if outcome == "error"
                      else "empty response")
            if _is_past_dated_core_season_chunk(cursor, chunk_end, today):
                raise MLBIngestionError(
                    f"Statcast chunk {cursor} -> {chunk_end} came back {reason} "
                    f"after {CHUNK_RETRIES} attempts in core regular-season "
                    f"months. Games in this window would silently drop from "
                    f"the decided frame. Refusing to proceed. Re-run after "
                    f"the transient failure clears (or set MLB_FULL_REPULL=1).",
                    source=SOURCE_STATCAST,
                    params={"chunk_start": str(cursor), "chunk_end": str(chunk_end)},
                    cause=last_exc,
                )
            logger.warning(
                "Statcast chunk %s -> %s empty (%s) — outside core season or "
                "future-dated; no completed games expected [%s]", cursor,
                chunk_end, reason, "mlb:offseason-or-future-empty-chunk")
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(pause_sec)
    return chunks


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply aliases, drop unused columns, filter game types, and ensure all
    catalog columns exist (missing -> NaN). Required schema columns must be present."""
    if df.empty:
        for col in STATCAST_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        return df

    n_before = len(df)
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Required column enforcement
    missing_req = [c for c in REQUIRED_STATCAST_COLS if c not in df.columns]
    if missing_req:
        raise MLBIngestionError(
            f"Statcast payload missing required catalog column(s): {missing_req}",
            source=SOURCE_STATCAST,
            params={"missing_columns": missing_req},
        )

    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.strip().isin(KEEP_GAME_TYPES)]
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d non-regular-season pitches (game_type S/E)",
                        dropped)
    else:
        logger.warning("game_type column missing — cannot filter spring training")
    for col in UNUSED_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in STATCAST_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality strings to category."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].isna().any():
            continue
        col_max, col_min = df[col].max(), df[col].min()
        if pd.isna(col_max) or pd.isna(col_min):
            continue
        if col_max < 32767 and col_min >= -32768:
            df[col] = df[col].astype("int16")
        else:
            df[col] = df[col].astype("int32")
    for col in ("game_type", "home_team", "away_team", "inning_topbot",
                "pitch_type", "description", "events", "p_throws", "stand"):
        if col in df.columns and str(df[col].dtype) == "object" \
                and df[col].nunique() < 100:
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pull_statcast(start_date: str | date, end_date: str | date,
                  out_path: str | Path = "pitches.parquet",
                  chunk_days: int = 7, pause_sec: float = 0.0,
                  *, full_repull: bool, resume: bool = True,
                  fetch=None, today: date | None = None) -> Path:
    """Pull Statcast data and save to Parquet.

    Args:
        start_date / end_date: inclusive window (YYYY-MM-DD or date).
        out_path: cache file.
        chunk_days: days per API chunk.
        pause_sec: seconds between chunks.
        full_repull: REQUIRED explicit flag (from MLB_FULL_REPULL). When
            True the cache is discarded and the full window re-pulled.
        resume: when True and a cache exists and ``full_repull`` is False,
            top up forward gaps and refresh the trailing tail window.
        fetch: injectable fetcher (tests).
        today: injectable clock (tests).

    Returns:
        Path to the written Parquet cache.

    Raises:
        MLBIngestionError: when the full pull returns no data at all.
    """
    fetch = fetch or _default_fetch
    out = Path(out_path)
    start = start_date if isinstance(start_date, date) \
        else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) \
        else date.fromisoformat(str(end_date))

    if full_repull:
        logger.info("FULL_REPULL: discarding cache and re-pulling full history")
        if out.exists():
            out.unlink()
    elif resume and out.exists():
        cached_lo, cached_hi = cache_bounds(out)
        if cached_hi is not None:
            # Backward gap first (history extension).
            if cached_lo is not None and start < cached_lo:
                back_end = cached_lo - timedelta(days=1)
                logger.info("Back-filling %s -> %s", start, back_end)
                chunks_back = _chunked_statcast(start, back_end, chunk_days,
                                                pause_sec, fetch, today)
                if chunks_back:
                    inc = downcast(normalize_columns(
                        pd.concat(chunks_back, ignore_index=True)))
                    out = _merge_and_save(out, inc, existing_first=False)
                else:
                    logger.warning("No Statcast data for %s -> %s",
                                   start, back_end)
                cached_lo, cached_hi = cache_bounds(out)

            # Forward top-up + tail refresh (fixes frozen mid-game finals).
            if cached_hi is not None:
                refresh_start = end - timedelta(days=REFRESH_TAIL_DAYS - 1)
                inc_start = min(cached_hi + timedelta(days=1), refresh_start)
                inc_start = max(inc_start, start)
                logger.info(
                    "Cache covers through %s — pulling %s -> %s "
                    "(includes %d-day tail refresh)", cached_hi, inc_start,
                    end, REFRESH_TAIL_DAYS)
                inc_chunks = _chunked_statcast(inc_start, end, chunk_days,
                                               pause_sec, fetch, today)
                if not inc_chunks:
                    if cached_hi < end:
                        logger.warning(
                            "No Statcast data posted for %s -> %s yet — "
                            "keeping cache (posting lags completion)",
                            inc_start, end)
                    else:
                        logger.error(
                            "Tail refresh returned ZERO pitches — STALE CACHE "
                            "KEPT: recent finals may be frozen. Wait and "
                            "re-run, or set MLB_FULL_REPULL=1.")
                    return out
                inc = downcast(normalize_columns(
                    pd.concat(inc_chunks, ignore_index=True)))
                out = _merge_and_save(out, inc, existing_first=True)
                new_lo, new_hi = cache_bounds(out)
                logger.info("Cache now covers %s -> %s", new_lo, new_hi)
                return out
            return out

    logger.info("Pulling Statcast: %s -> %s (chunk_days=%d)", start, end,
                chunk_days)
    chunks = _chunked_statcast(start, end, chunk_days, pause_sec, fetch, today)
    if not chunks:
        raise MLBIngestionError(
            f"No Statcast data for {start} to {end}",
            source=SOURCE_STATCAST,
            params={"start_date": str(start), "end_date": str(end)},
        )
    raw = pd.concat(chunks, ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(
        subset=[c for c in PITCH_IDENTITY_COLS if c in raw.columns],
        keep="first",
    )
    if len(raw) < before:
        logger.warning("Dropped %d duplicate pitches at chunk boundaries",
                       before - len(raw))
    raw = downcast(normalize_columns(raw))
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out, index=False)
    logger.info("Saved %d pitches -> %s", len(raw), out)
    return out


def load_pitches(path: str | Path) -> pd.DataFrame:
    """Load the raw pitches cache (already normalized)."""
    return pd.read_parquet(path)
