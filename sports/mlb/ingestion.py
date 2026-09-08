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
* normalization to the declared raw-field catalog (``sports.mlb.catalog``).

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

from sports.mlb.catalog import (
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
# hard abort (see _abort_on_exhausted_core_season_empty_chunk).
CHUNK_RETRIES = 3
CHUNK_RETRY_BASE_MS = 1000

# Re-pull this many trailing days on every resume so games captured
# mid-game (partial Statcast posts) get their real finals.
REFRESH_TAIL_DAYS = 3

_REGULAR_SEASON_CORE_MONTHS = {4, 5, 6, 7, 8, 9}


class MLBIngestionError(RuntimeError):
    """Raised when Statcast ingestion fails or would silently degrade."""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


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
        logger.warning("Could not read cache date bounds from %s: %s", path, exc)
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
    df = statcast(str(start), str(end))
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
                      pause_sec: float, fetch=_default_fetch,
                      today: date | None = None) -> list[pd.DataFrame]:
    """Pull Statcast in rate-limit-friendly chunks with retry/backoff.

    A past-dated core-season chunk that is STILL empty or erroring after all
    retries RAISES (``MLBIngestionError``) — it must never silently proceed
    to feature building with missing games. Future-dated and offseason
    empties are allowed (no games can exist there).
    """
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
                    f"the transient failure clears (or set MLB_FULL_REPULL=1)."
                )
            logger.warning(
                "Statcast chunk %s -> %s empty (%s) — outside core season or "
                "future-dated; no completed games expected", cursor, chunk_end,
                reason)
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(pause_sec)
    return chunks


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply aliases, drop unused columns, filter game types, and ensure all
    catalog columns exist (missing -> NaN)."""
    n_before = len(df)
    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.strip().isin(KEEP_GAME_TYPES)]
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d non-regular-season pitches (game_type S/E)",
                        dropped)
    else:
        logger.warning("game_type column missing — cannot filter spring training")
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
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
                  fetch=_default_fetch, today: date | None = None) -> Path:
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
        raise MLBIngestionError(f"No Statcast data for {start} to {end}")
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
