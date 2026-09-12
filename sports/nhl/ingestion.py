"""NHL data ingestion — NHL Stats API (api-web.nhle.com) with Parquet
caching.

Mirrors the proven platform contract (sports.mlb / sports.nfl):

* per-season Parquet caches under ``store/nhl/raw/`` (git-ignored);
* ``full_repull=True`` (from the mandatory ``NHL_FULL_REPULL`` env var)
  discards the cache and rebuilds from scratch;
* incremental mode reuses the cache and re-pulls only date windows whose
  results may have changed (the trailing tail — games finish and post
  finals hours after the scheduled date);
* regular-season filtering (gameType 2) happens here so every downstream
  module sees the eligible population only;
* a past-dated window that stays empty after retries ABORTS loudly —
  silent data loss is the failure mode this module exists to prevent;
  future-dated windows are allowed to be empty (no games can exist yet).

The schedule endpoint returns a WEEK per dated call, so the pull walks the
window in 7-day steps (bounded request volume: ~15 calls/week-window).
Goalie boxscores are pulled per-game ONLY for the games the goalie panel
needs (slate + trailing window), never for the full history.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sports.nhl.catalog import (
    KEEP_GAME_TYPES,
    SCHEDULE_COLS,
    normalize_season_code,
)

logger = logging.getLogger(__name__)

#: Days between retry attempts (rate-limit friendliness).
CHUNK_RETRIES = 3
CHUNK_RETRY_BASE_MS = 1000

#: Re-pull this many trailing days on every incremental resume so games
#: that finished after the last pull get their real finals.
REFRESH_TAIL_DAYS = 3

#: Schedule requests return a week per dated call.
WINDOW_DAYS = 7

#: A maximal run of at least this many consecutive EMPTY week-windows
#: (>= 15 days with WINDOW_DAYS=7) strictly between windows that returned
#: games is a GENUINE SEASON BREAK (offseason, pandemic pause) — not a
#: hole. In-season NHL stoppages never exceed ~2 weeks (All-Star/bye,
#: COVID pauses), so 3 empty windows separates a month-long offseason
#: from a real in-season data loss. Shorter bracketed empty runs, and
#: every ERROR window (unknown outcome — never assumed empty), remain
#: fatal. Documented policy, not a calendar guess.
GENUINE_GAP_WINDOWS = 3

_API_BASE = "https://api-web.nhle.com/v1"


class NHLIngestionError(RuntimeError):
    """Raised when NHL ingestion fails or would silently degrade.

    Carries typed context fields: ``source``, ``params``, and ``cause``.
    """

    def __init__(self, message: str, *, source: str = "nhl:api-web.nhle.com",
                 params: dict | None = None,
                 cause: Exception | None = None) -> None:
        super().__init__(message)
        self.source = source
        self.params = params or {}
        self.cause = cause


#: Declared external source registry: canonical source id -> the default
#: adapter function that owns the real transport. Enforced by the B-007
#: integration-smoke guardrail (tests/core/test_spec_guardrails.py).
SOURCE_SCHEDULE = "nhl:api-web.nhle.com/schedule"
SOURCE_BOXSCORE = "nhl:api-web.nhle.com/boxscore"

EXTERNAL_SOURCES: dict[str, str] = {
    SOURCE_SCHEDULE: "_default_fetch_schedule_window",
    SOURCE_BOXSCORE: "_default_fetch_boxscore",
}

#: Required (identity / date / score) source columns — a payload missing
#: any of these is schema drift and fails loudly. Optional catalog columns
#: are NaN-filled explicitly.
REQUIRED_COLS: tuple[str, ...] = (
    "game_id", "season", "game_type", "game_date", "home_team",
    "away_team", "home_score", "away_score",
)

#: Approved non-fatal degradations — every entry is an explicitly
#: reviewed exception to the no-silent-fallback guardrail.
APPROVED_DEGRADATIONS: dict[str, str] = {
    "nhl:cache-unreadable": (
        "a corrupt schedule cache is re-pulled from the source"),
    "nhl:tail-refresh-empty": (
        "an incremental tail refresh may return no rows before new "
        "games post; the existing cache is authoritative, typed + logged"),
    "nhl:genuine-gap": (
        "a bracketed run of empty windows >= GENUINE_GAP_WINDOWS is a "
        "genuine season break (offseason / pause), typed + logged"),
}


# ---------------------------------------------------------------------------
# API access (injectable for tests)
# ---------------------------------------------------------------------------


def _default_fetch_schedule_window(start: date, end: date) -> pd.DataFrame:
    """Pull the schedule week(s) covering [start, end] from the NHL API.

    The real production transport; failures raise a typed
    ``NHLIngestionError`` preserving source/params/cause.
    """
    import urllib.request

    rows: list[dict] = []
    cursor = start
    try:
        while cursor <= end:
            url = f"{_API_BASE}/schedule/{cursor.isoformat()}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "sports_prediction_platform/phase4"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            for day in payload.get("gameWeek", []):
                for g in day.get("games", []):
                    rows.append(_schedule_row(g))
            cursor += timedelta(days=WINDOW_DAYS)
    except NHLIngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 — transport failure is loud
        raise NHLIngestionError(
            f"NHL schedule transport failure for {start}..{end}: {exc}",
            source=SOURCE_SCHEDULE,
            params={"start": start.isoformat(), "end": end.isoformat()},
            cause=exc) from exc
    return pd.DataFrame(rows)


def _schedule_row(g: dict) -> dict:
    """Flatten one API schedule game into the catalog row shape."""
    home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
    venue = (g.get("venue") or {}).get("default")
    return {
        "game_id": str(g.get("id")),
        "season": normalize_season_code(g.get("season")),
        "game_type": g.get("gameType"),
        "game_date": g.get("gameDate") or g.get("startTimeUTC", "")[:10],
        "start_time_utc": g.get("startTimeUTC"),
        "home_team": home.get("abbrev"),
        "away_team": away.get("abbrev"),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "venue": venue,
        "neutral_site": bool(g.get("neutralSite", False)),
        "game_state": g.get("gameState"),
    }


def _default_fetch_boxscore(game_id: str) -> dict | None:
    """The boxscore payload for one game (goalie panel source).

    The real production transport; failures raise a typed
    ``NHLIngestionError`` (the previous swallow-to-None fallback is gone —
    a missing boxscore is never silently rendered as TBD).
    """
    import urllib.request

    url = f"{_API_BASE}/gamecenter/{game_id}/boxscore"
    req = urllib.request.Request(
        url, headers={"User-Agent": "sports_prediction_platform/phase4"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except NHLIngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 — transport failure is loud
        raise NHLIngestionError(
            f"NHL boxscore transport failure for game {game_id}: {exc}",
            source=SOURCE_BOXSCORE, params={"game_id": str(game_id)},
            cause=exc) from exc
    if not payload:
        raise NHLIngestionError(
            f"NHL boxscore payload for game {game_id} is EMPTY",
            source=SOURCE_BOXSCORE, params={"game_id": str(game_id)})
    return payload


# json import for the default fetchers
import json  # noqa: E402


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the catalog: game-type filter, dtypes, missing-column NaNs.

    REQUIRED (identity/date/score) columns must be present — a payload
    missing any of them is schema drift and raises a typed error;
    optional catalog columns are NaN-filled explicitly.
    """
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req and len(df):
        raise NHLIngestionError(
            f"NHL schedule payload missing REQUIRED column(s): "
            f"{missing_req}",
            source=SOURCE_SCHEDULE,
            params={"missing_required_columns": missing_req})
    if "game_type" in df.columns:
        n_before = len(df)
        df = df[pd.to_numeric(df["game_type"], errors="coerce")
                .isin(KEEP_GAME_TYPES)]
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d non-regular-season games (types 1/3)",
                        dropped)
        df = df.copy()
    else:
        logger.warning("game_type column missing — cannot filter preseason")
    for col in SCHEDULE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    # Season codes -> study end-year convention (20212022 -> 2022). Done
    # at the ingestion boundary so every study-geometry comparison
    # (warmup split, oof_first_season, fold masks) sees one convention.
    if "season" in df.columns:
        df["season"] = df["season"].map(normalize_season_code)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def cache_bounds(path: Path) -> tuple[date | None, date | None]:
    """(earliest, latest) game_date in a cached parquet; (None, None) if
    unreadable."""
    try:
        gd = pd.read_parquet(path, columns=["game_date"])["game_date"]
        lo = pd.Timestamp(gd.min()).date() if pd.notna(gd.min()) else None
        hi = pd.Timestamp(gd.max()).date() if pd.notna(gd.max()) else None
        return lo, hi
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read cache bounds from %s: %s [%s]", path,
                       exc, "nhl:cache-unreadable")
        return None, None


def _merge_and_save(out: Path, inc: pd.DataFrame) -> Path:
    """Merge an increment with the cache, dedupe on game identity (keep the
    NEWEST copy — refreshed finals overwrite stale in-progress rows)."""
    existing = pd.read_parquet(out)
    n_existing = len(existing)
    merged = pd.concat([existing, inc], ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "Cache updated: %d games now (%d new, %d overlap rows dropped)",
        len(merged), max(len(merged) - n_existing, 0),
        n_before - len(merged))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pull_schedule(start_date: str | date, end_date: str | date,
                  out_path: str | Path = "store/nhl/raw/schedule.parquet",
                  *, full_repull: bool, resume: bool = True,
                  fetch=None,
                  today: date | None = None,
                  pause_sec: float = 0.0) -> Path:
    """Pull NHL schedules for [start, end] into the Parquet cache.

    Args:
        start_date / end_date: inclusive window (YYYY-MM-DD or date).
        out_path: cache file.
        full_repull: REQUIRED explicit flag (from NHL_FULL_REPULL). True
            discards the cache and re-pulls the full window.
        resume: when True and a cache exists and ``full_repull`` is False,
            top up forward gaps and refresh the trailing tail window.
        fetch: injectable window fetcher (tests).
        today: injectable clock (tests).

    Returns:
        Path to the written Parquet cache.

    Raises:
        NHLIngestionError: when a past-dated in-season window stays empty
            after retries, or the full pull returns no data at all.
    """
    fetch = fetch or _default_fetch_schedule_window
    out = Path(out_path)
    start = start_date if isinstance(start_date, date) \
        else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) \
        else date.fromisoformat(str(end_date))
    today = today or date.today()

    if full_repull:
        logger.info("FULL_REPULL: discarding cache and re-pulling %s->%s",
                    start, end)
        if out.exists():
            out.unlink()
    elif resume and out.exists():
        cached_lo, cached_hi = cache_bounds(out)
        if cached_hi is not None:
            if cached_lo is not None and start < cached_lo:
                back_end = cached_lo - timedelta(days=1)
                logger.info("Back-filling %s -> %s", start, back_end)
                inc = _pull_windows(start, back_end, fetch, today,
                                    pause_sec)
                if inc is not None and len(inc):
                    out = _merge_and_save(out, inc)
                cached_lo, cached_hi = cache_bounds(out)
            if cached_hi is not None:
                refresh_start = end - timedelta(days=REFRESH_TAIL_DAYS - 1)
                inc_start = min(cached_hi + timedelta(days=1), refresh_start)
                inc_start = max(inc_start, start)
                logger.info(
                    "Cache covers through %s — pulling %s -> %s "
                    "(includes %d-day tail refresh)", cached_hi, inc_start,
                    end, REFRESH_TAIL_DAYS)
                inc = _pull_windows(inc_start, end, fetch, today, pause_sec)
                if inc is not None and len(inc):
                    out = _merge_and_save(out, inc)
                else:
                    logger.warning(
                        "Tail refresh returned no schedule rows for %s -> "
                        "%s — keeping cache (no games posted yet) [%s]",
                        inc_start, end, "nhl:tail-refresh-empty")
                return out
            return out

    logger.info("Pulling NHL schedule: %s -> %s", start, end)
    raw = _pull_windows(start, end, fetch, today, pause_sec)
    if raw is None or raw.empty:
        raise NHLIngestionError(
            f"No NHL schedule data for {start} to {end}",
            source=SOURCE_SCHEDULE,
            params={"start": str(start), "end": str(end)})
    raw = normalize_schedule(raw).drop_duplicates(subset=["game_id"],
                                                  keep="first")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out, index=False)
    logger.info("Saved %d games -> %s", len(raw), out)
    return out


def _pull_windows(start: date, end: date, fetch, today: date,
                  pause_sec: float) -> pd.DataFrame | None:
    """Walk [start, end] in week-sized windows with retry/backoff.

    Hole detection is DATA-DRIVEN, not calendar-guessed: the walk runs
    first (empty windows tolerated), then maximal runs of consecutive
    non-ok windows STRICTLY BETWEEN windows that returned games are
    classified: a run of >= GENUINE_GAP_WINDOWS consecutive EMPTY windows
    is a genuine season break (offseason / pandemic pause); a shorter
    empty run, or ANY run containing an error window, with games on both
    sides is fatal — silent data loss is the failure mode this module
    exists to prevent. Genuine gaps at the EDGES (leading/trailing empty
    runs: delayed seasons, unplayed future dates) pass.
    """
    frames: list[pd.DataFrame] = []
    results: list[tuple[date, date, str, Exception | None]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=WINDOW_DAYS - 1), end)
        outcome = "ok"
        last_exc: Exception | None = None
        for attempt in range(CHUNK_RETRIES):
            if attempt:
                backoff = (CHUNK_RETRY_BASE_MS * (2 ** (attempt - 1))) / 1000.0
                logger.info("  retrying %s -> %s (attempt %d/%d, %.1fs)",
                            cursor, chunk_end, attempt + 1, CHUNK_RETRIES,
                            backoff)
                time.sleep(backoff)
            try:
                df = fetch(cursor, chunk_end)
                if df is not None and not df.empty:
                    frames.append(df)
                    logger.info("  %s -> %s: %d games", cursor, chunk_end,
                                len(df))
                    outcome = "ok"
                    break
                outcome = "empty"
                last_exc = None
            except Exception as exc:  # noqa: BLE001
                outcome = "error"
                last_exc = exc
                logger.warning("  window %s -> %s attempt %d/%d failed: %s",
                               cursor, chunk_end, attempt + 1,
                               CHUNK_RETRIES, exc)
        results.append((cursor, chunk_end, outcome, last_exc))
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(pause_sec)

    # Hole detection: classify maximal runs of consecutive non-ok windows
    # strictly between windows that returned games.
    got = [o == "ok" for _, _, o, _ in results]
    i = 0
    while i < len(results):
        if results[i][2] == "ok":
            i += 1
            continue
        j = i
        while j < len(results) and results[j][2] != "ok":
            j += 1
        run = results[i:j]
        bracketed = any(got[:i]) and any(got[j:])
        if bracketed:
            run_start, run_end = run[0][0], run[-1][1]
            if any(o == "error" for _, _, o, _ in run):
                raise NHLIngestionError(
                    f"NHL schedule window {run_start} -> {run_end} FAILED "
                    f"({CHUNK_RETRIES} attempts) with games existing on "
                    f"BOTH sides. An error is an UNKNOWN outcome — never "
                    f"assumed empty. Refusing to proceed.",
                    source=SOURCE_SCHEDULE,
                    params={"window_start": str(run_start),
                            "window_end": str(run_end)})
            if len(run) < GENUINE_GAP_WINDOWS:
                raise NHLIngestionError(
                    f"NHL schedule window {run_start} -> {run_end} came "
                    f"back empty after {CHUNK_RETRIES} attempts but games "
                    f"exist on BOTH sides ({run_start} and {run_end} are "
                    f"bracketed by returned windows) and the empty run is "
                    f"shorter than {GENUINE_GAP_WINDOWS} consecutive "
                    f"windows. Games here would silently drop from the "
                    f"decided frame. Refusing to proceed.",
                    source=SOURCE_SCHEDULE,
                    params={"window_start": str(run_start),
                            "window_end": str(run_end)})
            logger.warning(
                "NHL schedule windows %s -> %s empty for %d consecutive "
                "windows — classified as a genuine season break "
                "(offseason / pause); no games expected [%s]", run_start,
                run_end, len(run), "nhl:genuine-gap")
        i = j
    for ws, we, outcome, exc in results:
        if outcome != "ok":
            reason = f"error: {exc}" if outcome == "error" \
                else "empty response"
            logger.warning(
                "NHL schedule window %s -> %s empty (%s) — genuine gap "
                "(offseason / delayed season / future dates); no games "
                "expected [%s]", ws, we, reason, "nhl:genuine-gap")
    if not frames:
        return None
    return normalize_schedule(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Goalie boxscores (per-game; display panel only — never a model feature)
# ---------------------------------------------------------------------------


def _goalie_rows_from_boxscore(payload: dict, game_id: str) -> list[dict]:
    """Flatten a boxscore's goalie rows into panel records."""
    rows = []
    season = payload.get("season")
    game_date = payload.get("gameDate")
    for side in ("homeTeam", "awayTeam"):
        team = (payload.get(side) or {}).get("abbrev")
        goalies = ((payload.get("playerByGameStats") or {})
                   .get(side, {}) or {}).get("goalies", []) or []
        for g in goalies:
            name = (g.get("name") or {}).get("default")
            rows.append({
                "game_id": game_id,
                "team": team,
                "player_id": g.get("playerId"),
                "player_name": name,
                "position": g.get("position", "G"),
                "starter": bool(g.get("starter", False)),
                "decision": g.get("decision"),
                "saves": g.get("saves"),
                "shots_against": g.get("shotsAgainst"),
                "goals_against": g.get("goalsAgainst"),
                "save_pctg": g.get("savePctg"),
                "toi": g.get("toi"),
                "game_date": game_date,
                "season": normalize_season_code(season),
            })
    return rows


def pull_goalie_boxscores(game_ids: list[str], out_path: str | Path, *,
                          fetch=None) -> Path:
    """Pull per-game boxscores for the given games and append the goalie
    rows to the goalie-panel cache.

    ``fetch`` is the injectable transport seam (defaults to the real NHL
    boxscore transport at call time). A transport failure or empty
    payload raises a typed ``NHLIngestionError`` — the row is never
    silently dropped and no cache is written after a failure. Returns the
    cache path.
    """
    fetch = fetch or _default_fetch_boxscore
    out = Path(out_path)
    frames: list[pd.DataFrame] = []
    existing_ids: set[str] = set()
    if out.exists():
        cached = pd.read_parquet(out)
        existing_ids = set(cached["game_id"].astype(str))
        frames.append(cached)
    for gid in game_ids:
        if str(gid) in existing_ids:
            continue
        payload = fetch(str(gid))
        if payload is None:
            raise NHLIngestionError(
                f"NHL boxscore adapter returned no payload for game {gid}",
                source=SOURCE_BOXSCORE, params={"game_id": str(gid)})
        rows = _goalie_rows_from_boxscore(payload, str(gid))
        if not rows:
            raise NHLIngestionError(
                f"NHL boxscore for game {gid} carried no goalie rows",
                source=SOURCE_BOXSCORE, params={"game_id": str(gid)})
        frames.append(pd.DataFrame(rows))
        time.sleep(0.2)  # rate-limit friendliness
    if not frames:
        return out
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id", "player_id"],
                                        keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    logger.info("Goalie panel cache now covers %d games",
                combined["game_id"].nunique())
    return out


def load_schedule_cache(path: str | Path) -> pd.DataFrame:
    """Load the cached (already normalized) schedule. Season codes are
    re-normalized on load (end-year convention) so caches written before
    the 8-digit-label fix still satisfy the study contract."""
    df = pd.read_parquet(path)
    if "season" in df.columns:
        df = df.copy()
        df["season"] = df["season"].map(normalize_season_code)
    return df


def load_goalie_cache(path: str | Path) -> pd.DataFrame | None:
    """Load the cached goalie panel rows (None when absent)."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)
