"""NBA data ingestion — NBA Stats API (stats.nba.com) with Parquet caching.

Mirrors the proven platform contract (sports.mlb / sports.nfl /
sports.nhl):

* per-season Parquet caches under ``store/nba/raw/`` (git-ignored);
* ``full_repull=True`` (from the mandatory ``NBA_FULL_REPULL`` env var)
  discards the cache and rebuilds from scratch;
* incremental mode reuses the cache and re-pulls only seasons whose
  results may have changed (the current season — games finish and post
  finals hours after the scheduled date);
* regular-season filtering (gameId type digit 2) happens here so every
  downstream module sees the eligible population only;
* the schedule endpoint returns a FULL SEASON per call (the cheapest of
  the three sports: ~2 calls/season), so the pull is season-granular;
* a past season that stays empty after retries ABORTS loudly — silent
  data loss is the failure mode this module exists to prevent.

Run-variable contract (runner): ``NBA_START_DATE`` / ``NBA_END_DATE`` /
``NBA_FULL_REPULL`` scope ONLY which days are ingested/served. Training
history (warmup seasons, folds, OOF rules) comes from study.yaml — never
from the run window.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sports.nba.catalog import (
    KEEP_GAME_TYPES,
    SCHEDULE_COLS,
    game_type_from_id,
    normalize_season_code,
)

logger = logging.getLogger(__name__)

#: Seconds between retry attempts (rate-limit friendliness).
CHUNK_RETRIES = 3
CHUNK_RETRY_BASE_MS = 1000

#: Request pacing (the stats API tolerates ~1-2 req/s sustained).
REQUEST_PAUSE_SEC = 0.5

_API_BASE = "https://stats.nba.com/stats"

#: Browser-like headers — the stats API rejects non-browser UAs (403).
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36"),
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}


class NBAIngestionError(RuntimeError):
    """Raised when NBA ingestion fails or would silently degrade."""


# ---------------------------------------------------------------------------
# Season helpers (NBA seasons span Oct–Jun; label = start year)
# ---------------------------------------------------------------------------


def seasons_for_window(start: date, end: date) -> list[str]:
    """The NBA season labels (start-year form, ``2023`` -> ``2023-24``)
    overlapping [start, end]. A season runs Oct 1 (start year) through
    Jun 30 (start year + 1): a window ending in July-September maps to
    the prior start year. Never fabricates a season outside the window.
    """
    def start_year(d: date) -> int:
        return d.year if d.month >= 10 else d.year - 1

    lo, hi = start_year(start), start_year(end)
    return [str(y) for y in range(lo, hi + 1)]


def season_label(season_start_year: str | int) -> str:
    """``2023`` -> ``2023-24`` (the API's Season parameter form)."""
    y = int(season_start_year)
    return f"{y}-{str((y + 1) % 100).zfill(2)}"


# ---------------------------------------------------------------------------
# API access (injectable for tests)
# ---------------------------------------------------------------------------


def _default_fetch_schedule(season_start_year: str) -> pd.DataFrame:
    """Pull the full-season schedule from the NBA Stats API."""
    import json
    import urllib.request
    import urllib.error

    label = season_label(season_start_year)
    url = (f"{_API_BASE}/scheduleleaguev2?LeagueID=00&Season={label}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    rows: list[dict] = []
    for day in (payload.get("leagueSchedule") or {}).get("gameDates", []):
        for g in day.get("games", []):
            rows.append(_schedule_row(g))
    return pd.DataFrame(rows)


def _schedule_row(g: dict) -> dict:
    """Flatten one API schedule game into the catalog row shape.

    The ``season`` column is the START-YEAR label (``2023``); it is
    re-normalized to the study's end-year convention (2023-24 -> 2024)
    at the ``normalize_schedule`` boundary.

    Score gating (real-data certified, Phase 5): the API reports
    ``homeTeam.score`` / ``awayTeam.score`` as the number 0 for games
    that have NOT finished (``gameStatus 1`` = scheduled, ``2`` = live).
    Scores are kept ONLY for ``gameStatus == 3`` (final); every other
    status yields NaN so a future slate can never be misread as
    1,206 decided 0-0 games (which would fabricate away-win outcomes
    downstream). """
    home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
    try:
        finished = int(g.get("gameStatus") or 0) == 3
    except (TypeError, ValueError):
        finished = False
    # gameDateTimeUTC is the real start instant; gameTimeUTC is a
    # 1900-01-01 time-of-day placeholder (documented in catalog.py).
    start = g.get("gameDateTimeUTC") or g.get("gameDateEst")
    return {
        "game_id": str(g.get("gameId")),
        "season": season_start_year_of_label(g),
        "game_type": game_type_from_id(str(g.get("gameId"))),
        "game_date": (g.get("gameDateEst") or "")[:10],
        "start_time_utc": start,
        "home_team": home.get("teamTricode"),
        "away_team": away.get("teamTricode"),
        "home_score": home.get("score") if finished else None,
        "away_score": away.get("score") if finished else None,
        "venue": " | ".join(
            x for x in (g.get("arenaName"), g.get("arenaCity")) if x
        ) or None,
        "neutral_site": bool(g.get("isNeutral", False)),
        "game_state": g.get("gameStatusText"),
    }


def season_start_year_of_label(g: dict) -> str:
    """The season start year for one schedule row.

    The scheduleleaguev2 payload does not repeat the season label per
    row; the caller pulls one season per call, so the ingestion layer
    stamps the requested season onto the rows (set by fetch wrapper).
    Falls back to the gameDate's Oct-June season-start inference when
    the stamp is absent (never guessed from an unrelated field).
    """
    return str(g.get("_season_start") or _infer_season_start(
        g.get("gameDateEst") or ""))


def _infer_season_start(game_date: str) -> str:
    """Season start year from a game date (Oct-Jun window)."""
    try:
        d = date.fromisoformat(str(game_date)[:10])
    except ValueError:
        return ""
    return str(d.year if d.month >= 10 else d.year - 1)


def _default_fetch_boxscore(game_id: str) -> dict | None:
    """The per-game player boxscore (participant-panel source).

    Real-data certified (Phase 5): ``boxscoretraditionalv2`` returns ZERO
    PlayerStats rows for current-era games (verified live: 2025-26 games
    return 0 rows while 2023-24/2024-25 return full rows), so the fetch
    uses ``boxscoretraditionalv3``, which serves every era (verified
    2015-16 -> 2025-26). The v3 payload is nested (homeTeam/awayTeam →
    players → statistics) and carries NO game date, so the caller stamps
    the game date from the schedule cache (never fabricated).
    """
    import json
    import urllib.request

    url = f"{_API_BASE}/boxscoretraditionalv3?gameId={game_id}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 — per-game degradation
        logger.warning("boxscore %s unavailable: %s", game_id, exc)
        return None
    try:
        return _flatten_v3_boxscore(payload)
    except Exception as exc:  # noqa: BLE001 — per-game degradation
        logger.warning("boxscore %s malformed: %s", game_id, exc)
        return None


def _flatten_v3_boxscore(payload: dict) -> dict:
    """Flatten the nested v3 payload into the v2-style resultSets dict the
    row parser already consumes. Player identity comes from the boxscore
    itself (``personId`` / ``nameI`` / ``position``; the team tricode from
    the parent team object) — never from a schedule lookup.
    """
    bs = payload.get("boxScoreTraditional")
    if not bs:
        raise ValueError("v3 payload missing boxScoreTraditional")
    rows: list[list] = []
    for side in ("homeTeam", "awayTeam"):
        team = bs.get(side) or {}
        tri = team.get("teamTricode")
        for p in (team.get("players") or []):
            st = p.get("statistics") or {}
            name = p.get("nameI")
            if not name and (p.get("firstName") or p.get("familyName")):
                name = " ".join(x for x in (p.get("firstName"),
                                            p.get("familyName")) if x)
            rows.append([
                p.get("personId"), name, p.get("position") or None, tri,
                st.get("minutes"), st.get("points"), st.get("assists"),
                st.get("reboundsTotal"),
            ])
    headers = ["PLAYER_ID", "PLAYER_NAME", "START_POSITION",
               "TEAM_ABBREVIATION", "MIN", "PTS", "AST", "REB"]
    return {"resultSets": [{"name": "PlayerStats",
                            "headers": headers, "rowSet": rows}]}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the catalog: game-type filter, dtypes, missing-column NaNs."""
    if "game_type" in df.columns:
        n_before = len(df)
        df = df[pd.to_numeric(df["game_type"], errors="coerce")
                .isin(KEEP_GAME_TYPES)]
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d non-regular-season games (types 1/4/5)",
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
    # Season labels -> study end-year convention (2023-24 -> 2024). Done
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
        logger.warning("Could not read cache bounds from %s: %s", path, exc)
        return None, None


def cached_seasons(path: Path) -> set[str]:
    """The season START-YEAR labels already present in the cache.

    The cache stores the study's END-YEAR convention (2023-24 -> 2024);
    start-year labels are recovered as end_year - 1 (an NBA season
    spans Oct Y through Jun Y+1, so the mapping is exact)."""
    try:
        seasons = pd.read_parquet(path, columns=["season"])["season"]
        return {str(int(s) - 1) for s in pd.to_numeric(
            seasons, errors="coerce").dropna()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read cached seasons from %s: %s",
                       path, exc)
        return set()


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
                  out_path: str | Path = "store/nba/raw/schedule.parquet",
                  *, full_repull: bool, resume: bool = True,
                  fetch=None, today: date | None = None,
                  pause_sec: float = REQUEST_PAUSE_SEC) -> Path:
    """Pull NBA schedules covering [start, end] into the Parquet cache.

    The pull is SEASON-granular (one API call covers a full season).
    Only seasons overlapping the window are pulled; seasons already in
    the cache are reused (incremental) unless ``full_repull``.

    Args:
        start_date / end_date: inclusive window (YYYY-MM-DD or date).
        out_path: cache file.
        full_repull: REQUIRED explicit flag (from NBA_FULL_REPULL). True
            discards the cache and re-pulls the full window.
        fetch: injectable season fetcher (tests).
        today: injectable clock (tests).

    Returns:
        Path to the written Parquet cache.

    Raises:
        NBAIngestionError: when a PAST season stays empty after retries,
            or the full pull returns no data at all. The CURRENT season
            (per ``today``) is allowed to be empty only before its first
            game; a past season with no rows is a hole.
    """
    fetch = fetch or _default_fetch_schedule
    out = Path(out_path)
    start = start_date if isinstance(start_date, date) \
        else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) \
        else date.fromisoformat(str(end_date))
    today = today or date.today()

    needed = seasons_for_window(start, end)
    current_start_year = today.year if today.month >= 10 \
        else today.year - 1

    if full_repull:
        logger.info("FULL_REPULL: discarding cache and re-pulling %s->%s",
                    start, end)
        if out.exists():
            out.unlink()
    elif resume and out.exists():
        have = cached_seasons(out)
        missing = [s for s in needed if s not in have]
        if not missing:
            logger.info("Cache covers every needed season %s — refresh "
                        "the current season only", needed)
            missing = ([current_start_year]
                       if current_start_year in needed else [])
            # refresh the current season only if its data may have changed
            # (games finish hours after tip); skip when it isn't needed
            missing = [s for s in missing if s in needed]
        else:
            logger.info("Cache missing seasons %s — pulling them", missing)
        for s in missing:
            df = _fetch_season_with_retry(fetch, s, current_start_year,
                                          today)
            if df is not None and len(df):
                out = _merge_and_save(out, normalize_schedule(df))
                time.sleep(pause_sec)
        if out.exists():
            return out

    logger.info("Pulling NBA schedule seasons %s for %s -> %s",
                needed, start, end)
    frames: list[pd.DataFrame] = []
    for s in needed:
        df = _fetch_season_with_retry(fetch, s, current_start_year, today)
        if df is not None and len(df):
            frames.append(df)
        time.sleep(pause_sec)
    if not frames:
        if int(needed[0]) >= current_start_year:
            # the ONLY needed season is the current one and it is empty:
            # the schedule has not posted yet — legitimate (no hole).
            logger.warning(
                "No NBA schedule data yet for current season(s) %s "
                "(schedule not posted) — writing an empty cache", needed)
            out.parent.mkdir(parents=True, exist_ok=True)
            normalize_schedule(pd.DataFrame(
                columns=SCHEDULE_COLS)).to_parquet(out, index=False)
            return out
        raise NBAIngestionError(
            f"No NBA schedule data for {start} to {end} (seasons {needed})")
    raw = normalize_schedule(pd.concat(frames, ignore_index=True))
    raw = raw.drop_duplicates(subset=["game_id"], keep="first")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out, index=False)
    logger.info("Saved %d games -> %s", len(raw), out)
    return out


def _fetch_season_with_retry(fetch, season: str, current_start_year: int,
                             today: date) -> pd.DataFrame | None:
    """One season with retries; loud failure for PAST seasons.

    A past season returning zero rows after retries is a hole (the NBA
    never plays a 0-game season) — refused loudly. The current season
    before its opening night legitimately returns games only after the
    schedule posts; an empty current-season response passes with a
    warning (no fabrication of a hole that may not exist).
    """
    last_exc: Exception | None = None
    for attempt in range(CHUNK_RETRIES):
        if attempt:
            backoff = (CHUNK_RETRY_BASE_MS * (2 ** (attempt - 1))) / 1000.0
            logger.info("  retrying season %s (attempt %d/%d, %.1fs)",
                        season, attempt + 1, CHUNK_RETRIES, backoff)
            time.sleep(backoff)
        try:
            df = fetch(season)
            if df is None or df.empty:
                if int(season) < current_start_year:
                    raise NBAIngestionError(
                        f"NBA schedule for season {season} came back "
                        f"EMPTY after {CHUNK_RETRIES} attempts and the "
                        f"season is PAST ({current_start_year} is "
                        f"current). A 0-game season is a data hole — "
                        f"refusing to proceed.")
                logger.warning(
                    "Season %s returned no games (current season — "
                    "schedule may not be posted yet)", season)
                return None
            # stamp the requested season onto the rows (the payload does
            # not repeat the label per row). The cache stores the study's
            # END-YEAR convention (start year + 1); normalize_schedule
            # passes plain integers through unmapped.
            df = df.copy()
            df["season"] = int(season) + 1
            df = df.drop(columns=["_season_start"], errors="ignore")
            return df
        except NBAIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("  season %s attempt %d/%d failed: %s",
                           season, attempt + 1, CHUNK_RETRIES, exc)
    raise NBAIngestionError(
        f"NBA schedule for season {season} FAILED after "
        f"{CHUNK_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Boxscores (per-game; participant panel only — never a model feature)
# ---------------------------------------------------------------------------


def _player_rows_from_boxscore(payload: dict, game_id: str) -> list[dict]:
    """Flatten a boxscore's PlayerStats rows into panel records."""
    rows = []
    for rs in (payload.get("resultSets") or []):
        if rs.get("name") != "PlayerStats":
            continue
        headers = rs.get("headers") or []
        idx = {h: i for i, h in enumerate(headers)}
        for r in rs.get("rowSet") or []:
            def _get(key):
                i = idx.get(key)
                return r[i] if i is not None and i < len(r) else None

            minutes = _get("MIN")
            rows.append({
                "game_id": game_id,
                "team": _get("TEAM_ABBREVIATION"),
                "player_id": _get("PLAYER_ID"),
                "player_name": _get("PLAYER_NAME"),
                "position": _get("START_POSITION") or None,
                "minutes": minutes,
                "points": _get("PTS"),
                "assists": _get("AST"),
                "rebounds": _get("REB"),
            })
    return rows


def pull_player_boxscores(game_ids: list[str], out_path: str | Path,
                          game_dates: dict[str, str] | None = None
                          ) -> Path:
    """Pull per-game boxscores for the given games and append the player
    rows to the participant-panel cache.

    ``game_dates`` maps game_id -> game_date (YYYY-MM-DD) from the cached
    schedule: the v3 boxscore carries NO date, and the panel's
    strictly-prior last-10 logic needs one — a game with no known date is
    SKIPPED (never guessed), so the cache stays point-in-time correct.
    A failed pull degrades to a missing row (the panel renders TBD) —
    never fabricated. Returns the cache path.
    """
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
        payload = _default_fetch_boxscore(str(gid))
        if payload is None:
            continue
        rows = _player_rows_from_boxscore(payload, str(gid))
        gd = (game_dates or {}).get(str(gid))
        if rows and gd:
            for r in rows:
                r["game_date"] = gd
            frames.append(pd.DataFrame(rows))
        elif rows:
            logger.warning(
                "boxscore %s has no schedule date — row skipped (the "
                "panel's strictly-prior logic requires real dates)", gid)
        time.sleep(REQUEST_PAUSE_SEC)  # rate-limit friendliness
    if not frames:
        return out
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id", "player_id"],
                                        keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    logger.info("Player panel cache now covers %d games",
                combined["game_id"].nunique())
    return out


def load_schedule_cache(path: str | Path) -> pd.DataFrame:
    """Load the cached (already normalized) schedule. Season labels are
    re-normalized on load (end-year convention) so caches written before
    a label change still satisfy the study contract."""
    df = pd.read_parquet(path)
    if "season" in df.columns:
        df = df.copy()
        df["season"] = df["season"].map(normalize_season_code)
    return df


def load_player_cache(path: str | Path) -> pd.DataFrame | None:
    """Load the cached participant panel rows (None when absent)."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)
