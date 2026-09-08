"""NFL data ingestion — nflverse pull with local Parquet caching.

Mirrors the reference repo's proven contract and ``sports.mlb.ingestion``:

* season-keyed Parquet caches beside the store (never inside the git tree);
* ``full_repull=True`` (from the mandatory ``NFL_FULL_REPULL`` env var)
  discards the cache and rebuilds from scratch;
* incremental mode (``full_repull=False``) reuses the cache and re-pulls
  only seasons whose tail may have changed (the current season — its
  results post progressively), then dedupes on game identity;
* regular-season filtering happens here so every downstream module sees
  the eligible population only;
* per-season pull failures for the CURRENT season warn and skip (a season
  without published data yet); failures for PAST seasons abort loudly —
  silent data loss is the failure mode to avoid.

All frames carry the nflverse column names; the feature engine is the only
place that interprets them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sports.nfl.catalog import (
    KEEP_GAME_TYPES,
    PBP_COLS,
    PLAYER_STATS_COLS,
    SCHEDULE_COLS,
    TEAM_ABBR_ALIASES,
    TEAMS_COLS,
)

logger = logging.getLogger(__name__)


class NFLIngestionError(RuntimeError):
    """Raised when nflverse ingestion fails or would silently degrade."""


def _polars_to_pandas(frame):
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return frame


def _default_load_schedules(seasons):
    from nflreadpy import load_schedules
    return _polars_to_pandas(load_schedules(seasons))


def _default_load_pbp(season):
    from nflreadpy import load_pbp
    return _polars_to_pandas(load_pbp(season))


def _default_load_player_stats(season):
    from nflreadpy import load_player_stats
    return _polars_to_pandas(load_player_stats([season]))


def _default_load_teams():
    from nflreadpy import load_teams
    return _polars_to_pandas(load_teams())


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def eligible_games(schedule: pd.DataFrame, oof_first_season: int,
                   warmup_first_season: int) -> pd.DataFrame:
    """Filter a schedules frame to the eligible population: settled+
    unsettled regular-season games inside the configured season window
    (warmup first season onward). Scores may be NaN for pending games.

    Franchise-rebrand aliases are normalized here (schedule abbr -> the
    canonical abbr nflverse PBP and the teams table use) so every
    downstream module — Elo, the trailing ladder, records, power rankings
    — sees ONE abbr per franchise across the whole decided timeline."""
    df = schedule.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    keep = df["season"].isin(range(warmup_first_season, oof_first_season)) \
        | (df["season"] >= oof_first_season)
    keep &= df["season"] >= warmup_first_season
    df = df[keep]
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(KEEP_GAME_TYPES)]
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].map(TEAM_ABBR_ALIASES).fillna(df[c])
    return df.reset_index(drop=True)


def _narrow(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("%s missing source columns (filled NaN): %s",
                       label, missing)
    out = df.reindex(columns=cols)
    for c in missing:
        out[c] = np.nan
    return out


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _season_cache_path(cache_dir: Path, kind: str, season: int) -> Path:
    return cache_dir / f"{kind}_{season}.parquet"


def _read_season(cache_dir: Path, kind: str, season: int) -> pd.DataFrame | None:
    p = _season_cache_path(cache_dir, kind, season)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001 — corrupt cache -> re-pull
        logger.warning("cache %s unreadable (%s) — re-pulling", p.name, exc)
        return None


def _write_season(cache_dir: Path, kind: str, season: int,
                  df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_season_cache_path(cache_dir, kind, season), index=False)


def clear_cache(cache_dir: Path, kinds: tuple[str, ...] = ("schedules",
                                                          "pbp",
                                                          "player_stats",
                                                          "teams")) -> None:
    """Remove all cached nflverse parquet artifacts (full rebuild)."""
    if not cache_dir.is_dir():
        return
    for p in cache_dir.glob("*.parquet"):
        if p.stem.split("_")[0] in kinds or any(
                p.stem.startswith(k) for k in kinds):
            p.unlink(missing_ok=True)
    for p in sorted(cache_dir.glob("*.parquet")):
        # exhaustive fallback: the cache dir is owned by this module
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_schedule(seasons: list[int], cache_dir: str | Path,
                  *, full_repull: bool = False,
                  load=_default_load_schedules) -> pd.DataFrame:
    """nflverse schedules for the given seasons, cached per season.

    Past seasons are cached-reused in incremental mode; the latest season
    is always re-pulled in incremental mode (results post progressively).
    A past-season pull failure aborts loudly; the latest season's failure
    degrades to the cached copy with a warning.
    """
    cache_dir = Path(cache_dir)
    latest = max(seasons)
    frames: list[pd.DataFrame] = []
    for season in sorted(seasons):
        cached = None if full_repull else _read_season(cache_dir,
                                                       "schedules", season)
        if cached is not None and season != latest:
            frames.append(cached)
            continue
        try:
            df = load([season])
        except Exception as exc:  # noqa: BLE001
            if season == latest and cached is not None:
                logger.warning("schedule re-pull for latest season %d failed "
                               "(%s) — using cached copy", season, exc)
                frames.append(cached)
                continue
            raise NFLIngestionError(
                f"nflverse schedule pull failed for season {season}: {exc}"
            ) from exc
        if df is None or df.empty:
            if season == latest:
                logger.warning("no schedule rows for latest season %d yet",
                               season)
                if cached is not None:
                    frames.append(cached)
                continue
            raise NFLIngestionError(
                f"nflverse schedule empty for past season {season} — "
                f"refusing to proceed with silent data loss")
        out = _narrow(df, SCHEDULE_COLS, "schedules")
        _write_season(cache_dir, "schedules", season, out)
        frames.append(out)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["game_id"], keep="last") \
        .reset_index(drop=True) if "game_id" in combined.columns else combined


def load_pbp(seasons: list[int], cache_dir: str | Path, *,
             full_repull: bool = False,
             load=_default_load_pbp) -> pd.DataFrame | None:
    """nflverse play-by-play narrowed to the rollup columns, cached per
    season. A season without published PBP (e.g. the in-progress season)
    warns and skips — never fatal; features degrade to NaN per the
    documented missing-value policy. Returns None only when NO season
    could be loaded."""
    cache_dir = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for season in sorted(seasons):
        cached = None if full_repull else _read_season(cache_dir, "pbp",
                                                       season)
        if cached is not None:
            frames.append(cached)
            continue
        try:
            df = load(season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pbp unavailable for %d: %s", season, exc)
            continue
        if df is None or df.empty:
            logger.warning("pbp empty for season %d — skipping", season)
            continue
        out = _narrow(df, PBP_COLS, "pbp")
        _write_season(cache_dir, "pbp", season, out)
        frames.append(out)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_player_stats(seasons: list[int], cache_dir: str | Path, *,
                      full_repull: bool = False,
                      load=_default_load_player_stats) -> dict[int, pd.DataFrame]:
    """QB player-stats frames per season (display-only enrichment).
    A failed season degrades to a missing entry — never fabricated."""
    cache_dir = Path(cache_dir)
    out: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        cached = None if full_repull else _read_season(cache_dir,
                                                       "player_stats", season)
        if cached is not None:
            out[season] = cached
            continue
        try:
            df = load(season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("player stats pull failed for %d: %s", season, exc)
            continue
        if df is None or df.empty:
            continue
        frame = _narrow(df, PLAYER_STATS_COLS, "player_stats")
        if "position" in frame.columns:
            frame = frame[frame["position"] == "QB"]
        if "season_type" in frame.columns:
            frame = frame[frame["season_type"] == "REG"]
        _write_season(cache_dir, "player_stats", season, frame)
        out[season] = frame
    return out


def load_team_names(cache_dir: str | Path, *,
                    load=_default_load_teams) -> dict[str, str]:
    """team abbr -> full team name (frontend display fields)."""
    cache_dir = Path(cache_dir)
    p = cache_dir / "teams.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        try:
            df = load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("teams pull failed: %s", exc)
            return {}
        if df is None or df.empty:
            return {}
        df = _narrow(df, TEAMS_COLS, "teams")
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    if "team_abbr" in df.columns and "team_name" in df.columns:
        return dict(zip(df["team_abbr"], df["team_name"]))
    return {}
