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

from sports.nfl.features.raw.catalog import (
    KEEP_GAME_TYPES,
    PBP_COLS,
    PLAYER_STATS_COLS,
    SCHEDULE_COLS,
    TEAM_ABBR_ALIASES,
    TEAMS_COLS,
)

logger = logging.getLogger(__name__)


class NFLIngestionError(RuntimeError):
    """Raised when nflverse ingestion fails or would silently degrade.

    Carries typed context fields: ``source``, ``params``, and ``cause``.
    """

    def __init__(self, message: str, *, source: str = "nfl:nflreadpy",
                 params: dict | None = None,
                 cause: Exception | None = None) -> None:
        super().__init__(message)
        self.source = source
        self.params = params or {}
        self.cause = cause


#: Declared external source registry: canonical source id -> the default
#: adapter function that owns the real transport. Enforced by the B-007
#: integration-smoke guardrail (tests/core/test_spec_guardrails.py).
SOURCE_SCHEDULES = "nfl:nflreadpy.load_schedules"
SOURCE_PBP = "nfl:nflreadpy.load_pbp"
SOURCE_PLAYER_STATS = "nfl:nflreadpy.load_player_stats"
SOURCE_TEAMS = "nfl:nflreadpy.load_teams"

EXTERNAL_SOURCES: dict[str, str] = {
    SOURCE_SCHEDULES: "_default_load_schedules",
    SOURCE_PBP: "_default_load_pbp",
    SOURCE_PLAYER_STATS: "_default_load_player_stats",
    SOURCE_TEAMS: "_default_load_teams",
}

#: Required (identity / date / score) source columns — a payload missing
#: any of these is schema drift and fails loudly. Optional catalog columns
#: are NaN-filled explicitly.
REQUIRED_COLS: dict[str, tuple[str, ...]] = {
    "schedules": ("game_id", "season", "game_type", "gameday",
                  "home_team", "away_team", "home_score", "away_score"),
    "pbp": ("game_id", "posteam", "defteam"),
    "player_stats": ("player_id", "player_name", "team", "season",
                     "week"),
    "teams": ("team_abbr", "team_name"),
}

#: Approved non-fatal degradations — every entry is an explicitly
#: reviewed exception to the no-silent-fallback guardrail. Any new
#: warn-and-continue path must add a key here first (guardrail-pinned).
APPROVED_DEGRADATIONS: dict[str, str] = {
    "nfl:latest-season-not-published": (
        "the newest season may legitimately have no rows before its "
        "schedule/PBP posts; typed + logged"),
    "nfl:latest-season-refresh-failed": (
        "an incremental re-pull of the newest season may fail while a "
        "cached copy exists; the cache is authoritative, typed + logged"),
    "nfl:cache-unreadable": (
        "a corrupt per-season cache is re-pulled from the source"),
}


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


def _narrow(df: pd.DataFrame, cols: list[str], label: str, *,
            required: tuple[str, ...] = (),
            source: str = "nfl:nflreadpy") -> pd.DataFrame:
    """Narrow to the catalog, NaN-filling OPTIONAL columns and failing
    loudly on missing REQUIRED (identity/date/score) columns."""
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise NFLIngestionError(
            f"{label} payload missing REQUIRED source column(s): "
            f"{missing_req}",
            source=source,
            params={"missing_required_columns": missing_req},
        )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.info("%s optional source columns filled NaN: %s",
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
        logger.warning("cache %s unreadable (%s) — re-pulling [%s]", p.name,
                       exc, "nfl:cache-unreadable")
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
                  load=None) -> pd.DataFrame:
    """nflverse schedules for the given seasons, cached per season.

    Past seasons are cached-reused in incremental mode; the latest season
    is always re-pulled in incremental mode (results post progressively).
    A past-season pull failure aborts loudly; the only approved
    degradations are the two newest-season cases declared in
    ``APPROVED_DEGRADATIONS`` (typed + logged) — everything else raises a
    typed ``NFLIngestionError`` carrying source/params/cause.

    ``load`` is the injectable transport seam (defaults to the real
    nflverse loader); tests inject deterministic fakes and never touch
    the network.
    """
    load = load or _default_load_schedules
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
        except NFLIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            if season == latest and cached is not None:
                logger.warning(
                    "schedule re-pull for latest season %d failed (%s) — "
                    "using cached copy [%s]", season, exc,
                    "nfl:latest-season-refresh-failed")
                frames.append(cached)
                continue
            raise NFLIngestionError(
                f"nflverse schedule pull failed for season {season}: {exc}",
                source=SOURCE_SCHEDULES, params={"season": season},
                cause=exc) from exc
        if df is None or df.empty:
            if season == latest:
                logger.warning(
                    "no schedule rows for latest season %d yet [%s]",
                    season, "nfl:latest-season-not-published")
                if cached is not None:
                    frames.append(cached)
                continue
            raise NFLIngestionError(
                f"nflverse schedule empty for past season {season} — "
                f"refusing to proceed with silent data loss",
                source=SOURCE_SCHEDULES, params={"season": season})
        out = _narrow(df, SCHEDULE_COLS, "schedules",
                      required=REQUIRED_COLS["schedules"],
                      source=SOURCE_SCHEDULES)
        _write_season(cache_dir, "schedules", season, out)
        frames.append(out)
    if not frames:
        raise NFLIngestionError(
            f"nflverse schedules: no rows for any requested season "
            f"{seasons}",
            source=SOURCE_SCHEDULES, params={"seasons": list(seasons)})
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["game_id"], keep="last") \
        .reset_index(drop=True) if "game_id" in combined.columns else combined


def load_pbp(seasons: list[int], cache_dir: str | Path, *,
             full_repull: bool = False,
             load=None) -> pd.DataFrame | None:
    """nflverse play-by-play narrowed to the rollup columns, cached per
    season.

    The ONE approved non-fatal case is a newest season whose PBP has not
    been published yet (declared in ``APPROVED_DEGRADATIONS``, typed +
    logged). Every other transport failure, empty past season, or schema
    drift raises a typed ``NFLIngestionError``. Returns None only when NO
    season could be loaded.
    """
    load = load or _default_load_pbp
    cache_dir = Path(cache_dir)
    latest = max(seasons)
    frames: list[pd.DataFrame] = []
    for season in sorted(seasons):
        cached = None if full_repull else _read_season(cache_dir, "pbp",
                                                       season)
        if cached is not None:
            frames.append(cached)
            continue
        try:
            df = load(season)
        except NFLIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NFLIngestionError(
                f"nflverse pbp pull failed for season {season}: {exc}",
                source=SOURCE_PBP, params={"season": season},
                cause=exc) from exc
        if df is None or df.empty:
            if season == latest:
                logger.warning(
                    "pbp not published for latest season %d yet [%s]",
                    season, "nfl:latest-season-not-published")
                continue
            raise NFLIngestionError(
                f"nflverse pbp EMPTY for past season {season} — refusing "
                f"to proceed with silent data loss",
                source=SOURCE_PBP, params={"season": season})
        out = _narrow(df, PBP_COLS, "pbp", required=REQUIRED_COLS["pbp"],
                      source=SOURCE_PBP)
        _write_season(cache_dir, "pbp", season, out)
        frames.append(out)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_player_stats(seasons: list[int], cache_dir: str | Path, *,
                      full_repull: bool = False,
                      load=None) -> dict[int, pd.DataFrame]:
    """QB player-stats frames per season (display-only enrichment).

    Never fabricated: a transport failure or schema drift raises a typed
    ``NFLIngestionError``; only a newest season whose stats have not been
    published yet degrades (typed + logged, declared in
    ``APPROVED_DEGRADATIONS``).
    """
    load = load or _default_load_player_stats
    cache_dir = Path(cache_dir)
    latest = max(seasons) if seasons else None
    out: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        cached = None if full_repull else _read_season(cache_dir,
                                                       "player_stats", season)
        if cached is not None:
            out[season] = cached
            continue
        try:
            df = load(season)
        except NFLIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NFLIngestionError(
                f"nflverse player-stats pull failed for season {season}: "
                f"{exc}",
                source=SOURCE_PLAYER_STATS, params={"season": season},
                cause=exc) from exc
        if df is None or df.empty:
            if season == latest:
                logger.warning(
                    "player stats not published for latest season %d yet "
                    "[%s]", season, "nfl:latest-season-not-published")
                continue
            raise NFLIngestionError(
                f"nflverse player stats EMPTY for past season {season} — "
                f"refusing to proceed with silent data loss",
                source=SOURCE_PLAYER_STATS, params={"season": season})
        frame = _narrow(df, PLAYER_STATS_COLS, "player_stats",
                        required=REQUIRED_COLS["player_stats"],
                        source=SOURCE_PLAYER_STATS)
        if "position" in frame.columns:
            frame = frame[frame["position"] == "QB"]
        if "season_type" in frame.columns:
            frame = frame[frame["season_type"] == "REG"]
        _write_season(cache_dir, "player_stats", season, frame)
        out[season] = frame
    return out


def load_team_names(cache_dir: str | Path, *, load=None) -> dict[str, str]:
    """team abbr -> full team name (frontend display fields).

    Fails LOUDLY (typed ``NFLIngestionError``) on transport failure,
    empty payload, or a missing required identity column — the previous
    silent ``{}`` fallback is gone (no-silent-fallback guardrail).
    """
    load = load or _default_load_teams
    cache_dir = Path(cache_dir)
    p = cache_dir / "teams.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        try:
            df = load()
        except NFLIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NFLIngestionError(
                f"nflverse teams pull failed: {exc}",
                source=SOURCE_TEAMS, cause=exc) from exc
        if df is None or df.empty:
            raise NFLIngestionError(
                "nflverse teams payload is EMPTY — refusing to degrade to "
                "blank display names",
                source=SOURCE_TEAMS)
        df = _narrow(df, TEAMS_COLS, "teams",
                     required=REQUIRED_COLS["teams"], source=SOURCE_TEAMS)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    missing = [c for c in REQUIRED_COLS["teams"] if c not in df.columns]
    if missing:
        raise NFLIngestionError(
            f"teams payload missing REQUIRED column(s): {missing}",
            source=SOURCE_TEAMS, params={"missing_required_columns": missing})
    return dict(zip(df["team_abbr"], df["team_name"]))
