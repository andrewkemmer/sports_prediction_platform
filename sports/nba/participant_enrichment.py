"""Participant-panel serving-contract enrichment — display information ONLY.

Player data never enters a production feature unless explicitly admitted
through the leakage-safe feature-admission process (it is not, today).

Per-side fields: player_{home,away}_name/_ppg/_apg/_games. Values are
each team's HIGHEST-SCORING player's per-game aggregates over his team's
LAST 10 completed regular-season games strictly before the slate
(point-in-time, trailing across the season boundary). The player is the
top scorer of the trailing window by total points, named by his most
recent appearance. Never fabricated: no prior appearances, or a failed
pull, yields the missing representation (None) — the frontend renders
TBD with preserved null statistics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLAYER_FIELDS = [
    "player_home_name", "player_home_ppg", "player_home_apg",
    "player_home_games",
    "player_away_name", "player_away_ppg", "player_away_apg",
    "player_away_games",
]

LAST10_WINDOW = 10


def _aggregate_prior_games(p_rows: pd.DataFrame) -> dict | None:
    """Per-game aggregates for one player's prior games."""
    if p_rows is None or not len(p_rows):
        return None
    pts = pd.to_numeric(p_rows.get("points"), errors="coerce")
    ast = pd.to_numeric(p_rows.get("assists"), errors="coerce")
    n = int(len(p_rows))
    if pts.notna().sum() == 0:
        return None
    name = p_rows["player_name"].iloc[0] if "player_name" in p_rows.columns \
        else None
    return {
        "name": str(name) if name is not None else None,
        "ppg": float(pts.mean()) if pts.notna().any() else None,
        "apg": float(ast.mean()) if ast.notna().any() else None,
        "n_games": n,
    }


def _last10_aggregates(p_rows: pd.DataFrame) -> dict | None:
    """Per-game aggregates over a player's LAST 10 completed
    regular-season appearances (sorted newest first, truncated)."""
    if p_rows is None or not len(p_rows):
        return None
    rows = p_rows.copy()
    rows["_gd"] = pd.to_datetime(rows.get("game_date"), errors="coerce")
    rows = rows.sort_values("_gd", ascending=False).head(LAST10_WINDOW)
    return _aggregate_prior_games(rows)


def _derive_top_scorer(team_rows: pd.DataFrame) -> str | None:
    """De-facto top scorer: the player with the most TOTAL points in the
    trailing window, named by his most recent appearance. None when
    unusable."""
    if team_rows is None or not len(team_rows):
        return None
    rows = team_rows.copy()
    rows["_pts"] = pd.to_numeric(rows.get("points"), errors="coerce") \
        if "points" in rows.columns else np.nan
    if "player_id" in rows.columns:
        by_player = rows.groupby("player_id")["_pts"].sum(min_count=1)
    else:
        by_player = rows.groupby("player_name")["_pts"].sum(min_count=1)
    if not len(by_player) or by_player.max() is np.nan \
            or not np.isfinite(by_player.max()):
        return None
    top_id = by_player.idxmax()
    sel = (rows["player_id"] == top_id) if "player_id" in rows.columns \
        else (rows["player_name"] == top_id)
    latest = rows[sel].copy()
    latest["_gd"] = pd.to_datetime(latest.get("game_date"), errors="coerce")
    latest = latest.sort_values("_gd", ascending=False)
    name = latest["player_name"].iloc[0] if "player_name" in latest.columns \
        else None
    return str(name) if isinstance(name, str) and name.strip() else None


def _team_last10(team_rows: pd.DataFrame) -> pd.DataFrame:
    """A team's player rows over its LAST 10 completed games."""
    if team_rows is None or not len(team_rows):
        return pd.DataFrame()
    rows = team_rows.copy()
    rows["_gd"] = pd.to_datetime(rows.get("game_date"), errors="coerce")
    return rows.sort_values("_gd", ascending=False).head(
        LAST10_WINDOW * 20)  # ~15 players/game headroom


def enrich_slate(slate_df: pd.DataFrame,
                 player_cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the participant-panel contract fields to a slate frame.

    ``player_cache`` is the cached boxscore player-row frame (injected
    from the bounded per-game pulls); a missing cache degrades to missing
    fields for all games (TBD). Matching is by the de-facto top-scorer
    identity over strictly-prior appearances only.
    """
    out = slate_df.copy()
    for f in PLAYER_FIELDS:
        out[f] = None

    if player_cache is None or not len(player_cache):
        return out

    cache = player_cache.copy()
    cache["_gd"] = pd.to_datetime(cache.get("game_date"), errors="coerce")

    def _side_fields(row, side: str) -> dict:
        prefix = f"player_{side}"
        empty = {f: None for f in PLAYER_FIELDS if f.startswith(prefix)}
        team = row.get("home_team" if side == "home" else "away_team")
        gd = pd.to_datetime(row.get("game_date"), errors="coerce")
        if team is None or pd.isna(gd):
            return empty
        # Strictly prior appearances for this team (current game excluded).
        window = _team_last10(cache[(cache.get("team") == team)
                                    & (cache["_gd"] < gd)])
        if not len(window):
            return empty
        name = _derive_top_scorer(window)
        if not name:
            return empty
        own = (window[window["player_name"] == name]
               if "player_name" in window.columns else pd.DataFrame())
        if not len(own):
            return {f"{prefix}_name": name, **empty}
        agg = _last10_aggregates(own)
        if agg is None:
            return {f"{prefix}_name": name, **empty}
        return {
            f"{prefix}_name": agg["name"],
            f"{prefix}_ppg": agg["ppg"],
            f"{prefix}_apg": agg["apg"],
            f"{prefix}_games": agg["n_games"],
        }

    enriched = []
    for _, row in out.iterrows():
        fields = {}
        fields.update(_side_fields(row, "home"))
        fields.update(_side_fields(row, "away"))
        enriched.append(fields)
    if enriched:
        for f in PLAYER_FIELDS:
            out[f] = [e.get(f) for e in enriched]
    return out
