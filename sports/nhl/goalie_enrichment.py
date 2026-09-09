"""Goalie serving-contract enrichment — display information ONLY.

Goalie data never enters a production feature unless explicitly admitted
through the leakage-safe feature-admission process (it is not, today).

Per-side fields: goalie_{home,away}_name/_save_pct/_gaa/_starts. Values
are the STARTING GOALTENDER's per-appearance aggregates over his LAST 10
completed regular-season appearances strictly before the slate
(point-in-time, trailing across the season boundary). The starter is the
boxscore-verified starter of each prior game; for a FUTURE game no
reliable announcement exists in the NHL API, so the starter is the
de-facto starter: the goalie with the most STARTS over the team's last
10 completed games. Never fabricated: no prior appearances, or a failed
pull, yields the missing representation (None) — the frontend renders
TBD.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOALIE_FIELDS = [
    "goalie_home_name", "goalie_home_save_pct", "goalie_home_gaa",
    "goalie_home_starts",
    "goalie_away_name", "goalie_away_save_pct", "goalie_away_gaa",
    "goalie_away_starts",
]

LAST10_WINDOW = 10


def _aggregate_prior_games(g_rows: pd.DataFrame) -> dict | None:
    """Per-appearance aggregates for one goalie's prior games."""
    if g_rows is None or not len(g_rows):
        return None
    sa = pd.to_numeric(g_rows.get("shots_against"), errors="coerce")
    ga = pd.to_numeric(g_rows.get("goals_against"), errors="coerce")
    n = int(len(g_rows))
    if sa.notna().sum() == 0 or ga.notna().sum() == 0:
        return None
    save_pct = float(1.0 - ga.sum() / sa.sum()) if sa.sum() > 0 else None
    gaa = float(ga.sum() / n)
    name = g_rows["player_name"].iloc[0] if "player_name" in g_rows.columns \
        else None
    return {
        "name": str(name) if name is not None else None,
        "save_pct": save_pct,
        "gaa": gaa,
        "n_games": n,
    }


def _last10_aggregates(g_rows: pd.DataFrame) -> dict | None:
    """Per-appearance aggregates over a goalie's LAST 10 completed
    regular-season appearances (sorted newest first, truncated)."""
    if g_rows is None or not len(g_rows):
        return None
    rows = g_rows.copy()
    rows["_gd"] = pd.to_datetime(rows.get("game_date"), errors="coerce")
    rows = rows.sort_values("_gd", ascending=False).head(LAST10_WINDOW)
    return _aggregate_prior_games(rows)


def _derive_starter(team_rows: pd.DataFrame) -> str | None:
    """De-facto starting goalie: the player with the most starts in the
    trailing window, named by his most recent appearance. None when
    unusable."""
    if team_rows is None or not len(team_rows):
        return None
    rows = team_rows.copy()
    rows["_st"] = rows.get("starter", False).astype(bool).astype(int) \
        if "starter" in rows.columns else 0
    by_player = (rows.groupby("player_id")["_st"].sum()
                 if "player_id" in rows.columns
                 else rows.groupby("player_name")["_st"].sum())
    if not len(by_player) or by_player.max() <= 0:
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
    """A team's goalie appearance rows over its LAST 10 completed games."""
    if team_rows is None or not len(team_rows):
        return pd.DataFrame()
    rows = team_rows.copy()
    rows["_gd"] = pd.to_datetime(rows.get("game_date"), errors="coerce")
    return rows.sort_values("_gd", ascending=False).head(LAST10_WINDOW)


def enrich_slate(slate_df: pd.DataFrame,
                 goalie_cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the goalie contract fields to a slate frame.

    ``goalie_cache`` is the cached boxscore goalie-row frame (injected
    from the bounded per-game pulls); a missing cache degrades to missing
    fields for all games (TBD). Matching is by the de-facto starter
    identity over strictly-prior appearances only.
    """
    out = slate_df.copy()
    for f in GOALIE_FIELDS:
        out[f] = None

    if goalie_cache is None or not len(goalie_cache):
        return out

    cache = goalie_cache.copy()
    cache["_gd"] = pd.to_datetime(cache.get("game_date"), errors="coerce")

    def _side_fields(row, side: str) -> dict:
        prefix = f"goalie_{side}"
        empty = {f: None for f in GOALIE_FIELDS if f.startswith(prefix)}
        team = row.get("home_team" if side == "home" else "away_team")
        gd = pd.to_datetime(row.get("game_date"), errors="coerce")
        if team is None or pd.isna(gd):
            return empty
        # Strictly prior appearances for this team (current game excluded).
        window = _team_last10(cache[(cache.get("team") == team)
                                    & (cache["_gd"] < gd)])
        if not len(window):
            return empty
        name = _derive_starter(window)
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
            f"{prefix}_save_pct": agg["save_pct"],
            f"{prefix}_gaa": agg["gaa"],
            f"{prefix}_starts": agg["n_games"],
        }

    enriched = []
    for _, row in out.iterrows():
        fields = {}
        fields.update(_side_fields(row, "home"))
        fields.update(_side_fields(row, "away"))
        enriched.append(fields)
    if enriched:
        for f in GOALIE_FIELDS:
            out[f] = [e.get(f) for e in enriched]
    return out
