"""NFL point-in-time feature engine (DuckDB/Parquet-backed).

Single owner of every production NFL feature, ported from the reference
repo's proven semantics:

LEAKAGE CONTRACT (enforced and asserted):
  feature(game_t) = f(information available STRICTLY BEFORE game_t)

- Elo: iterated strictly chronologically; each row's ``elo_entering`` is
  the team's rating at kickoff (updated only after that game settles).
- Trailing windowed / EWM statistics: computed per team over its own games
  in chronological order, then ``shift(1)`` — the current and all future
  games are excluded from a row's own value.
- Team state (records, Elo) updates only after a game settles.
- Venue/schedule facts (roof, division, stadium, kickoff hour) are static
  pre-game facts.

DuckDB/Parquet discipline: the per-(game, team) play-by-play rollup and the
team ladder are materialized as Parquet through DuckDB SQL so the full
historical play-by-play never has to sit in RAM at once — the rollup is a
``GROUP BY`` executed in-engine over the cached Parquet file.

Model-family representations:
  - linear view: difference features + the constant ``is_home`` anchor
  - tree view: differences + raw home/away side values
  - mlp view: linear columns, standard-scaled at fit time
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports.nfl.study_config import load_nfl_study

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trailing-window configuration — sourced from study.yaml (never hard-coded
# drift): the loader pins them to the reference-frozen values.
# ---------------------------------------------------------------------------


def _cfg():
    study = load_nfl_study()
    return study


EARTH_RADIUS_MILES = 3958.8

REQUIRED_GAME_COLS = ["game_id", "season", "week", "gameday", "home_team",
                      "away_team", "home_score", "away_score"]


# ---------------------------------------------------------------------------
# Events: long-form (team, game) view
# ---------------------------------------------------------------------------


def team_events(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with team-perspective scores."""
    missing = [c for c in REQUIRED_GAME_COLS if c not in games.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")
    gd = pd.to_datetime(games["gameday"], errors="coerce")
    home = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd,
        "team": games["home_team"], "opponent": games["away_team"],
        "is_home": True,
        "for": games["home_score"].astype(float),
        "against": games["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd,
        "team": games["away_team"], "opponent": games["home_team"],
        "is_home": False,
        "for": games["away_score"].astype(float),
        "against": games["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=0.5)
    return ev


# ---------------------------------------------------------------------------
# Elo — pre-game entering rating, updated only after a game settles
# ---------------------------------------------------------------------------


def _elo_apply(events: pd.DataFrame, elo_k: float, elo_prior: float,
               elo_scale: float) -> tuple[pd.DataFrame, dict]:
    ev = events.sort_values(
        ["gameday", "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    for game_id, rows in ev.groupby("game_id", sort=False):
        a, b = list(rows.itertuples(index=False))[:2]
        ra, rb = rating.get(a.team, elo_prior), rating.get(b.team, elo_prior)
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / elo_scale))
        exp_b = 1.0 / (1.0 + 10.0 ** ((ra - rb) / elo_scale))
        rating[a.team] = ra + elo_k * (a.team_win - exp_a)
        rating[b.team] = rb + elo_k * (b.team_win - exp_b)
    ev = ev.copy()
    ev["elo_entering"] = [entering.get((gid, team), elo_prior)
                          for gid, team in zip(ev["game_id"], ev["team"])]
    return ev, rating


def compute_elo(events: pd.DataFrame, elo_k: float, elo_prior: float,
                elo_scale: float) -> tuple[pd.DataFrame, dict]:
    """Attach pre-game ``elo_entering`` (strictly prior information only)."""
    return _elo_apply(events, elo_k, elo_prior, elo_scale)


# ---------------------------------------------------------------------------
# Trailing primitives — per-team, strictly-prior via shift(1)
# ---------------------------------------------------------------------------


def _trailing_per_team(srt: pd.DataFrame, value_col: str,
                       window: int) -> np.ndarray:
    roll = srt.groupby("team", sort=False)[value_col].rolling(
        window, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


def _trailing_ewm(srt: pd.DataFrame, value_col: str,
                  halflife: float) -> np.ndarray:
    roll = srt.groupby("team", sort=False)[value_col].ewm(
        halflife=halflife, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


# ---------------------------------------------------------------------------
# PBP rollup — persisted as Parquet via DuckDB (bounded RAM)
# ---------------------------------------------------------------------------


def pbp_team_agg(pbp: pd.DataFrame | None, out_path=None) -> pd.DataFrame:
    """Per-(game_id, posteam) play aggregates via DuckDB over Parquet.

    When ``out_path`` is given the play-by-play is first persisted to
    Parquet and the GROUP BY runs inside DuckDB — the full historical PBP
    is never held in memory twice. Every column is a per-game sum/rate;
    the trailing shift downstream keeps them strictly-prior. Absent source
    columns degrade to NaN (never fabricated).
    """
    cols = ["game_id", "team", "total_yards", "n_plays", "elapsed_min"]
    if pbp is None or len(pbp) == 0 or "posteam" not in pbp.columns:
        return pd.DataFrame(columns=cols)
    p = pbp.dropna(subset=["posteam"])
    if "game_id" not in p.columns or "yards_gained" not in p.columns:
        return pd.DataFrame(columns=cols)
    p = p.copy()
    p["yards_gained"] = pd.to_numeric(p["yards_gained"], errors="coerce")
    p["game_seconds_remaining"] = pd.to_numeric(
        p.get("game_seconds_remaining"), errors="coerce")

    import duckdb
    if out_path is not None:
        out_path = __import__("pathlib").Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        p.to_parquet(out_path, index=False)
        g = duckdb.sql("""
            SELECT game_id, posteam AS team,
                   sum(yards_gained) AS total_yards,
                   count(yards_gained) AS n_plays
            FROM read_parquet(?)
            GROUP BY game_id, posteam
        """, params=[str(out_path)]).df()
        last = duckdb.sql("""
            SELECT game_id,
                   (3600.0 - min(game_seconds_remaining)) / 60.0
                       AS elapsed_min
            FROM read_parquet(?)
            WHERE game_seconds_remaining IS NOT NULL
            GROUP BY game_id
        """, params=[str(out_path)]).df()
        if len(last):
            g = g.merge(last, on="game_id", how="left")
        else:
            g["elapsed_min"] = np.nan
    else:
        agg = {"total_yards": ("yards_gained", "sum"),
               "n_plays": ("yards_gained", "count")}
        g = p.groupby(["game_id", "posteam"], as_index=False).agg(**agg)
        last = (p.dropna(subset=["game_seconds_remaining"])
                 .sort_values("game_seconds_remaining")
                 .drop_duplicates("game_id", keep="first"))
        last = last.assign(
            elapsed_min=(3600.0 - last["game_seconds_remaining"]) / 60.0)
        g = g.merge(last[["game_id", "elapsed_min"]], on="game_id",
                    how="left")
    return g.rename(columns={"posteam": "team"})[cols]


# ---------------------------------------------------------------------------
# The ladder — team state + trailing stats, one row per (game_id, team)
# ---------------------------------------------------------------------------


def team_stats_ladder(events: pd.DataFrame,
                      team_game_agg: pd.DataFrame | None,
                      *, form_window: int, winpct_window: int,
                      ypp_window: int, ewm_halflife: float,
                      pace_window: int) -> pd.DataFrame:
    """Per-(game_id, team) point-in-time state: elo_entering, form_pts,
    win_pct, rest_days, ypp, ewm_net_pts, ewm_ypp, pace_plays_min,
    short_rest — every trailing value strictly-prior (asserted)."""
    ev = events.copy()
    if team_game_agg is not None and len(team_game_agg):
        agg = team_game_agg.rename(columns={"total_yards": "tot_yd",
                                            "n_plays": "npl"})
        agg["ypp_game"] = agg["tot_yd"] / agg["npl"].replace(0, np.nan)
        agg["pace_plays_min_game"] = agg["npl"] / agg["elapsed_min"].replace(
            0, np.nan)
        ev = ev.merge(agg[["game_id", "team", "ypp_game",
                           "pace_plays_min_game"]],
                      on=["game_id", "team"], how="left")

    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    # LEAKAGE GATE: gameday strictly increasing within each team
    diffs = srt.groupby("team", sort=False)["gameday"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            "team_stats_ladder: team gameday not strictly increasing "
            f"({len(bad)} rows) — trailing features could leak")

    srt["form_pts"] = _trailing_per_team(srt, "net_from_team", form_window)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", winpct_window)
    srt["rest_days"] = srt.groupby("team", sort=False)["gameday"].diff().dt.days
    srt["short_rest"] = np.where(
        srt["rest_days"].notna(),
        (srt["rest_days"] < 7).astype(float), np.nan)
    srt["ypp"] = (_trailing_per_team(srt, "ypp_game", ypp_window)
                  if "ypp_game" in srt.columns else np.nan)
    srt["ewm_net_pts"] = _trailing_ewm(srt, "net_from_team", ewm_halflife)
    srt["ewm_ypp"] = (_trailing_ewm(srt, "ypp_game", ewm_halflife)
                      if "ypp_game" in srt.columns else np.nan)
    srt["pace_plays_min"] = (
        _trailing_per_team(srt, "pace_plays_min_game", pace_window)
        if "pace_plays_min_game" in srt.columns else np.nan)
    return srt


def _home_minus_away(ladder: pd.DataFrame, game_ids: pd.Index,
                     col: str) -> np.ndarray:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids) - away.reindex(game_ids)).to_numpy()


def _per_side(ladder: pd.DataFrame, game_ids: pd.Index,
              col: str) -> tuple[np.ndarray, np.ndarray]:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids).to_numpy(),
            away.reindex(game_ids).to_numpy())


# ---------------------------------------------------------------------------
# Venue facts (committed stadiums table committed with the package)
# ---------------------------------------------------------------------------


def _load_venue_table(path) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["stadium", "teams", "lat", "lon",
                                     "altitude_ft", "tz"])
    p = Path_ref(path)
    if not p.is_file():
        return pd.DataFrame(columns=["stadium", "teams", "lat", "lon",
                                     "altitude_ft", "tz"])
    return pd.read_csv(p)


def Path_ref(path):
    from pathlib import Path
    return Path(path)


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float)
                              for a in (lat1, lon1, lat2, lon2))
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2)
    d = 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.where(np.isfinite(d), d, np.nan)


def _attach_venue_facts(df: pd.DataFrame, venue_csv) -> pd.DataFrame:
    """Attach travel_miles_diff / altitude_home from the committed stadiums
    table. NaN when unknown — never fabricated."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None
    venue = _load_venue_table(venue_csv)
    facts: dict[str, dict] = {}
    team_home: dict[str, str] = {}
    for r in venue.itertuples(index=False):
        facts[str(r.stadium)] = {
            "lat": float(r.lat) if pd.notna(getattr(r, "lat", np.nan)) else np.nan,
            "lon": float(r.lon) if pd.notna(getattr(r, "lon", np.nan)) else np.nan,
            "altitude_ft": (float(r.altitude_ft)
                            if pd.notna(getattr(r, "altitude_ft", np.nan))
                            else np.nan),
        }
        teams = getattr(r, "teams", "")
        if isinstance(teams, str) and teams.strip():
            for team in teams.split(","):
                team_home.setdefault(team.strip(), str(r.stadium))

    stadium = (df["stadium"] if "stadium" in df.columns
               else pd.Series(np.nan, index=df.index))

    def _game_fact(name: str) -> np.ndarray:
        return stadium.map(
            lambda s: facts.get(s, {}).get(name, np.nan)).to_numpy()

    def _team_fact(team_col: str, name: str) -> np.ndarray:
        return df[team_col].map(
            lambda t: facts.get(team_home.get(t, ""), {}).get(name, np.nan)
        ).to_numpy()

    game_lat, game_lon = _game_fact("lat"), _game_fact("lon")
    home_lat, home_lon = _team_fact("home_team", "lat"), _team_fact(
        "home_team", "lon")
    away_lat, away_lon = _team_fact("away_team", "lat"), _team_fact(
        "away_team", "lon")
    df = df.copy()
    df["travel_miles_diff"] = (
        _haversine_miles(home_lat, home_lon, game_lat, game_lon)
        - _haversine_miles(away_lat, away_lon, game_lat, game_lon))
    df["altitude_home"] = _game_fact("altitude_ft")

    gametime = (df["gametime"].astype(str) if "gametime" in df.columns
                else pd.Series("", index=df.index))
    hour = pd.to_numeric(gametime.str.split(":").str[0], errors="coerce")
    study = _cfg()
    df["prime_time"] = np.where(
        hour >= study.prime_time_hour, 1.0,
        np.where(hour.isna(), np.nan, 0.0))
    return df


def _records_string(events: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Cumulative W-L record per team over the decided timeline."""
    rec = events.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
        ties=("team_win", lambda s: float((s == 0.5).sum())),
    )

    def _fmt(team: str) -> str:
        if team not in rec.index:
            return ""
        r = rec.loc[team]
        base = f"{int(r['wins'])}-{int(r['losses'])}"
        return f"{base}-{int(r['ties'])}" if r["ties"] > 0 else base

    return rec, _fmt


def _apply_diffs(df: pd.DataFrame, ladder: pd.DataFrame,
                 venue_csv) -> pd.DataFrame:
    """Shared diff/per-side/venue block for decided + slate builders."""
    gids = df["game_id"]
    df = df.copy()
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids,
                                                 "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    roof = df.get("roof", pd.Series(np.nan, index=df.index))
    df["is_dome_home"] = np.where(
        roof.isin(["dome", "closed"]), 1.0,
        np.where(roof.isin(["outdoors"]), 0.0, np.nan))
    df = _attach_venue_facts(df, venue_csv)

    for side_col, lad_col in (
            ("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
            ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
            ("ewm_net_pts_home", "ewm_net_pts"),
            ("ewm_net_pts_away", "ewm_net_pts"),
            ("ewm_ypp_home", "ewm_ypp"), ("ewm_ypp_away", "ewm_ypp"),
            ("rest_days_home", "rest_days"),
            ("rest_days_away", "rest_days")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v
    return df


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_game_features(games: pd.DataFrame, pbp: pd.DataFrame | None,
                        venue_csv=None,
                        pbp_rollup_path=None) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the warmup timeline so early games carry real
    priors; every trailing value is shifted strictly prior. Returns the
    served diff features + per-side values the tree view needs + targets.
    """
    study = _cfg()
    ev = compute_elo(team_events(games), study.elo_k, study.elo_prior,
                     study.elo_scale)[0]
    ladder = team_stats_ladder(
        ev, pbp_team_agg(pbp, pbp_rollup_path),
        form_window=study.form_window, winpct_window=study.winpct_window,
        ypp_window=study.ypp_window, ewm_halflife=study.ewm_halflife,
        pace_window=study.pace_window)

    df = games.copy().reset_index(drop=True)
    df = _apply_diffs(df, ladder, venue_csv)

    rec, fmt = _records_string(ev)
    df["home_record"] = df["home_team"].map(fmt)
    df["away_record"] = df["away_team"].map(fmt)

    # targets (kept beside features for OOF assembly; never model inputs)
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["total"] = df["home_score"].astype(float) + df["away_score"].astype(float)
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df


def build_slate_features(schedule: pd.DataFrame, pbp: pd.DataFrame | None,
                         venue_csv=None) -> pd.DataFrame:
    """Point-in-time feature frame for SCHEDULED (undecided) games.

    The ladder spans the full decided timeline; the scheduled rows are the
    latest events so their trailing stats come from strictly-prior decided
    games only (same shift(1) discipline, monotonicity still asserted).
    """
    study = _cfg()
    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    pending = sched[sched["home_score"].isna() | sched["away_score"].isna()]
    if pending.empty:
        return pd.DataFrame()

    ev_decided, ratings = _elo_apply(team_events(decided), study.elo_k,
                                     study.elo_prior, study.elo_scale)
    ev_pending = team_events(pending)
    ev_pending["elo_entering"] = ev_pending["team"].map(
        lambda t: ratings.get(t, study.elo_prior))
    combined = pd.concat([ev_decided, ev_pending], ignore_index=True)
    ladder = team_stats_ladder(
        combined, pbp_team_agg(pbp),
        form_window=study.form_window, winpct_window=study.winpct_window,
        ypp_window=study.ypp_window, ewm_halflife=study.ewm_halflife,
        pace_window=study.pace_window)

    df = pending.copy().reset_index(drop=True)
    # market-independence: drop any odds columns at the boundary
    for m in ("spread_line", "total_line", "home_moneyline",
              "away_moneyline", "over_odds", "under_odds",
              "away_spread_odds", "home_spread_odds"):
        if m in df.columns:
            df = df.drop(columns=m)

    df = _apply_diffs(df, ladder, venue_csv)

    rec, fmt = _records_string(ev_decided)
    df["home_record"] = df["home_team"].map(fmt)
    df["away_record"] = df["away_team"].map(fmt)
    return df


# ---------------------------------------------------------------------------
# Model-family feature views (deterministic ordering + dimensionality)
# ---------------------------------------------------------------------------


def served_diff_columns(study=None) -> list[str]:
    study = study or _cfg()
    return list(study.feature_columns)


def linear_view(df: pd.DataFrame, study=None) -> pd.DataFrame:
    """Difference-oriented linear/MLP matrix (+ is_home anchor)."""
    study = study or _cfg()
    cols = [c for c in list(study.feature_columns) + ["is_home"]
            if c in df.columns]
    return df.reindex(columns=cols).astype(float)


def tree_view(df: pd.DataFrame, study=None) -> pd.DataFrame:
    """Tree-family matrix: differences + raw home/away side values."""
    study = study or _cfg()
    diff_cols = [c for c in study.feature_columns if c in df.columns]
    side_cols = [c for c in ("elo_home", "elo_away", "win_pct_home",
                             "win_pct_away", "ewm_net_pts_home",
                             "ewm_net_pts_away", "ewm_ypp_home",
                             "ewm_ypp_away", "rest_days_home",
                             "rest_days_away") if c in df.columns]
    return df.reindex(columns=diff_cols + side_cols).astype(float)


def feature_coverage_report(df: pd.DataFrame, study=None) -> pd.DataFrame:
    """Coverage + missingness diagnostics per served feature."""
    study = study or _cfg()
    rows = []
    for f in study.feature_columns:
        if f not in df.columns:
            rows.append({"feature": f, "n_games": len(df),
                         "coverage_pct": 0.0, "mean": np.nan, "std": np.nan})
            continue
        v = pd.to_numeric(df[f], errors="coerce")
        rows.append({
            "feature": f, "n_games": int(len(df)),
            "coverage_pct": round(100.0 * float(v.notna().mean()), 2),
            "mean": float(v.mean()) if v.notna().any() else np.nan,
            "std": float(v.std()) if v.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)
