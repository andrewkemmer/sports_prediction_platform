"""NHL point-in-time feature engine (schedule-derivable, PIT-safe).

LEAKAGE CONTRACT (enforced and asserted):
  feature(game_t) = f(information available STRICTLY BEFORE game_t)

- Elo: iterated strictly chronologically; each row's ``elo_entering`` is
  the team's rating at puck drop (updated only after that game settles).
- Trailing windowed / EWM statistics: computed per team over its own
  decided games in chronological order, then ``shift(1)`` — the current
  and all future games are excluded from a row's own value.
- Team state (records, Elo) updates only after a game settles.
- Venue/schedule facts (division, prime-time hour, venue coordinates) are
  static pre-game facts.

NHL-specific derivations (schedule-derivable by design — see study.yaml):

- ``b2b_home`` / ``b2b_away``: back-to-back flag — the team played the
  previous calendar day (rest_days == 1).
- ``ewm_net_goals_diff``: EWM net goals/game difference (the hockey
  analogue of the NFL's ewm_net_pts_diff).
- ``travel_miles_diff``: haversine(away_arena → venue) −
  haversine(home_arena → venue) from the committed NHL arena table.

The ladder's leakage gate asserts chronological monotonicity per team
(the same gate as NFL Phase 3 / MLB Phase 2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sports.nhl.study_config import load_nhl_study

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.8

ARENAS_CSV = Path(__file__).resolve().parent / "nhl_arenas.csv"

REQUIRED_GAME_COLS = ["game_id", "game_date", "home_team", "away_team",
                      "home_score", "away_score"]


def _cfg():
    return load_nhl_study()


# ---------------------------------------------------------------------------
# Events: long-form (team, game) view
# ---------------------------------------------------------------------------


def team_events(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with team-perspective scores.

    Ordering key: ``_event_ts`` — the game's instant (``start_time_utc``,
    a static pre-game fact) falling back to the calendar ``game_date``.
    The instant (not the calendar date) is the correct point-in-time
    ordering key: real NHL back-to-backs that cross a local midnight
    (e.g. WPG vs CHI 2021-11-05 local, 00:00Z Nov 6) collapse to one UTC
    calendar date, and a date-ordered ladder would misorder them.
    """
    missing = [c for c in REQUIRED_GAME_COLS if c not in games.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")
    gd = pd.to_datetime(games["game_date"], errors="coerce")
    if "start_time_utc" in games.columns:
        inst = pd.to_datetime(games["start_time_utc"], errors="coerce",
                              utc=True).dt.tz_localize(None)
    else:
        inst = pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns]")
    # instant when known; calendar date otherwise (never NaT-ordered)
    event_ts = inst.fillna(gd)
    home = pd.DataFrame({
        "game_id": games["game_id"], "game_date": gd,
        "_event_ts": event_ts,
        "team": games["home_team"], "opponent": games["away_team"],
        "is_home": True,
        "for": games["home_score"].astype(float),
        "against": games["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "game_date": gd,
        "_event_ts": event_ts,
        "team": games["away_team"], "opponent": games["home_team"],
        "is_home": False,
        "for": games["away_score"].astype(float),
        "against": games["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    # Full-game resolution: overtime/shootout winner gets the win (the
    # schedule score already reflects the final score incl. OT/SO goals).
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
        ["_event_ts", "game_date", "game_id", "is_home"]).reset_index(drop=True)
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
# The ladder — team state + trailing stats, one row per (game_id, team)
# ---------------------------------------------------------------------------


def team_stats_ladder(events: pd.DataFrame, *, form_window: int,
                      winpct_window: int, ewm_halflife: float
                      ) -> pd.DataFrame:
    """Per-(game_id, team) point-in-time state: elo_entering, form_gls,
    win_pct, rest_days, b2b, ewm_net_goals — every trailing value
    strictly-prior (asserted)."""
    srt = events.sort_values(
        ["team", "_event_ts", "game_date", "game_id"]).reset_index(drop=True)

    # LEAKAGE GATE: a team's game instants strictly increasing. Ordered by
    # the game instant (start_time_utc; game_date fallback), this holds on
    # REAL NHL data — same-calendar-day repeats that are genuinely distinct
    # instants (local-midnight-crossing back-to-backs) order correctly,
    # while a true same-instant repeat for one team (data corruption) still
    # fails loudly ("trailing features could leak"), as in NFL Phase 3.
    diffs = srt.groupby("team", sort=False)["_event_ts"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            "team_stats_ladder: team game_date not strictly increasing "
            f"({len(bad)} rows) — trailing features could leak")

    srt["form_gls"] = _trailing_per_team(srt, "net_from_team", form_window)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", winpct_window)
    srt["rest_days"] = (srt.groupby("team", sort=False)["game_date"]
                        .diff().dt.days)
    srt["b2b"] = np.where(
        srt["rest_days"].notna(), (srt["rest_days"] == 1).astype(float),
        np.nan)
    srt["ewm_net_goals"] = _trailing_ewm(srt, "net_from_team", ewm_halflife)
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
# Venue facts (committed NHL arena table)
# ---------------------------------------------------------------------------


def _load_arena_table(path=None) -> pd.DataFrame:
    p = Path(path) if path is not None else ARENAS_CSV
    if not p.is_file():
        logger.warning("NHL arena table missing at %s — travel features "
                       "will be unavailable (never fabricated)", p)
        return pd.DataFrame(columns=["arena", "teams", "lat", "lon"])
    return pd.read_csv(p)


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


def _attach_venue_facts(df: pd.DataFrame, arena_csv=None) -> pd.DataFrame:
    """Attach travel_miles_diff from the committed arena table. NaN when
    unknown — never fabricated."""
    arena = _load_arena_table(arena_csv)
    facts: dict[str, dict] = {}
    team_home: dict[str, str] = {}
    for r in arena.itertuples(index=False):
        facts[str(r.arena)] = {
            "lat": float(r.lat) if pd.notna(getattr(r, "lat", np.nan)) else np.nan,
            "lon": float(r.lon) if pd.notna(getattr(r, "lon", np.nan)) else np.nan,
        }
        teams = getattr(r, "teams", "")
        if isinstance(teams, str) and teams.strip():
            for team in teams.split(","):
                team_home.setdefault(team.strip(), str(r.arena))

    game_arena = (df["venue"] if "venue" in df.columns
                  else pd.Series(np.nan, index=df.index))

    def _game_fact(name: str) -> np.ndarray:
        return game_arena.map(
            lambda s: facts.get(s, {}).get(name, np.nan)).to_numpy()

    def _team_fact(team_col: str, name: str) -> np.ndarray:
        return df[team_col].map(
            lambda t: facts.get(team_home.get(t, ""), {}).get(name, np.nan)
        ).to_numpy()

    df = df.copy()
    df["travel_miles_diff"] = (
        _haversine_miles(_team_fact("away_team", "lat"),
                         _team_fact("away_team", "lon"),
                         _game_fact("lat"), _game_fact("lon"))
        - _haversine_miles(_team_fact("home_team", "lat"),
                           _team_fact("home_team", "lon"),
                           _game_fact("lat"), _game_fact("lon")))

    # Division flag: committed team→division mapping.
    divisions = _division_map()
    home_div = df["home_team"].map(divisions)
    away_div = df["away_team"].map(divisions)
    df["div_game"] = np.where(
        home_div.notna() & away_div.notna(),
        (home_div == away_div).astype(float), np.nan)

    hour = pd.to_numeric(
        pd.to_datetime(df["start_time_utc"], errors="coerce",
                       utc=True).dt.hour
        if "start_time_utc" in df.columns
        else pd.Series(np.nan, index=df.index),
        errors="coerce")
    study = _cfg()
    df["prime_time"] = np.where(
        hour >= study.prime_time_hour, 1.0,
        np.where(hour.isna(), np.nan, 0.0))
    return df


_DIVISIONS = {
    # Atlantic
    "BOS": "Atlantic", "BUF": "Atlantic", "DET": "Atlantic",
    "FLA": "Atlantic", "MTL": "Atlantic", "OTT": "Atlantic",
    "TBL": "Atlantic", "TOR": "Atlantic",
    # Metropolitan
    "CAR": "Metro", "CBJ": "Metro", "NJD": "Metro", "NYI": "Metro",
    "NYR": "Metro", "PHI": "Metro", "PIT": "Metro", "WSH": "Metro",
    # Central
    "CHI": "Central", "COL": "Central", "DAL": "Central", "MIN": "Central",
    "NSH": "Central", "STL": "Central", "UTA": "Central", "WPG": "Central",
    # Pacific
    "ANA": "Pacific", "CGY": "Pacific", "EDM": "Pacific", "LAK": "Pacific",
    "SEA": "Pacific", "SJS": "Pacific", "UTAH": "Pacific", "VAN": "Pacific",
    "VGK": "Pacific",
    # Legacy abbreviations (schedule rows may predate a rebrand)
    "ARI": "Central", "PHX": "Central", "NJ": "Metro", "SJ": "Pacific",
    "TB": "Atlantic", "LA": "Pacific", "MTL": "Atlantic",
}


def _division_map() -> dict[str, str]:
    return dict(_DIVISIONS)


# ---------------------------------------------------------------------------
# Diff/per-side application (shared decided + slate block)
# ---------------------------------------------------------------------------


def _apply_diffs(df: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    gids = df["game_id"]
    df = df.copy()
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_goals_diff"] = _home_minus_away(ladder, gids,
                                                "ewm_net_goals")
    home_b2b, away_b2b = _per_side(ladder, gids, "b2b")
    df["b2b_home"] = home_b2b
    df["b2b_away"] = away_b2b
    df = _attach_venue_facts(df)

    for side_col, lad_col in (
            ("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
            ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
            ("ewm_net_goals_home", "ewm_net_goals"),
            ("ewm_net_goals_away", "ewm_net_goals"),
            ("rest_days_home", "rest_days"),
            ("rest_days_away", "rest_days")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v
    return df


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_game_features(games: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the warmup timeline so early games carry real
    priors; every trailing value is shifted strictly prior. Returns the
    served diff features + per-side values the tree view needs + targets.
    """
    study = _cfg()
    ev = compute_elo(team_events(games), study.elo_k, study.elo_prior,
                     study.elo_scale)[0]
    ladder = team_stats_ladder(
        ev, form_window=study.form_window,
        winpct_window=study.winpct_window,
        ewm_halflife=study.ewm_halflife)

    df = games.copy().reset_index(drop=True)
    df = _apply_diffs(df, ladder)

    # Full-game targets (kept beside features for OOF assembly; never
    # model inputs). NOTE: a regulation tie is NOT a full-game tie — the
    # NHL resolves every full game via OT/SO, so home_win is 0/1 and the
    # regulation 3-way market is a separate contract (adapter).
    df["margin"] = (df["home_score"].astype(float)
                    - df["away_score"].astype(float))
    df["total"] = (df["home_score"].astype(float)
                   + df["away_score"].astype(float))
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df


def build_slate_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time feature frame for SCHEDULED (undecided) games.

    The ladder spans the full decided timeline; the scheduled rows are
    appended as the latest events so their trailing stats come from
    strictly-prior decided games only (same shift(1) discipline;
    monotonicity still asserted). No targets are produced.
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
        combined, form_window=study.form_window,
        winpct_window=study.winpct_window,
        ewm_halflife=study.ewm_halflife)

    df = pending.copy().reset_index(drop=True)
    df = _apply_diffs(df, ladder)
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
                             "win_pct_away", "ewm_net_goals_home",
                             "ewm_net_goals_away", "rest_days_home",
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
