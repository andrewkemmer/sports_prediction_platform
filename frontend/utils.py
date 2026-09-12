"""Shared artifact loader + sport adapters for the multi-sport frontend.

Resolves each sport's physical artifacts (CSV snapshots for MLB; JSON/CSV
families for NFL/NHL/NBA) under ``sports/<sport>/data_delivery`` and
normalizes them into ONE semantic page model shared by the page
components:

    games         — board rows (sports/<sport> slate / MLB todays_games)
    game_status   — Final | Live | Scheduled
    evening_game  / day_game
    *_team_name   — full team names
    model_pick    / model_correct / is_upset / is_coin_flip
    participant   — per-game dicts from the sport's matchup artifact
    calibration / model_monitor / power_rankings / markets / shap

Backend filenames and schemas are never altered. A missing artifact yields
an empty frame / None — pages render an explicit no-data state; nothing is
synthesized or reused from another sport.

Streamlit is imported lazily (only for caching) so the module — and the
whole adapter layer — imports cleanly under plain pytest without a
Streamlit runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

import frontend.sports_config as sports_config

try:  # caching is a runtime nicety, not a test dependency
    import streamlit as st

    _cache = st.cache_data(ttl=300, show_spinner=False)
except Exception:  # pragma: no cover - streamlit absent (pytest)
    st = None

    def _cache(fn):  # identity decorator
        return fn


# ---------------------------------------------------------------------------
# Theme tokens (mirror the reference frontend's dark palette)
# ---------------------------------------------------------------------------
PRIMARY = "#10B981"      # emerald accent
BLUE = "#3B82F6"
RED = "#EF4444"
AMBER = "#F59E0B"
SLATE = "#94A3B8"
TEXT = "#E2E8F0"
CARD_BG = "#131B2E"
BORDER = "#1E293B"
PAGE_BG = "#0A1128"

UPSET_PROB_THRESHOLD = 0.35
COIN_FLIP_THRESHOLD = 0.02

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Low-level resolution
# ---------------------------------------------------------------------------
def is_demo_mode() -> bool:
    """True ONLY when demo mode is explicitly opted in.

    Sources: the FRONTEND_DEMO_MODE environment variable set to exactly
    "1", or the equivalent explicit Streamlit session flag
    ``st.session_state["frontend_demo_mode"] is True``. Anything else
    (unset, "0", any other value) is off — sample artifacts are never
    used implicitly and pages show the explicit no-data state.
    """
    if os.environ.get("FRONTEND_DEMO_MODE", "").strip() == "1":
        return True
    if st is not None:
        try:
            return st.session_state.get("frontend_demo_mode") is True
        except Exception:
            return False
    return False


def artifact_source(sport: str | None, family: str) -> str:
    """Where the newest artifact for `family` comes from:
    "real" (data_delivery), "sample" (tracked fixtures, demo mode only),
    or "missing" (nothing — the page shows the no-data state)."""
    path = resolve_sport_artifact(sport, family)
    if path is None:
        return "missing"
    return "sample" if "sample_artifacts" in str(path) else "real"


def _latest_artifact_path(sport: str, family: str) -> Optional[Path]:
    """Newest dated file matching the sport's glob for `family`.

    Filenames sort chronologically because dates are zero-padded YYYYMMDD.
    The real data_delivery dir is ALWAYS preferred. The tracked sample
    artifacts under tests/fixtures/sample_artifacts/<sport> are used only
    when demo mode is explicitly enabled (FRONTEND_DEMO_MODE=1) AND no
    real artifact exists — never as a silent fallback. Returns None when
    neither applies (the page's explicit no-data state).
    """
    sport = sports_config.normalize_sport_key(sport)
    pattern = sports_config.artifact_patterns(sport).get(family)
    if not pattern:
        return None
    base = sports_config.data_delivery_dir(sport)
    if base.is_dir():
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]
    if not is_demo_mode():
        return None
    sample_base = sports_config.sample_artifacts_dir() / sport
    if sample_base.is_dir():
        matches = sorted(sample_base.glob(pattern))
        if matches:
            return matches[-1]
    return None


def resolve_sport_artifact(sport: str | None, family: str) -> Optional[Path]:
    return _latest_artifact_path(sports_config.normalize_sport_key(sport), family)


def latest_artifact_date(sport: str | None, family: str) -> Optional[str]:
    """The YYYYMMDD run date of the newest artifact in `family`, or None."""
    path = resolve_sport_artifact(sport, family)
    if path is None:
        return None
    stem = path.stem  # e.g. nba_moneyline_v1_20270105 / todays_games_20260907
    parts = stem.split("_")
    for part in reversed(parts):
        if len(part) == 8 and part.isdigit():
            return part
    return None


def _dated_paths(sport: str, family: str) -> list[Path]:
    """All dated files in `family`, oldest → newest.

    Real data_delivery files always come first; the tracked sample
    artifacts are appended ONLY when demo mode is explicitly enabled
    (FRONTEND_DEMO_MODE=1) — never as a silent fallback."""
    sport = sports_config.normalize_sport_key(sport)
    pattern = sports_config.artifact_patterns(sport).get(family)
    if not pattern:
        return []
    paths: list[Path] = []
    base = sports_config.data_delivery_dir(sport)
    if base.is_dir():
        paths.extend(sorted(base.glob(pattern)))
    if not is_demo_mode():
        return paths
    sample_base = sports_config.sample_artifacts_dir() / sport
    if sample_base.is_dir():
        seen = {p.name for p in paths}
        paths.extend(p for p in sorted(sample_base.glob(pattern))
                     if p.name not in seen)
    return paths


def artifact_dates(sport: str | None, family: str) -> list[str]:
    """Every run date (YYYYMMDD) available in `family` for the sport."""
    dates = []
    for path in _dated_paths(sports_config.normalize_sport_key(sport), family):
        stem = path.stem
        for part in reversed(stem.split("_")):
            if len(part) == 8 and part.isdigit():
                dates.append(part)
                break
    return dates


# ---------------------------------------------------------------------------
# Raw loaders (cached)
# ---------------------------------------------------------------------------
@_cache
def _load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@_cache
def _load_json(path_str: str) -> Optional[dict]:
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return None


def load_csv_artifact(sport: str | None, family: str) -> pd.DataFrame:
    path = resolve_sport_artifact(sport, family)
    if path is None:
        return pd.DataFrame()
    return _load_csv(str(path))


def load_json_artifact(sport: str | None, family: str) -> Optional[dict]:
    path = resolve_sport_artifact(sport, family)
    if path is None:
        return None
    return _load_json(str(path))


def load_json_games(sport: str | None, family: str) -> pd.DataFrame:
    """Flatten a `{games: [...]}` JSON family (moneyline/matchup) to a frame."""
    data = load_json_artifact(sport, family)
    if not data or not isinstance(data.get("games"), list):
        return pd.DataFrame()
    return pd.DataFrame(data["games"])


# ---------------------------------------------------------------------------
# Team-name mapping (shared display fallback; artifacts carry *_team_name
# where the backend publishes full names — MLB abbreviations only)
# ---------------------------------------------------------------------------
TEAM_NAMES: dict[str, str] = {
    # MLB
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
    # NFL
    "ARI_NFL": "Arizona Cardinals", "ATL_NFL": "Atlanta Falcons", "BAL_NFL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN_NFL": "Cincinnati Bengals", "CLE_NFL": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET_NFL": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU_NFL": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC_NFL": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA_NFL": "Miami Dolphins", "MIN_NFL": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI_NFL": "Philadelphia Eagles", "PIT_NFL": "Pittsburgh Steelers",
    "SF_NFL": "San Francisco 49ers", "SEA_NFL": "Seattle Seahawks", "TB_NFL": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WSH_NFL": "Washington Commanders",
    # NHL (division names as fallback; moneyline JSON carries *_team_name)
    # NBA (tricode → city nickname fallback; moneyline JSON carries *_team_name)
}


def _team_display(abbr, fallback_name=""):
    """Best-effort display name for a team abbrev. Never fabricates: falls
    back to the artifact's own *_team_name value, then the raw abbrev."""
    if abbr is None or (isinstance(abbr, float) and pd.isna(abbr)):
        return ""
    s = str(abbr).strip()
    return TEAM_NAMES.get(s) or str(fallback_name or s)


# ---------------------------------------------------------------------------
# Shared board normalization — the dashboard semantics contract
# ---------------------------------------------------------------------------
def _status_from_row(row: pd.Series) -> str:
    """Final | Live | Scheduled, from explicit fields only.

    Priority: moneyline game_status (pre/post/live/final) → MLB game_state
    (post/in/pre) → outcome presence (home_win notnull) → scheduled start
    in the future → Scheduled; anything else Live (never fabricate Final).
    """
    for col in ("game_status", "game_state"):
        raw = row.get(col)
        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
            s = str(raw).strip().lower()
            if s in ("final", "post"):
                return "Final"
            if s in ("live", "in"):
                return "Live"
            if s in ("pre", "preview", "scheduled"):
                return "Scheduled"
    win = row.get("home_win")
    if win is not None and not (isinstance(win, float) and pd.isna(win)):
        return "Final"
    start = row.get("start_time_utc")
    if start:
        ts = pd.to_datetime(start, errors="coerce", utc=True)
        if pd.notna(ts) and ts > pd.Timestamp.now(tz="UTC"):
            return "Scheduled"
    return "Live"


def _is_evening_start(iso) -> bool:
    """Evening = 7 PM ET or later (matches the reference definition)."""
    if not iso:
        return False
    try:
        ts = pd.to_datetime(iso, errors="coerce", utc=True)
        if pd.isna(ts):
            return False
        return ts.tz_convert("America/New_York").hour >= 19
    except (TypeError, ValueError):
        return False


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Derive every shared display column the game cards expect.

    Adds: game_status, evening_game, day_game, home/away_team_name,
    model_pick (from probabilities when absent), model_correct,
    is_upset, is_coin_flip. Purely presentational — no outcome or
    probability is ever synthesized for a row the backend left null.
    """
    df = df.copy()
    if df.empty:
        for col in ("game_status", "evening_game", "day_game", "home_team_name",
                    "away_team_name", "model_pick", "model_correct", "is_upset",
                    "is_coin_flip"):
            df[col] = pd.Series(dtype="object")
        return df

    df["game_status"] = df.apply(_status_from_row, axis=1)

    if "start_time_utc" in df.columns:
        df["evening_game"] = df["start_time_utc"].map(_is_evening_start)
    else:
        df["evening_game"] = False
    df["day_game"] = ~df["evening_game"].astype(bool)

    for abbr_col, name_col in (("home_team", "home_team_name"),
                               ("away_team", "away_team_name")):
        if name_col not in df.columns:
            df[name_col] = df.get(abbr_col, pd.Series(dtype="object")).map(
                lambda a: _team_display(a))
        else:
            df[name_col] = df.apply(
                lambda r: _team_display(r.get(abbr_col), r.get(name_col)), axis=1)

    # Model pick: prefer the backend field; derive from probabilities ONLY
    # when the artifact omitted it (same derivation the reference uses).
    if "home_win_prob_model" in df.columns:
        probs = pd.to_numeric(df["home_win_prob_model"], errors="coerce")
    else:
        probs = pd.Series(float("nan"), index=df.index)
    if "model_pick" not in df.columns:
        df["model_pick"] = ""
        df.loc[probs >= 0.5, "model_pick"] = df.loc[probs >= 0.5, "home_team"]
        df.loc[probs < 0.5, "model_pick"] = df.loc[probs < 0.5, "away_team"]
    pick = df["model_pick"].fillna("").astype(str).str.strip()

    if "home_win" in df.columns:
        win = pd.to_numeric(df["home_win"], errors="coerce")
    else:
        win = pd.Series(float("nan"), index=df.index)
    finished = win.notna()
    actual = pd.Series(pd.NA, index=df.index, dtype="object")
    actual = actual.mask(finished & (win == 1), df["home_team"])
    actual = actual.mask(finished & (win == 0), df["away_team"])

    if "model_correct" in df.columns:
        df["model_correct"] = df["model_correct"].astype("boolean")
    else:
        df["model_correct"] = (finished & (pick == actual)).astype("boolean")

    winner_prob = probs.where(actual == df["home_team"], 1 - probs)
    df["is_upset"] = finished & (winner_prob <= UPSET_PROB_THRESHOLD)
    df["is_coin_flip"] = (pick == "") | ((probs - 0.5).abs() < COIN_FLIP_THRESHOLD)

    # Defensive: never let a duplicate game_id reach the card loop.
    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Semantic loaders (per page)
# ---------------------------------------------------------------------------
def load_games_for_date(sport: str | None, date_str: str) -> pd.DataFrame:
    """Normalized board rows for `date_str` (YYYY-MM-DD or YYYYMMDD).

    MLB: the todays_games_*.csv snapshot. NFL/NHL/NBA: the moneyline JSON
    slate filtered to the date. Empty frame when nothing exists.
    """
    sport = sports_config.normalize_sport_key(sport)
    compact = str(date_str).replace("-", "")
    dashed = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"

    if sport == "mlb":
        # MLB board files are dated snapshots; prefer the exact date.
        base = sports_config.data_delivery_dir(sport)
        path = base / f"todays_games_{compact}.csv"
        if path.is_file():
            df = _load_csv(str(path))
        else:
            df = load_csv_artifact(sport, "games")
        if df.empty:
            return df
        if "game_date" in df.columns:
            df = df[df["game_date"].astype(str).str[:10] == dashed]
        return normalize_games(df)

    df = load_json_games(sport, "moneyline_json")
    if df.empty:
        return df
    if "game_date" in df.columns:
        df = df[df["game_date"].astype(str).str[:10] == dashed]
    return normalize_games(df)


def board_dates(sport: str | None) -> list[str]:
    """Distinct YYYYMMDD game dates available for the sport's board,
    ascending — drives date navigation and calendar highlighting."""
    sport = sports_config.normalize_sport_key(sport)
    frames: list[pd.DataFrame] = []
    if sport == "mlb":
        for path in _dated_paths(sport, "games"):
            df = _load_csv(str(path))
            if not df.empty:
                frames.append(df)
    else:
        df = load_json_games(sport, "moneyline_json")
        if not df.empty:
            frames.append(df)
    dates: set[str] = set()
    for df in frames:
        if "game_date" in df.columns:
            dates |= set(df["game_date"].astype(str).str[:10].str.replace("-", ""))
    return sorted(dates)


def load_calibration(sport: str | None) -> Optional[dict]:
    sport = sports_config.normalize_sport_key(sport)
    family = "calibration_json"
    data = load_json_artifact(sport, family)
    if data is None:
        return None
    cal = {
        "date": data.get("date"),
        "n_games": data.get("n_games"),
        "metrics": data.get("metrics", {}) or {},
        "calibration": data.get("calibration", {}) or {},
        "calibration_buckets": data.get("calibration_buckets", []) or [],
        "daily": data.get("daily", []) or [],
    }
    if sport == "mlb":
        cal["league_total"] = data.get("league_total")
        cal["evening_games_league"] = data.get("evening_games_league")
        cal["today_record"] = cal.get("daily", [{}])[-1] if cal.get("daily") else {}
    return cal


def load_model_monitor(sport: str | None) -> Optional[dict]:
    sport = sports_config.normalize_sport_key(sport)
    data = load_json_artifact(sport, "model_monitor")
    if data is None:
        return None
    return {
        "date": data.get("date") or latest_artifact_date(sport, "model_monitor"),
        "version": data.get("version"),
        "last_retrained": data.get("last_retrained"),
        "next_retrain": data.get("next_retrain"),
        "metrics": data.get("metrics", {}) or {},
        "drift_summary": data.get("drift_summary", {}) or {},
        "feature_drift": data.get("feature_drift", []) or [],
        "feature_coverage": data.get("feature_coverage", []) or [],
        "rolling_brier": data.get("rolling_brier", []) or [],
        "brier_baseline": data.get("brier_baseline"),
        "brier_baseline_label": data.get("brier_baseline_label"),
        "ensemble": data.get("ensemble", []) or [],
        "version_history": data.get("version_history", []) or [],
    }


def load_power_rankings(sport: str | None) -> pd.DataFrame:
    return load_csv_artifact(sport, "power_rankings")


def load_predictions_history(sport: str | None) -> pd.DataFrame:
    return load_csv_artifact(sport, "predictions_history")


def load_markets(sport: str | None) -> tuple[pd.DataFrame, Optional[dict]]:
    """(markets frame, meta dict) for the Totals & Run Lines page."""
    sport = sports_config.normalize_sport_key(sport)
    return (
        load_csv_artifact(sport, "markets"),
        load_json_artifact(sport, "markets_meta"),
    )


def load_markets_monitor(sport: str | None) -> Optional[dict]:
    return load_json_artifact(sport, "markets_monitor")


def load_participant_matchup(sport: str | None) -> pd.DataFrame:
    """The sport's participant-box rows (qb/goalie/player matchup), keyed
    by game_id. MLB returns an empty frame — its sp_* fields are inline on
    the games CSV."""
    sport = sports_config.normalize_sport_key(sport)
    if sports_config.participant_spec(sport).get("source") == "inline":
        return pd.DataFrame()
    return load_json_games(sport, "participant_matchup")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
@_cache
def _load_shap_cached(sport: str, game_id: str, demo: bool) -> pd.DataFrame:
    for path in reversed(_dated_paths(sport, "shap_game")):
        if str(game_id) in path.stem:
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def load_shap(sport: str | None, game_id: str) -> pd.DataFrame:
    """The SHAP rows for one game, newest matching artifact first.
    Empty frame when no file exists (card renders the no-data caption).
    The demo-mode flag is part of the cache key so enabling/disabling demo
    mode never serves a stale result."""
    sport = sports_config.normalize_sport_key(sport)
    if not game_id:
        return pd.DataFrame()
    return _load_shap_cached(sport, str(game_id), is_demo_mode())


def shap_dates(sport: str | None) -> list[str]:
    """Run dates that have at least one shap_game artifact (used to show
    SHAP availability for a slate date)."""
    sport = sports_config.normalize_sport_key(sport)
    return artifact_dates(sport, "shap_game")


# ---------------------------------------------------------------------------
# Participant-box adapter — one dict per card side
# ---------------------------------------------------------------------------
def participant_rows_for_game(sport: str | None, game_id: str,
                              matchup_df: pd.DataFrame | None = None) -> dict:
    """Per-game participant panel: {home: {...}|None, away: {...}|None}.

    Each side dict carries: name, stats (list of raw floats/None in
    registry `stats` order), labels (display labels), games (appearances
    backing the L10 averages or None). TBD/null pass through exactly as
    the backend emitted them — never fabricated.
    """
    sport = sports_config.normalize_sport_key(sport)
    spec = sports_config.participant_spec(sport)
    prefix = spec.get("prefix") or spec.get("renderer", "generic")
    stats_fields = spec.get("stats", [])
    labels = spec.get("stat_labels", stats_fields)

    def _side(prefix_side: str) -> Optional[dict]:
        row = None
        if spec.get("source") == "inline":
            # MLB: inline sp_* columns are read off the game row by the card
            # builder; the adapter only defines the shape here.
            return None
        if matchup_df is None or matchup_df.empty or "game_id" not in matchup_df.columns:
            return None
        match = matchup_df[matchup_df["game_id"].astype(str) == str(game_id)]
        if match.empty:
            return None
        row = match.iloc[-1]
        name = row.get(f"{prefix}_{prefix_side}_name")
        name = None if name is None or str(name).strip() in ("", "nan", "None") else str(name)
        stats = []
        for f in stats_fields:
            v = row.get(f"{prefix}_{prefix_side}_{f}")
            if v is None or (isinstance(v, float) and pd.isna(v)):
                stats.append(None)
            else:
                try:
                    stats.append(float(v))
                except (TypeError, ValueError):
                    stats.append(None)
        games = row.get(f"{prefix}_{prefix_side}_games")
        try:
            games = int(games) if games is not None and not pd.isna(games) else None
        except (TypeError, ValueError):
            games = None
        return {"name": name, "stats": stats, "labels": list(labels), "games": games}

    return {"home": _side("home"), "away": _side("away")}


def participant_box_html(sport: str | None, game, matchup_df=None) -> str:
    """Render one game's two participant panels as an HTML string.

    Pure (no Streamlit calls) so tests can assert on every sport's
    rendering. MLB reads inline ``sp_*`` fields off ``game``; the other
    sports resolve their matchup row through ``participant_rows_for_game``.
    TBD/null renders as 'TBD' with an explicit unavailable caption —
    never a fabricated participant.
    """
    sport = sports_config.normalize_sport_key(sport)
    spec = sports_config.participant_spec(sport)

    def _box(name, stats, labels, games) -> str:
        if name is None or str(name).strip() in ("", "None", "nan"):
            name_html = '<div class="pname">TBD</div>'
            stats_html = '<div class="pstats">Awaiting lineup</div>'
        else:
            name_html = f'<div class="pname">{name}</div>'
            parts = []
            for v, label in zip(stats, labels):
                parts.append(label + " —" if v is None
                             else f"{label} {v:.2f}")
            n = f" (last {games})" if games else ""
            stats_html = f'<div class="pstats">{" · ".join(parts)}{n}</div>'
        return f'<div class="fb-participant">{name_html}{stats_html}</div>'

    if spec.get("source") == "inline":
        # MLB: sp_* columns live on the game row itself.
        def _fmt_inline(field: str):
            v = game.get(field)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _name_inline(field: str):
            v = game.get(field)
            if v is None or str(v).strip() in ("", "nan", "None"):
                return None
            return str(v).strip()

        home = _box(_name_inline("sp_name_home"),
                    [_fmt_inline("sp_era_home"), _fmt_inline("sp_k9_home")],
                    spec.get("stat_labels", []), None)
        away = _box(_name_inline("sp_name_away"),
                    [_fmt_inline("sp_era_away"), _fmt_inline("sp_k9_away")],
                    spec.get("stat_labels", []), None)
        return f'<div class="fb-participants">{home}{away}</div>'

    gid = str(game.get("game_id", "") if hasattr(game, "get") else game)
    panel = participant_rows_for_game(sport, gid, matchup_df)

    def _render(side):
        if side is None:
            return ('<div class="fb-participant"><div class="pname">TBD</div>'
                    '<div class="pstats">Matchup data unavailable</div></div>')
        return _box(side["name"], side["stats"], side["labels"],
                    side.get("games"))

    return (f'<div class="fb-participants">{_render(panel["home"])}'
            f'{_render(panel["away"])}</div>')


# ---------------------------------------------------------------------------
# Sidebar / shared UI helpers
# ---------------------------------------------------------------------------
def get_sport() -> str:
    """The active sport key from Streamlit session state (default MLB).
    Falls back to the registry default outside a Streamlit runtime."""
    if st is None:
        return sports_config.DEFAULT_SPORT
    try:
        return sports_config.normalize_sport_key(st.session_state.get("sport"))
    except Exception:
        return sports_config.DEFAULT_SPORT


def source_label() -> str:
    """Human description of where artifacts are being read from."""
    sport = get_sport()
    try:
        base = sports_config.data_delivery_dir(sport)
        exists = base.is_dir()
    except Exception:
        exists = False
    return f"Committed artifacts · sports/{sport}/data_delivery" if exists else \
        f"No artifacts yet · sports/{sport}/data_delivery (expected)"


def start_time_et(iso) -> str:
    """Format a UTC ISO start as 'h:mm ET', or '' when absent/unparseable."""
    if not iso:
        return ""
    try:
        ts = pd.to_datetime(iso, errors="coerce", utc=True)
        if pd.isna(ts):
            return ""
        # Portable 12-hour formatting: the glibc-only ``%-I`` directive is
        # unsupported on Windows (raises -> the caller returns ''), so
        # format zero-padded and strip the leading zero explicitly.
        hour12 = ts.tz_convert("America/New_York").strftime("%I:%M %p ET")
        return hour12[1:] if hour12.startswith("0") else hour12
    except (TypeError, ValueError):
        return ""


def has_any_artifacts(sport: str | None) -> bool:
    """True when at least one real artifact file exists for the sport
    (sample fixtures do NOT count — they back tests/offline demo only)."""
    sport = sports_config.normalize_sport_key(sport)
    base = sports_config.data_delivery_dir(sport)
    if not base.is_dir():
        return False
    return any(base.glob(pattern) for pattern in
               sports_config.artifact_patterns(sport).values())


def last_updated_line(sport: str | None) -> str:
    """Sidebar 'Last updated' caption from the newest artifact run date."""
    sport = sports_config.normalize_sport_key(sport)
    best = None
    for family in sports_config.artifact_patterns(sport):
        d = latest_artifact_date(sport, family)
        if d and (best is None or d > best):
            best = d
    if not best:
        return "No artifacts published yet"
    pretty = f"{best[:4]}-{best[4:6]}-{best[6:]}"
    return f"Last updated {pretty}"


# ---------------------------------------------------------------------------
# Shared page rendering (CSS, sidebar blocks, no-data states, SHAP chart)
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Reference dark-theme CSS (same palette/classes as the MLB app)."""
    if st is None:
        return
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PAGE_BG}; }}
        .block-container {{ padding-top: 1.4rem; }}
        .fb-card {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 14px 16px 12px; margin-bottom: 12px;
        }}
        .fb-top {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
        .fb-top .spacer {{ flex: 1; }}
        .fb-tag {{ color: {SLATE}; font-size: 0.78rem; }}
        .fb-pill {{ border-radius: 999px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; }}
        .fb-pill.correct {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-pill.miss {{ background: rgba(239,68,68,.15); color: #F87171; }}
        .fb-pill.final {{ background: rgba(59,130,246,.15); color: #93C5FD; }}
        .fb-pill.upset {{ background: rgba(245,158,11,.18); color: #FBBF24; }}
        .fb-pill.coinflip {{ background: rgba(250,204,21,.18); color: #FDE047; }}
        .fb-pill.live {{ background: rgba(239,68,68,.18); color: #F87171; }}
        .fb-pill.pick {{ background: rgba(59,130,246,.9); color: #fff; }}
        .fb-score {{ display: flex; align-items: center; justify-content: center; gap: 26px; margin: 6px 0 4px; }}
        .fb-score .side {{ text-align: center; }}
        .fb-score .num-wrap {{ display: flex; align-items: center; justify-content: center; }}
        .fb-score .win-bar {{ width: 4px; border-radius: 3px; height: 26px; margin-right: 7px; }}
        .fb-score .num {{ font-size: 2.1rem; font-weight: 800; line-height: 1; color: {TEXT}; }}
        .fb-score .abbr {{ color: {SLATE}; font-size: 0.8rem; letter-spacing: 1px; }}
        .fb-score .mid {{ color: {SLATE}; font-size: 0.95rem; font-weight: 600; }}
        .fb-team {{ display: flex; align-items: center; gap: 8px; margin: 7px 0 3px; }}
        .fb-accent {{ width: 4px; border-radius: 3px; align-self: stretch; min-height: 26px; }}
        .fb-team .name {{ font-weight: 700; color: {TEXT}; }}
        .fb-team .sub {{ color: {SLATE}; font-size: 0.8rem; }}
        .fb-team .pct {{ margin-left: auto; font-weight: 800; font-size: 1.05rem; }}
        .fb-bar {{ height: 7px; border-radius: 5px; background: #0F172A; overflow: hidden; display: flex; margin: 3px 0 2px; }}
        .fb-bar .fill {{ height: 100%; }}
        .fb-pregame {{ text-align: center; color: #64748B; font-size: 0.78rem; margin: 6px 0 8px; }}
        .fb-participants {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 4px 0; }}
        .fb-participant {{ background: #0F172A; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 9px; }}
        .fb-participant .pname {{ font-weight: 700; color: {TEXT}; font-size: 0.86rem; }}
        .fb-participant .pstats {{ color: {SLATE}; font-size: 0.76rem; margin-top: 2px; }}
        .fb-venue {{ color: {SLATE}; font-size: 0.8rem; margin: 6px 0 2px; }}
        .fb-banner {{ border-radius: 10px; padding: 8px 12px; text-align: center; font-size: 0.85rem; font-weight: 700; }}
        .fb-banner.green {{ background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.45); color: #34D399; }}
        .fb-banner.red {{ background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.45); color: #F87171; }}
        .fb-banner.amber {{ background: rgba(250,204,21,.10); border: 1px solid rgba(250,204,21,.5); color: #FDE047; }}
        .fb-banner.blue {{ background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.45); color: #93C5FD; }}
        .fb-kpi {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px; text-align: center; }}
        .fb-kpi .label {{ color: {SLATE}; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }}
        .fb-kpi .value {{ font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px; }}
        .fb-kpi .cap {{ color: {SLATE}; font-size: 0.72rem; }}
        .fb-box {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; }}
        .fb-status-pill {{ border-radius: 999px; padding: 1px 10px; font-size: 0.72rem; font-weight: 700; }}
        .fb-status-pill.ok {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-status-pill.warn {{ background: rgba(245,158,11,.15); color: #FBBF24; }}
        .fb-status-pill.alert {{ background: rgba(239,68,68,.15); color: #F87171; }}
        table.fb-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
        table.fb-table th {{ text-align: left; color: {SLATE}; font-size: 0.72rem; letter-spacing: 1px; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
        table.fb-table td {{ padding: 8px 10px; border-bottom: 1px solid rgba(30,41,59,.6); color: {TEXT}; }}
        table.fb-table tr:nth-child(even) td {{ background: rgba(30,41,59,.25); }}
        .fb-accent-cell {{ display: inline-flex; align-items: center; gap: 8px; }}
        .fb-accent-line {{ width: 4px; border-radius: 3px; height: 20px; display: inline-block; }}
        [data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
        [data-testid="stSidebarHeader"] {{ order: 1; flex-shrink: 0; }}
        [data-testid="stSidebarUserContent"] {{ order: 2; flex-shrink: 0; }}
        [data-testid="stSidebarNav"] {{ order: 3; flex-shrink: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    """Sidebar brand block — title/subtitle from the ACTIVE sport's registry
    entry, rendered above the dashboard list via the sidebar-reorder CSS."""
    if st is None:
        return
    cfg = sports_config.resolve_sport(get_sport())
    st.markdown(
        f"<div style='font-size:1.25rem;font-weight:800;color:#E2E8F0;'>"
        f"{cfg['emoji']} {cfg['title']}</div>"
        f"<div style='color:#64748B;font-size:0.8rem;margin-bottom:10px;'>"
        f"{cfg['subtitle']}</div>",
        unsafe_allow_html=True,
    )


def render_source_note() -> None:
    """Defensive empty-state note (only when the sport ships no artifacts)."""
    if st is None:
        return
    if not has_any_artifacts(get_sport()):
        st.caption("⚠️ No artifacts found for this sport yet")


def render_last_updated(sport_key: str | None = None) -> None:
    """Small muted 'Last updated' caption wired to the active sport."""
    if st is None:
        return
    st.caption(last_updated_line(sport_key or get_sport()))


def no_data_state(message: str, hint: str = "") -> None:
    """Explicit no-data state: an info box plus an optional hint caption.
    Used by every page when an artifact family is missing — never a
    synthesized table or another sport's data."""
    st.info(message)
    if hint:
        st.caption(hint)


def demo_banner_html(sport: str | None, families: list[str]) -> Optional[str]:
    """Pure builder: the DEMO DATA banner HTML when the given page's
    artifact families are being served from the tracked sample fixtures
    (demo mode on + no real artifact), else None. Testable without a
    Streamlit runtime."""
    sport = sports_config.normalize_sport_key(sport)
    if not is_demo_mode():
        return None
    served_sample = any(artifact_source(sport, f) == "sample" for f in families)
    if not served_sample:
        return None
    return ('<div class="fb-banner amber" style="margin-bottom:10px;">'
            '🧪 DEMO DATA — sample fixtures (FRONTEND_DEMO_MODE=1); '
            'no committed artifacts exist for this sport yet.</div>')


def render_demo_banner(sport: str | None, families: list[str]) -> None:
    """Render the visible DEMO DATA banner on pages served sample data."""
    html = demo_banner_html(sport, families)
    if html and st is not None:
        st.markdown(html, unsafe_allow_html=True)


def shap_chart(df: pd.DataFrame):
    """Horizontal signed-SHAP bar chart (altair), or None when empty."""
    if df is None or df.empty or "shap_value" not in df.columns:
        return None
    try:
        import altair as alt

        d = df.copy()
        d["shap_value"] = pd.to_numeric(d["shap_value"], errors="coerce")
        d = d.dropna(subset=["shap_value"])
        d = d.reindex(d["shap_value"].abs().sort_values(ascending=False).index).head(15)
        d = d.iloc[::-1]
        chart = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="SHAP impact on win probability"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.shap_value > 0,
                    alt.value(PRIMARY),
                    alt.value(RED),
                ),
            )
            .properties(height=280)
        )
        return chart
    except Exception:
        return None


def show_chart(chart) -> None:
    """Render an altair chart when both altair and Streamlit are present."""
    if chart is not None and st is not None:
        st.altair_chart(chart, width="stretch")


def nearest_valid_date(valid: list[str] | tuple[str, ...],
                       target: str | None = None) -> Optional[str]:
    """Nearest available YYYYMMDD date to `target` (default: today ET).
    Falls back to the latest available date when none is in the future."""
    if not valid:
        return None
    if target is None:
        target = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d")
    try:
        t = pd.to_datetime(target, format="%Y%m%d")
        dates = [pd.to_datetime(v, format="%Y%m%d") for v in valid]
    except (TypeError, ValueError):
        return valid[-1]
    upcoming = [d for d in dates if d >= t]
    pool = upcoming or dates
    return min(pool, key=lambda d: abs((d - t).days)).strftime("%Y%m%d")
