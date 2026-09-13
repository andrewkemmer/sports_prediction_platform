"""Sport registry for the shared Streamlit frontend (Phase 6).

Single source of truth for:

* which sports exist and their labels/emojis/branding,
* the artifact directory each sport publishes from
  (``sports/<sport>/data_delivery``),
* which dashboards each sport supports (sidebar order contract),
* the artifact-family glob patterns the loader resolves,
* the participant-box mapping (matchup artifact family + renderer key +
  per-side stat fields) for Today's Games cards,
* market-page semantics (the page title is identical for every sport;
  content structure adapts to the sport's market artifact columns).

This module stays free of streamlit/pandas imports so backend tests can
import it without a Streamlit runtime. The sidebar sport toggle in
``app.py`` writes ``st.session_state["sport"]``; ``readers/loaders.py`` resolves
artifact paths and adapters through this registry.
"""

from pathlib import Path

# Sidebar page order contract — mirrored by app.py's literal `pages` list.
ALL_PAGE_URL_PATHS = [
    "todays-games",
    "power-rankings",
    "calibration",
    "model-monitor",
    "markets",
]

MARKETS_PAGE_LABEL = "Totals & Run Lines"

_SPORT_DIR = "sports/{sport}/data_delivery"


def _common_json_families(prefix: str, participant_family: str) -> dict:
    """Artifact-family globs shared by the NFL/NHL/NBA JSON-artifact sports.

    ``prefix`` is the sport's filename prefix (e.g. ``nfl``); participant
    families are ``qb_matchup`` / ``goalie_matchup`` / ``player_matchup``.
    """
    return {
        "moneyline_json": f"{prefix}_moneyline_v1_*.json",
        "feature_json": f"{prefix}_feature_v1_*.json",
        "calibration_json": f"{prefix}_calibration_*.json",
        "predictions_history": f"{prefix}_predictions_history_*.csv",
        "power_rankings": f"{prefix}_power_rankings_*.csv",
        "markets": f"{prefix}_run_engine_markets_*.csv",
        "markets_meta": f"{prefix}_run_engine_markets_*.meta.json",
        "markets_monitor": f"{prefix}_run_engine_monitor_*.json",
        "model_monitor": f"{prefix}_model_monitor_*.json",
        "participant_matchup": f"{prefix}_{participant_family}_*.json",
        "shap_game": f"{prefix}_shap_game_*.csv",
    }


SPORTS = {
    "mlb": {
        "label": "MLB",
        "emoji": "⚾",
        "title": "MLB Predictions",
        "subtitle": "MLB betting model dashboard",
        "artifact_dir": _SPORT_DIR.format(sport="mlb"),
        "pages": list(ALL_PAGE_URL_PATHS),
        # MLB publishes the card board as a CSV snapshot with inline
        # starting-pitcher columns (sp_*) — no separate matchup artifact.
        "artifacts": {
            "games": "todays_games_*.csv",
            "calibration_json": "calibration_*.json",
            "predictions_history": "predictions_history_*.csv",
            "power_rankings": "power_rankings_*.csv",
            "markets": "run_engine_markets_*.csv",
            "markets_meta": "run_engine_markets_*.meta.json",
            "markets_monitor": "run_engine_monitor_*.json",
            "model_monitor": "model_monitor_*.json",
            "rolling_brier": "rolling_brier_*.json",
            # §22 Amendment 11 support artifact: game_pk → team-names
            # bridge for the markets ladder (markets rows carry game_pk
            # only) and board-finals reconciliation.
            "game_level_features": "game_level_features_*.csv",
            "shap_game": "shap_game_*.csv",
        },
        "participant": {
            "renderer": "pitcher",
            "prefix": "sp",  # artifact field prefix (sp_era_home, ...)
            "source": "inline",  # sp_* columns live on the games CSV
            "stats": ["era", "k9"],
            "stat_labels": ["ERA", "K/9"],
        },
        "market_kind": "mlb_grid",
    },
    "nfl": {
        "label": "NFL",
        "emoji": "🏈",
        "title": "NFL Predictions",
        "subtitle": "NFL betting model dashboard",
        "artifact_dir": _SPORT_DIR.format(sport="nfl"),
        "pages": list(ALL_PAGE_URL_PATHS),
        "artifacts": _common_json_families("nfl", "qb_matchup"),
        "participant": {
            "renderer": "quarterback",
            "prefix": "qb",  # artifact field prefix (qb_home_rating, ...)
            "source": "qb_matchup",
            "stats": ["rating", "td_per_game"],
            "stat_labels": ["Passer rtg L10", "TD/gm L10"],
        },
        "market_kind": "lines_row",
    },
    "nhl": {
        "label": "NHL",
        "emoji": "🏒",
        "title": "NHL Predictions",
        "subtitle": "NHL betting model dashboard",
        "artifact_dir": _SPORT_DIR.format(sport="nhl"),
        "pages": list(ALL_PAGE_URL_PATHS),
        "artifacts": _common_json_families("nhl", "goalie_matchup"),
        "participant": {
            "renderer": "goalie",
            "prefix": "goalie",  # artifact field prefix (goalie_home_gaa, ...)
            "source": "goalie_matchup",
            "stats": ["save_pct", "gaa"],
            "stat_labels": ["Sv% L10", "GAA L10"],
        },
        "market_kind": "lines_row",
    },
    "nba": {
        "label": "NBA",
        "emoji": "🏀",
        "title": "NBA Predictions",
        "subtitle": "NBA betting model dashboard",
        "artifact_dir": _SPORT_DIR.format(sport="nba"),
        "pages": list(ALL_PAGE_URL_PATHS),
        "artifacts": _common_json_families("nba", "player_matchup"),
        "participant": {
            "renderer": "scorer",
            "prefix": "player",  # artifact field prefix (player_home_ppg, ...)
            "source": "player_matchup",
            "stats": ["ppg", "apg"],
            "stat_labels": ["PPG L10", "APG L10"],
        },
        "market_kind": "lines_row",
    },
}

DEFAULT_SPORT = "mlb"


def resolve_sport(sport_key: str) -> dict:
    """Resolve a sport key to its registry entry, never raising.

    Unknown / missing keys fall back to DEFAULT_SPORT so a degraded state
    always renders a valid sport.
    """
    return SPORTS[normalize_sport_key(sport_key)]


def normalize_sport_key(sport_key: str) -> str:
    """Normalize a sport key; unknown/empty values fall back to the default."""
    key = str(sport_key or "").strip().lower()
    if key not in SPORTS:
        key = DEFAULT_SPORT
    return key


def is_unknown_sport(sport_key: str) -> bool:
    """True only for a genuinely unknown NON-EMPTY sport value."""
    key = str(sport_key or "").strip().lower()
    if not key or key == "none":
        return False
    return key not in SPORTS


def artifact_patterns(sport_key: str) -> dict:
    """Family name → glob pattern under the sport's data_delivery dir."""
    return dict(resolve_sport(sport_key).get("artifacts", {}) or {})


def data_delivery_dir(sport_key: str) -> Path:
    """Local data_delivery directory for the sport, resolved from the
    repo root (the parent of this frontend/ package)."""
    return Path(__file__).resolve().parents[2] / resolve_sport(sport_key)["artifact_dir"]


def sample_artifacts_dir() -> Path:
    """Tracked sample-artifact directory used by tests (and any offline
    demo mode): tests/fixtures/sample_artifacts. Layout mirrors a real
    data_delivery tree per sport so loaders resolve identically."""
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_artifacts"


def active_page_url_paths(sport_key: str) -> list[str]:
    """Ordered subset of ALL_PAGE_URL_PATHS the given sport renders.

    Every registered sport currently ships the full five-page set; a
    sport with an empty/partial page list degrades to the full set so the
    sidebar never renders blank.
    """
    cfg = resolve_sport(sport_key)
    allowed = cfg.get("pages", []) or []
    active = [p for p in ALL_PAGE_URL_PATHS if p in allowed]
    if not active:
        return list(ALL_PAGE_URL_PATHS)
    return active


def participant_spec(sport_key: str) -> dict:
    """The sport's participant-box mapping (renderer key, source family,
    stat fields and display labels)."""
    return dict(resolve_sport(sport_key).get("participant", {}) or {})


def market_kind(sport_key: str) -> str:
    """Which market semantic view the Totals & Run Lines page renders."""
    return resolve_sport(sport_key).get("market_kind", "lines_row")
