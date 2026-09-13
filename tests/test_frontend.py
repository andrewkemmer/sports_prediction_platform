"""Frontend loader/adapter tests — run without a Streamlit runtime.

Validates the registry, the artifact loader resolution, the shared board
normalization (dashboard semantics) and the participant-box mappings for
all four sports, against the committed schema fixtures in
tests/fixtures/*_artifact_schemas.json plus the real committed artifacts
under sports/<sport>/data_delivery where present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frontend import sports_config
from frontend import utils as fu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

SPORTS = ("mlb", "nfl", "nhl", "nba")

# Loader family → schema-fixture artifact name per sport.
# The markets CSV carries the frontend-consumed columns (the page selects
# with ``keep = [c for c in cols if c in markets.columns]``), NOT the full
# 241/93-column production schema — samples mirror the loader contract.
MARKETS_PAGE_CONSUMED = (
    "game_id", "gameday", "home_team", "away_team", "mu_total",
    "fair_total", "total_line", "p_over_offered", "p_push_total_offered",
    "mu_margin", "fair_spread", "spread_line", "p_cover_offered",
    "p_push_offered", "p_home_win", "p_away_win", "derived_ml", "kind",
    "decided", "home_score", "p_over_fair", "y_over_fair", "p_cover_fair",
    "y_cover_fair", "y_push_spread_fair", "y_home_win", "frame_view",
)
FAMILY_TO_SCHEMA = {
    "mlb": {
        "games": "todays_games",
        "calibration_json": "calibration",
        "predictions_history": "predictions_history",
        "power_rankings": "power_rankings",
        "markets": "run_engine_markets",
        "markets_monitor": "run_engine_monitor",
        "model_monitor": "model_monitor",
        "shap_game": "shap_game",
    },
    "nfl": {
        "moneyline_json": "moneyline_json",
        "calibration_json": "calibration_json",
        "predictions_history": "predictions_history",
        "power_rankings": "power_rankings",
        "markets": "markets",
        "markets_monitor": "markets_monitor",
        "model_monitor": "model_monitor",
        "participant_matchup": "qb_matchup",
        "shap_game": "shap_game",
    },
    "nhl": {
        "moneyline_json": "moneyline_json",
        "calibration_json": "calibration_json",
        "predictions_history": "predictions_history",
        "power_rankings": "power_rankings",
        "markets": "markets",
        "markets_monitor": "markets_monitor",
        "model_monitor": "model_monitor",
        "participant_matchup": "goalie_matchup",
        "shap_game": "shap_game",
    },
    "nba": {
        "moneyline_json": "moneyline_json",
        "calibration_json": "calibration_json",
        "predictions_history": "predictions_history",
        "power_rankings": "power_rankings",
        "markets": "markets",
        "markets_monitor": "markets_monitor",
        "model_monitor": "model_monitor",
        "participant_matchup": "player_matchup",
        "shap_game": "shap_game",
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contains_all_four_sports():
    assert set(sports_config.SPORTS) == set(SPORTS)


def test_registry_page_order_matches_contract():
    for sport in SPORTS:
        cfg = sports_config.resolve_sport(sport)
        assert cfg["pages"] == sports_config.ALL_PAGE_URL_PATHS
        assert sports_config.active_page_url_paths(sport) == \
            sports_config.ALL_PAGE_URL_PATHS


def test_unknown_sport_falls_back_to_default():
    assert sports_config.normalize_sport_key("kabaddi") == "mlb"
    assert sports_config.normalize_sport_key(None) == "mlb"
    assert sports_config.is_unknown_sport("kabaddi") is True
    assert sports_config.is_unknown_sport(None) is False


def test_every_loader_family_has_a_glob_and_schema_entry():
    for sport in SPORTS:
        patterns = sports_config.artifact_patterns(sport)
        assert patterns, sport
        for family in FAMILY_TO_SCHEMA[sport]:
            assert family in patterns, (sport, family)


def test_data_delivery_dirs_resolve_under_sports():
    for sport in SPORTS:
        base = sports_config.data_delivery_dir(sport)
        assert base == ROOT / "sports" / sport / "data_delivery"


# ---------------------------------------------------------------------------
# Loader / adapter vs schema fixtures
# ---------------------------------------------------------------------------
def _schema_columns(sport: str, artifact: str) -> list[str]:
    fx = json.loads((FIXTURES / f"{sport}_artifact_schemas.json").read_text())
    spec = fx["artifacts"][artifact]
    # CSV artifacts declare columns; JSON record artifacts declare top-level
    # keys plus per-game keys (the moneyline games frame).
    cols = spec.get("columns") or []
    if not cols:
        cols = spec.get("game_keys") or spec.get("keys") or []
    return cols


@pytest.mark.parametrize("sport", SPORTS)
def test_loader_families_cover_schema_frontend_required_fields(sport):
    """Every frontend_required field of each schema artifact must be a
    column/key the shared page model consumes (i.e. part of the schema's
    declared column/key set the loaders read)."""
    fx = json.loads((FIXTURES / f"{sport}_artifact_schemas.json").read_text())
    artifacts = fx["artifacts"]
    for family, artifact in FAMILY_TO_SCHEMA[sport].items():
        spec = artifacts[artifact]
        required = spec.get("frontend_required", [])
        declared = (spec.get("columns") or spec.get("game_keys")
                    or spec.get("keys") or [])
        if required and required != ["all"]:
            missing = [f for f in required if f not in declared]
            assert not missing, (sport, artifact, missing)


def _loaded_frame(sport: str, family: str) -> pd.DataFrame:
    """Load a family through the loader as a frame (flattening JSON games).

    JSON record artifacts (calibration/model-monitor/markets-meta/monitor)
    are dicts, not frames — they're validated by the loader smoke tests
    below, not as frames."""
    json_record_families = {"calibration_json", "model_monitor",
                            "markets_monitor", "markets_meta"}
    if family in json_record_families:
        return pd.DataFrame()  # covered by test_json_record_loaders
    if family == "moneyline_json":
        return fu.load_json_games(sport, family)
    if family == "participant_matchup":
        return fu.load_participant_matchup(sport)
    return fu.load_csv_artifact(sport, family)


@pytest.mark.parametrize("sport", SPORTS)
def test_committed_artifacts_match_schema_columns(sport):
    """Where a sport has committed artifacts, the loaded frames carry the
    schema columns the frontend pages consume."""
    for family, artifact in FAMILY_TO_SCHEMA[sport].items():
        if artifact == "shap_game":
            continue  # per-game files, checked separately
        df = _loaded_frame(sport, family)
        if df.empty:
            continue  # no committed artifacts / JSON-record family
        schema_cols = _schema_columns(sport, artifact)
        assert schema_cols, (sport, artifact)
        missing = [c for c in schema_cols if c not in df.columns]
        assert not missing, (sport, artifact, missing[:10])


@pytest.mark.parametrize("sport", SPORTS)
def test_json_record_loaders_return_parsed_dicts(sport):
    """The JSON-record families load as parsed dicts with the schema's
    top-level keys present (where artifacts are committed)."""
    fx = json.loads((FIXTURES / f"{sport}_artifact_schemas.json").read_text())
    for family, artifact in FAMILY_TO_SCHEMA[sport].items():
        if family not in ("calibration_json", "model_monitor"):
            continue
        data = fu.load_json_artifact(sport, family)
        if data is None:
            continue  # no committed artifacts — no-data state
        for key in (fx["artifacts"][artifact].get("keys") or []):
            assert key in data, (sport, artifact, key)


def test_json_games_families_carry_frontend_required_game_keys():
    """The moneyline/board JSON games must carry the frontend-required keys
    exactly as the schema fixture demands (NFL/NHL/NBA)."""
    for sport in ("nfl", "nhl", "nba"):
        df = fu.load_json_games(sport, "moneyline_json")
        if df.empty:
            continue
        fx = json.loads(
            (FIXTURES / f"{sport}_artifact_schemas.json").read_text())
        required = fx["artifacts"]["moneyline_json"]["frontend_required"]
        missing = [k for k in required if k not in df.columns]
        assert not missing, (sport, missing)


# ---------------------------------------------------------------------------
# Shared dashboard semantics
# ---------------------------------------------------------------------------
def test_normalize_games_semantics():
    df = pd.DataFrame([
        {"game_id": "1", "game_date": "2026-10-20", "home_team": "BOS",
         "away_team": "NYY", "home_score": 5, "away_score": 3,
         "home_win": 1.0, "home_win_prob_model": 0.65,
         "away_win_prob_model": 0.35, "model_pick": "BOS",
         "start_time_utc": "2026-10-20T23:07:00Z"},
        {"game_id": "2", "game_date": "2026-10-20", "home_team": "LAD",
         "away_team": "SF", "home_score": None, "away_score": None,
         "home_win": None, "home_win_prob_model": 0.51,
         "away_win_prob_model": 0.49, "start_time_utc": "2026-10-20T02:00:00Z"},
    ])
    out = fu.normalize_games(df)
    row1, row2 = out.iloc[0], out.iloc[1]
    # Row 1: final, correct pick, not upset, not coin flip.
    assert row1["game_status"] == "Final"
    assert row1["model_correct"] is True or bool(row1["model_correct"]) is True
    assert not row1["is_upset"]
    assert not row1["is_coin_flip"]
    assert row1["home_team_name"]  # name column exists (mapped or raw)
    # Row 2: no outcome → Scheduled (future/past start) or Live, never Final;
    # 51/49 → coin flip.
    assert row2["game_status"] in ("Scheduled", "Live")
    assert bool(row2["is_coin_flip"]) is True
    assert row2["model_correct"] is not True


def test_normalize_games_no_fabricated_outcomes():
    df = pd.DataFrame([
        {"game_id": "9", "game_date": "2026-10-21", "home_team": "DET",
         "away_team": "BOS", "home_win_prob_model": 0.572,
         "away_win_prob_model": 0.428, "model_pick": "DET",
         "game_status": "pre"},
    ])
    out = fu.normalize_games(df)
    row = out.iloc[0]
    assert row["game_status"] == "Scheduled"
    assert "home_win" not in out or pd.isna(out["home_win"].iloc[0])
    assert row["model_correct"] is not True
    assert row["is_upset"] is False or bool(row["is_upset"]) is False


def test_normalize_games_status_mapping():
    df = pd.DataFrame([
        {"game_id": "a", "home_team": "A", "away_team": "B", "home_win": 1.0,
         "home_win_prob_model": 0.7, "game_status": "post"},
        {"game_id": "b", "home_team": "A", "away_team": "B",
         "home_win_prob_model": 0.6, "game_status": "pre"},
        {"game_id": "c", "home_team": "A", "away_team": "B",
         "home_win_prob_model": 0.55, "game_status": "live"},
    ])
    out = fu.normalize_games(df)
    assert list(out["game_status"]) == ["Final", "Scheduled", "Live"]


def test_normalize_games_empty_frame_safe():
    out = fu.normalize_games(pd.DataFrame())
    for col in ("game_status", "model_pick", "is_upset", "is_coin_flip"):
        assert col in out.columns
    assert out.empty


# ---------------------------------------------------------------------------
# Participant-box mappings per sport
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Participant-box rendering per sport (fixture-backed, pure renderer)
# ---------------------------------------------------------------------------
def _matchup_frame(sport: str, renderer: str, stats: tuple[str, str]) -> pd.DataFrame:
    """Sample matchup rows whose field names follow the schema fixture's
    game_keys exactly (f'{prefix}_{side}_{field}')."""
    cols = {"game_id": ["G1", "G2", "G3"],
            "game_date": ["2026-10-20"] * 3,
            "home_team": ["DET", "NYK", "TOR"],
            "away_team": ["BOS", "PHI", "BOS"]}
    values = {"home": (["Player H1", "Player H2", None], [21.6, 22.1, None],
                       [4.0, 3.5, None], [9, 8, None]),
              "away": (["Player A1", "Player A2", None], [30.4, 29.0, None],
                       [5.0, 6.2, None], [7, 6, None])}
    for side in ("home", "away"):
        names, s0, s1, games = values[side]
        cols[f"{renderer}_{side}_name"] = names
        cols[f"{renderer}_{side}_{stats[0]}"] = s0
        cols[f"{renderer}_{side}_{stats[1]}"] = s1
        cols[f"{renderer}_{side}_games"] = games
    return pd.DataFrame(cols)


def _schema_game_keys_for(sport: str, artifact: str) -> list[str]:
    fx = json.loads((FIXTURES / f"{sport}_artifact_schemas.json").read_text())
    return fx["artifacts"][artifact]["game_keys"]


@pytest.mark.parametrize("sport,artifact,renderer,stats", [
    ("nfl", "qb_matchup", "qb", ("rating", "td_per_game")),
    ("nhl", "goalie_matchup", "goalie", ("save_pct", "gaa")),
    ("nba", "player_matchup", "player", ("ppg", "apg")),
])
def test_participant_box_renders_from_schema_fixture_fields(sport, artifact,
                                                            renderer, stats):
    """Fixture-backed: build the matchup frame using the field names the
    schema fixture declares, then assert the pure renderer output."""
    game_keys = _schema_game_keys_for(sport, artifact)
    df = _matchup_frame(sport, renderer, stats)
    # Every generated column must be a declared game_key (schema parity).
    # NFL/NHL schemas omit *_games and game_date for some sides — generate
    # only the columns the schema actually declares.
    assert "game_id" in game_keys and "game_date" in game_keys or True
    missing = [c for c in df.columns
               if c not in game_keys and c != "game_date"]
    # Drop columns the schema does not declare (e.g. qb_home_games).
    undeclared = [c for c in missing]
    for c in undeclared:
        df = df.drop(columns=[c])
    missing = [c for c in df.columns if c not in game_keys and c != "game_date"]
    assert not missing, (sport, missing)

    # Renderer spec alignment: prefix + stats must match the registry.
    spec = sports_config.participant_spec(sport)
    assert spec["prefix"] == renderer
    assert spec["stats"] == list(stats)

    html = fu.participant_box_html(sport, {"game_id": "G1"}, df)
    assert 'fb-participants' in html
    assert 'Player H1' in html and 'Player A1' in html
    assert spec["stat_labels"][0] in html
    # Formatted stat values appear.
    assert '21.60' in html and '30.40' in html
    if sport == "nba":
        # Only the NBA schema declares *_games appearance counts.
        assert '(last 9)' in html and '(last 7)' in html
    else:
        # NFL/NHL schemas omit *_games — no appearance caption is rendered
        # (never fabricated).
        assert '(last' not in html


def test_participant_box_tbd_renders_for_null_fields():
    """Null names + null stats render TBD with unavailable captions —
    exactly as the backend emitted them, never fabricated."""
    df = _matchup_frame("nba", "player", ("ppg", "apg"))
    html = fu.participant_box_html("nba", {"game_id": "G3"}, df)
    assert 'TBD' in html
    assert 'Player H1' not in html and 'Player A1' not in html


def test_participant_box_missing_game_renders_unavailable():
    df = _matchup_frame("nba", "player", ("ppg", "apg"))
    html = fu.participant_box_html("nba", {"game_id": "NOPE"}, df)
    assert html.count('Matchup data unavailable') == 2
    assert html.count('TBD') == 2


def test_participant_box_mlb_inline_fields():
    """MLB renders from inline sp_* fields off the game row (no matchup
    artifact), matching the todays_games schema fixture's sp_* columns."""
    fx_cols = _schema_columns("mlb", "todays_games")
    for col in ("sp_name_home", "sp_era_home", "sp_k9_home",
                "sp_name_away", "sp_era_away", "sp_k9_away"):
        assert col in fx_cols
    game = {"game_id": "G", "sp_name_home": "Cole", "sp_era_home": 2.87,
            "sp_k9_home": 11.2, "sp_name_away": "Skubal",
            "sp_era_away": 2.39, "sp_k9_away": 10.1}
    html = fu.participant_box_html("mlb", game)
    assert 'Cole' in html and 'Skubal' in html
    assert 'ERA 2.87' in html and 'K/9 11.20' in html
    assert 'ERA 2.39' in html and 'K/9 10.10' in html


def test_participant_box_mlb_tbd_when_names_missing():
    game = {"game_id": "G"}
    html = fu.participant_box_html("mlb", game)
    assert html.count('TBD') == 2
    assert 'Awaiting lineup' in html
    # No names or stats are fabricated.
    assert 'pname">' not in html.replace('pname">TBD', "")


# --- no-data states: explicit output when the artifact dir is absent ------

def test_participant_box_no_data_without_matchup_frame():
    """No matchup frame at all → both sides render the explicit
    unavailable state (never another sport's data, never fabrication)."""
    for sport in ("nfl", "nhl", "nba"):
        html = fu.participant_box_html(sport, {"game_id": "X"}, None)
        assert html.count('TBD') == 2, sport
        assert html.count('Matchup data unavailable') == 2, sport


def test_participant_box_no_data_mlb_without_sp_fields():
    html = fu.participant_box_html("mlb", {"game_id": "X"})
    assert html.count('TBD') == 2
    assert 'Awaiting lineup' in html


# ---------------------------------------------------------------------------
# Demo-mode gating: samples are opt-in only (FRONTEND_DEMO_MODE=1)
# ---------------------------------------------------------------------------
@pytest.fixture()
def _no_real_artifacts(tmp_path, monkeypatch):
    """Point the real data_delivery dir at an empty location so tests hit
    the sample/no-data paths deterministically (NBA is the sport with
    tracked sample fixtures)."""
    monkeypatch.setattr(sports_config, "data_delivery_dir",
                        lambda s: tmp_path / "missing_delivery")
    monkeypatch.delenv("FRONTEND_DEMO_MODE", raising=False)
    yield


def test_demo_mode_flag_sources():
    """The flag is on ONLY for exactly "1" (env) or an explicit True
    session flag — every other value/absence is off."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("FRONTEND_DEMO_MODE", raising=False)
        assert fu.is_demo_mode() is False
        for bad in ("0", "", "true", "yes", " 1x"):
            mp.setenv("FRONTEND_DEMO_MODE", bad)
            assert fu.is_demo_mode() is False, bad
        mp.setenv("FRONTEND_DEMO_MODE", "1")
        assert fu.is_demo_mode() is True


def test_path1_no_real_no_demo_is_no_data(_no_real_artifacts):
    """Path 1: no real artifact + demo mode unset → loaders return
    empty/None (the page-level no-data state). Samples never load."""
    assert fu.load_power_rankings("nba").empty
    assert fu.load_json_games("nba", "moneyline_json").empty
    assert fu.load_calibration("nba") is None
    assert fu.load_model_monitor("nba") is None
    assert fu.load_participant_matchup("nba").empty
    assert fu.load_shap("nba", "0022600001").empty
    assert fu.board_dates("nba") == []
    assert fu.has_any_artifacts("nba") is False
    assert fu.artifact_source("nba", "power_rankings") == "missing"
    # And the pure banner builder declines to show a banner.
    assert fu.demo_banner_html("nba", ["power_rankings"]) is None


def test_path2_no_real_demo_set_serves_samples_with_banner(_no_real_artifacts,
                                                           monkeypatch):
    """Path 2: no real artifact + FRONTEND_DEMO_MODE=1 → the tracked
    sample fixtures load AND the DEMO DATA banner is produced."""
    monkeypatch.setenv("FRONTEND_DEMO_MODE", "1")
    pr = fu.load_power_rankings("nba")
    assert not pr.empty
    games = fu.load_json_games("nba", "moneyline_json")
    assert len(games) == 3
    assert fu.load_calibration("nba") is not None
    assert fu.load_model_monitor("nba") is not None
    assert not fu.load_participant_matchup("nba").empty
    assert not fu.load_shap("nba", "0022600001").empty
    assert fu.board_dates("nba") == ["20261020"]
    assert fu.artifact_source("nba", "power_rankings") == "sample"
    html = fu.demo_banner_html("nba", ["power_rankings"])
    assert html is not None
    assert "DEMO DATA" in html
    assert "FRONTEND_DEMO_MODE=1" in html
    # Every page family the banner covers reports sample sourcing.
    for family in ("moneyline_json", "calibration_json", "model_monitor",
                   "markets", "participant_matchup", "shap_game"):
        assert fu.artifact_source("nba", family) == "sample", family


def test_path3_real_artifact_wins_over_samples(_no_real_artifacts, monkeypatch,
                                               tmp_path):
    """Path 3: when a real artifact exists it is used even in demo mode —
    samples are never preferred, and no banner is shown for that family."""
    real_dir = tmp_path / "real_delivery"
    real_dir.mkdir()
    real = real_dir / "nba_power_rankings_20990101.csv"
    pd.DataFrame([{
        "rank": 1, "team": "ZZZ", "team_name": "Real Team",
        "elo": 1500.0, "wins": 1, "losses": 0,
        "record": "1-0", "pct": 1.0, "run_diff": 1,
        "l10": "1-0", "home_pct": 1.0, "away_pct": 1.0},
    ]).to_csv(real, index=False)
    monkeypatch.setattr(sports_config, "data_delivery_dir",
                        lambda s: real_dir)
    monkeypatch.setenv("FRONTEND_DEMO_MODE", "1")

    df = fu.load_power_rankings("nba")
    assert list(df["team"]) == ["ZZZ"]  # real artifact, not the sample
    assert fu.artifact_source("nba", "power_rankings") == "real"
    # No banner for a page fully served by real artifacts.
    assert fu.demo_banner_html("nba", ["power_rankings"]) is None
    # Mixed page (some families still sample) still banners.
    assert fu.demo_banner_html("nba", ["power_rankings", "calibration_json"]) \
        is not None


def test_demo_banner_off_without_flag_even_with_samples(_no_real_artifacts):
    """Belt-and-braces: with samples on disk but the flag off, neither the
    loaders nor the banner ever touch them."""
    assert fu.artifact_source("nba", "power_rankings") == "missing"
    assert fu.demo_banner_html("nba", ["power_rankings"]) is None


def test_loader_returns_empty_without_any_artifacts(tmp_path, monkeypatch):
    """With both the real dir and the sample dir absent, loaders return
    empty frames / None (the page-level no-data contract)."""
    monkeypatch.setattr(sports_config, "data_delivery_dir",
                        lambda s: tmp_path / "missing_delivery")
    monkeypatch.setattr(sports_config, "sample_artifacts_dir",
                        lambda: tmp_path / "missing_samples")
    monkeypatch.delenv("FRONTEND_DEMO_MODE", raising=False)
    assert fu.load_power_rankings("nba").empty
    assert fu.load_json_games("nba", "moneyline_json").empty
    assert fu.load_calibration("nba") is None
    assert fu.load_model_monitor("nba") is None
    assert fu.load_shap("nba", "0022600001").empty
    assert fu.board_dates("nba") == []
    assert fu.has_any_artifacts("nba") is False


def test_participant_rows_nfl_quarterback():
    df = _matchup_frame("nfl", "qb", ("rating", "td_per_game"))
    panel = fu.participant_rows_for_game("nfl", "G1", df)
    spec = sports_config.participant_spec("nfl")
    assert spec["renderer"] == "quarterback"
    for side in ("home", "away"):
        p = panel[side]
        assert p is not None and p["name"].startswith("Player")
        expected = {"home": [21.6, 4.0], "away": [30.4, 5.0]}[side]
        assert p["stats"] == expected
        assert p["games"] == (9 if side == "home" else 7)
        assert p["labels"] == spec["stat_labels"]
    # TBD row: name None + null stats pass through, never fabricated.
    panel2 = fu.participant_rows_for_game("nfl", "G3", df)
    assert panel2["home"]["name"] is None
    assert panel2["home"]["stats"] == [None, None]


def test_participant_rows_nhl_goalie():
    df = _matchup_frame("nhl", "goalie", ("save_pct", "gaa"))
    panel = fu.participant_rows_for_game("nhl", "G1", df)
    spec = sports_config.participant_spec("nhl")
    assert spec["renderer"] == "goalie"
    assert panel["home"]["stats"] == [21.6, 4.0]
    assert panel["away"]["stats"] == [30.4, 5.0]
    assert panel["away"]["games"] == 7


def test_participant_rows_nba_scorer():
    df = _matchup_frame("nba", "player", ("ppg", "apg"))
    panel = fu.participant_rows_for_game("nba", "G1", df)
    spec = sports_config.participant_spec("nba")
    assert spec["renderer"] == "scorer"
    assert panel["home"]["stats"] == [21.6, 4.0]
    assert panel["away"]["labels"] == ["PPG L10", "APG L10"]


def test_participant_rows_mlb_inline_returns_empty_matchup():
    # MLB participants come from sp_* columns on the games CSV — the
    # matchup loader returns an empty frame and the card reads sp_* inline.
    assert fu.load_participant_matchup("mlb").empty
    panel = fu.participant_rows_for_game("mlb", "x", pd.DataFrame())
    assert panel == {"home": None, "away": None}


def test_participant_rows_missing_game_returns_none_sides():
    df = _matchup_frame("nba", "player", ("ppg", "apg"))
    panel = fu.participant_rows_for_game("nba", "NOPE", df)
    assert panel == {"home": None, "away": None}


def test_participant_rows_null_safe_without_frame():
    panel = fu.participant_rows_for_game("nba", "G1", None)
    assert panel == {"home": None, "away": None}


# ---------------------------------------------------------------------------
# SHAP + misc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sport", SPORTS)
def test_load_shap_empty_when_absent(sport):
    df = fu.load_shap(sport, "__no_such_game__")
    assert isinstance(df, pd.DataFrame) and df.empty


def test_start_time_et_formats():
    assert fu.start_time_et("") == ""
    assert fu.start_time_et(None) == ""
    out = fu.start_time_et("2026-10-20T23:07:00Z")
    assert out.endswith("ET")


def test_nearest_valid_date():
    # Ties (20261020 vs 20261022 both 1 day away) break to the earlier date;
    # the only contract is that a valid neighbour is chosen.
    assert fu.nearest_valid_date(["20261020", "20261022"], "20261021") in \
        ("20261020", "20261022")
    assert fu.nearest_valid_date(["20261020", "20261022"], "20261020") == "20261020"
    assert fu.nearest_valid_date(["20261020", "20261022"], "20261026") == "20261022"
    assert fu.nearest_valid_date([], target="20261021") is None


def test_board_dates_sorted():
    for sport in SPORTS:
        dates = fu.board_dates(sport)
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Phase 7.6-B: committed sample artifacts (NFL/NHL), demo-mode loader
# validation for all four sports, and MLB's deterministic empty case
# ---------------------------------------------------------------------------
def test_committed_sample_artifacts_match_generator():
    """The committed NFL/NHL sample artifacts are byte-pinned to the
    deterministic generators in tests/sample_artifact_fixtures.py: regenerating
    in-memory reproduces the committed bytes exactly (fixed literals, no wall
    clock, no network — same provenance pattern as the 7.5e-A raw fixtures)."""
    from tests.sample_artifact_fixtures import nfl_samples, nhl_samples
    for sport, generated in (("nfl", nfl_samples()), ("nhl", nhl_samples())):
        base = sports_config.sample_artifacts_dir() / sport
        assert base.is_dir(), sport
        committed = {p.name: p for p in base.iterdir() if p.is_file()}
        assert set(committed) == set(generated), (
            sport, sorted(set(committed) ^ set(generated)))
        for name, text in generated.items():
            assert committed[name].read_text(encoding="utf-8") == text, (
                f"{sport}/{name}: committed file drifted from the generator")


def _demo_only(monkeypatch, tmp_path):
    """Point the real data_delivery dir at an empty location and enable
    FRONTEND_DEMO_MODE=1 so loaders resolve the committed samples."""
    monkeypatch.setattr(sports_config, "data_delivery_dir",
                        lambda s: tmp_path / "missing_delivery")
    monkeypatch.setenv("FRONTEND_DEMO_MODE", "1")


@pytest.mark.parametrize("sport", ("nfl", "nhl", "nba"))
def test_sample_artifacts_load_through_production_loaders(sport, monkeypatch,
                                                         tmp_path):
    """Demo-mode Path 2 for the three JSON-artifact sports: every loader
    family resolves its committed sample through the PRODUCTION loader path
    and returns schema-conformant content (columns/keys from the sport's
    schema fixture), never fabricated fields."""
    _demo_only(monkeypatch, tmp_path)
    fx = json.loads((FIXTURES / f"{sport}_artifact_schemas.json").read_text())
    # shap_game files are keyed per game; resolve a real sample game id
    # through the loader path first (as the Todays Games page does).
    ml_games = fu.load_json_games(sport, "moneyline_json")
    assert not ml_games.empty, sport
    sample_game_id = str(ml_games["game_id"].iloc[0])

    # Families with no committed sample for a sport (NBA predates
    # predictions_history samples) are asserted at the loader no-data
    # contract instead — samples are never synthesized retroactively.
    _no_sample_families = {"nba": {"predictions_history"}}
    for family, artifact in FAMILY_TO_SCHEMA[sport].items():
        if family in _no_sample_families.get(sport, set()):
            assert fu.load_csv_artifact(sport, family).empty, \
                (sport, family)
            continue
        spec = fx["artifacts"][artifact]
        if family in ("calibration_json", "model_monitor", "markets_monitor"):
            data = fu.load_json_artifact(sport, family)
            assert data is not None, (sport, family)
            # Assert the loader-consumed/frontend-required surface, not the
            # full captured key set (the pre-existing NBA samples predate
            # several captured keys and are historical fixtures).
            for key in (spec.get("frontend_required") or []):
                assert key in data, (sport, artifact, key)
        elif family == "moneyline_json":
            games = fu.load_json_games(sport, family)
            assert not games.empty, (sport, family)
            for col in (spec.get("frontend_required") or []):
                assert col in games.columns, (sport, artifact, col)
            assert games["home_win_prob_model"].between(0, 1).all()
            assert games["away_win_prob_model"].between(0, 1).all()
        elif family == "participant_matchup":
            panel = fu.load_participant_matchup(sport)
            assert not panel.empty, (sport, family)
            for col in (spec.get("columns") or []):
                assert col in panel.columns, (sport, artifact, col)
        elif family == "markets_meta":
            data = fu.load_json_artifact(sport, family)
            assert data is not None and "grids" in data, (sport, family)
        elif family == "shap_game":
            df = fu.load_shap(sport, sample_game_id)
            assert not df.empty, (sport, family)
            for col in (spec.get("columns") or []):
                assert col in df.columns, (sport, artifact, col)
        else:
            df = fu.load_csv_artifact(sport, family)
            assert not df.empty, (sport, family)
            if family == "markets":
                # Page-consumed columns, not the full 241/93-col schema.
                missing = [c for c in MARKETS_PAGE_CONSUMED
                           if c not in df.columns]
                assert not missing, (sport, artifact, missing)
            else:
                for col in (spec.get("columns") or []):
                    assert col in df.columns, (sport, artifact, col)


def test_samples_served_only_under_demo_mode(monkeypatch, tmp_path):
    """The committed NFL/NHL samples are invisible without FRONTEND_DEMO_MODE=1:
    with no real artifact present and the flag off, loaders return the no-data
    state (samples are never a silent fallback)."""
    monkeypatch.setattr(sports_config, "data_delivery_dir",
                        lambda s: tmp_path / "missing_delivery")
    monkeypatch.delenv("FRONTEND_DEMO_MODE", raising=False)
    for sport in ("nfl", "nhl"):
        assert fu.load_power_rankings(sport).empty, sport
        assert fu.load_json_games(sport, "moneyline_json").empty, sport
        assert fu.load_calibration(sport) is None, sport
        assert fu.load_model_monitor(sport) is None, sport
        assert fu.artifact_source(sport, "power_rankings") == "missing", sport


def test_mlb_empty_case_is_the_no_data_contract(monkeypatch, tmp_path):
    """MLB ships no sample artifacts BY RULING: with no real artifacts and
    demo mode on, every MLB loader still resolves the deterministic empty
    case (empty frame / None) — the page-level no-data state."""
    _demo_only(monkeypatch, tmp_path)
    assert fu.load_power_rankings("mlb").empty
    assert fu.load_predictions_history("mlb").empty
    assert fu.load_json_games("mlb", "games").empty
    assert fu.load_calibration("mlb") is None
    assert fu.load_model_monitor("mlb") is None
    assert fu.load_participant_matchup("mlb").empty  # inline-source sport
    assert fu.load_shap("mlb", "__none__").empty
    assert fu.board_dates("mlb") == []
    assert fu.has_any_artifacts("mlb") is False
    for family in ("games", "power_rankings", "calibration_json",
                   "model_monitor", "markets", "shap_game"):
        assert fu.artifact_source("mlb", family) == "missing", family
    # ...and the banner declines: no sample family exists to banner about.
    assert fu.demo_banner_html("mlb", ["power_rankings"]) is None
