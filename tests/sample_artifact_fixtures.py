"""Deterministic NFL/NHL sample artifacts (demo-mode fixtures).

Builds the committed sample artifacts under
``tests/fixtures/sample_artifacts/{nfl,nhl}`` — the same demo-mode surface
the NBA samples already provide. Values are FIXED LITERALS (no wall clock,
no unseeded randomness, no network): calling the builders twice produces
byte-identical content. The pinning tests in ``tests/test_frontend.py``
compare the committed files against these builders through identical
serialization, so any drift between disk and generator fails loudly.

Structural provenance: every artifact mirrors the keys/columns declared in
``tests/fixtures/{nfl,nhl}_artifact_schemas.json`` (the captured real
artifact headers from the read-only reference repository), restricted to
the frontend-relevant surface — exactly the mirroring rule the committed
NBA samples follow. These are demo-mode stand-ins only: loaders serve them
solely under FRONTEND_DEMO_MODE=1 with no real artifact present (Path 2),
never as a silent fallback, and no production code reads them.
"""

from __future__ import annotations

import json
from typing import Callable

# Fixed slate: one board date, three games each (mirrors the NBA samples'
# 3-game slate shape so loader assertions stay uniform across sports).
RUN_DATE = "20270106"           # compact artifact date (Y-M-D, no wall clock)
BOARD_DATE = "2026-10-20"       # ISO slate date the games carry

NFL_TEAMS = ["KC", "BUF", "PHI", "DAL", "SF", "NYG"]
NFL_TEAM_NAMES = {
    "KC": "Kansas City Chiefs", "BUF": "Buffalo Bills",
    "PHI": "Philadelphia Eagles", "DAL": "Dallas Cowboys",
    "SF": "San Francisco 49ers", "NYG": "New York Giants",
}
NHL_TEAMS = ["TOR", "BOS", "MTL", "NYR", "TBL", "VGK"]

# Deterministic per-game probability ladders (i-indexed, all in (0, 1)).
NFL_P_HOME = (0.615, 0.532, 0.478)
NHL_P_HOME = (0.587, 0.508, 0.461)
# Fixed kickoff instants (UTC) for the sample slate rows.
NFL_STARTS = ("2026-10-20T17:00:00Z", "2026-10-20T21:05:00Z",
              "2026-10-21T00:15:00Z")
NHL_STARTS = ("2026-10-20T23:00:00Z", "2026-10-21T00:00:00Z",
              "2026-10-20T19:00:00Z")

_ENSEMBLE = (
    {"name": "xgboost", "weight": 0.2140, "auc": 0.7012,
     "brier": 0.2158, "logloss": 0.6231},
    {"name": "lightgbm", "weight": 0.1865, "auc": 0.6998,
     "brier": 0.2164, "logloss": 0.6243},
    {"name": "logistic", "weight": 0.3110, "auc": 0.7051,
     "brier": 0.2141, "logloss": 0.6208},
    {"name": "randomforest", "weight": 0.1825, "auc": 0.6964,
     "brier": 0.2172, "logloss": 0.6252},
    {"name": "mlp", "weight": 0.1060, "auc": 0.6923,
     "brier": 0.2188, "logloss": 0.6286},
)
_CAL_BUCKETS = (
    {"bucket": "50\u201360%", "mean_predicted": 0.5512, "mean_actual": 0.5401,
     "count": 1204, "gap": 0.0111},
    {"bucket": "60\u201370%", "mean_predicted": 0.6478, "mean_actual": 0.6522,
     "count": 987, "gap": -0.0044},
    {"bucket": "70\u201380%", "mean_predicted": 0.7440, "mean_actual": 0.7381,
     "count": 512, "gap": 0.0059},
    {"bucket": "80\u201390%", "mean_predicted": 0.8415, "mean_actual": 0.8502,
     "count": 178, "gap": -0.0087},
    {"bucket": "90\u2013100%", "mean_predicted": 0.9188, "mean_actual": 0.8901,
     "count": 31, "gap": 0.0287},
)
_METRICS = {"auc": 0.7021, "brier": 0.2152, "logloss": 0.6217,
            "ece": 0.0128, "ece_calibrated": 0.0071, "auc_calibrated": 0.7021,
            "brier_calibrated": 0.2150, "logloss_calibrated": 0.6214}
_CALIBRATION = {
    "method": "platt",
    "params": {"a": 0.9301, "b": -0.0198, "n": 240},
    "metrics_raw": {"brier": 0.2152, "logloss": 0.6217, "ece": 0.0128},
    "metrics_calibrated": {"brier": 0.2150, "logloss": 0.6214,
                           "ece": 0.0071},
    "calibration_buckets_calibrated": list(_CAL_BUCKETS),
}


def _config(feature_set_version: str) -> dict:
    return {
        "feature_set_version": feature_set_version,
        "warmup_seasons": [2015],
        "oof_first_season": 2016,
        "retrain_cadence_days": 7,
        "ensemble_members": [m["name"] for m in _ENSEMBLE],
        "random_seed": 42,
        "market_independence": True,
    }


def _games(prefix: str, feature_set_version: str, *, n_games: int,
           teams: list[str], team_names: dict[str, str] | None,
           p_home: tuple[float, ...], starts: tuple[str, ...],
           records: tuple[str, str] | None, venue_key: str) -> dict:
    """The moneyline_json card payload (fixture: moneyline_json)."""
    games = []
    for i in range(n_games):
        home, away = teams[(2 * i) % len(teams)], teams[(2 * i + 1) % len(teams)]
        ph = p_home[i % len(p_home)]
        pick = home if ph >= 0.5 else away
        record = {"home_record": records[0], "away_record": records[1]} \
            if records else {}
        games.append({
            "game_id": f"2026{i + 1:02d}{away}{home}",
            "game_date": BOARD_DATE,
            "start_time_utc": starts[i % len(starts)],
            "home_team": home,
            "away_team": away,
            "home_team_name": (team_names or {}).get(home, home),
            "away_team_name": (team_names or {}).get(away, away),
            "venue": f"{home} Stadium" if venue_key == "stadium"
                     else f"{home} Arena",
            "game_status": "pre",
            "home_score": None,
            "away_score": None,
            "home_win_prob_model": ph,
            "away_win_prob_model": 1.0 - ph,
            "model_pick": pick,
            "model_correct": None,
            **record,
        })
    return {
        "created_utc": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
        "config": _config(feature_set_version),
        "slate_date": BOARD_DATE,
        "n_games": n_games,
        "games": games,
    }


def _calibration_payload(sport_prefix: str, n_games: int) -> dict:
    """The calibration_json record (fixture: calibration_json)."""
    return {
        "date": RUN_DATE,
        "trained_at": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
        "n_games": n_games,
        "created_utc": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
        "config": _config(f"{sport_prefix}-moneyline-v75"),
        "metrics": dict(_METRICS),
        "calibration": _CALIBRATION,
        "calibration_buckets": list(_CAL_BUCKETS),
        "daily": [
            {"date": "20261018", "n_games": 3, "wins": 2, "losses": 1,
             "metrics": {"auc": 0.7, "brier": 0.21, "logloss": 0.62,
                         "ece": 0.03}},
            {"date": "20261019", "n_games": 3, "wins": 1, "losses": 2,
             "metrics": {"auc": 0.68, "brier": 0.22, "logloss": 0.63,
                         "ece": 0.04}},
        ],
    }


def _model_monitor_payload(sport_prefix: str) -> dict:
    """The model_monitor record (fixture: model_monitor)."""
    return {
        "last_retrained": "2027-01-06",
        "last_retrained_note": "sample fixture",
        "next_retrain": "2027-01-13",
        "next_retrain_note": None,
        "upset_note": None,
        "feature_drift": [],
        "features_metadata": [],
        "feature_coverage": [],
        "ensemble": [dict(m) for m in _ENSEMBLE],
        "rolling_brier": [],
        "brier_baseline": 0.4355,
        "brier_baseline_label": "Constant home-edge",
        "rolling_brier_meta": {"window_days": 14, "min_games": 20},
        "version_history": [
            {"version": RUN_DATE, "date": "2027-01-06",
             "weights": {m["name"]: m["weight"] for m in _ENSEMBLE},
             "auc": _METRICS["auc"], "logloss": _METRICS["logloss"],
             "ece_calibrated": _METRICS["ece_calibrated"],
             "note": "sample fixture",
             "calibration": {"a": 0.9301, "b": -0.0198}},
        ],
        "fold_geometry": {"n_folds": 2, "val_share": 0.2},
        "config": _config(f"{sport_prefix}-moneyline-v75"),
    }


def _markets_meta_payload(record: str, n_rows: int, grids: dict) -> dict:
    """The markets_meta record (fixture: markets_meta)."""
    return {
        "record": record,
        "written_utc": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
        "config": _config(record.split("_")[0] + "-moneyline-v75"),
        "columns": None,  # filled by the caller
        "n_rows": n_rows,
        "n_oof": n_rows,
        "n_slate": 3,
        "grids": grids,
    }


def _markets_monitor_payload(iso_date: str, market_card: str,
                             feature_set_version: str) -> dict:
    """The markets_monitor record (fixture: markets_monitor)."""
    card = {"n": 240, "brier": 0.2491, "logloss": 0.6923,
            "predicted_mean": 0.4877}
    return {
        "schema": "run-engine-monitor/v2",
        "date": iso_date,
        "markets_persisted": True,
        "markets_persist_error": None,
        "winner_cards": {
            "over_under": dict(card),
            market_card: dict(card),
            "derived_ml": {**card, "ml_reference": {
                "source": "ml_win_prob", "n": 240, "win_rate": 0.6312,
                "predicted_mean": 0.5402}},
        },
        "slate_history": [
            {"card": key, "date": iso_date, **card}
            for key in ("over_under", market_card, "derived_ml")
        ],
        "fit": {},
        "market_metrics": {},
        "config": _config(feature_set_version),
    }


def _power_rankings_csv(teams: list[str],
                        team_names: dict[str, str] | None) -> str:
    """The power_rankings CSV (fixture: power_rankings)."""
    lines = ["rank,team,team_name,elo,wins,losses,record,pct,run_diff,"
             "l10,home_pct,away_pct"]
    for i, t in enumerate(teams, start=1):
        name = (team_names or {}).get(t, t)
        lines.append(
            f"{i},{t},{name},{1600 + 37 * (len(teams) - i)}.{i},"
            f"{10 - i},{i - 1},{10 - i}-{i - 1},"
            f"{round((10 - i) / 9, 3)},{90 - 17 * i},6-3,"
            f"0.{600 + i * 7},0.{500 + i * 9}")
    return "\n".join(lines) + "\n"


def _shap_csv(team: str) -> str:
    """The shap_game CSV (fixture: shap_game)."""
    rows = ["feature,shap_value,signed_effect,perspective_team",
            f"elo_diff,0.131937,positive,{team}",
            f"rest_days_diff,-0.029233,negative,{team}",
            f"b2b_home,0.027501,positive,{team}"]
    return "\n".join(rows) + "\n"


def _predictions_history_csv(games: list[dict]) -> str:
    """The predictions_history CSV (fixture: predictions_history) — decided
    sample rows for the history page (no fabrication concerns: sample)."""
    header = ("game_id,game_date,home_team,away_team,home_score,away_score,"
              "home_win,home_win_prob_model,home_win_prob_model_calibrated,"
              "model_pick,actual_winner,correct")
    lines = [header]
    for i, g in enumerate(games):
        ph = g["home_win_prob_model"]
        home, away = g["home_team"], g["away_team"]
        hs, as_ = (24, 17) if i % 2 == 0 else (10, 20)
        hw = 1.0 if hs > as_ else 0.0
        winner = home if hs > as_ else away
        correct = "true" if g["model_pick"] == winner else "false"
        lines.append(
            f"{g['game_id']},{g['game_date']},{home},{away},{hs},{as_},"
            f"{hw},{ph},{round(min(ph * 1.02, 0.99), 4)},{g['model_pick']},"
            f"{winner},{correct}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public builders: sport → {filename: text}
# ---------------------------------------------------------------------------

def nfl_samples() -> dict[str, str]:
    """All NFL sample artifacts, keyed by committed filename."""
    n = 3
    games = _games("nfl", "nfl-moneyline-v75", n_games=n, teams=NFL_TEAMS,
                   team_names=NFL_TEAM_NAMES, p_home=NFL_P_HOME,
                   starts=NFL_STARTS, records=("5-2", "4-3"),
                   venue_key="stadium")["games"]
    grids = {"spread": [-14, 14], "total": [35, 55], "half_stops": []}
    meta = _markets_meta_payload("nfl_run_engine_markets", 6, grids)
    meta["columns"] = 241
    out = {
        f"nfl_moneyline_v1_{RUN_DATE}.json": json.dumps(
            _games("nfl", "nfl-moneyline-v75", n_games=n, teams=NFL_TEAMS,
                   team_names=NFL_TEAM_NAMES, p_home=NFL_P_HOME,
                   starts=NFL_STARTS, records=("5-2", "4-3"),
                   venue_key="stadium"), indent=1) + "\n",
        f"nfl_calibration_{RUN_DATE}.json": json.dumps(
            _calibration_payload("nfl", 240), indent=1) + "\n",
        f"nfl_model_monitor_{RUN_DATE}.json": json.dumps(
            _model_monitor_payload("nfl"), indent=1) + "\n",
        f"nfl_run_engine_markets_{RUN_DATE}.csv": _markets_csv_nfl(games),
        f"nfl_run_engine_markets_{RUN_DATE}.meta.json": json.dumps(
            meta, indent=1) + "\n",
        f"nfl_run_engine_monitor_{RUN_DATE}.json": json.dumps(
            _markets_monitor_payload("2027-01-06", "run_line",
                                     "nfl-moneyline-v75"), indent=1) + "\n",
        f"nfl_predictions_history_{RUN_DATE}.csv":
            _predictions_history_csv(games),
        f"nfl_power_rankings_{RUN_DATE}.csv":
            _power_rankings_csv(NFL_TEAMS, NFL_TEAM_NAMES),
        f"nfl_qb_matchup_{RUN_DATE}.json": json.dumps(
            _qb_payload(games), indent=1) + "\n",
        f"nfl_shap_game_{RUN_DATE}_202601BUFKC.csv":
            _shap_csv("KC"),
    }
    return out


def nhl_samples() -> dict[str, str]:
    """All NHL sample artifacts, keyed by committed filename."""
    n = 3
    payload = _games("nhl", "nhl-moneyline-v75", n_games=n, teams=NHL_TEAMS,
                     team_names=None, p_home=NHL_P_HOME, starts=NHL_STARTS,
                     records=("10-6-2", "9-7-2"), venue_key="venue")
    games = payload["games"]
    grids = {"spread": [-4, 4], "total": [5, 12], "half_stops": [-1.5, 1.5]}
    meta = _markets_meta_payload("nhl_run_engine_markets", 6, grids)
    meta["columns"] = 93
    out = {
        f"nhl_moneyline_v1_{RUN_DATE}.json": json.dumps(payload, indent=1) + "\n",
        f"nhl_calibration_{RUN_DATE}.json": json.dumps(
            _calibration_payload("nhl", 240), indent=1) + "\n",
        f"nhl_model_monitor_{RUN_DATE}.json": json.dumps(
            _model_monitor_payload("nhl"), indent=1) + "\n",
        f"nhl_run_engine_markets_{RUN_DATE}.csv": _markets_csv_nhl(games),
        f"nhl_run_engine_markets_{RUN_DATE}.meta.json": json.dumps(
            meta, indent=1) + "\n",
        f"nhl_run_engine_monitor_{RUN_DATE}.json": json.dumps(
            _markets_monitor_payload("2027-01-06", "puck_line",
                                     "nhl-moneyline-v75"), indent=1) + "\n",
        f"nhl_predictions_history_{RUN_DATE}.csv":
            _predictions_history_csv(games),
        f"nhl_power_rankings_{RUN_DATE}.csv":
            _power_rankings_csv(NHL_TEAMS, None),
        f"nhl_goalie_matchup_{RUN_DATE}.json": json.dumps(
            _goalie_payload(games), indent=1) + "\n",
        f"nhl_shap_game_{RUN_DATE}_202601BOSTOR.csv":
            _shap_csv("TOR"),
    }
    return out


def _markets_rows(games: list[dict], *, season: int, week: int,
                  extra_slate: list[str], extra_oof: list[str],
                  scores: list[tuple[int, int]],
                  total_line: float) -> tuple[list[str], list[list[str]]]:
    """Shared markets-row builder: per game one slate row (undecided) and
    one settled OOF row. ``extra_slate``/``extra_oof`` carry the columns
    unique to that sport's schema (appended after the shared tail); values
    are pre-stringified and written via the csv module so field counts
    always match the header."""
    shared = ["kind", "game_id", "gameday", "season", "week", "home_team",
              "away_team", "p_home_win", "p_away_win", "p_tie", "derived_ml",
              "p_home_win_derived", "p_away_win_derived", "mu_h", "mu_a",
              "mu_margin", "mu_total", "fair_spread", "fair_total",
              "spread_line", "total_line", "has_offer", "p_cover_offered",
              "p_push_offered", "p_over_offered", "p_push_total_offered",
              "p_under_fair", "home_score", "away_score", "total", "margin",
              "p_over_fair", "p_cover_fair", "y_over_fair", "y_under_fair",
              "y_push_fair", "y_cover_fair", "y_push_spread_fair",
              "y_home_win", "decided", "frame_view", "pred_home", "pred_away"]
    header = ([c for c in shared if c != "week"] if week is None else
              shared)
    header = [c if c != "week" else "week" for c in header]
    rows: list[list[str]] = []
    for i, g in enumerate(games):
        ph = g["home_win_prob_model"]
        pa = round(1.0 - ph, 4)
        base_slate = {
            "kind": "slate", "game_id": g["game_id"],
            "gameday": g["game_date"], "season": season, "week": week,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "p_home_win": ph, "p_away_win": pa, "p_tie": 0.0,
            "derived_ml": ph, "p_home_win_derived": ph,
            "p_away_win_derived": pa,
            "mu_h": round(ph * 24 + 14, 1), "mu_a": round(pa * 21 + 14, 1),
            "mu_margin": 3.5, "mu_total": 45.5, "fair_spread": -3.5,
            "fair_total": 45.5, "spread_line": -3.5, "total_line": total_line,
            "has_offer": 1, "p_cover_offered": round(ph * 0.9, 4),
            "p_push_offered": 0.05, "p_over_offered": round(ph * 0.85, 4),
            "p_push_total_offered": 0.08, "p_under_fair": 0.52,
            "home_score": "", "away_score": "", "total": "", "margin": "",
            "p_over_fair": "", "p_cover_fair": "", "y_over_fair": "",
            "y_under_fair": "", "y_push_fair": "", "y_cover_fair": "",
            "y_push_spread_fair": "", "y_home_win": "", "decided": False,
            "frame_view": "slate", "pred_home": round(ph * 24 + 14, 1),
            "pred_away": round(pa * 21 + 14, 1),
        }
        hs, as_ = scores[i % len(scores)]
        base_oof = {
            **base_slate,
            "kind": "oof", "mu_margin": hs - as_, "mu_total": hs + as_,
            "home_score": hs, "away_score": as_, "total": hs + as_,
            "margin": hs - as_, "p_over_fair": 0.52,
            "p_cover_fair": ph,
            "y_over_fair": int(hs + as_ > total_line),
            "y_under_fair": int(hs + as_ < total_line),
            "y_push_fair": int(hs + as_ == total_line),
            "y_cover_fair": int((hs - as_) > -3.5),
            "y_push_spread_fair": 0, "y_home_win": int(hs > as_),
            "decided": True, "frame_view": "oof", "pred_home": 24.0,
            "pred_away": 21.0,
        }
        for base, extra in ((base_slate, extra_slate), (base_oof, extra_oof)):
            row = [base[c] for c in header]
            rows.append([str(v) for v in row] + [str(v) for v in extra])
    return header, rows


def _markets_csv_nfl(games: list[dict]) -> str:
    """NFL markets CSV — the frontend-consumed columns of the 241-col
    schema, one slate row + one settled OOF row per game. The NFL-leading
    venue/kickoff columns are prepended; values are written via the csv
    module so field counts always match the header."""
    import csv
    import io
    header, rows = _markets_rows(
        games, season=2026, week=12, extra_slate=[], extra_oof=[],
        scores=[(24, 17), (10, 20), (27, 24)], total_line=45.5)
    header = ["stadium", "gametime", "home_record", "away_record"] + header
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for i, row in enumerate(rows):
        g = games[i % len(games)]
        w.writerow([g.get("venue", ""), "13:00", g.get("home_record", ""),
                    g.get("away_record", "")] + row)
    return buf.getvalue()


def _markets_csv_nhl(games: list[dict]) -> str:
    """NHL markets CSV — the frontend-consumed columns of the 93-col
    schema, one slate row + one settled OOF row per game. The NHL-leading
    venue/records and regulation columns wrap the shared row body."""
    import csv
    import io
    header, rows = _markets_rows(
        games, season=2026, week=None, extra_slate=[], extra_oof=[],
        scores=[(4, 2), (1, 3), (5, 2)], total_line=5.5)
    # Shared header omits `week` for NHL; splice venue/records after
    # away_team and regulation probabilities after p_away_win_derived.
    away_idx = header.index("away_team")
    pawd_idx = header.index("p_away_win_derived")
    header = (header[:away_idx + 1] + ["venue", "home_record", "away_record"]
              + header[away_idx + 1:pawd_idx + 1]
              + ["p_home_regulation", "p_away_regulation"]
              + header[pawd_idx + 1:])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for i, row in enumerate(rows):
        g = games[i % len(games)]
        row = row[:]  # shared row, same order as _markets_rows header
        row = (row[:away_idx + 1]
               + [g.get("venue", ""), "10-6-2", "9-7-2"]
               + row[away_idx + 1:pawd_idx + 1]
               + [0.72, 0.68]
               + row[pawd_idx + 1:])
        w.writerow(row)
    return buf.getvalue()


def _qb_payload(games: list[dict]) -> dict:
    """The qb_matchup record (fixture: qb_matchup)."""
    out = []
    for i, g in enumerate(games):
        out.append({
            "game_id": g["game_id"], "gameday": g["game_date"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "qb_home_name": f"QB {g['home_team']}",
            "qb_home_rating": 100.2 + i,
            "qb_home_td_per_game": 2.1,
            "qb_home_cmp_pct": 66.8,
            "qb_home_yards_per_attempt": 8.2,
            "qb_home_ints": 0.6,
            "qb_away_name": f"QB {g['away_team']}",
            "qb_away_rating": 94.7 + i,
            "qb_away_td_per_game": 1.7,
            "qb_away_cmp_pct": 63.1,
            "qb_away_yards_per_attempt": 7.4,
            "qb_away_ints": 0.9,
        })
    return {"created_utc": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
            "n_games": len(out), "games": out}


def _goalie_payload(games: list[dict]) -> dict:
    """The goalie_matchup record (fixture: goalie_matchup)."""
    out = []
    for i, g in enumerate(games):
        out.append({
            "game_id": g["game_id"], "game_date": g["game_date"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "goalie_home_name": f"G {g['home_team']}",
            "goalie_home_save_pct": 0.917 + i * 0.001,
            "goalie_home_gaa": 2.31 - i * 0.03,
            "goalie_home_starts": 12 - i,
            "goalie_away_name": f"G {g['away_team']}",
            "goalie_away_save_pct": 0.910 + i * 0.002,
            "goalie_away_gaa": 2.58 - i * 0.04,
            "goalie_away_starts": 11 - i,
        })
    return {"created_utc": f"{RUN_DATE[:4]}-{RUN_DATE[4:6]}-{RUN_DATE[6:]}T12:00:00+00:00",
            "n_games": len(out), "games": out}
