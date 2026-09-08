"""Enrich the raw NFL fixture (extracted verbatim headers/keys) with dtype,
null-behavior, and frontend-required metadata documented from the reference
repo's serving/monitoring source. Run once after extract_nfl_schema_fixtures;
the result is treated as immutable and reviewed by diff.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path("tests/fixtures/nfl_artifact_schemas.json")


def main() -> int:
    f = json.loads(FIXTURE.read_text(encoding="utf-8"))
    a = f["artifacts"]

    a["moneyline_json"]["game_keys"] = [
        "game_id", "game_date", "start_time_utc", "home_team", "away_team",
        "home_team_name", "away_team_name", "home_record", "away_record",
        "venue", "game_status", "home_score", "away_score",
        "home_win_prob_model", "away_win_prob_model", "model_pick",
        "model_correct",
    ]
    a["moneyline_json"]["game_keys_dtypes"] = {
        "game_id": "str", "game_date": "date(YYYY-MM-DD)",
        "start_time_utc": "iso8601|null", "home_team": "str (abbrev)",
        "away_team": "str (abbrev)", "home_team_name": "str",
        "away_team_name": "str", "home_record": "str|null (W-L)",
        "away_record": "str|null (W-L)", "venue": "str|null",
        "game_status": "str (pre)", "home_score": "null until final",
        "away_score": "null until final",
        "home_win_prob_model": "float|null",
        "away_win_prob_model": "float|null", "model_pick": "str|null",
        "model_correct": "null for slate rows",
    }
    a["moneyline_json"]["frontend_required"] = [
        "game_id", "game_date", "home_team", "away_team", "home_team_name",
        "away_team_name", "home_win_prob_model", "away_win_prob_model",
        "model_pick", "game_status",
    ]

    a["calibration_json"]["metrics_keys"] = [
        "auc", "brier", "logloss", "ece", "ece_calibrated", "auc_calibrated",
        "brier_calibrated", "logloss_calibrated",
    ]
    a["calibration_json"]["calibration_keys"] = [
        "method", "params", "metrics_raw", "metrics_calibrated",
        "calibration_buckets_calibrated",
    ]
    a["calibration_json"]["bucket_keys"] = [
        "bucket", "mean_predicted", "mean_actual", "count", "gap",
    ]
    a["calibration_json"]["frontend_required"] = [
        "date", "n_games", "metrics", "calibration", "calibration_buckets",
        "daily",
    ]

    a["predictions_history"]["dtypes"] = {
        "game_id": "str", "game_date": "date(YYYY-MM-DD)",
        "home_team": "str (abbrev)", "away_team": "str (abbrev)",
        "home_score": "float", "away_score": "float", "home_win": "float 0/1",
        "home_win_prob_model": "float",
        "home_win_prob_model_calibrated": "float",
        "model_pick": "str", "actual_winner": "str (team|TIE)",
        "correct": "bool",
    }
    a["predictions_history"]["frontend_required"] = ["all"]

    a["power_rankings"]["dtypes"] = {
        "rank": "int", "team": "str (abbrev)", "team_name": "str",
        "elo": "float", "wins": "int", "losses": "int",
        "record": "str (W-L)", "pct": "float", "run_diff": "int",
        "l10": "str", "home_pct": "float|null", "away_pct": "float|null",
    }
    a["power_rankings"]["frontend_required"] = ["all"]

    # markets: the 241-column header was extracted verbatim; the grid
    # columns follow the base MARKETS_BASE_COLS + spread/total/half-stop
    # grid labels documented in serving.py.
    a["markets"]["dtypes"] = {
        "kind": "str (oof|slate)", "game_id": "str",
        "gameday": "date(YYYY-MM-DD)", "season": "int", "week": "int",
        "home_team": "str", "away_team": "str", "stadium": "str|null",
        "gametime": "str|null", "home_record": "str|null",
        "away_record": "str|null", "p_home_win": "float",
        "p_away_win": "float", "p_tie": "float (push_0 mass)",
        "derived_ml": "float", "p_home_win_derived": "float",
        "p_away_win_derived": "float", "mu_h": "float", "mu_a": "float",
        "mu_margin": "float", "mu_total": "float", "fair_spread": "float",
        "fair_total": "float", "spread_line": "float|null (no offers)",
        "total_line": "float|null", "has_offer": "bool",
        "p_cover_offered": "float|null", "p_push_offered": "float|null",
        "p_over_offered": "float|null", "p_under_offered": "float|null",
        "p_push_total_offered": "float|null", "home_score": "float|null",
        "away_score": "float|null", "total": "float|null",
        "margin": "float|null", "p_over_fair": "float",
        "p_cover_fair": "float", "y_over_fair": "float|null (oof rows)",
        "y_under_fair": "float|null", "y_push_fair": "float|null",
        "y_cover_fair": "float|null", "y_push_spread_fair": "float|null",
        "y_over_offered": "null (no offers)",
        "y_under_offered": "null", "y_push_total_offered": "null",
        "y_cover_offered": "null", "y_push_spread_offered": "null",
        "y_home_win": "float|null", "decided": "bool",
        "frame_view": "str (oof|slate)", "pred_home": "float",
        "pred_away": "float",
        "p_home_cover_<L>": "float (L=-14..14 int; m-prefix negative)",
        "p_push_<L>": "float (integer lines only)",
        "p_home_cover_m0_5|p_home_cover_0_5": "float (half stops, no push)",
        "p_over_<U>": "float (U=24..66 int)", "p_under_<U>": "float",
        "p_push_<U>": "float (integer totals only)",
    }
    a["markets"]["frontend_required"] = ["all"]

    a["markets_meta"]["config_keys"] = [
        "feature_set_version", "warmup_seasons", "oof_first_season",
        "retrain_cadence_days", "ensemble_members", "random_seed",
        "market_independence",
    ]
    a["markets_meta"]["grids"] = {"spread": [-14, 14], "total": [24, 66],
                                  "half_stops": [-0.5, 0.5]}
    a["markets_meta"]["frontend_required"] = ["all"]

    a["markets_monitor"]["winner_card_keys"] = [
        "n", "ece_calibrated", "brier", "logloss", "predicted_mean",
    ]
    a["markets_monitor"]["frontend_required"] = [
        "date", "markets_persisted", "winner_cards", "slate_history", "fit",
        "market_metrics",
    ]

    a["qb_matchup"]["game_keys"] = [
        "game_id", "gameday", "home_team", "away_team",
        "qb_home_name", "qb_home_rating", "qb_home_td_per_game",
        "qb_home_cmp_pct", "qb_home_yards_per_attempt", "qb_home_ints",
        "qb_away_name", "qb_away_rating", "qb_away_td_per_game",
        "qb_away_cmp_pct", "qb_away_yards_per_attempt", "qb_away_ints",
    ]
    a["qb_matchup"]["game_keys_dtypes"] = {
        "game_id": "str", "gameday": "date(YYYY-MM-DD)",
        "home_team": "str", "away_team": "str",
        "qb_home_name": "str|null (TBD when unannounced)",
        "qb_home_rating": "float|null", "qb_home_td_per_game": "float|null",
        "qb_home_cmp_pct": "float|null",
        "qb_home_yards_per_attempt": "float|null",
        "qb_home_ints": "float|null",
        "qb_away_name": "str|null", "qb_away_rating": "float|null",
        "qb_away_td_per_game": "float|null", "qb_away_cmp_pct": "float|null",
        "qb_away_yards_per_attempt": "float|null", "qb_away_ints": "float|null",
    }
    a["qb_matchup"]["frontend_required"] = ["all"]

    a["feature_json"]["coverage_row_keys"] = [
        "feature", "n_games", "coverage_pct", "mean", "std",
    ]
    a["feature_json"]["frontend_required"] = [
        "feature_set_version", "served_columns", "coverage", "fold_geometry",
    ]

    a["model_monitor"]["frontend_required"] = [
        "last_retrained", "next_retrain", "feature_drift", "features_metadata",
        "feature_coverage", "ensemble", "rolling_brier", "brier_baseline",
        "brier_baseline_label", "rolling_brier_meta", "version_history",
        "fold_geometry",
    ]

    a["shap_game"]["dtypes"] = {
        "feature": "str", "shap_value": "float",
        "signed_effect": "str (positive|negative)",
        "perspective_team": "str (favored team abbrev)",
    }
    a["shap_game"]["frontend_required"] = ["all"]
    a["shap_game"]["count_rule"] = (
        "exactly one file per undecided prediction-slate game; zero files "
        "for an empty slate; no fixed minimum")

    FIXTURE.write_text(json.dumps(f, indent=2) + "\n", encoding="utf-8")
    print(f"enriched {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
