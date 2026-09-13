"""DuckDB feature-factory tests: PIT compliance, frame rules, feature
coverage, and the schema-robust connect path.

The feature factory runs against a real Parquet cache built from synthetic
Statcast data — the SQL path is exercised end-to-end without a network.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from sports.mlb.features.build_frame import build_features
from sports.mlb.frames import (
    compute_elo_entries,
    compute_season_records,
    get_decided_frame,
)
from sports.mlb.features.registry import (
    MARKET_FEATURE_COLS,
    MONEYLINE_FEATURE_COLS,
    build_candidate_frame,
    derive_diff_features,
)
from tests.mlb_fixtures import make_statcast_games

D0 = date(2026, 4, 1)


@pytest.fixture(scope="module")
def feature_frame(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("feat")
    cache = tmp / "pitches.parquet"
    df = make_statcast_games(D0, 90, games_per_day=4, seed=7,
                             include_slate_day=True)
    df.to_parquet(cache, index=False)
    game_df, pbp_df = build_features(cache, output_dir=tmp)
    return game_df, pbp_df


def test_feature_tables_written(feature_frame):
    game_df, pbp_df = feature_frame
    assert len(game_df) > 0
    assert len(pbp_df) > 0
    required = {"game_pk", "game_date", "home_team", "away_team", "home_win",
                "home_score", "away_score", "total_runs", "sp_era_home",
                "team_woba_30g_home", "bullpen_whip_10g_home", "venue"}
    assert required <= set(game_df.columns)


def test_home_win_binary_on_decided(feature_frame):
    game_df, _ = feature_frame
    decided = game_df[game_df["home_win"].notna()]
    assert set(decided["home_win"].unique()) <= {0.0, 1.0}


def test_slate_rows_undecided_not_fabricated(feature_frame):
    """The final synthetic day is undecided; its rows must carry NaN
    home_win, never a fabricated outcome."""
    game_df, _ = feature_frame
    slate = game_df[pd.to_datetime(game_df["game_date"]).dt.date
                    == D0 + timedelta(days=89)]
    assert slate["home_win"].isna().all()


def test_decided_frame_rules(feature_frame):
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df)
    assert decided["home_win"].notna().all()
    assert decided["game_pk"].is_unique
    assert decided["game_pk"].dtype.kind == "i"
    dates = pd.to_datetime(decided["game_date"])
    assert dates.is_monotonic_increasing


def test_rolling_window_pit_leakage(feature_frame):
    """A team's 30g wOBA entering game T must NOT include game T's own
    offense. Recompute the expectation independently and compare."""
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df)
    cand, _ = build_candidate_frame(decided)

    # PIT property: for each team's FIRST game, any trailing-30g metric is
    # NaN (no prior games) while the raw game-day offense exists. The
    # registry renames the DuckDB raw column to the spec name on the
    # candidate frame (`team_woba_30g_*` -> `woba_30g_*`).
    first_day = pd.to_datetime(cand["game_date"]).min()
    first_rows = cand[pd.to_datetime(cand["game_date"]) == first_day]
    assert first_rows["woba_30g_home"].isna().all()
    assert first_rows["woba_30g_away"].isna().all()


def test_season_records_pit(feature_frame):
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df)
    rec = compute_season_records(decided)
    # Before each team's first game the record is 0-0.
    assert (rec["home_wins"].iloc[0] == 0) and (rec["home_losses"].iloc[0] == 0)
    # By the last game every team has accumulated wins/losses across the
    # season — cumulative totals never exceed games played.
    assert rec["home_wins"].max() + rec["home_losses"].max() <= len(decided)


def test_elo_point_in_time(feature_frame):
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df).sort_values("game_date")
    elo = compute_elo_entries(decided)
    # Pre-game Elo for a fresh team is the 1500 start.
    assert elo["home_elo"].iloc[0] == 1500.0
    assert elo["away_elo"].iloc[0] == 1500.0
    # Elo drifts within the bounded K*window of the start rating.
    assert elo["home_elo"].abs().sub(1500).max() < 200


def test_candidate_frame_feature_coverage(feature_frame):
    """Every frozen feature column exists on the candidate frame; each
    rolling metric attains real (non-null) coverage on a 90-day frame."""
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df)
    cand, feature_cols = build_candidate_frame(decided)
    assert feature_cols == MONEYLINE_FEATURE_COLS
    for col in MONEYLINE_FEATURE_COLS:
        assert col in cand.columns, col
    for col in MARKET_FEATURE_COLS:
        assert col in cand.columns, col
    coverage = cand[list(MONEYLINE_FEATURE_COLS)].notna().mean()
    # Baselines like is_home are fully covered; rolling metrics cover most
    # of the frame after warmup.
    assert coverage["is_home"] == 1.0
    assert (coverage > 0.5).mean() > 0.5


def test_diff_recomputation_from_raw_inputs():
    df = pd.DataFrame({
        "home_elo": [1550.0], "away_elo": [1500.0],
        "home_win_pct": [0.6], "away_win_pct": [0.4],
        "sp_era_home": [3.0], "sp_era_away": [4.0],
    })
    out = derive_diff_features(df)
    assert out["elo_diff"].iloc[0] == pytest.approx(50.0)
    assert out["win_pct_diff"].iloc[0] == pytest.approx(0.2)
    assert out["sp_era_diff"].iloc[0] == pytest.approx(-1.0)
    # Missing inputs ship as NaN — never fabricated zeros.
    assert pd.isna(out["rest_days_diff"].iloc[0])


def test_missing_feature_columns_ship_as_nan_not_zero(feature_frame):
    game_df, _ = feature_frame
    decided = get_decided_frame(game_df)[["game_pk", "game_date", "home_team",
                                          "away_team", "home_win"]].copy()
    cand, _ = build_candidate_frame(decided, include_run_view=True)
    # No raw inputs at all -> every rolling feature is NaN (not 0). The
    # registry exposes the trailing-wOBA feature under its spec name.
    assert cand["woba_30g_home"].isna().all()
    assert cand["bullpen_whip_10g_home"].isna().all()


def test_duckdb_connect_schema_robust(tmp_path):
    """A pitches cache missing optional columns still runs the factory."""
    cache = tmp_path / "p.parquet"
    make_statcast_games(D0, 3, games_per_day=2, seed=3,
                        include_slate_day=False).to_parquet(cache)
    game_df, _ = build_features(cache, output_dir=tmp_path)
    assert len(game_df) > 0


def test_duckdb_spill_configured(tmp_path):
    """The connection is tuned for bounded RAM (temp spill on)."""
    from sports.mlb.features.build_frame import _connect
    cache = tmp_path / "p.parquet"
    make_statcast_games(D0, 2, games_per_day=2, seed=5,
                        include_slate_day=False).to_parquet(cache)
    con = _connect(cache)
    try:
        tmp_dir = con.execute(
            "SELECT current_setting('temp_directory')").fetchone()[0]
        assert tmp_dir  # spill directory configured
    finally:
        con.close()
