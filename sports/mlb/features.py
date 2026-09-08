"""Pure DuckDB SQL feature engineering for MLB Statcast data.

ZERO pandas during feature engineering. All rolling windows, shifted
features, groupby aggregations, and joins are DuckDB SQL window functions
operating on Parquet files on disk — the historical dataset is never fully
loaded into RAM.

PIT compliance: every rolling metric uses LAG() first (shift), then a
window over the SHIFTED values, so Game T features use only data from games
strictly before T. Season-cumulative windows are season-partitioned so the
prior October never leaks into a new season's expanding mean.

Layout (mirrors the reference repo's proven SQL, reduced to the production
feature set):

    pitches.parquet -> DuckDB SQL -> game_level_features.parquet
                                    + pbp_level_features.parquet
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

VENUE_MAP: dict[str, str] = {
    "ARI": "Chase Field", "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards", "BOS": "Fenway Park",
    "CHC": "Wrigley Field", "CWS": "Rate Field",
    "CIN": "Great American Ball Park", "CLE": "Progressive Field",
    "COL": "Coors Field", "DET": "Comerica Park", "HOU": "Minute Maid Park",
    "KC": "Kauffman Stadium", "LAA": "Angel Stadium", "LAD": "Dodger Stadium",
    "MIA": "loanDepot park", "MIL": "American Family Field",
    "MIN": "Target Field", "NYM": "Citi Field", "NYY": "Yankee Stadium",
    "OAK": "Sutter Health Park", "PHI": "Citizens Bank Park",
    "PIT": "PNC Park", "SD": "Petco Park", "SF": "Oracle Park",
    "SEA": "T-Mobile Park", "STL": "Busch Stadium", "TB": "Steinbrenner Field",
    "TEX": "Globe Life Field", "TOR": "Rogers Centre", "WSH": "Nationals Park",
}

PA_END_EVENTS = (
    "'single', 'double', 'triple', 'home_run',"
    "'strikeout', 'strikeout_double_play',"
    "'walk', 'hit_by_pitch',"
    "'field_out', 'field_error', 'fielders_choice', 'fielders_choice_out',"
    "'grounded_into_double_play', 'double_play', 'triple_play',"
    "'sac_fly', 'sac_bunt', 'sac_fly_double_play',"
    "'catcher_interf', 'batter_interference',"
    "'force_out', 'sacrifice_bunt_double_play'"
)

# Rolling window sizes (reference config constants).
WOBA_WINDOW = 30          # games
BULLPEN_WHIP_WINDOW = 10  # games
SP_ERA_WINDOW = 30        # games (season-to-date; the 30g label is legacy)
REST_DAYS_CAP = 6


def _connect(pitches_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB in-memory, load pitches from Parquet, tune for low RAM."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '4GB'")
    _duckdb_tmp = os.path.join(tempfile.gettempdir(), "duckdb_temp") \
        .replace("\\", "/")
    con.execute(f"SET temp_directory = '{_duckdb_tmp}'")
    con.execute("SET max_temp_directory_size = '50GB'")
    con.execute(f"CREATE TABLE pitches AS SELECT * FROM '{pitches_path}'")
    con.execute("""
        ALTER TABLE pitches
        ALTER COLUMN game_date TYPE DATE
        USING CAST(game_date AS DATE)
    """)
    existing = {r[0] for r in con.execute("DESCRIBE pitches").fetchall()}
    for col in ("fielder_2", "fielder_3", "fielder_4", "fielder_5",
                "fielder_6", "fielder_7", "fielder_8", "fielder_9",
                "if_fielding_alignment", "of_fielding_alignment",
                "post_home_score", "post_away_score",
                "event", "type", "launch_speed_angle"):
        if col in existing:
            con.execute(f'ALTER TABLE pitches DROP COLUMN "{col}"')
    # Ensure ALL columns referenced by SQL queries exist (schema-robust).
    _required = {
        "game_pk", "game_date", "game_type", "home_team", "away_team",
        "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
        "on_1b", "on_2b", "on_3b",
        "at_bat_number", "pitch_number", "pitcher", "batter",
        "p_throws", "stand",
        "pitch_type", "release_speed", "description", "events",
        "barrel", "hard_contact", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
        "zone", "home_score", "away_score", "spin_rate",
        "woba_value", "babip_value", "iso_value",
        "delta_home_win_exp", "delta_run_exp", "player_name",
        "hit_distance_sc", "release_pos_x", "release_pos_z",
        "release_spin_rate", "release_extension", "pfx_x", "pfx_z",
    }
    _str_cols = {
        "pitch_type", "events", "description", "player_name",
        "home_team", "away_team", "pitcher", "batter",
        "stand", "p_throws", "game_type", "inning_topbot",
    }
    existing2 = {r[0] for r in con.execute("DESCRIBE pitches").fetchall()}
    for col in _required - existing2:
        dtype = "VARCHAR" if col in _str_cols else "DOUBLE"
        con.execute(f'ALTER TABLE pitches ADD COLUMN "{col}" {dtype}')
        con.execute(f'UPDATE pitches SET "{col}" = NULL')
    return con


def _build_rest_days(con: duckdb.DuckDBPyConnection) -> None:
    """Rest days per team entering each game (capped at REST_DAYS_CAP)."""
    con.execute(f"""
        CREATE TABLE rest_days AS
        WITH games AS (
            SELECT DISTINCT game_pk, game_date, home_team, away_team
            FROM pitches
        ),
        appearances AS (
            SELECT game_pk, game_date, home_team AS team FROM games
            UNION ALL
            SELECT game_pk, game_date, away_team AS team FROM games
        ),
        seq AS (
            SELECT *,
                LAG(game_date) OVER (PARTITION BY team ORDER BY game_date, game_pk)
                    AS prev_date
            FROM appearances
        )
        SELECT game_pk, team,
            CASE WHEN prev_date IS NULL THEN NULL
                 ELSE LEAST(CAST(game_date - prev_date AS INT),
                            {REST_DAYS_CAP}) END AS rest_days
        FROM seq
    """)


def _build_pitcher_features(con: duckdb.DuckDBPyConnection) -> None:
    """Pitcher rolling features (PIT: LAG first, then rolling over shifted)."""
    con.execute(f"""
        CREATE TABLE pa_boundary AS
        WITH lastp AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   game_pk, inning, inning_topbot, at_bat_number,
                   pitcher, events,
                   launch_speed, launch_angle,
                   COALESCE(home_score, 0) + COALESCE(away_score, 0) AS tot_score,
                   estimated_woba_using_speedangle AS xwoba_val,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk, inning, inning_topbot, at_bat_number
                       ORDER BY pitch_number DESC
                   ) AS rn
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        ),
        lp AS (SELECT * FROM lastp WHERE rn = 1),
        seq AS (
            SELECT *,
                LAG(tot_score) OVER (
                    PARTITION BY game_pk ORDER BY at_bat_number
                ) AS prev_tot
            FROM lp
        )
        SELECT game_date, game_pk, pitcher, events,
               CASE WHEN prev_tot IS NULL THEN tot_score
                    ELSE GREATEST(tot_score - prev_tot, 0) END AS runs_on_pa,
               CASE events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END AS outs_on_pa,
               xwoba_val,
               CASE WHEN launch_speed >= 98 AND launch_angle BETWEEN 22 AND 36
                    THEN 1.0 ELSE 0.0 END AS barrel_flag,
               CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag
        FROM seq
    """)
    con.execute("""
        CREATE TABLE pitcher_game_stats AS
        SELECT game_date, game_pk, pitcher,
               COUNT(*) AS n_batters_faced,
               SUM(outs_on_pa) / 3.0 AS ip,
               SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play')
                        THEN 1 ELSE 0 END) AS ks,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bbs,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbps,
               SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run')
                        THEN 1 ELSE 0 END) AS hits_allowed,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs_allowed,
               SUM(runs_on_pa) AS runs,
               AVG(xwoba_val) AS xwoba,
               AVG(barrel_flag) AS barrel_rate,
               AVG(hard_flag) AS hard_contact_rate
        FROM pa_boundary
        GROUP BY game_date, game_pk, pitcher
    """)
    con.execute("""
        CREATE TABLE pitcher_shifted AS
        SELECT *,
            LAG(ip, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ip,
            LAG(runs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ks,
            LAG(bbs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_bbs,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hits,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hrs,
            LAG(hbps, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hbps,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_xwoba
        FROM pitcher_game_stats
    """)
    # Season-to-date ERA / K/9 (season-partitioned LAGs so the prior October
    # never leaks into a new season's cumulative) + last-5-start twins
    # (cross-season, no season-start gap).
    con.execute("""
        CREATE TABLE pitcher_shifted_season AS
        SELECT *,
            LAG(ip, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ip_s,
            LAG(runs, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_runs_s,
            LAG(ks, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ks_s,
            LAG(bbs, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_bbs_s,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hits_s,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hrs_s,
            LAG(hbps, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hbps_s,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_xwoba_s
        FROM (SELECT *, EXTRACT(YEAR FROM game_date) AS season FROM pitcher_shifted)
    """)
    con.execute("""
        CREATE TABLE pitcher_rolling AS
        SELECT game_date, game_pk, pitcher,
            SUM(_s_runs) OVER w AS _roll_runs,
            SUM(_s_ks) OVER w AS _roll_ks,
            SUM(_s_bbs) OVER w AS _roll_bbs,
            SUM(_s_hits) OVER w AS _roll_hits,
            SUM(_s_hrs) OVER w AS _roll_hrs,
            SUM(_s_hbps) OVER w AS _roll_hbps,
            SUM(_s_ip) OVER w AS _roll_ip,
            AVG(_s_xwoba) OVER w AS _roll_xwoba,
            SUM(_s_runs_s) OVER ws AS _s_runs_ss,
            SUM(_s_ks_s) OVER ws AS _s_ks_ss,
            SUM(_s_ip_s) OVER ws AS _s_ip_ss,
            SUM(_s_bbs_s) OVER ws AS _s_bbs_ss,
            SUM(_s_hits_s) OVER ws AS _s_hits_ss,
            SUM(_s_hbps_s) OVER ws AS _s_hbps_ss,
            AVG(_s_xwoba_s) OVER ws AS _s_xwoba_ss,
            SUM(_s_runs) OVER w5 AS _roll5_runs,
            SUM(_s_ks) OVER w5 AS _roll5_ks,
            SUM(_s_ip) OVER w5 AS _roll5_ip
        FROM pitcher_shifted_season
        WINDOW w AS (PARTITION BY pitcher ORDER BY game_date
                     ROWS BETWEEN 5 PRECEDING AND CURRENT ROW),
               w5 AS (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
               ws AS (PARTITION BY pitcher, season ORDER BY game_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE pitcher_features AS
        SELECT game_date, game_pk, pitcher,
            _s_runs_ss / NULLIF(_s_ip_ss, 0) * 9.0 AS sp_era,
            _s_ks_ss / NULLIF(_s_ip_ss, 0) * 9.0 AS sp_k9,
            _roll5_runs / NULLIF(_roll5_ip, 0) * 9.0 AS sp_era_5g,
            _roll5_ks / NULLIF(_roll5_ip, 0) * 9.0 AS sp_k9_5g,
            _roll_bbs / NULLIF(_roll_ip, 0) * 9.0 AS sp_bb9_30g,
            (_roll_bbs + _roll_hits) / NULLIF(_roll_ip, 0) AS sp_whip_30g,
            (13 * _roll_hrs + 3 * (_roll_bbs + _roll_hbps) - 2 * _roll_ks)
                / NULLIF(_roll_ip, 0) AS sp_fip_30g,
            _roll_xwoba AS sp_xwoba_30g,
            _s_bbs_ss / NULLIF(_s_ip_ss, 0) * 9.0 AS sp_bb9_std,
            (_s_bbs_ss + _s_hits_ss) / NULLIF(_s_ip_ss, 0) AS sp_whip_std,
            _s_xwoba_ss AS sp_xwoba_std
        FROM pitcher_rolling
    """)


def _build_starters_and_venues(con: duckdb.DuckDBPyConnection) -> None:
    """Starting pitchers (first PA of each half of the 1st inning) + venues.

    Top of the 1st: AWAY team bats, so the HOME team's starter pitches.
    These were swapped before in the reference repo (home SP stats read off
    the away starter) — the orientation here is the corrected one.
    """
    con.execute("""
        CREATE TABLE starters AS
        WITH first_pa_top AS (
            SELECT game_pk, pitcher AS home_starter_id,
                   p_throws AS home_starter_hand,
                   ROW_NUMBER() OVER (PARTITION BY game_pk
                                      ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Top'
        ),
        first_pa_bot AS (
            SELECT game_pk, pitcher AS away_starter_id,
                   p_throws AS away_starter_hand,
                   ROW_NUMBER() OVER (PARTITION BY game_pk
                                      ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Bot'
        )
        SELECT DISTINCT p.game_pk,
               CAST(p.game_date AS DATE) AS game_date,
               p.home_team, p.away_team,
               t.home_starter_id, t.home_starter_hand,
               b.away_starter_id, b.away_starter_hand
        FROM (SELECT DISTINCT game_pk, game_date, home_team, away_team
              FROM pitches) p
        LEFT JOIN first_pa_top t ON p.game_pk = t.game_pk AND t.rn = 1
        LEFT JOIN first_pa_bot b ON p.game_pk = b.game_pk AND b.rn = 1
    """)
    venue_lines = "\n".join(
        f"            WHEN '{k}' THEN '{v}'" for k, v in VENUE_MAP.items())
    con.execute(f"""
        CREATE TABLE venues AS
        SELECT DISTINCT game_pk, home_team,
            CASE home_team
{venue_lines}
                ELSE 'Unknown'
            END AS venue
        FROM pitches
    """)


def _build_team_offense(con: duckdb.DuckDBPyConnection) -> None:
    """Team offense rolling features (30g wOBA/ISO/K%/BB% + season baselines)."""
    con.execute(f"""
        CREATE TABLE team_offense_raw AS
        WITH pa_events AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   events
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        ),
        game_agg AS (
            SELECT game_date, game_pk, batting_team,
                   COUNT(*) AS n_pa,
                   SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS singles,
                   SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS doubles,
                   SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS triples,
                   SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
                   SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
                   SUM(CASE WHEN events IN ('strikeout',
                                            'strikeout_double_play')
                            THEN 1 ELSE 0 END) AS ks
            FROM pa_events GROUP BY game_date, game_pk, batting_team
        )
        SELECT *,
            (0.690*bb + 0.722*hbp + 0.878*singles + 1.242*doubles
             + 1.568*triples + 2.007*hrs) / NULLIF(n_pa, 0) AS team_woba_game,
            (doubles + 2*triples + 3*hrs) / NULLIF(n_pa - bb - hbp, 0)
                AS team_iso_game,
            ks::DOUBLE / NULLIF(n_pa, 0) AS team_k_rate_game,
            bb::DOUBLE / NULLIF(n_pa, 0) AS team_bb_rate_game
        FROM game_agg
    """)
    con.execute("""
        CREATE TABLE team_off_shifted AS
        SELECT *,
            LAG(team_woba_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_woba,
            LAG(team_iso_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_iso,
            LAG(team_k_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_krate,
            LAG(team_bb_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_bbrate
        FROM team_offense_raw
    """)
    con.execute("""
        CREATE TABLE team_offense_rolling AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_woba) OVER w AS team_woba_30g,
            AVG(_s_iso) OVER w AS team_iso_30g,
            AVG(_s_krate) OVER w AS team_k_rate_30g,
            AVG(_s_bbrate) OVER w AS team_bb_rate_30g
        FROM team_off_shifted
        WINDOW w AS (PARTITION BY batting_team ORDER BY game_date
                     ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE team_off_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_woba) OVER w AS team_woba_std,
            AVG(_s_iso) OVER w AS team_iso_std,
            AVG(_s_krate) OVER w AS team_k_rate_std,
            AVG(_s_bbrate) OVER w AS team_bb_rate_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(team_woba_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_woba,
                     LAG(team_iso_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_iso,
                     LAG(team_k_rate_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_krate,
                     LAG(team_bb_rate_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_bbrate
              FROM team_offense_raw)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)


def _build_bullpen(con: duckdb.DuckDBPyConnection) -> None:
    """Bullpen rolling features (10g/3g WHIP, 10g ERA, 3-day workload).

    Reliever workload is attributed to the FIELDING team (Top half -> home
    team pitchers, Bottom half -> away team), and BOTH starting pitchers
    are excluded (the reference repo's join-fanout bug).
    """
    reliever_events = PA_END_EVENTS
    con.execute(f"""
        CREATE TABLE bullpen_raw AS
        WITH re AS (
            SELECT CAST(p.game_date AS DATE) AS game_date,
                   p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                        ELSE p.away_team END AS fielding_team,
                   p.events
            FROM pitches p
            JOIN starters s ON p.game_pk = s.game_pk
            WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
              AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
              AND p.events IN ({reliever_events})
        )
        SELECT game_date, game_pk, fielding_team AS team,
            SUM(CASE events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END) / 3.0 AS bullpen_ip,
            SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play')
                     THEN 1 ELSE 0 END) AS bullpen_ks,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bullpen_bbs,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run')
                     THEN 1 ELSE 0 END) AS bullpen_hits,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run',
                                     'walk', 'hit_by_pitch')
                     THEN 1 ELSE 0 END) AS bullpen_runs
        FROM re
        GROUP BY game_date, game_pk, fielding_team
    """)
    con.execute("""
        CREATE TABLE bp_daily AS
        SELECT CAST(p.game_date AS DATE) AS day,
               CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                    ELSE p.away_team END AS team,
               COUNT(*) AS pitches,
               SUM(CASE p.events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END) / 3.0 AS ip
        FROM pitches p
        JOIN starters s ON p.game_pk = s.game_pk
        WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
          AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TABLE bp_fatigue AS
        WITH games AS (
            SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                   home_team, away_team FROM pitches
        ),
        home_load AS (
            SELECT g.game_pk, SUM(d.pitches) AS pitches_3d, SUM(d.ip) AS ip_3d
            FROM games g JOIN bp_daily d
              ON d.team = g.home_team
             AND d.day < g.gd AND d.day >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        ),
        away_load AS (
            SELECT g.game_pk, SUM(d.pitches) AS pitches_3d, SUM(d.ip) AS ip_3d
            FROM games g JOIN bp_daily d
              ON d.team = g.away_team
             AND d.day < g.gd AND d.day >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        )
        SELECT g.game_pk,
               h.pitches_3d AS bullpen_pitches_3d_home,
               h.ip_3d AS bullpen_ip_3d_home,
               a.pitches_3d AS bullpen_pitches_3d_away,
               a.ip_3d AS bullpen_ip_3d_away
        FROM games g
        LEFT JOIN home_load h ON g.game_pk = h.game_pk
        LEFT JOIN away_load a ON g.game_pk = a.game_pk
    """)
    con.execute("""
        CREATE TABLE bullpen_shifted AS
        SELECT *,
            LAG(bullpen_bbs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_bbs,
            LAG(bullpen_hits, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_hits,
            LAG(bullpen_ip, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_ip,
            LAG(bullpen_runs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_runs
        FROM bullpen_raw
    """)
    con.execute("""
        CREATE TABLE bullpen_rolling AS
        SELECT game_date, game_pk, team,
            (SUM(_s_bbs) OVER w10 + SUM(_s_hits) OVER w10)
                / NULLIF(SUM(_s_ip) OVER w10, 0) AS bullpen_whip_10g,
            SUM(_s_runs) OVER w10 / NULLIF(SUM(_s_ip) OVER w10, 0) * 9.0
                AS bullpen_era_10g,
            (SUM(_s_bbs) OVER w3 + SUM(_s_hits) OVER w3)
                / NULLIF(SUM(_s_ip) OVER w3, 0) AS bullpen_whip_3g
        FROM bullpen_shifted
        WINDOW w10 AS (PARTITION BY team ORDER BY game_date
                       ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
               w3 AS (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE bullpen_season AS
        SELECT game_date, game_pk, team,
            (SUM(_s_bbs) OVER w + SUM(_s_hits) OVER w)
                / NULLIF(SUM(_s_ip) OVER w, 0) AS bullpen_whip_std,
            SUM(_s_runs) OVER w / NULLIF(SUM(_s_ip) OVER w, 0) * 9.0
                AS bullpen_era_std
        FROM (SELECT game_date, game_pk, team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(bullpen_bbs, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_bbs,
                     LAG(bullpen_hits, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_hits,
                     LAG(bullpen_ip, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_ip,
                     LAG(bullpen_runs, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_runs
              FROM bullpen_raw)
        WINDOW w AS (PARTITION BY team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)


def _build_contact_and_lineups(con: duckdb.DuckDBPyConnection) -> None:
    """Team contact form (15g barrel/hard-hit/exit-velo) + lineup wOBA."""
    con.execute("""
        CREATE TABLE team_contact_raw AS
        WITH bip AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   CASE WHEN launch_speed >= 98
                             AND launch_angle BETWEEN 26 AND 30
                        THEN 1.0 ELSE 0.0 END AS barrel_flag,
                   CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag,
                   launch_speed
            FROM pitches
            WHERE description = 'hit_into_play' AND launch_speed IS NOT NULL
        )
        SELECT game_date, game_pk, batting_team,
               AVG(barrel_flag) AS barrel_rate,
               AVG(hard_flag) AS hardhit_rate,
               AVG(launch_speed) AS exitvelo
        FROM bip
        GROUP BY game_date, game_pk, batting_team
    """)
    con.execute("""
        CREATE TABLE team_contact_shifted AS
        SELECT *,
            LAG(barrel_rate, 1) OVER w AS _s_barrel,
            LAG(hardhit_rate, 1) OVER w AS _s_hardhit,
            LAG(exitvelo, 1) OVER w AS _s_exitvelo
        FROM team_contact_raw
        WINDOW w AS (PARTITION BY batting_team ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE team_contact_rolling AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_barrel) OVER w15 AS team_barrel_15g,
            AVG(_s_hardhit) OVER w15 AS team_hardhit_15g,
            AVG(_s_exitvelo) OVER w15 AS team_exitvelo_15g
        FROM team_contact_shifted
        WINDOW w15 AS (PARTITION BY batting_team ORDER BY game_date
                       ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE team_contact_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_barrel) OVER w AS team_barrel_std,
            AVG(_s_hardhit) OVER w AS team_hardhit_std,
            AVG(_s_exitvelo) OVER w AS team_exitvelo_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(barrel_rate, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_barrel,
                     LAG(hardhit_rate, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_hardhit,
                     LAG(exitvelo, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_exitvelo
              FROM team_contact_raw)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    # Lineup composition: per-batter trailing-30g wOBA shrunk toward the
    # point-in-time league mean by PA count (empirical Bayes), aggregated
    # over the expected top-9 by playing time.
    con.execute(f"""
        CREATE TABLE batter_game_stats AS
        WITH pa AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   batter, events
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        )
        SELECT game_date, game_pk, batting_team, batter,
               COUNT(*) AS pa,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
               SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS s,
               SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS d,
               SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS t,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM pa
        GROUP BY game_date, game_pk, batting_team, batter
    """)
    con.execute("""
        CREATE TABLE batter_shifted AS
        SELECT *,
            LAG(pa, 1) OVER w AS _pa,
            LAG(bb, 1) OVER w AS _bb,
            LAG(hbp, 1) OVER w AS _hbp,
            LAG(s, 1) OVER w AS _s,
            LAG(d, 1) OVER w AS _d,
            LAG(t, 1) OVER w AS _t,
            LAG(hr, 1) OVER w AS _hr
        FROM batter_game_stats
        WINDOW w AS (PARTITION BY batter ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE batter_rolling AS
        SELECT game_date, game_pk, batting_team, batter,
            SUM(0.690*_bb + 0.722*_hbp + 0.878*_s + 1.242*_d
                + 1.568*_t + 2.007*_hr) OVER w30 AS _woba_num,
            SUM(_pa - _bb - _hbp) OVER w30 AS _ab,
            SUM(_pa) OVER w30 AS _pa30
        FROM batter_shifted
        WINDOW w30 AS (PARTITION BY batter ORDER BY game_date
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE batter_league AS
        SELECT game_date,
            SUM(_woba_num) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / NULLIF(SUM(_ab) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
              AS lg_woba
        FROM (
            SELECT game_date, SUM(_woba_num) AS _woba_num, SUM(_ab) AS _ab
            FROM batter_rolling GROUP BY game_date
        )
    """)
    con.execute("""
        CREATE TABLE batter_ratings AS
        SELECT r.game_date, r.game_pk, r.batting_team, r.batter,
               CASE WHEN COALESCE(r._ab, 0) > 0 THEN
                   (r._woba_num + COALESCE(l.lg_woba, 0.315) * 120)
                   / (r._ab + 120)
               END AS shrunk_woba,
               r._pa30
        FROM batter_rolling r
        LEFT JOIN batter_league l USING (game_date)
    """)
    con.execute("""
        CREATE TABLE lineup_agg AS
        WITH ranked AS (
            SELECT game_date, game_pk, batting_team, shrunk_woba,
                   ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team
                                      ORDER BY _pa30 DESC) AS rn
            FROM batter_ratings WHERE shrunk_woba IS NOT NULL
        ),
        top9 AS (SELECT * FROM ranked WHERE rn <= 9)
        SELECT game_date, game_pk, batting_team,
               AVG(shrunk_woba) AS lineup_woba_mean,
               AVG(CASE WHEN rn <= 3 THEN shrunk_woba END) AS lineup_woba_top3,
               STDDEV(shrunk_woba) AS lineup_woba_std
        FROM top9
        GROUP BY game_date, game_pk, batting_team
    """)
    con.execute("""
        CREATE TABLE lineup_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_lwm) OVER w AS lineup_woba_mean_std,
            AVG(_s_lwt) OVER w AS lineup_woba_top3_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(lineup_woba_mean, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_lwm,
                     LAG(lineup_woba_top3, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_lwt
              FROM lineup_agg)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)


def _build_travel_and_closer(con: duckdb.DuckDBPyConnection) -> None:
    """Travel fatigue (time zones crossed in the last 3 days) + closer
    availability (whether the team's highest-save reliever threw yesterday)."""
    con.execute("""
        CREATE TABLE game_venue_tz AS
        SELECT DISTINCT game_pk, game_date, home_team, away_team,
            CASE home_team
                WHEN 'LAD' THEN 'PT' WHEN 'SF' THEN 'PT' WHEN 'SD' THEN 'PT'
                WHEN 'LAA' THEN 'PT' WHEN 'ARI' THEN 'PT' WHEN 'SEA' THEN 'PT'
                WHEN 'COL' THEN 'MT' WHEN 'TEX' THEN 'CT' WHEN 'HOU' THEN 'CT'
                WHEN 'KC' THEN 'CT' WHEN 'MIN' THEN 'CT' WHEN 'CWS' THEN 'CT'
                WHEN 'STL' THEN 'CT' WHEN 'MIL' THEN 'CT'
                WHEN 'ATL' THEN 'ET' WHEN 'NYM' THEN 'ET' WHEN 'NYY' THEN 'ET'
                WHEN 'BOS' THEN 'ET' WHEN 'PHI' THEN 'ET' WHEN 'TOR' THEN 'ET'
                WHEN 'BAL' THEN 'ET' WHEN 'WSH' THEN 'ET' WHEN 'MIA' THEN 'ET'
                WHEN 'TB' THEN 'ET' WHEN 'PIT' THEN 'ET' WHEN 'CIN' THEN 'ET'
                WHEN 'CLE' THEN 'ET' WHEN 'DET' THEN 'ET'
                ELSE 'ET' END AS tz
        FROM pitches
    """)
    con.execute("""
        CREATE TABLE team_travel_raw AS
        SELECT game_pk, game_date, team, tz FROM (
            SELECT game_pk, game_date, home_team AS team, tz FROM game_venue_tz
            UNION ALL
            SELECT game_pk, game_date, away_team AS team, tz FROM game_venue_tz
        )
    """)
    con.execute("""
        CREATE TABLE travel_seq AS
        SELECT *,
            LAG(tz, 1) OVER (PARTITION BY team ORDER BY game_date) AS prev_tz_1,
            LAG(tz, 2) OVER (PARTITION BY team ORDER BY game_date) AS prev_tz_2,
            LAG(tz, 3) OVER (PARTITION BY team ORDER BY game_date) AS prev_tz_3
        FROM team_travel_raw
    """)
    con.execute("""
        CREATE TABLE travel_cross AS
        SELECT game_pk, game_date, team,
            (CASE WHEN prev_tz_1 IS NOT NULL AND tz != prev_tz_1 THEN 1 ELSE 0 END
           + CASE WHEN prev_tz_2 IS NOT NULL AND prev_tz_1 != prev_tz_2 THEN 1 ELSE 0 END
           + CASE WHEN prev_tz_3 IS NOT NULL AND prev_tz_2 != prev_tz_3 THEN 1 ELSE 0 END)
            AS zones_crossed_3d
        FROM travel_seq
    """)
    con.execute("""
        CREATE TABLE travel_fatigue AS
        SELECT g.game_pk,
            COALESCE(h.zones_crossed_3d, 0) AS time_zones_crossed_last_3d_home,
            COALESCE(a.zones_crossed_3d, 0) AS time_zones_crossed_last_3d_away
        FROM (SELECT DISTINCT game_pk FROM pitches) g
        LEFT JOIN (SELECT game_pk, zones_crossed_3d FROM travel_cross
                   WHERE team = (SELECT home_team FROM pitches p
                                 WHERE p.game_pk = travel_cross.game_pk LIMIT 1)) h
               ON g.game_pk = h.game_pk
        LEFT JOIN (SELECT game_pk, zones_crossed_3d FROM travel_cross
                   WHERE team = (SELECT away_team FROM pitches p
                                 WHERE p.game_pk = travel_cross.game_pk LIMIT 1)) a
               ON g.game_pk = a.game_pk
    """)
    # Closer availability: the team's most-used late-inning reliever
    # (highest pitch count in the trailing 30 days) did NOT pitch yesterday.
    con.execute("""
        CREATE TABLE late_relief AS
        SELECT p.game_pk, CAST(p.game_date AS DATE) AS game_date,
               CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                    ELSE p.away_team END AS fielding_team,
               p.pitcher, p.inning
        FROM pitches p
        JOIN starters s ON p.game_pk = s.game_pk
        WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
          AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
          AND p.inning >= 8
    """)
    con.execute("""
        CREATE TABLE rel_team AS
        SELECT fielding_team AS team, pitcher, SUM(1) AS n_pitches
        FROM late_relief
        GROUP BY fielding_team, pitcher
    """)
    con.execute("""
        CREATE TABLE team_closer_pit AS
        SELECT team, pitcher FROM (
            SELECT team, pitcher,
                   ROW_NUMBER() OVER (PARTITION BY team ORDER BY n_pitches DESC)
                   AS rn
            FROM rel_team
        ) WHERE rn = 1
    """)
    con.execute("""
        CREATE TABLE rel_daily AS
        SELECT game_date AS day, fielding_team AS team, pitcher,
               COUNT(*) AS n_pitches
        FROM late_relief
        GROUP BY game_date, fielding_team, pitcher
    """)
    con.execute("""
        CREATE TABLE rel_cum AS
        SELECT d.day, d.team, d.pitcher,
               SUM(d.n_pitches) OVER w30 AS n_pitches_30d
        FROM rel_daily d
        WINDOW w30 AS (PARTITION BY d.team, d.pitcher ORDER BY d.day
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE closer_avail AS
        WITH games AS (
            SELECT DISTINCT game_pk, game_date, home_team, away_team
            FROM pitches
        ),
        threw_yesterday AS (
            SELECT g.game_pk, g.home_team, g.away_team,
                   MAX(CASE WHEN rd.team = g.home_team THEN 1 ELSE 0 END)
                       AS home_closer_threw,
                   MAX(CASE WHEN rd.team = g.away_team THEN 1 ELSE 0 END)
                       AS away_closer_threw
            FROM games g
            LEFT JOIN rel_daily rd
              ON ((rd.team = g.home_team OR rd.team = g.away_team))
             AND rd.day < g.game_date
             AND rd.day >= g.game_date - INTERVAL 1 DAY
             AND rd.pitcher IN (SELECT pitcher FROM team_closer_pit
                                WHERE team = rd.team)
            GROUP BY g.game_pk, g.home_team, g.away_team
        )
        SELECT game_pk,
               CASE WHEN home_closer_threw = 1 THEN 0.0 ELSE 1.0 END
                   AS closer_available_home,
               CASE WHEN away_closer_threw = 1 THEN 0.0 ELSE 1.0 END
                   AS closer_available_away
        FROM threw_yesterday
    """)


def _build_game_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build the game_level table via pure DuckDB SQL."""
    # 1. Game winners (last pitch of each game).
    con.execute("""
        CREATE TABLE game_winners AS
        WITH last_pitch AS (
            SELECT game_pk, game_date, home_team, away_team,
                   home_score, away_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk
                       ORDER BY at_bat_number DESC, pitch_number DESC
                   ) AS rn
            FROM pitches
        )
        SELECT game_pk,
               CAST(game_date AS DATE) AS game_date,
               home_team, away_team,
               home_score, away_score,
               CASE WHEN home_score > away_score THEN 1.0
                    WHEN away_score > home_score THEN 0.0
                    ELSE NULL END AS home_win,
               (COALESCE(home_score, 0) + COALESCE(away_score, 0)) AS total_runs
        FROM last_pitch WHERE rn = 1
    """)
    _build_starters_and_venues(con)
    _build_rest_days(con)
    _build_pitcher_features(con)
    _build_team_offense(con)
    _build_bullpen(con)
    _build_contact_and_lineups(con)
    _build_travel_and_closer(con)

    con.execute("""
        CREATE TABLE game_level AS
        SELECT
            w.game_pk, w.game_date, w.home_team, w.away_team,
            w.home_score, w.away_score, w.home_win, w.total_runs,
            s.home_starter_id, s.away_starter_id, v.venue,
            rh.rest_days AS rest_days_home, ra.rest_days AS rest_days_away,
            sh.sp_era AS sp_era_home, sh.sp_k9 AS sp_k9_home,
            sa.sp_era AS sp_era_away, sa.sp_k9 AS sp_k9_away,
            sh.sp_era_5g AS sp_era_5g_home, sh.sp_k9_5g AS sp_k9_5g_home,
            sa.sp_era_5g AS sp_era_5g_away, sa.sp_k9_5g AS sp_k9_5g_away,
            sh.sp_bb9_30g AS sp_bb9_home, sh.sp_whip_30g AS sp_whip_home,
            sh.sp_fip_30g AS sp_fip_home, sh.sp_xwoba_30g AS sp_xwoba_home,
            sa.sp_bb9_30g AS sp_bb9_away, sa.sp_whip_30g AS sp_whip_away,
            sa.sp_fip_30g AS sp_fip_away, sa.sp_xwoba_30g AS sp_xwoba_away,
            th.team_woba_30g AS team_woba_30g_home,
            th.team_iso_30g AS team_iso_30g_home,
            th.team_k_rate_30g AS team_k_rate_30g_home,
            th.team_bb_rate_30g AS team_bb_rate_30g_home,
            ta.team_woba_30g AS team_woba_30g_away,
            ta.team_iso_30g AS team_iso_30g_away,
            ta.team_k_rate_30g AS team_k_rate_30g_away,
            ta.team_bb_rate_30g AS team_bb_rate_30g_away,
            tosh.team_woba_std AS team_woba_std_home,
            tosh.team_iso_std AS team_iso_std_home,
            tosh.team_k_rate_std AS team_k_rate_std_home,
            tosh.team_bb_rate_std AS team_bb_rate_std_home,
            tosa.team_woba_std AS team_woba_std_away,
            tosa.team_iso_std AS team_iso_std_away,
            tosa.team_k_rate_std AS team_k_rate_std_away,
            tosa.team_bb_rate_std AS team_bb_rate_std_away,
            bh.bullpen_whip_10g AS bullpen_whip_10g_home,
            bh.bullpen_era_10g AS bullpen_era_10g_home,
            ba.bullpen_whip_10g AS bullpen_whip_10g_away,
            ba.bullpen_era_10g AS bullpen_era_10g_away,
            bh.bullpen_whip_3g AS bullpen_whip_3g_home,
            ba.bullpen_whip_3g AS bullpen_whip_3g_away,
            bphs.bullpen_whip_std AS bullpen_whip_std_home,
            bphs.bullpen_era_std AS bullpen_era_std_home,
            bpsa.bullpen_whip_std AS bullpen_whip_std_away,
            bpsa.bullpen_era_std AS bullpen_era_std_away,
            ch.team_barrel_15g AS team_barrel_15g_home,
            ch.team_hardhit_15g AS team_hardhit_15g_home,
            ch.team_exitvelo_15g AS team_exitvelo_15g_home,
            ca.team_barrel_15g AS team_barrel_15g_away,
            ca.team_hardhit_15g AS team_hardhit_15g_away,
            ca.team_exitvelo_15g AS team_exitvelo_15g_away,
            tcsh.team_barrel_std AS team_barrel_std_home,
            tcsh.team_hardhit_std AS team_hardhit_std_home,
            tcsh.team_exitvelo_std AS team_exitvelo_std_home,
            tcsa.team_barrel_std AS team_barrel_std_away,
            tcsa.team_hardhit_std AS team_hardhit_std_away,
            tcsa.team_exitvelo_std AS team_exitvelo_std_away,
            lh.lineup_woba_mean AS lineup_woba_mean_home,
            lh.lineup_woba_top3 AS lineup_woba_top3_home,
            lh.lineup_woba_std AS lineup_woba_std_home,
            la.lineup_woba_mean AS lineup_woba_mean_away,
            la.lineup_woba_top3 AS lineup_woba_top3_away,
            la.lineup_woba_std AS lineup_woba_std_away,
            lsh.lineup_woba_mean_std AS lineup_woba_mean_std_home,
            lsh.lineup_woba_top3_std AS lineup_woba_top3_std_home,
            lsa.lineup_woba_mean_std AS lineup_woba_mean_std_away,
            lsa.lineup_woba_top3_std AS lineup_woba_top3_std_away,
            tf.time_zones_crossed_last_3d_home,
            tf.time_zones_crossed_last_3d_away,
            cl.closer_available_home, cl.closer_available_away,
            bf.bullpen_pitches_3d_home, bf.bullpen_ip_3d_home,
            bf.bullpen_pitches_3d_away, bf.bullpen_ip_3d_away,
            sh.sp_era_5g - sh.sp_era AS sp_era_delta_home,
            sh.sp_k9_5g - sh.sp_k9 AS sp_k9_delta_home,
            sh.sp_bb9_30g - sh.sp_bb9_std AS sp_bb9_delta_home,
            sh.sp_whip_30g - sh.sp_whip_std AS sp_whip_delta_home,
            sh.sp_xwoba_30g - sh.sp_xwoba_std AS sp_xwoba_delta_home,
            th.team_woba_30g - tosh.team_woba_std AS woba_delta_home,
            th.team_iso_30g - tosh.team_iso_std AS team_iso_delta_home,
            th.team_k_rate_30g - tosh.team_k_rate_std AS team_k_rate_delta_home,
            th.team_bb_rate_30g - tosh.team_bb_rate_std AS team_bb_rate_delta_home,
            ch.team_barrel_15g - tcsh.team_barrel_std AS team_barrel_delta_home,
            ch.team_hardhit_15g - tcsh.team_hardhit_std AS team_hardhit_delta_home,
            ch.team_exitvelo_15g - tcsh.team_exitvelo_std AS team_exitvelo_delta_home,
            bh.bullpen_whip_10g - bphs.bullpen_whip_std AS bullpen_whip_delta_home,
            bh.bullpen_era_10g - bphs.bullpen_era_std AS bullpen_era_delta_home,
            lh.lineup_woba_mean - lsh.lineup_woba_mean_std AS lineup_woba_mean_delta_home,
            lh.lineup_woba_top3 - lsh.lineup_woba_top3_std AS lineup_woba_top3_delta_home,
            sa.sp_era_5g - sa.sp_era AS sp_era_delta_away,
            sa.sp_k9_5g - sa.sp_k9 AS sp_k9_delta_away,
            sa.sp_bb9_30g - sa.sp_bb9_std AS sp_bb9_delta_away,
            sa.sp_whip_30g - sa.sp_whip_std AS sp_whip_delta_away,
            sa.sp_xwoba_30g - sa.sp_xwoba_std AS sp_xwoba_delta_away,
            ta.team_woba_30g - tosa.team_woba_std AS woba_delta_away,
            ta.team_iso_30g - tosa.team_iso_std AS team_iso_delta_away,
            ta.team_k_rate_30g - tosa.team_k_rate_std AS team_k_rate_delta_away,
            ta.team_bb_rate_30g - tosa.team_bb_rate_std AS team_bb_rate_delta_away,
            ca.team_barrel_15g - tcsa.team_barrel_std AS team_barrel_delta_away,
            ca.team_hardhit_15g - tcsa.team_hardhit_std AS team_hardhit_delta_away,
            ca.team_exitvelo_15g - tcsa.team_exitvelo_std AS team_exitvelo_delta_away,
            ba.bullpen_whip_10g - bpsa.bullpen_whip_std AS bullpen_whip_delta_away,
            ba.bullpen_era_10g - bpsa.bullpen_era_std AS bullpen_era_delta_away,
            la.lineup_woba_mean - lsa.lineup_woba_mean_std AS lineup_woba_mean_delta_away,
            la.lineup_woba_top3 - lsa.lineup_woba_top3_std AS lineup_woba_top3_delta_away,
            s.home_starter_hand, s.away_starter_hand
        FROM game_winners w
        LEFT JOIN starters s ON w.game_pk = s.game_pk
        LEFT JOIN venues v ON w.game_pk = v.game_pk
        LEFT JOIN rest_days rh ON w.game_pk = rh.game_pk AND w.home_team = rh.team
        LEFT JOIN rest_days ra ON w.game_pk = ra.game_pk AND w.away_team = ra.team
        LEFT JOIN pitcher_features sh ON w.game_pk = sh.game_pk AND sh.pitcher = s.home_starter_id
        LEFT JOIN pitcher_features sa ON w.game_pk = sa.game_pk AND sa.pitcher = s.away_starter_id
        LEFT JOIN team_offense_rolling th ON w.game_pk = th.game_pk AND w.home_team = th.batting_team
        LEFT JOIN team_offense_rolling ta ON w.game_pk = ta.game_pk AND w.away_team = ta.batting_team
        LEFT JOIN team_off_season tosh ON w.game_pk = tosh.game_pk AND w.home_team = tosh.batting_team
        LEFT JOIN team_off_season tosa ON w.game_pk = tosa.game_pk AND w.away_team = tosa.batting_team
        LEFT JOIN bullpen_rolling bh ON w.game_pk = bh.game_pk AND w.home_team = bh.team
        LEFT JOIN bullpen_rolling ba ON w.game_pk = ba.game_pk AND w.away_team = ba.team
        LEFT JOIN bullpen_season bphs ON w.game_pk = bphs.game_pk AND w.home_team = bphs.team
        LEFT JOIN bullpen_season bpsa ON w.game_pk = bpsa.game_pk AND w.away_team = bpsa.team
        LEFT JOIN team_contact_rolling ch ON w.game_pk = ch.game_pk AND w.home_team = ch.batting_team
        LEFT JOIN team_contact_rolling ca ON w.game_pk = ca.game_pk AND w.away_team = ca.batting_team
        LEFT JOIN team_contact_season tcsh ON w.game_pk = tcsh.game_pk AND w.home_team = tcsh.batting_team
        LEFT JOIN team_contact_season tcsa ON w.game_pk = tcsa.game_pk AND w.away_team = tcsa.batting_team
        LEFT JOIN lineup_agg lh ON w.game_pk = lh.game_pk AND w.home_team = lh.batting_team
        LEFT JOIN lineup_agg la ON w.game_pk = la.game_pk AND w.away_team = la.batting_team
        LEFT JOIN lineup_season lsh ON w.game_pk = lsh.game_pk AND w.home_team = lsh.batting_team
        LEFT JOIN lineup_season lsa ON w.game_pk = lsa.game_pk AND w.away_team = lsa.batting_team
        LEFT JOIN travel_fatigue tf ON w.game_pk = tf.game_pk
        LEFT JOIN closer_avail cl ON w.game_pk = cl.game_pk
        LEFT JOIN bp_fatigue bf ON w.game_pk = bf.game_pk
    """)

    for tbl in (
        "game_winners", "starters", "venues", "rest_days",
        "pa_boundary", "pitcher_game_stats", "pitcher_shifted",
        "pitcher_shifted_season", "pitcher_rolling", "pitcher_features",
        "team_offense_raw", "team_off_shifted", "team_offense_rolling",
        "team_off_season",
        "bullpen_raw", "bullpen_shifted", "bullpen_rolling", "bullpen_season",
        "bp_daily", "bp_fatigue",
        "team_contact_raw", "team_contact_shifted", "team_contact_rolling",
        "team_contact_season",
        "batter_game_stats", "batter_shifted", "batter_rolling",
        "batter_league", "batter_ratings", "lineup_agg", "lineup_season",
        "game_venue_tz", "team_travel_raw", "travel_seq", "travel_cross",
        "travel_fatigue",
        "late_relief", "rel_daily", "rel_cum", "rel_team", "team_closer_pit",
        "closer_avail",
    ):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")


def _build_pbp_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build the lean pitch-level cache table via pure DuckDB SQL."""
    con.execute(f"""
        CREATE TABLE pbp_level AS
        SELECT
            game_pk, game_date, home_team, away_team,
            inning, inning_topbot, at_bat_number, pitch_number,
            pitcher, batter, player_name,
            p_throws, stand,
            pitch_type, release_speed,
            description, events,
            launch_speed, launch_angle, hit_distance_sc,
            estimated_woba_using_speedangle,
            home_score, away_score,
            delta_home_win_exp, delta_run_exp
        FROM pitches
    """)


def build_features(pitches_path: str | Path, output_dir: str | Path = ".",
                   validate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pure-DuckDB feature engineering on a pitches Parquet file.

    Reads pitches.parquet, builds game_level + pbp_level via SQL, writes
    them to disk, and returns pandas frames ONLY for the model/training
    stage. The historical dataset is never fully loaded into RAM during
    feature engineering (DuckDB spills to disk under the memory limit).

    Returns:
        (game_df, pbp_df) as pandas DataFrames.
    """
    pitches_path = Path(pitches_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game_out = out_dir / "game_level_features.parquet"
    pbp_out = out_dir / "pbp_level_features.parquet"

    logger.info("=== DuckDB Feature Engineering ===")
    con = _connect(pitches_path)
    try:
        _build_game_level(con)
        con.execute(
            f"COPY (SELECT * FROM game_level) TO '{game_out}' (FORMAT PARQUET)")
        logger.info("Exported game_level: %.1f MB", game_out.stat().st_size / 1e6)
        _build_pbp_level(con)
        con.execute(
            f"COPY (SELECT * FROM pbp_level) TO '{pbp_out}' (FORMAT PARQUET)")
        logger.info("Exported pbp_level: %.1f MB", pbp_out.stat().st_size / 1e6)
        con.execute("DROP TABLE IF EXISTS pbp_level")
    finally:
        con.close()

    game_df = pd.read_parquet(game_out)
    pbp_df = pd.read_parquet(pbp_out)

    for df in (game_df, pbp_df):
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            if df[col].isna().any():
                continue
            if df[col].max() < 32767 and df[col].min() >= -32768:
                df[col] = df[col].astype("int16")

    logger.info("=== Complete: %d games, %d pitches ===",
                len(game_df), len(pbp_df))
    return game_df, pbp_df
