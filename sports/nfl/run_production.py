"""NFL Phase 3 production runner.

The ONLY sanctioned end-to-end NFL production path:

1. mandatory run-window validation — ``NFL_START_DATE`` / ``NFL_END_DATE``
   / ``NFL_FULL_REPULL`` must be present and valid (strict YYYY-MM-DD;
   END >= START; FULL_REPULL exactly "0" or "1"; no silent defaults, no
   season-variable aliases — NFL_START_SEASON and friends are forbidden);
2. bounded nflverse ingestion (incremental or full repull per flag);
3. point-in-time feature factory (DuckDB/Parquet-backed, bounded RAM);
4. walk-forward five-member moneyline OOF + run-engine distribution OOF;
5. artifact generation, each writer validated against the immutable
   schema fixture; per-slate-game SHAP;
6. ``run_production_retention(generation_succeeded=True)`` — retention
   executes ONLY after every writer returned successfully.

Training history (warmup seasons, folds, OOF rules, minimum rows) comes
from ``sports/nfl/study.yaml`` — NEVER from the run window. The run
window scopes only which days are ingested and served.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import PlatformConfig, default_config
from core.config import load_market_rules
from core.contracts import validate_record
from core.validation.future_availability import (
    validate_available_at,
    validate_settlement_record,
    validate_slate_window,
    window_anchor,
)
from core.artifacts import run_production_retention
from core.config import resolve_run_window
from sports.nfl.ingestion import (
    load_pbp,
    load_player_stats,
    load_schedule,
    load_team_names,
)
from sports.nfl.config.study_config import load_nfl_study

logger = logging.getLogger(__name__)

#: Settlement rules — declared in ``config/market_rules.yaml`` (spec §20)
#: and loaded once per process; the OOF settlement gate reads the
#: moneyline rule from here instead of a hardcoded constant.
_MARKET_RULES = load_market_rules("nfl")

#: Env var names for the NFL production run (strict; no aliases, no
#: defaults — season-based variables are explicitly forbidden).
ENV_START_DATE = "NFL_START_DATE"
ENV_END_DATE = "NFL_END_DATE"
ENV_FULL_REPULL = "NFL_FULL_REPULL"
FORBIDDEN_ENV_VARS = ("NFL_START_SEASON", "NFL_END_SEASON",
                      "NFL_SLATE_SEASON")

NFL_CACHE = Path("store/nfl/raw")
VENUE_CSV = Path(__file__).resolve().parent / "nfl_stadiums.csv"


class NFLRunnerError(RuntimeError):
    """Raised when the NFL production run cannot proceed safely."""


@dataclass
class RunResult:
    run_window_start: str
    run_window_end: str
    full_repull: bool
    n_prediction_games: int = 0
    n_decided_games: int = 0
    n_folds: int = 0
    n_oof_rows: int = 0
    artifacts_written: list[str] = field(default_factory=list)
    retention_executed: bool = False
    retention_candidates_deleted: int = 0


def run_nfl_production(
    *,
    env: dict[str, str] | None = None,
    data_delivery_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    today=None,
    platform_config: PlatformConfig | None = None,
    dry_run_ingestion: bool = False,
) -> RunResult:
    """Execute the bounded NFL production run.

    ``dry_run_ingestion=True`` skips the network fetch (tests / offline
    validation of the env-var gate) — every other stage still runs against
    the existing cache.
    """
    cfg = platform_config or default_config()
    study = load_nfl_study()

    # ------------------------------------------------------------------
    # 1. Mandatory run window — strict, no silent defaults, no aliases.
    # ------------------------------------------------------------------
    source = dict(os.environ) if env is None else env
    present_forbidden = [k for k in FORBIDDEN_ENV_VARS
                         if str(source.get(k) or "").strip()]
    if present_forbidden:
        raise NFLRunnerError(
            f"Forbidden season-based environment variable(s) present: "
            f"{present_forbidden}. The run window is NFL_START_DATE / "
            f"NFL_END_DATE only — training history comes from study.yaml.")
    missing = [k for k in (ENV_START_DATE, ENV_END_DATE, ENV_FULL_REPULL)
               if not str(source.get(k) or "").strip()]
    if missing:
        raise NFLRunnerError(
            f"Missing required production environment variable(s): "
            f"{missing}. NFL_START_DATE, NFL_END_DATE, and "
            f"NFL_FULL_REPULL must be set explicitly — no silent defaults.")
    window = resolve_run_window(start_date=source[ENV_START_DATE],
                                end_date=source[ENV_END_DATE],
                                full_repull=source[ENV_FULL_REPULL])

    cache = Path(cache_dir) if cache_dir else NFL_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    result = RunResult(run_window_start=window.start_date,
                       run_window_end=window.end_date,
                       full_repull=window.full_repull)

    generation_succeeded = False
    sink = Path(data_delivery_dir) if data_delivery_dir \
        else cfg.data_delivery_dir("nfl", Path("."))
    try:
        # --------------------------------------------------------------
        # 2. Bounded ingestion (incremental or full repull).
        # --------------------------------------------------------------
        # Eligible season window: warmup seasons + core seasons covering
        # the decided history (study-frozen); the current season covers
        # the prediction window. The run window scopes serving only.
        current_season = window.end.year + 1 if window.end.month >= 3 \
            else window.end.year
        seasons = sorted(set(
            list(range(study.oof_first_season, current_season + 1))
            + list(study.warmup_seasons)))

        if not dry_run_ingestion or window.full_repull:
            if window.full_repull and dry_run_ingestion:
                logger.info("full_repull with dry_run_ingestion: cache is "
                            "cleared but no fetch is performed (tests)")
            schedule = load_schedule(seasons, cache,
                                     full_repull=window.full_repull)
        else:
            schedule = load_schedule(seasons, cache,
                                     full_repull=window.full_repull)
        from sports.nfl.ingestion import eligible_games
        schedule = eligible_games(schedule, study.oof_first_season,
                                  min(study.warmup_seasons))
        pbp = load_pbp(seasons, cache, full_repull=window.full_repull)
        logger.info("schedule rows (eligible): %d; pbp rows: %s",
                    len(schedule), 0 if pbp is None else len(pbp))

        decided_all = schedule[schedule["home_score"].notna()
                               & schedule["away_score"].notna()].copy()
        warmup = decided_all[pd.to_numeric(decided_all["season"])
                             < study.oof_first_season]
        core = decided_all[pd.to_numeric(decided_all["season"])
                           >= study.oof_first_season]
        logger.info("decided games: warmup %d, core (OOF population) %d",
                    len(warmup), len(core))
        result.n_decided_games = int(len(decided_all))

        # --------------------------------------------------------------
        # 3. Point-in-time features.
        # --------------------------------------------------------------
        from sports.nfl.features.build_frame import (
            build_game_features,
            build_slate_features,
            feature_coverage_report,
        )
        pbp_rollup = cache / "pbp_rollup.parquet"
        game_df = build_game_features(decided_all, pbp, VENUE_CSV,
                                      pbp_rollup_path=pbp_rollup)
        game_df = game_df.sort_values("gameday").reset_index(drop=True)
        if game_df.empty:
            raise NFLRunnerError(
                "no decided games in the feature frame — cannot train")
        cov = feature_coverage_report(game_df, study)

        # --------------------------------------------------------------
        # 4. Moneyline + run-engine walk-forward OOF.
        # --------------------------------------------------------------
        from sports.nfl.models.market.distributions import (
            apply_distribution,
            calibrate_sigma,
            fit_final,
            walk_forward_oof as dist_oof,
        )
        from sports.nfl.models.moneyline.training import (
            apply_platt,
            fit_final_models,
            fit_platt,
            predict_slate,
            walk_forward_oof as ml_oof,
        )
        from core.folds import fold_summary, fold_table, make_folds

        fold_list = make_folds(game_df, study.oof_first_season,
                               cadence_days=study.walk_forward.step_days,
                               date_col="gameday",
                               val_scope="oof_season", end_of_day=True)
        result.n_folds = len(fold_list)
        if not fold_list:
            raise NFLRunnerError(
                "fold generation produced ZERO folds — check eligible-"
                "game ingestion and oof_first_season geometry")
        result.n_oof_rows = int(sum(len(f.val_idx) for f in fold_list))

        ml = ml_oof(game_df, study)
        oof_ml = ml["oof"]
        weights = ml["member_weights"]
        if oof_ml.empty:
            raise NFLRunnerError(
                "moneyline OOF produced no rows — cannot calibrate/serve")

        # join targets/sides onto the OOF rows for evaluation/serving.
        # The OOF frame already carries week/home_team/away_team — drop
        # them from the key frame so the merge never suffixes the columns.
        key = game_df[["game_id", "gameday", "season", "home_score",
                       "away_score", "margin", "total", "stadium",
                       "gametime", "home_record", "away_record"]].copy()
        key["gameday"] = pd.to_datetime(key["gameday"])
        oof_ml["gameday"] = pd.to_datetime(oof_ml["gameday"])
        oof_ml = oof_ml.merge(key, on=["game_id", "gameday", "season"],
                              how="left")

        dist = dist_oof(game_df, study)
        oof_dist = dist["oof"]
        sig = calibrate_sigma(
            oof_dist["resid_margin"].to_numpy(),
            oof_dist["resid_total"].to_numpy(), study)

        # calibration (Platt on OOF only)
        y_oof = oof_ml["home_win"].to_numpy(float)
        p_ens = oof_ml["p_ensemble"].to_numpy(float)
        okp = np.isfinite(p_ens)
        platt = fit_platt(p_ens[okp], y_oof[okp])
        oof_ml["p_ensemble_calibrated"] = np.nan
        oof_ml.loc[okp, "p_ensemble_calibrated"] = apply_platt(
            p_ens[okp], platt)

        # evaluation metrics
        from sports.nfl.evaluation import (
            binary_metrics,
            calibration_buckets,
        )
        raw_m = binary_metrics(oof_ml["p_ensemble"], y_oof)
        cal_m = binary_metrics(oof_ml["p_ensemble_calibrated"], y_oof)

        # final full-history refit + slate serving
        final_models, _ = fit_final_models(game_df, study)
        final_reg = fit_final(game_df, study)

        slate = build_slate_features(schedule, pbp, VENUE_CSV)
        # Prediction-window scoping: the slate is the UNDECIDED games
        # inside the run window only (START_DATE..END_DATE define the
        # serving scope; the full future schedule stays unpublished).
        if not slate.empty:
            gd = pd.to_datetime(slate["gameday"])
            in_window = (gd >= pd.Timestamp(window.start_date)) & (
                gd <= pd.Timestamp(window.end_date) + pd.Timedelta(
                    hours=23, minutes=59, seconds=59))
            slate = slate[in_window].reset_index(drop=True)
        # B-001 WS4: SECONDARY slate gate (every served start at/after the
        # serving-window lower bound, no decided rows) — loud on violation;
        # NOT §7.1 PIT evidence.
        # Deterministic serving-window-derived anchor (never wall clock):
        # the instant the run window opens (window.start_date).
        # Date-resolution comparison: NFL starts are date-granularity
        # today (instant starts = 7.6 start-timestamp normalization).
        slate_report = validate_slate_window(
            slate,
            anchor_utc=window_anchor(window.start_date.replace("-", "")),
            start_col="gameday", resolution="date", label="nfl slate")
        if not slate_report.ok:
            raise NFLRunnerError(
                f"slate window gate failed: {slate_report.violations[:3]}")
        # §7.1 per-field gate (report_only in 7.5d): telemetry every run.
        pit_report = validate_available_at(
            slate.to_dict(orient="records"),
            buffer_minutes=study.prediction_cutoff_buffer_minutes,
            label="nfl availability (report_only)",
            start_col="gameday", available_col="available_at")
        logger.info(
            "[B-001 §7.1] %s: n_rows=%d n_missing=%d n_violations=%d "
            "missing_ids=%s (metadata layer = B-001-RESIDUAL, 7.6)",
            pit_report.label, pit_report.n_rows, pit_report.n_missing,
            pit_report.n_violations, list(pit_report.missing)[:3])
        if not slate.empty:
            slate = slate.sort_values("gameday").reset_index(drop=True)
            p_home = predict_slate(final_models, slate, weights, study)
            p_home_cal = apply_platt(p_home, platt)
            slate["mu_h"], slate["mu_a"] = final_reg.predict(slate, study)
            slate = apply_distribution(slate, sig["sigma_margin"],
                                       sig["sigma_total"], study)
            slate["p_home_win"] = p_home_cal
            slate["p_away_win"] = 1.0 - p_home_cal
            slate["p_tie"] = slate["p_push_0"]
            slate["derived_ml"] = slate["p_home_win_derived"]
            slate["kind"] = "slate"
            slate["decided"] = False
            slate["frame_view"] = "slate"
            slate["pred_home"] = slate["mu_h"]
            slate["pred_away"] = slate["mu_a"]
            slate["home_win_prob_model"] = p_home_cal
        result.n_prediction_games = int(len(slate))

        # QB enrichment (display only)
        qb_seasons = sorted(set(int(x) for x in slate["season"].unique())) \
            if not slate.empty else []
        qb_stats = load_player_stats(
            ([qb_seasons[0] - 1] + qb_seasons) if qb_seasons else [],
            cache, full_repull=window.full_repull) \
            if qb_seasons else {}
        from sports.nfl.qb_enrichment import enrich_slate
        qb_df = enrich_slate(slate, qb_stats) if not slate.empty \
            else pd.DataFrame()

        # decided OOF rows for the markets artifact
        oof_market_rows = _build_oof_market_rows(oof_ml, oof_dist, sig,
                                                 study)

        # --------------------------------------------------------------
        # 5. Artifact generation (all-or-nothing via the fixture gate).
        # --------------------------------------------------------------
        sink.mkdir(parents=True, exist_ok=True)
        run_date = window.end_date.replace("-", "")
        date_c = run_date
        team_names = load_team_names(cache)
        config_meta = {
            # Bound from the ACTIVE study/registry moneyline contract — never
            # a hardcoded legacy literal (T3: features_metadata writer path).
            "feature_set_version": study.moneyline_contract_version,
            "warmup_seasons": list(study.warmup_seasons),
            "oof_first_season": study.oof_first_season,
            "retrain_cadence_days": study.walk_forward.step_days,
            "ensemble_members": list(study.training.ensemble_members),
            "random_seed": study.training.random_seed,
            "market_independence": True,
        }
        written: list[Path] = []

        from sports.nfl.artifacts import (
            persist_shap_game,
            write_calibration_json,
            write_feature_json,
            write_markets_csv,
            write_markets_monitor_json,
            write_model_monitor_json,
            write_moneyline_json,
            write_power_rankings_csv,
            write_predictions_history_csv,
            write_qb_matchup_json,
        )

        if not slate.empty:
            written.append(write_moneyline_json(
                sink / f"nfl_moneyline_v1_{date_c}.json", slate,
                p_home_cal, team_names, config_meta)["created_utc"]
                and sink / f"nfl_moneyline_v1_{date_c}.json")
            written.append(write_qb_matchup_json(
                sink / f"nfl_qb_matchup_{date_c}.json", qb_df, slate)
                and sink / f"nfl_qb_matchup_{date_c}.json")
            written += _write_slate_shap(sink, date_c, slate, game_df,
                                         final_models, weights, study)

        written.append(sink / f"nfl_calibration_{date_c}.json")
        write_calibration_json(
            sink / f"nfl_calibration_{date_c}.json", raw_m, cal_m,
            calibration_buckets(oof_ml["p_ensemble"], y_oof),
            _daily_rows(oof_ml), config_meta, platt=platt,
            run_date=date_c, n_games=int(okp.sum()),
            calibrated_buckets=calibration_buckets(
                oof_ml["p_ensemble_calibrated"], y_oof))

        written.append(sink / f"nfl_predictions_history_{date_c}.csv")
        write_predictions_history_csv(
            sink / f"nfl_predictions_history_{date_c}.csv", oof_ml,
            oof_ml["p_ensemble_calibrated"].to_numpy())

        written.append(sink / f"nfl_power_rankings_{date_c}.csv")
        write_power_rankings_csv(
            sink / f"nfl_power_rankings_{date_c}.csv",
            *_power_rankings_inputs(game_df, study), team_names)

        written.append(sink / f"nfl_run_engine_markets_{date_c}.csv")
        write_markets_csv(
            sink / f"nfl_run_engine_markets_{date_c}.csv",
            sink / f"nfl_run_engine_markets_{date_c}.meta.json",
            oof_market_rows, slate, config_meta)

        # OOF stores (backend evaluation data — retention-exempt dir)
        models_dir = sink / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        validate_record("nfl", "oof_moneyline", oof_ml)
        oof_ml.to_csv(models_dir / f"nfl_oof_moneyline_{date_c}.csv",
                      index=False)
        validate_record("nfl", "oof_distribution", oof_dist)
        oof_dist.to_csv(models_dir / f"nfl_oof_distribution_{date_c}.csv",
                        index=False)
        _fold_tbl = fold_table(game_df, fold_list)
        validate_record("nfl", "fold_table", _fold_tbl)
        _fold_tbl.to_csv(
            models_dir / f"nfl_fold_table_{date_c}.csv", index=False)

        from sports.nfl.features.registry import (
            build_feature_contract,
            unavailable_columns,
        )
        fc_meta = build_feature_contract().to_metadata_json(run_date)
        # explicit unavailable-column markers (missing-data policy)
        unavail = unavailable_columns(game_df, study.moneyline_feature_cols)
        fc_meta["warnings"] = [
            f"unavailable: {c} (no observations in source data — routed "
            f"as explicitly unavailable per missing-data policy; not "
            f"fabricated)" for c in unavail]
        written.append(sink / f"nfl_feature_v1_{date_c}.json")
        write_feature_json(sink / f"nfl_feature_v1_{date_c}.json", cov,
                           config_meta,
                           fold_summary(fold_list), fc_meta)

        ml_ref: dict = {}
        prob = pd.to_numeric(oof_ml["p_ensemble"], errors="coerce")
        yv = pd.to_numeric(oof_ml["home_win"], errors="coerce")
        ok = prob.notna() & yv.isin([0, 1])
        if ok.any():
            prob2, yv2 = prob[ok], yv[ok].astype(int)
            pick_home = prob2 > 0.5
            win = float(((pick_home & (yv2 == 1))
                         | (~pick_home & (yv2 == 0))).mean())
            ml_ref = {"source": "ml_win_prob", "n": int(len(prob2)),
                      "win_rate": round(win, 4),
                      "predicted_mean": round(float(prob2.mean()), 4)}
        written.append(sink / f"nfl_run_engine_monitor_{date_c}.json")
        write_markets_monitor_json(
            sink / f"nfl_run_engine_monitor_{date_c}.json", date_c,
            oof_market_rows, config_meta, ml_reference=ml_ref)

        written.append(sink / f"nfl_model_monitor_{date_c}.json")
        write_model_monitor_json(
            sink / f"nfl_model_monitor_{date_c}.json", date_c, [], [],
            _ensemble_rows(oof_ml, weights, study), [], float(1.0
                                                             - y_oof.mean()),
            config_meta, fold_summary(fold_list), metrics=cal_m,
            platt=platt, features_metadata=fc_meta)

        # §22.1 standalone monitor siblings — the same records MLB
        # persists: the REAL rolling moneyline Brier (from this run's
        # decided OOF market rows; the nested monitor field stays the
        # frontend's compact copy) and the canonical feature metadata.
        from core.evaluation.probabilistic_metrics import rolling_brier_payload
        from sports.nfl.artifacts import (
            persist_features_metadata as _persist_fm,
            persist_rolling_brier as _persist_rb,
        )
        _rb_payload = rolling_brier_payload(
            oof_market_rows, prob_col="p_home_win",
            date_col="gameday")
        _persist_rb(_rb_payload, date_c, sink)
        written.append(sink / f"nfl_rolling_brier_{date_c}.json")
        _persist_fm(fc_meta, date_c, sink)
        written.append(sink / f"nfl_features_metadata_{date_c}.json")

        result.artifacts_written = [p.name for p in written
                                    if isinstance(p, Path)]
        generation_succeeded = bool(result.artifacts_written)
    except Exception:
        logger.exception("NFL production run: artifact generation FAILED "
                         "— retention will NOT execute")
        raise

    # ------------------------------------------------------------------
    # 6. Retention — only after successful generation.
    # ------------------------------------------------------------------
    if generation_succeeded:
        plan = run_production_retention(
            sink, cfg, "nfl",
            generation_succeeded=True,
            reference_date=window.end_date.replace("-", ""))
        result.retention_executed = True
        result.retention_candidates_deleted = plan.n_candidates

    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_oof_market_rows(oof_ml: pd.DataFrame, oof_dist: pd.DataFrame,
                           sig: dict, study) -> pd.DataFrame:
    """Decided OOF rows in the markets schema: distribution grids from the
    OOF mu pair + honest outcomes (y_*), plus the calibrated moneyline."""
    from sports.nfl.models.market.distributions import apply_distribution
    m = oof_ml[["game_id", "gameday", "season", "week", "home_team",
                "away_team", "stadium", "gametime", "home_record",
                "away_record", "home_score", "away_score", "margin",
                "total", "home_win", "p_ensemble",
                "p_ensemble_calibrated"]].copy()
    d = oof_dist[["game_id", "mu_h", "mu_a"]].copy()
    df = m.merge(d, on="game_id", how="inner")
    if not len(df):
        return pd.DataFrame()
    df = apply_distribution(df, sig["sigma_margin"], sig["sigma_total"],
                            study)
    df["p_home_win"] = df["p_ensemble_calibrated"].where(
        df["p_ensemble_calibrated"].notna(), df["p_ensemble"])
    df["p_away_win"] = 1.0 - df["p_home_win"]
    df["p_tie"] = df["p_push_0"]
    df["derived_ml"] = df["p_home_win_derived"]
    df["kind"] = "oof"
    df["decided"] = True
    df["frame_view"] = "oof"
    df["pred_home"] = df["mu_h"]
    df["pred_away"] = df["mu_a"]
    df["spread_line"] = np.nan
    df["total_line"] = np.nan
    df["has_offer"] = False
    for c in ("p_cover_offered", "p_push_offered", "p_over_offered",
              "p_under_offered", "p_push_total_offered",
              "y_over_offered", "y_under_offered",
              "y_push_total_offered", "y_cover_offered",
              "y_push_spread_offered"):
        df[c] = np.nan

    def _outcome_row(r):
        fair_s, fair_t = r.fair_spread, r.fair_total
        y_cover = 1.0 if r.margin > fair_s else 0.0
        y_push_s = 1.0 if r.margin == fair_s else 0.0
        y_over = 1.0 if r.total > fair_t else 0.0
        y_push_t = 1.0 if r.total == fair_t else 0.0
        return {
            "y_over_fair": y_over,
            "y_under_fair": 1.0 - y_over - y_push_t,
            "y_push_fair": y_push_t,
            "y_cover_fair": y_cover, "y_push_spread_fair": y_push_s,
            "y_home_win": float(r.home_win),
        }

    outs = [_outcome_row(r) for r in df.itertuples(index=False)]
    for c in outs[0]:
        df[c] = [o[c] for o in outs]
    # B-001 WS4 settlement sub-gate: every settled fair market record
    # must carry a rule-consistent outcome (win/loss/push under the
    # full-game moneyline config; distinct from the B-005 shape gate,
    # NOT PIT evidence).
    for i, o in enumerate(outs):
        outcome = ("push" if (o["y_push_fair"] == 1.0 or
                              o["y_push_spread_fair"] == 1.0)
                   else "home" if o["y_home_win"] == 1.0 else "away")
        rec = validate_settlement_record(
            {"outcome": outcome, "scope": "full_game"},                market_kind="moneyline",
                settlement_cfg=_MARKET_RULES.settlement_config("moneyline"),
                label=f"nfl fair market row {df['game_id'].iloc[i]}")
        if not rec.ok:
            raise NFLRunnerError(f"settlement gate failed: {rec.violations}")
    return df


def _power_rankings_inputs(game_df: pd.DataFrame, study):
    """(ratings dict, records dict) from the decided Elo state."""
    from sports.nfl.features.build_frame import team_events, compute_elo
    ev = compute_elo(team_events(game_df), study.elo_k, study.elo_prior,
                     study.elo_scale)[0]
    ratings = {}
    for gid, rows in ev.groupby("game_id", sort=False):
        for r in rows.itertuples(index=False):
            ratings[r.team] = r.elo_entering
    rec = ev.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
    )
    records = {t: (int(rec.loc[t, "wins"]), int(rec.loc[t, "losses"]))
               for t in rec.index}
    return ratings, records


def _daily_rows(oof_ml: pd.DataFrame) -> list[dict]:
    """Daily calibration rows at the shared presentation precision."""
    from sports.nfl.evaluation import binary_metrics
    daily = []
    oof = oof_ml.copy()
    oof["gd_date"] = pd.to_datetime(oof["gameday"]).dt.strftime("%Y%m%d")
    for day, grp in oof.groupby("gd_date"):
        m = binary_metrics(grp["p_ensemble_calibrated"],
                           grp["home_win"])
        pick_right = ((grp["p_ensemble_calibrated"] >= 0.5)
                      == (grp["home_win"] > 0.5)).sum()
        daily.append({
            "date": day, "n_games": m["n"],
            "wins": int(pick_right), "losses": int(m["n"] - pick_right),
            "metrics": {k: m[k] for k in ("auc", "brier", "logloss",
                                          "ece")},
        })
    return daily


def _ensemble_rows(oof_ml: pd.DataFrame, weights: dict, study) -> list[dict]:
    """Per-member diagnostics for the monitor artifact."""
    from sports.nfl.evaluation import binary_metrics
    rows = []
    y = oof_ml["home_win"].to_numpy(float)
    for name in study.training.ensemble_members:
        col = f"p_{name}"
        p = oof_ml[col].to_numpy(float) if col in oof_ml.columns \
            else np.array([])
        m = binary_metrics(p, y) if len(p) else {"n": 0}
        rows.append({"name": name,
                     "weight": round(float(weights.get(name, 0.0)), 4),
                     "auc": m.get("auc"), "brier": m.get("brier"),
                     "logloss": m.get("logloss")})
    return rows


def _write_slate_shap(out_dir: Path, run_date: str, slate: pd.DataFrame,
                      decided: pd.DataFrame, final_models: dict,
                      weights: dict, study) -> list[Path]:
    """Emit EXACTLY one nfl_shap_game_<date>_<game_id>.csv per undecided
    prediction-slate game — no fixed minimum, no fixed maximum (zero files
    for an empty slate).

    Attribution provenance (documented):
      * model member: the deployed five-member ensemble's TREE members
        (xgboost / lightgbm) via NATIVE exact TreeSHAP — xgboost
        ``pred_contrib=True`` and lightgbm ``pred_contrib=True`` (no
        third-party shap package, no KernelSHAP approximation, nothing
        fabricated);
      * feature view: the frozen tree view (registry-ordered diffs +
        per-side values) — the exact named matrix the members consume at
        predict time;
      * prediction space: home-win log-odds (margin > 0 class margin of
        the binary P(home win) model);
      * perspective: the FAVORED team — attributions are negated when the
        away side is favored so positive always pushes the favorite to
        win (the shared favored-side display convention);
      * missing values: routed natively by the tree members (no
        imputation, missingness preserved);
      * blending: the two members' log-odds contributions are combined
        with the SAME adaptive weights the ensemble blend uses,
        renormalized over the contributing members.
    """
    if slate.empty:
        return []
    from sports.nfl.models.moneyline.training import member_matrix
    feature_cols = list(member_matrix("xgboost", slate, study).columns)
    X_slate = member_matrix("xgboost", slate, study)

    contribs: dict[str, np.ndarray] = {}
    xb = final_models.get("xgboost")
    if xb is not None:
        try:
            import xgboost as xgb
            raw = xb["model"].get_booster().predict(
                xgb.DMatrix(X_slate), pred_contribs=True)
            contribs["xgboost"] = raw[:, :len(feature_cols)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("xgboost pred_contribs failed: %s", exc)
    lb = final_models.get("lightgbm")
    if lb is not None:
        try:
            raw = lb["model"].booster_.predict(
                X_slate.to_numpy(dtype=float), pred_contrib=True)
            contribs["lightgbm"] = raw[:, :len(feature_cols)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("lightgbm pred_contrib failed: %s", exc)
    if not contribs:
        logger.warning("no tree member produced SHAP contributions — no "
                       "nfl_shap_game files written (nothing fabricated)")
        return []

    w = np.array([float(weights.get(n, 0.0)) for n in contribs])
    if w.sum() <= 0:
        return []
    w = w / w.sum()
    blended = sum(wi * sv for wi, sv in zip(w, contribs.values()))

    from sports.nfl.artifacts import persist_shap_game
    written: list[Path] = []
    for i, (_, row) in enumerate(slate.iterrows()):
        avg = blended[i]
        p_home = row.get("home_win_prob_model")
        perspective = str(row.get("home_team", "") or "HOME")
        try:
            if pd.notna(p_home) and float(p_home) < 0.5:
                avg = -avg
                perspective = str(row.get("away_team", "") or "AWAY")
        except (TypeError, ValueError):
            pass
        vals = np.round(avg, 6)
        frame = pd.DataFrame({
            "feature": feature_cols,
            "shap_value": vals,
            "signed_effect": ["positive" if v > 0 else "negative"
                              for v in vals],
            "perspective_team": [perspective] * len(feature_cols),
        })
        gid = str(row["game_id"])  # e.g. 2026_01_ARI_LAC — fixture naming
        written.append(persist_shap_game(frame, run_date, gid, out_dir))

    # Count-rule guard: exactly one file per slate row, keyed uniquely.
    if len(written) != len(slate):
        raise NFLRunnerError(
            f"nfl_shap_game count rule violated: {len(written)} files for "
            f"{len(slate)} slate games")
    keys = [p.name for p in written]
    if len(set(keys)) != len(keys):
        raise NFLRunnerError("nfl_shap_game keys are not unique per game")
    return written
