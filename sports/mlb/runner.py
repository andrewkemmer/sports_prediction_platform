"""MLB Phase 2 production runner.

The ONLY sanctioned end-to-end MLB production path:

1. mandatory run-window validation — ``MLB_START_DATE`` / ``MLB_END_DATE`` /
   ``MLB_FULL_REPULL`` must be present and valid (no silent defaults; the
   runner fails loudly, it never invents a date);
2. bounded Statcast ingestion (incremental or full repull per flag);
3. DuckDB feature factory (point-in-time-safe, Parquet-backed);
4. wide candidate frame → walk-forward moneyline training + run-engine OOF;
5. artifact generation, each writer validated against the immutable
   schema fixture;
6. ``run_production_retention(generation_succeeded=True)`` — retention
   executes ONLY after every writer returned successfully.

Training history (warmup, folds, OOF rules, minimum rows) comes from
``sports/mlb/study.yaml`` — NEVER from the run window. ``MLB_START_DATE`` /
``MLB_END_DATE`` scope only which days are ingested and served.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import PlatformConfig, default_config
from core.markets import FULL_GAME_NO_TIE
from core.prediction_window import resolve_prediction_window
from core.validation.future_availability import (
    validate_available_at,
    validate_settlement_record,
    validate_slate_window,
    window_anchor,
)
from core.retention import run_production_retention
from core.runconfig import resolve_run_window
from sports.mlb.artifacts import write_all_artifacts
from sports.mlb.ingestion import pull_statcast
from sports.mlb.study_config import load_mlb_study

logger = logging.getLogger(__name__)

#: Env var names for the MLB production run (strict; no aliases, no defaults).
ENV_START_DATE = "MLB_START_DATE"
ENV_END_DATE = "MLB_END_DATE"
ENV_FULL_REPULL = "MLB_FULL_REPULL"

PITCHES_CACHE = Path("store/mlb/raw/pitches.parquet")


class MLBRunnerError(RuntimeError):
    """Raised when the MLB production run cannot proceed safely."""


@dataclass
class RunResult:
    run_window_start: str
    run_window_end: str
    full_repull: bool
    n_prediction_games: int = 0
    n_decided_games: int = 0
    artifacts_written: list[str] = field(default_factory=list)
    retention_executed: bool = False
    retention_candidates_deleted: int = 0


def run_mlb_production(
    *,
    env: dict[str, str] | None = None,
    data_delivery_dir: Path | str | None = None,
    pitches_cache: Path | str | None = None,
    fetch=None,
    today=None,
    platform_config: PlatformConfig | None = None,
    dry_run_ingestion: bool = False,
) -> RunResult:
    """Execute the bounded MLB production run.

    ``dry_run_ingestion=True`` skips the network fetch (tests / offline
    validation of the env-var gate) — every other stage still runs against
    the existing cache.
    """
    cfg = platform_config or default_config()
    study = load_mlb_study()

    # ------------------------------------------------------------------
    # 1. Mandatory run window — strict, no silent defaults.
    # ------------------------------------------------------------------
    source = dict(os.environ) if env is None else env
    # Whitespace-only values count as missing — no silent whitespace defaults.
    missing = [k for k in (ENV_START_DATE, ENV_END_DATE, ENV_FULL_REPULL)
               if not str(source.get(k) or "").strip()]
    if missing:
        raise MLBRunnerError(
            f"Missing required production environment variable(s): "
            f"{missing}. MLB_START_DATE, MLB_END_DATE, and MLB_FULL_REPULL "
            f"must be set explicitly — no silent defaults.")
    window = resolve_run_window(start_date=source[ENV_START_DATE],
                                end_date=source[ENV_END_DATE],
                                full_repull=source[ENV_FULL_REPULL])

    # Prediction window: which slate days to serve (run window only).
    mlb_sport = cfg.sport("mlb")
    pred_window = resolve_prediction_window(window.start_date.replace("-", ""),
                                            mlb_sport)

    cache = Path(pitches_cache) if pitches_cache else PITCHES_CACHE

    # ------------------------------------------------------------------
    # 2. Bounded ingestion (incremental or full repull; explicit flag).
    # ------------------------------------------------------------------
    if not dry_run_ingestion:
        # fetch=None means "use the real Statcast fetcher" — never pass None
        # down, which would override pull_statcast's own real default.
        pull_statcast(window.start_date, window.end_date, out_path=cache,
                      full_repull=window.full_repull,
                      **({"fetch": fetch} if fetch is not None else {}),
                      **({"today": today} if today is not None else {}))
    else:
        logger.info("dry_run_ingestion: skipping Statcast fetch")

    # ------------------------------------------------------------------
    # 3-5. Features → training → run engine → artifacts.
    # ------------------------------------------------------------------
    result = RunResult(run_window_start=window.start_date,
                       run_window_end=window.end_date,
                       full_repull=window.full_repull)

    generation_succeeded = False
    written: list[Path] = []
    try:
        from sports.mlb.features import build_features
        from sports.mlb.frames import get_decided_frame, \
            enrich_elo_and_records
        from sports.mlb.feature_registry import build_candidate_frame
        from sports.mlb.training import train_moneyline_ensemble
        from sports.mlb.run_engine import (
            derive_markets_v3,
            predict_slate_runs,
            run_oof,
        )

        game_df, pbp_df = build_features(cache,
                                         output_dir=cache.parent)
        decided = get_decided_frame(game_df)
        if decided.empty:
            raise MLBRunnerError(
                "no decided games in the feature frame — cannot train")
        result.n_decided_games = int(len(decided))

        # B-001 WS4 settlement sub-gate: each settled OOF moneyline row
        # must carry a rule-consistent win/loss outcome (full-game MLB
        # moneyline config; distinct from the B-005 shape gate, NOT PIT
        # evidence).
        for gid, y in zip(decided["game_pk"], decided["home_win"]):
            outcome = "home" if float(y) == 1.0 else "away"
            rec = validate_settlement_record(
                {"outcome": outcome, "scope": "full_game"},
                market_kind="moneyline", settlement_cfg=FULL_GAME_NO_TIE,
                label=f"mlb oof moneyline row {gid}")
            if not rec.ok:
                raise MLBRunnerError(
                    f"settlement gate failed: {rec.violations}")

        # Prediction-window slate rows (pre-game, PIT-carried state).
        slate = _build_slate(game_df, pred_window)
        # B-001 WS4: SECONDARY slate gate (every served start at/after the
        # serving-window lower bound, no decided rows) — loud on violation;
        # NOT §7.1 PIT evidence.
        # Deterministic serving-window-derived anchor (never wall clock):
        # the instant the window opens (first prediction-window date);
        # instant-resolution comparison (MLB carries start_time_utc).
        slate_report = validate_slate_window(
            slate,
            anchor_utc=window_anchor(pred_window.primary_date()),
            start_col="start_time_utc", resolution="instant",
            label="mlb slate")
        if not slate_report.ok:
            raise MLBRunnerError(
                f"slate window gate failed: {slate_report.violations[:3]}")
        # §7.1 per-field gate (report_only in 7.5d): telemetry every run.
        pit_report = validate_available_at(
            slate.to_dict(orient="records"),
            buffer_minutes=study.prediction_cutoff_buffer_minutes,
            label="mlb availability (report_only)",
            start_col="start_time_utc", available_col="available_at")
        logger.info(
            "[B-001 §7.1] %s: n_rows=%d n_missing=%d n_violations=%d "
            "missing_ids=%s (metadata layer = B-001-RESIDUAL, 7.6)",
            pit_report.label, pit_report.n_rows, pit_report.n_missing,
            pit_report.n_violations, list(pit_report.missing)[:3])
        result.n_prediction_games = int(len(slate))

        candidate, feature_cols = build_candidate_frame(decided)
        moneyline = train_moneyline_ensemble(
            candidate, feature_cols, study,
            min_val_games=study.min_val_games,
            min_train_games=study.walk_forward.min_train_games)

        # Board probabilities for the slate (reference predict_games path:
        # final-fit ensemble applied to pre-game rows). The slate is PIT-
        # enriched with the SAME Elo/records + diff pipeline the candidate
        # frame uses — pre-game state only (compute_elo_entries/records are
        # point-in-time by construction; undecided rows update nothing).
        from sports.mlb.frames import enrich_elo_and_records
        from sports.mlb.feature_registry import derive_diff_features, \
            ensure_feature_columns
        _obs = [c for c in slate.columns if slate[c].notna().any()]
        slate = derive_diff_features(
            enrich_elo_and_records(slate[_obs], rename_team_woba=True))
        slate = ensure_feature_columns(slate, feature_cols)

        from sports.mlb.training import predict_slate_moneyline
        slate = predict_slate_moneyline(decided, slate, feature_cols,
                                        weights=moneyline.get("weights"))

        engine = run_oof(
            candidate,
            retrain_cadence_days=study.walk_forward.step_days,
            min_val_games=study.min_val_games,
            min_train_games=study.walk_forward.min_train_games,
        )
        markets_v3 = derive_markets_v3(
            engine["oof"],
            moneyline_probs=moneyline.get("predictions_history"),
        )

        slate_markets = predict_slate_runs(
            decided, slate, engine["summary"]["final_fit_rounds"],
            markets_v3["_curves"])

        # agreement_slate (reference-exact): compare the run engine's
        # slate λ-derived win prob vs the moneyline ensemble's probability
        # ON THE SLATE ROWS (the board's predict_games output), matched by
        # game_pk. The OOF history is NOT the slate source — the reference
        # daily merges target_games' own home_win_prob_model column.
        if not slate_markets.empty and "home_win_prob_model" in slate.columns:
            # predict_slate_runs unifies game_pk to str; normalize both sides.
            prob_map = {str(k): v for k, v in
                        zip(slate["game_pk"],
                            slate["home_win_prob_model"])}
            m = slate_markets["game_pk"].astype(str).map(prob_map)
            known = m.notna()
            if known.any():
                from sports.mlb.run_engine import agreement_stats
                from sports.mlb.market_config import AGREEMENT_FILTER_DELTA
                slate_markets.loc[known, "ml_win_prob"] = m[known]
                slate_markets.loc[known, "agreement_conflict"] = (
                    (slate_markets.loc[known, "p_home_win_derived"]
                     - slate_markets.loc[known, "ml_win_prob"]).abs()
                    > AGREEMENT_FILTER_DELTA)
                slate_stats = agreement_stats(
                    slate_markets.loc[known,
                                      "p_home_win_derived"].to_numpy(),
                    m[known].to_numpy(), delta=AGREEMENT_FILTER_DELTA)
                slate_stats["scope"] = "today_slate"
                markets_v3["summary"]["agreement_slate"] = slate_stats
            else:
                logger.warning(
                    "Run engine: no slate rows matched board probabilities "
                    "— agreement_slate not computed")

        # ------------------------------------------------------------ #
        # Artifact generation (all-or-nothing via the fixture gate).   #
        # ------------------------------------------------------------ #
        sink = Path(data_delivery_dir) if data_delivery_dir \
            else cfg.data_delivery_dir("mlb", Path("."))
        target = pred_window.primary_date().replace("-", "")
        all_markets = pd.concat([markets_v3["markets"], slate_markets],
                                ignore_index=True) \
            if not slate_markets.empty else markets_v3["markets"]
        written = write_all_artifacts(
            target_date_str=target,
            out_dir=sink,
            markets=all_markets,
            markets_summary=markets_v3["summary"],
            oof=engine["oof"],
            todays_games=_todays_games_frame(slate, moneyline),
            power_rankings=_power_rankings_frame(decided),
            predictions_history=moneyline.get("predictions_history_frame",
                                              pd.DataFrame()),
            calibration=_calibration_payload(moneyline, pred_window),
            model_monitor=_model_monitor_payload(moneyline, study),
            rolling_brier=_rolling_brier_payload(markets_v3),
            features_metadata=_features_metadata_payload(
                feature_cols, target, candidate,
                study.moneyline_contract_version),
            run_engine_monitor=_run_engine_monitor_payload(markets_v3),
            feature_drift=pd.DataFrame(
                columns=["feature", "current_mean", "baseline_mean", "psi",
                         "psi_adjusted", "noise_floor", "mean_shift",
                         "shift_se", "location_shift", "status",
                         "weight_pct", "n_baseline", "n_current"]),
            feature_coverage=pd.DataFrame(
                columns=["feature", "window", "n_games", "n_nonnull",
                         "pct_nonnull", "n_measured", "pct_measured",
                         "n_default_zero", "status"]),
        )
        # Per-slate-game SHAP artifacts (reference emits one file per game):
        # exact TreeSHAP from a final-fit LightGBM member over the frozen
        # feature view — no fabricated attributions.
        written += _write_slate_shap(sink, target, slate, decided,
                                     feature_cols)
        result.artifacts_written = [p.name for p in written]
        generation_succeeded = bool(written)
    except Exception:
        logger.exception("MLB production run: artifact generation FAILED — "
                         "retention will NOT execute")
        raise

    # ------------------------------------------------------------------
    # 6. Retention — only after successful generation.
    # ------------------------------------------------------------------
    if generation_succeeded:
        plan = run_production_retention(
            sink,
            cfg, "mlb",
            generation_succeeded=True,
            reference_date=pred_window.primary_date())
        result.retention_executed = True
        result.retention_candidates_deleted = plan.n_candidates \
            if hasattr(plan, "n_candidates") else 0

    return result


def _write_slate_shap(out_dir: Path, target_date_str: str,
                      slate: pd.DataFrame, decided: pd.DataFrame,
                      feature_cols: tuple[str, ...]) -> list[Path]:
    """Emit EXACTLY one shap_game_<date>_<AWAY>@<HOME>.csv per undecided
    prediction-slate game — no fixed minimum, no fixed maximum. The count
    rule is: ``len(files) == len(undecided slate games)`` (0 when the
    slate is empty).

    Attribution source: exact TreeSHAP (``pred_contrib=True``) from a
    LightGBM classifier refit on ALL decided games with the frozen
    LIGHTGBM_PARAMS, evaluated on the slate rows' moneyline feature
    matrix. Contributions are in home-win log-odds space; the perspective
    team is the game's home side (the model's prediction target).
    """
    from sports.mlb.artifacts import persist_shap_game
    from sports.mlb.training import LIGHTGBM_PARAMS

    if slate.empty or feature_cols is None or not len(feature_cols):
        return []
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("lightgbm unavailable — shap_game artifacts skipped")
        return []

    X_tr = decided.reindex(columns=list(feature_cols)).to_numpy(dtype=float)
    y_tr = decided["home_win"].to_numpy(dtype=float)
    model = LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(X_tr, y_tr)
    X_slate = slate.reindex(columns=list(feature_cols)).to_numpy(dtype=float)
    contrib = model.booster_.predict(X_slate, pred_contrib=True)  # (n, F+1)

    written: list[Path] = []
    for i, (_, row) in enumerate(slate.iterrows()):
        game_key = f"{row['away_team']}@{row['home_team']}"
        frame = pd.DataFrame({
            "feature": list(feature_cols),
            "shap_value": np.round(contrib[i, :len(feature_cols)], 6),
            "signed_effect": np.round(contrib[i, :len(feature_cols)], 6),
            "perspective_team": [row["home_team"]] * len(feature_cols),
        })
        written.append(persist_shap_game(frame, target_date_str, game_key,
                                         out_dir))
    # Count-rule guard: exactly one file per slate row, keyed uniquely.
    if len(written) != len(slate):
        raise MLBRunnerError(
            f"shap_game count rule violated: {len(written)} files for "
            f"{len(slate)} slate games")
    keys = [p.name.split("_", 2)[2] for p in written]
    if len(set(keys)) != len(keys):
        raise MLBRunnerError("shap_game keys are not unique per slate game")
    return written


# ---------------------------------------------------------------------------
# Slate + payload helpers (Phase-2 minimal but schema-honest)
# ---------------------------------------------------------------------------
def _build_slate(game_df: pd.DataFrame, pred_window) -> pd.DataFrame:
    """Pre-game rows for the prediction window with PIT-carried state.

    Keeps every candidate game whose game_date falls inside the window and
    has no decided outcome yet — never fabricates an outcome.
    """
    if game_df.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(game_df["game_date"]).dt.strftime("%Y%m%d")
    in_win = dates.isin(list(pred_window.slate_dates))
    undecided = game_df["home_win"].isna() \
        if "home_win" in game_df.columns else pd.Series(True, index=game_df.index)
    slate = game_df[in_win & undecided].copy()
    return slate.reset_index(drop=True)


def _todays_games_frame(slate: pd.DataFrame, moneyline: dict) -> pd.DataFrame:
    cols = ["game_id", "game_date", "start_time_utc", "home_team",
            "away_team", "home_record", "away_record", "home_win_prob_model",
            "away_win_prob_model", "edge_home", "edge_away", "sp_name_home",
            "sp_name_away", "sp_era_home", "sp_k9_home", "sp_era_away",
            "sp_k9_away", "venue", "model_pick", "home_win", "home_score",
            "away_score", "total_runs", "game_state", "game_status_detail"]
    if slate.empty:
        return pd.DataFrame(columns=cols)
    out = slate.reindex(columns=cols)
    # Reference game_id convention: YYYYMMDD_AWAY@HOME (slate-key convention).
    if out["game_id"].isna().all() and {"away_team", "home_team"} \
            <= set(slate.columns):
        out["game_id"] = (
            pd.to_datetime(slate["game_date"]).dt.strftime("%Y%m%d") + "_"
            + slate["away_team"].astype(str) + "@"
            + slate["home_team"].astype(str))
    if "home_win_prob_model" not in slate.columns:
        out["home_win_prob_model"] = pd.NA
    return out


def _power_rankings_frame(decided: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "team", "team_name", "elo", "wins", "losses", "record",
            "pct", "run_diff", "l10", "home_pct", "away_pct"]
    frame = decided.reindex(columns=cols)
    if frame.empty or "team" not in decided.columns:
        return pd.DataFrame(columns=cols)
    return frame


def _calibration_payload(moneyline: dict, pred_window) -> dict:
    return {
        "date": pred_window.primary_date(),
        "n_games": int(moneyline.get("n_games", 0)),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "metrics": moneyline.get("metrics", {}),
        "calibration_buckets": [],
        "calibration": moneyline.get("calibration", {}),
        "daily": [],
        "league_total": int(moneyline.get("n_games", 0)),
        "evening_games_league": 0,
    }


def _model_monitor_payload(moneyline: dict, study) -> dict:
    return {
        "date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "version": moneyline.get("version", "vphase2"),
        "last_retrained": moneyline.get("last_retrained", ""),
        "next_retrain": moneyline.get("next_retrain", ""),
        "metrics": moneyline.get("metrics", {}),
        "drift_summary": {"warnings": [], "alerts": [], "features": []},
        "feature_drift": [],
        "feature_coverage": [],
        "rolling_brier": [],
        "brier_baseline": 0.25,
        "brier_baseline_label": "always-home (league base rate)",
        "rolling_brier_meta": {},
        "features_metadata": {},
        "run_engine": {},
        "ensemble": moneyline.get("ensemble", []),
        "model_history": [],
        "version_history": [],
    }


def _rolling_brier_payload(markets_v3: dict) -> dict:
    return {
        "window_days": 30,
        "min_games_per_day": 1,
        "source_column": "home_win_prob_model",
        "calibration": "prequential",
        "map_scope_note": "league-wide rolling brier",
        "calibrator_is_identity": False,
        "n_points": 0,
        "n_games_total": 0,
        "excluded_sparse_days": 0,
        "history_mean_brier": None,
        "series": [],
        "n_games_in_series": 0,
    }


def _features_metadata_payload(feature_cols, target_date_str: str = "",
                               decided: pd.DataFrame | None = None,
                               version: str | None = None) -> dict:
    """The real registry metadata (fixture shape: features[<name>] = spec).

    Entirely-unavailable features (all-NaN across the decided frame or
    absent from it) are EXPLICITLY marked in ``warnings`` — never silently
    dropped or silently substituted (missing-data policy, Phase 2.1).

    ``version`` binds the ACTIVE moneyline contract version from the study
    call site (T3); it defaults to the registry's current version so the
    helper stays callable standalone.
    """
    from sports.mlb.feature_registry import (MONEYLINE_CONTRACT_VERSION,
                                             build_feature_contract)
    import numpy as np
    warnings: list[str] = []
    if decided is not None and len(feature_cols):
        X = decided.reindex(columns=list(feature_cols)).to_numpy(dtype=float)
        for c in feature_cols:
            col = decided[c] if c in decided.columns else None
            if col is None or (col.notna().sum() == 0):
                warnings.append(
                    f"unavailable: {c} (no observations in source data — "
                    f"routed as explicitly unavailable per missing-data "
                    f"policy; not fabricated)")
    payload = build_feature_contract(
        version or MONEYLINE_CONTRACT_VERSION
    ).to_metadata_json(target_date_str, warnings=warnings)
    payload["n_features"] = len(feature_cols)
    return payload


def _run_engine_monitor_payload(markets_v3: dict) -> dict:
    return {
        "schema": "run-engine-monitor/v2",
        "date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "markets_persisted": True,
        "markets_persist_error": None,
        "winner_cards": {"over_under": {}, "run_line": {},
                         "derived_ml": {}},
        "rolling": {"over_under": {}, "run_line": {}, "derived_ml": {}},
        "fit": markets_v3["summary"],
        "market_metrics": {},
        "phase1": {"n_folds": markets_v3["summary"].get("n_folds", 0),
                   "n_games": markets_v3["summary"].get("n_games", 0),
                   "dispersion_ratio": None,
                   "final_fit_rounds":
                       markets_v3["summary"].get("final_fit_rounds", {})},
    }
