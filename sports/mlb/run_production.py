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

import sys as _sys
from pathlib import Path as _Path

# §3 direct execution bootstrap: `python sports/<sport>/run_production.py`
# must work without PYTHONPATH — insert the repo root before the `core`
# imports below.
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import PlatformConfig, default_config
from core.config import load_market_rules
from core.folds import brier_score
from core.config import resolve_prediction_window
from core.validation.future_availability import (
    validate_settlement_record,
    validate_slate_window,
    window_anchor,
)
from core.features import per_field_gate
from sports.mlb.features.registry import (
    MARKET_FEATURE_COLS,
    MONEYLINE_FEATURE_COLS,
)

from sports.mlb.frames import (ELO_HOME_ADV, ELO_K, MLB_TEAM_NAMES,
                               compute_elo_entries)

#: §7.1 per-field availability coverage (B-001-RESIDUAL, r6): the union
#: of both frozen contracts — every field the slate predictions may
#: consume carries a declared availability class and is gated.
_AVAILABILITY_FEATURES = tuple(dict.fromkeys(
    (*MONEYLINE_FEATURE_COLS, *MARKET_FEATURE_COLS)))
from core.artifacts import run_production_retention
from core.config import resolve_run_window
from sports.mlb.artifacts import write_all_artifacts
from sports.mlb.ingestion import pull_schedule, pull_statcast
from sports.mlb.config.study_config import load_mlb_study

#: Settlement rules — declared in ``config/market_rules.yaml`` (spec §20)
#: and loaded once per process; the OOF settlement gate reads the
#: moneyline rule from here instead of a hardcoded constant.
_MARKET_RULES = load_market_rules("mlb")

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
    # §4: FULL_REPULL rebuilds the COMPLETE historical store from the
    # STUDY-defined start date (data_start_date) — never just the run
    # window (the r10 live smoke caught the window-only scope). The run
    # window scopes SERVING only; training history comes from the study.
    # A future-dated window end is clamped to today: no pitch data can
    # exist past the present, and the chunker treats future chunks as
    # tolerated empties anyway.
    # study.data_start_date is compact (YYYYMMDD) — normalize to a date.
    _ds = str(study.data_start_date).replace("-", "")
    ingest_start = (f"{_ds[:4]}-{_ds[4:6]}-{_ds[6:]}" if len(_ds) == 8
                    else study.data_start_date)
    ingest_end = min(window.end, date.today())  # `end` is the date property
    logger.info("ingestion scope: %s .. %s (study data_start_date; run "
                "window scopes serving only)", ingest_start, ingest_end)
    if not dry_run_ingestion:
        # fetch=None means "use the real Statcast fetcher" — never pass None
        # down, which would override pull_statcast's own real default.
        pull_statcast(ingest_start, ingest_end, out_path=cache,
                      full_repull=window.full_repull,
                      **({"fetch": fetch} if fetch is not None else {}),
                      **({"today": today} if today is not None else {}))
        # §5 schedule source: StatsAPI supplies the first-pitch instants
        # Statcast does not carry (proven by the r10 live smoke: the §7.1
        # instant gates fail closed without them). Scoped to the study
        # history like Statcast, but NOT clamped to today — the future
        # slate days' instants are exactly what the pregame gates need.
        pull_schedule(ingest_start, window.end,
                      out_path=cache.parent / "schedule.parquet",
                      full_repull=window.full_repull)
    else:
        logger.info("dry_run_ingestion: skipping Statcast + schedule fetch")

    # ------------------------------------------------------------------
    # 3-5. Features → training → run engine → artifacts.
    # ------------------------------------------------------------------
    result = RunResult(run_window_start=window.start_date,
                       run_window_end=window.end_date,
                       full_repull=window.full_repull)

    generation_succeeded = False
    written: list[Path] = []
    try:
        from sports.mlb.features.build_frame import build_features
        from sports.mlb.frames import get_decided_frame, \
            enrich_elo_and_records
        from sports.mlb.features.registry import build_candidate_frame
        from sports.mlb.models.moneyline.training import train_moneyline_ensemble
        from sports.mlb.models.market.run_engine import (
            derive_markets_v3,
            predict_slate_runs,
            run_oof,
        )

        game_df, pbp_df = build_features(cache,
                                         output_dir=cache.parent)
        # §5 schedule merge: fail-closed population of start_time_utc by
        # game_pk. Rows already carrying a source instant (fixture/evidence
        # stores) pass through untouched; a decided Statcast game with no
        # schedule match is a real integrity failure and must be loud.
        if "start_time_utc" not in game_df.columns:
            game_df["start_time_utc"] = pd.NaT
        _need = game_df["start_time_utc"].isna()
        if _need.any():
            _sched_path = cache.parent / "schedule.parquet"
            if not _sched_path.exists():
                raise MLBRunnerError(
                    "schedule cache missing (store/mlb/raw/schedule.parquet): "
                    "Statcast carries no first-pitch instant and the §5 "
                    "schedule source was not pulled — cannot serve")
            _sched = pd.read_parquet(_sched_path)
            _instants = _sched.set_index("game_pk")["start_time_utc"]
            _instants.index = pd.to_numeric(_instants.index,
                                            errors="coerce").astype("Int64")
            _keys = pd.to_numeric(game_df.loc[_need, "game_pk"],
                                  errors="coerce").astype("Int64")
            _mapped = _keys.map(_instants)
            _unmatched = int(_mapped.isna().sum())
            if _unmatched:
                _sample = game_df.loc[_need, "game_pk"][
                    _mapped.isna()].head(5).tolist()
                raise MLBRunnerError(
                    f"{_unmatched} game row(s) have no StatsAPI schedule "
                    f"match for start_time_utc (e.g. {_sample}) — fail-closed")
            game_df.loc[_need, "start_time_utc"] = _mapped
            logger.info("schedule merge: populated start_time_utc for %d "
                        "row(s)", int(_need.sum()))
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
                market_kind="moneyline",
                settlement_cfg=_MARKET_RULES.settlement_config("moneyline"),
                label=f"mlb oof moneyline row {gid}")
            if not rec.ok:
                raise MLBRunnerError(
                    f"settlement gate failed: {rec.violations}")

        # Prediction-window slate rows (pre-game, PIT-carried state).
        slate = _build_slate(game_df, pred_window)
        # §5 schedule source: pregame-known probable pitchers (source-
        # populated StatsAPI fields, never fabricated) — the card's
        # ``sp_name_*`` come from here when the feature frame cannot
        # supply them.
        _sched_path = cache.parent / "schedule.parquet"
        if _sched_path.exists() and not slate.empty:
            _sp = pd.read_parquet(_sched_path)
            if {"game_pk", "sp_name_home", "sp_name_away"} \
                    <= set(_sp.columns):
                _sp = _sp.dropna(subset=["game_pk"]).drop_duplicates(
                    "game_pk").set_index("game_pk")
                _pk = pd.to_numeric(slate["game_pk"], errors="coerce")\
                    .astype("Int64")
                for _col in ("sp_name_home", "sp_name_away"):
                    if _col not in slate.columns \
                            or slate[_col].isna().any():
                        slate[_col] = slate[_col] if _col in slate.columns \
                            else pd.NA
                        slate[_col] = slate[_col].fillna(
                            _pk.map(_sp[_col]))
                logger.info("slate probable pitchers: %d/%d populated",
                            int(slate["sp_name_home"].notna().sum()),
                            len(slate))
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
        # §7.1 per-field gate (B-001-RESIDUAL metadata layer, r6): every
        # contract feature is stamped from its declared availability class
        # and gated strictly against prediction_cutoff. In `enforce` mode
        # (the activated 7.6 default) a non-ok gate FAILS the run before
        # any prediction artifact is generated.
        pit_report = per_field_gate(
            slate,
            sport="mlb",
            start_col="start_time_utc",
            features=_AVAILABILITY_FEATURES,
            buffer_minutes=study.prediction_cutoff_buffer_minutes,
            label="mlb availability",
            game_id_col="game_pk")  # MLB's canonical key (Statcast+schedule)
        logger.info(
            "[B-001 §7.1 per-field] %s: fields=%d rows=%d n_missing=%d "
            "n_violations=%d failing=%s (mode=%s)",
            pit_report.label, pit_report.n_fields,
            pit_report.n_rows_per_field, pit_report.n_missing,
            pit_report.n_violations,
            list(pit_report.failing_fields())[:3],
            study.availability_enforcement)
        if study.availability_enforcement == "enforce" and not pit_report.ok:
            raise MLBRunnerError(
                "§7.1 availability gate failed (enforce): "
                f"{pit_report.failing_fields()[:5]}")
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
        from sports.mlb.features.registry import derive_diff_features, \
            ensure_feature_columns
        _obs = [c for c in slate.columns if slate[c].notna().any()]
        slate = derive_diff_features(
            enrich_elo_and_records(slate[_obs], rename_team_woba=True))
        slate = ensure_feature_columns(slate, feature_cols)

        from sports.mlb.models.moneyline.training import predict_slate_moneyline
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
                from sports.mlb.models.market.run_engine import agreement_stats
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
            # §22 Amendment 11 support artifact: the decided frame's core
            # columns — board-finals reconciliation + markets team bridge.
            game_level_features=decided.reindex(
                columns=["game_pk", "game_date", "home_team", "away_team",
                         "home_score", "away_score", "home_win"]),
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
    from sports.mlb.models.moneyline.training import LIGHTGBM_PARAMS

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
    """League-wide power rankings from the decided frame's FINAL state —
    the most recent season-to-date Elo/record per team (display only;
    every number is a settled result, never a projection).

    The reference dashboard's rankings table reads rank / team / team_name
    / elo / wins / losses / record / pct / run_diff / l10 / home_pct /
    away_pct; L10 and splits are computed from the decided games themselves.
    Empty frame (honest no-data) when the decided frame lacks the
    enrichment columns.
    """
    cols = ["rank", "team", "team_name", "elo", "wins", "losses", "record",
            "pct", "run_diff", "l10", "home_pct", "away_pct"]
    need = {"game_date", "home_team", "away_team", "home_win",
            "home_score", "away_score"}
    if decided.empty or not need <= set(decided.columns):
        return pd.DataFrame(columns=cols)
    d = decided.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d.dropna(subset=["game_date"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    last_season = int(d["game_date"].dt.year.max())
    cur = d[d["game_date"].dt.year == last_season]
    if cur.empty:
        return pd.DataFrame(columns=cols)

    # Final season Elo per team: run the PIT engine over the season's
    # chronological games, then apply each team's last-game update (the
    # engine's exact K/home-adv formula) so the table shows the CURRENT
    # rating, not the last pre-game one.
    srt = cur.sort_values("game_date", kind="stable").reset_index(drop=True)
    elo_pre = compute_elo_entries(srt)
    final_elo: dict[str, float] = {}
    for idx in range(len(srt)):
        row = srt.iloc[idx]
        h, a = str(row["home_team"]), str(row["away_team"])
        he, ae = float(elo_pre.iloc[idx]["home_elo"]), \
            float(elo_pre.iloc[idx]["away_elo"])
        hw = row.get("home_win")
        if pd.isna(hw):
            continue
        hw = float(hw)
        exp_h = 1.0 / (1.0 + 10 ** ((ae - he - ELO_HOME_ADV) / 400))
        final_elo[h] = he + ELO_K * (hw - exp_h)
        final_elo[a] = ae + ELO_K * ((1 - hw) - (1 - exp_h))

    rows: list[dict] = []
    for team in sorted(set(cur["home_team"]) | set(cur["away_team"])):
        g = cur[(cur["home_team"] == team) | (cur["away_team"] == team)]
        g = g.sort_values("game_date")
        if g.empty:
            continue
        is_home = g["home_team"] == team
        won = np.where(is_home, g["home_win"], 1 - g["home_win"]).astype(float)
        scored = np.where(is_home, g["home_score"], g["away_score"]).astype(float)
        allowed = np.where(is_home, g["away_score"], g["home_score"]).astype(float)
        wins, losses = int(won.sum()), int((1 - won).sum())
        n = wins + losses
        if not n:
            continue
        # last 10: most recent completed games, most recent first
        l10 = won[-10:]
        l10_str = f"{int(l10.sum())}-{int((1 - l10).sum())}"
        home_mask = is_home.to_numpy()
        home_n = int(home_mask.sum())
        away_n = n - home_n
        home_w = float(won[home_mask].sum()) if home_n else 0.0
        away_w = float(won[~home_mask].sum()) if away_n else 0.0
        rows.append({
            "team": team,
            "team_name": MLB_TEAM_NAMES.get(team, team),
            "elo": round(final_elo.get(team, float("nan")), 1)
            if final_elo.get(team) is not None else np.nan,
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "pct": round(wins / n, 3),
            "run_diff": int((scored - allowed).sum()),
            "l10": l10_str,
            "home_pct": round(home_w / home_n, 3) if home_n else np.nan,
            "away_pct": round(away_w / away_n, 3) if away_n else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = out.sort_values(["elo", "pct", "run_diff"], ascending=False)\
        .reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[cols]


def _calibration_payload(moneyline: dict, pred_window) -> dict:
    """Calibration record — buckets/daily come from the training module's
    OOF-derived payloads (real per-bin reliability + per-day record), never
    the [] stubs the pre-r12 artifact shipped."""
    return {
        "date": pred_window.primary_date(),
        "n_games": int(moneyline.get("n_games", 0)),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "metrics": moneyline.get("metrics", {}),
        "calibration_buckets": moneyline.get("calibration_buckets", []),
        "calibration": moneyline.get("calibration", {}),
        "daily": moneyline.get("daily", []),
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
        "rolling_brier": [],  # standalone record (r5); monitor reads it directly
        "brier_baseline": 0.25,
        "brier_baseline_label": "always-home (league base rate)",
        "rolling_brier_meta": {},
        "features_metadata": {},
        "run_engine": {},
        "ensemble": moneyline.get("ensemble", []),  # per-member OOF metrics
        "model_history": [],
        "version_history": [],
    }


def _rolling_brier_payload(markets_v3: dict) -> dict:
    """League-wide rolling moneyline Brier from THIS run's decided OOF games.

    The series is the artifact contract's own series: ``source_column`` names
    ``home_win_prob_model`` (the moneyline probability, carried on the markets
    frame as ``ml_win_prob``) scored against the settled
    ``home_score > away_score`` outcome, one point per calendar day. The
    per-day value is the canonical Brier formula (mean squared error between
    predicted probability and actual outcome — ``core.oof.brier_score``), the
    same one MLB's run-engine scoring uses.

    Games whose moneyline probability or final score is unavailable are
    EXCLUDED from the series (never fabricated as 0.0/0.5); days left with
    fewer than ``min_games_per_day`` usable games are counted in
    ``excluded_sparse_days``. ``history_mean_brier`` is null only when no day
    qualified — it is a contract-nullable field, not a placeholder.

    Pre-7.5e this helper took ``markets_v3`` and ignored it entirely,
    returning a hardcoded ``n_points=0`` / ``series=[]`` payload, so every
    persisted ``rolling_brier_<date>.json`` reported an empty series even on
    runs with hundreds of settled OOF games. The signature is unchanged
    (the markets frame carries both the probability and the outcome).

    SEMANTICS — this artifact is MONEYLINE, not totals:

    * probabilities come from the moneyline model's ``home_win_prob_model``
      column (carried on the markets frame as ``ml_win_prob``);
    * outcomes come from settled results, ``home_score > away_score``;
    * scoring uses the canonical ``core.oof.brier_score``
      (``mean((p - y) ** 2)``), the same formula MLB's run-engine scoring
      and ``core/optimization/metrics.py`` use;
    * unavailable or non-finite (NaN) probability/outcome rows are EXCLUDED
      — never imputed or fabricated;
    * the existing ``rolling_brier`` schema, keys, order, and dtypes are
      preserved exactly.

    DELIBERATELY NOT WIRED: ``sports/mlb/run_engine.py::
    compute_rolling_totals_brier`` scores a DIFFERENT market — the OVER/UNDER
    TOTALS series (``p_over_9_0`` against ``total_runs >= 9.5``). Consuming it
    here would repoint a moneyline artifact at a totals market and is a
    market-semantics regression, so it stays unwired dead code and is
    deferred as separate follow-up work (no blocker ID is created for it).
    """
    window_days = 30
    min_games_per_day = 1
    series: list[dict] = []
    excluded = 0
    markets = markets_v3.get("markets")
    if markets is not None and not markets.empty:
        required = ("ml_win_prob", "home_score", "away_score", "game_date")
        if set(required) <= set(markets.columns):
            usable = markets.dropna(subset=list(required)).copy()
            usable["_day"] = pd.to_datetime(usable["game_date"]).dt.date
            usable["_y_home_win"] = (
                usable["home_score"] > usable["away_score"]).astype(float)
            for day, grp in usable.groupby("_day"):
                if len(grp) < min_games_per_day:
                    excluded += 1
                    continue
                series.append({
                    "date": str(day),
                    "brier": round(float(brier_score(
                        [float(v) for v in grp["ml_win_prob"]],
                        [float(v) for v in grp["_y_home_win"]])), 5),
                    "games": int(len(grp)),
                })
            series.sort(key=lambda r: r["date"])
    n_games = int(sum(r["games"] for r in series))
    return {
        "window_days": window_days,
        "min_games_per_day": min_games_per_day,
        "source_column": "home_win_prob_model",
        "calibration": "prequential",
        "map_scope_note": "league-wide rolling brier",
        "calibrator_is_identity": False,
        "n_points": len(series),
        "n_games_total": n_games,
        "excluded_sparse_days": excluded,
        "history_mean_brier": (round(float(np.mean(
            [r["brier"] for r in series])), 5) if series else None),
        "series": series,
        "n_games_in_series": n_games,
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
    from sports.mlb.features.registry import (MONEYLINE_CONTRACT_VERSION,
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


def _run_and_log() -> int:
    """Resolve the window from the environment, print the §3 banner
    (resolved prediction window + repull mode FIRST), run, and print
    the final summary. Exit 0 on success; the failure is logged loudly
    and the process exits 1."""
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("mlb.run_production")
    up = "mlb".upper()
    try:
        window = resolve_run_window(
            start_date=os.environ[f"{up}_START_DATE"],
            end_date=os.environ[f"{up}_END_DATE"],
            full_repull=os.environ[f"{up}_FULL_REPULL"])
    except KeyError as exc:
        log.error("missing required environment variable: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        log.error("invalid run window configuration: %s", exc)
        return 1
    log.info("[%s] resolved prediction window: %s .. %s | FULL_REPULL=%s",
             up, window.start_date, window.end_date, window.full_repull)
    try:
        result = run_mlb_production()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        log.error("[%s] PRODUCTION RUN FAILED: %s: %s",
                  up, type(exc).__name__, exc)
        return 1
    log.info("[%s] PRODUCTION RUN COMPLETE: %s", up, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_and_log())
