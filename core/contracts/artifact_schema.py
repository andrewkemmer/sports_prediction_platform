"""Pre-write record-type validation (Phase 7.5d / B-005 — hardened rule 10).

Every active production artifact writer in the four sport verticals must
call :func:`validate_record` on the fully-normalized record or frame
IMMEDIATELY BEFORE serialization (``to_csv`` / ``write_text`` /
``_atomic_write_*``). A violation raises
:class:`RecordValidationError` before any bytes hit the sink — no
partial files, no silent coercion.

Registry keys are ``(sport, family)`` where ``family`` is the EXACT
emitted family name (not the legacy delivery-allowlist aliases): MLB
emits ``run_engine_markets``/``run_engine_monitor`` while the legacy
allowlist keys ``markets``/``markets_monitor`` resolve to those same
files via the frontend globs; the prefixed ``feature_drift`` /
``feature_coverage`` allowlist keys are file-embedded fields of
``<sport>_model_monitor.json`` for NFL/NHL/NBA and standalone MLB
families. ``.meta.json`` companions are SEPARATE registry entries with
a ``.meta`` suffix (their schemas differ from the primary artifact).

Exclusions (documented in docs/TRACKER.md, B-005):
* raw ingestion caches (``sports/mlb/ingestion.py`` parquet writes) —
  raw source payloads, exempt from hardened-rule scope;
* ``core/storage.py`` generic writers — zero production callers.

Reserved durable names ``model_history`` / ``model_version_history``
have NO registry entry (zero production writers); any future writer
must add one in the same change — enforced by the bidirectional AST
alignment test in ``tests/core/test_spec_guardrails.py``.

Validation semantics (approved design):
* runs post-normalization, pre-serialization;
* frames: required columns must exist; probability columns (declared
  ``probability_fields`` PLUS any column named ``p_*``) must be finite
  and within the closed interval [0, 1] (NaN/inf rejected unless the
  column is declared nullable); declared ``mass_groups`` are checked
  row-wise, read-only (never renormalized), skipping rows with nulls;
* JSON records: required keys must exist; NaN/inf never permitted;
  ``None`` only in declared ``nullable_fields``; declared probability
  keys within [0, 1]; mass groups read-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Tolerance for read-only probability-mass checks (floating-point).
_MASS_EPS = 1e-9


class RecordValidationError(ValueError):
    """Raised when a record violates its family's contract pre-write."""


@dataclass(frozen=True)
class RecordType:
    """The per-record-type contract enforced before a sink write."""

    record_kind: str  # "frame" | "json"
    required_fields: tuple[str, ...]
    nullable_fields: frozenset[str] = frozenset()
    probability_fields: tuple[str, ...] = ()
    mass_groups: tuple[tuple[str, ...], ...] = ()
    # JSON families whose records embed a ``games`` list of per-game
    # cards (e.g. moneyline_v1): declared card fields are probability-
    # range-checked within EVERY card. None is contract-permitted per
    # card (undecided/unknown), any present value must be in [0, 1].
    game_card_probability_fields: tuple[str, ...] = ()
    description: str = ""
    # Null-tolerant probability columns: nulls permitted (values still
    # range-checked when present). Used for the NNX market grids, whose
    # writers NaN-fill grid columns the engine did not produce for the
    # run (documented writer behavior; MLB's writer forbids NaN instead).
    null_tolerant_prob: bool = False


def _is_bad_float(value: object) -> bool:
    """True for NaN/inf floats (JSON-strict: nullable slots use None)."""
    return isinstance(value, float) and not math.isfinite(value)


def _check_prob_value(where: str, name: str, value: object) -> None:
    if _is_bad_float(value):
        raise RecordValidationError(
            f"{where} field {name!r}: NaN/inf probability rejected")
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(
            f"{where} field {name!r}: not a numeric probability "
            f"({value!r})") from exc
    if not 0.0 <= p <= 1.0:
        raise RecordValidationError(
            f"{where} field {name!r}: probability {p} outside [0, 1]")


def _is_prob_field(name: str, spec: RecordType) -> bool:
    """A field is a probability if declared, or named ``p_*`` (markets
    convention across all four engines)."""
    return name in spec.probability_fields or name.startswith("p_")


def _validate_frame(sport: str, family: str, spec: RecordType,
                    frame: pd.DataFrame) -> None:
    where = f"({sport}, {family})"
    if not isinstance(frame, pd.DataFrame):
        raise RecordValidationError(
            f"{where}: expected DataFrame, got {type(frame).__name__}")
    missing = [c for c in spec.required_fields if c not in frame.columns]
    if missing:
        raise RecordValidationError(
            f"{where}: missing required columns {missing}")
    prob_cols = [c for c in frame.columns if _is_prob_field(c, spec)]
    for col in prob_cols:
        for v in frame[col]:
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                # Frame nullability: nullable columns serialize as empty
                # CSV cells (the frame-level NaN/None convention).
                if col in spec.nullable_fields or spec.null_tolerant_prob:
                    continue
                raise RecordValidationError(
                    f"{where} column {col!r}: null probability in a "
                    "non-nullable column")
            if _is_bad_float(v):
                raise RecordValidationError(
                    f"{where} column {col!r}: inf probability rejected")
            _check_prob_value(where, col, v)
    for group in spec.mass_groups:
        if not all(g in frame.columns for g in group):
            # Optional grid columns (e.g. the NFL totals grid spans only
            # the NFL schema): an absent group has nothing to check.
            continue
        sub = frame[list(group)]
        for _, row in sub.iterrows():
            vals = row.to_dict()
            if any(v is None or pd.isna(v) for v in vals.values()):
                continue  # read-only check: skip incomplete rows
            total = sum(float(v) for v in vals.values())
            if not 0.0 <= total <= 1.0 + _MASS_EPS:
                raise RecordValidationError(
                    f"{where}: probability mass {group} sums to {total} "
                    f"(outside [0, 1+eps])")


def _validate_json(sport: str, family: str, spec: RecordType,
                   record: dict) -> None:
    where = f"({sport}, {family})"
    if not isinstance(record, dict):
        raise RecordValidationError(
            f"{where}: expected dict, got {type(record).__name__}")
    missing = [k for k in spec.required_fields if k not in record]
    if missing:
        raise RecordValidationError(
            f"{where}: missing required keys {missing}")
    for k, v in record.items():
        if _is_bad_float(v):
            raise RecordValidationError(
                f"{where} key {k!r}: NaN/inf not JSON-strict (use None "
                "for contract-permitted nulls)")
        if v is None and k not in spec.nullable_fields:
            raise RecordValidationError(
                f"{where} key {k!r}: None not permitted by contract")
    for name in spec.probability_fields:
        if name in record and record[name] is not None:
            _check_prob_value(where, name, record[name])
    if spec.game_card_probability_fields:
        games = record.get("games")
        if not isinstance(games, list):
            raise RecordValidationError(
                f"{where}: 'games' must be a list of per-game cards")
        for i, card in enumerate(games):
            if not isinstance(card, dict):
                raise RecordValidationError(
                    f"{where}: games[{i}] is not a dict")
            for name in spec.game_card_probability_fields:
                v = card.get(name)
                if v is not None:
                    _check_prob_value(f"{where} games[{i}]", name, v)
    for group in spec.mass_groups:
        if not all(g in record for g in group):
            continue  # optional grid keys: absent group -> nothing to check
        vals = {g: record[g] for g in group}
        if any(v is None for v in vals.values()):
            continue  # read-only check: skip incomplete groups
        total = sum(float(v) for v in vals.values())
        if not 0.0 <= total <= 1.0 + _MASS_EPS:
            raise RecordValidationError(
                f"{where}: probability mass {group} sums to {total} "
                f"(outside [0, 1+eps])")


def validate_record(sport: str, family: str, record) -> None:
    """Validate a frame or JSON record against its registry entry.

    Raises :class:`RecordValidationError` on the first violation; the
    writer must call this AFTER normalization and BEFORE serialization
    so that zero bytes reach the sink on failure.
    """
    key = (sport, family)
    spec = REGISTRY.get(key)
    if spec is None:
        raise RecordValidationError(
            f"no registry entry for ({sport!r}, {family!r}) — writers "
            "must validate against a declared family")
    if spec.record_kind == "frame":
        _validate_frame(sport, family, spec, record)
    else:
        _validate_json(sport, family, spec, record)


# ---------------------------------------------------------------------------
# Registry — active writers only (see module docstring + TRACKER B-005).
# Schemas are in-module literals mirroring tests/fixtures/*_artifact_
# schemas.json (production code must not import from tests/).
# ---------------------------------------------------------------------------

# --- shared column lists ----------------------------------------------------
_PRED_COLS = ("game_id", "game_date", "home_team", "away_team", "home_score",
              "away_score", "home_win", "home_win_prob_model",
              "home_win_prob_model_calibrated", "model_pick", "actual_winner",
              "correct")
_RANK_COLS = ("rank", "team", "team_name", "elo", "wins", "losses", "record",
              "pct", "run_diff", "l10", "home_pct", "away_pct")
_SHAP_COLS = ("feature", "shap_value", "signed_effect", "perspective_team")

# --- MLB JSON key sets (fixture: mlb_artifact_schemas.json) ------------------
_MLB_CALIBRATION_KEYS = ("date", "n_games", "trained_at", "metrics",
                         "calibration_buckets", "calibration", "daily",
                         "league_total", "evening_games_league")
_MLB_MODEL_MONITOR_KEYS = ("date", "version", "last_retrained",
                           "next_retrain", "metrics", "drift_summary",
                           "feature_drift", "feature_coverage",
                           "rolling_brier", "brier_baseline",
                           "brier_baseline_label", "rolling_brier_meta",
                           "features_metadata", "run_engine", "ensemble",
                           "model_history", "version_history")
_MLB_ROLLING_BRIER_KEYS = ("window_days", "min_games_per_day",
                           "source_column", "calibration", "map_scope_note",
                           "calibrator_is_identity", "n_points",
                           "n_games_total", "excluded_sparse_days",
                           "history_mean_brier", "series",
                           "n_games_in_series")
_MLB_ROLLING_BRIER_NULLABLE = frozenset({"history_mean_brier"})
_MLB_FEATURES_METADATA_KEYS = ("generated_for", "n_features", "warnings",
                               "features", "categorical_context")
_MLB_RUN_ENGINE_MONITOR_KEYS = ("schema", "date", "markets_persisted",
                                "markets_persist_error", "winner_cards",
                                "rolling", "fit", "market_metrics", "phase1")
_MLB_RUN_ENGINE_MONITOR_NULLABLE = frozenset({"markets_persist_error"})
_MLB_MARKETS_META_KEYS = (
    # Always emitted by derive_markets_v3 (sports/mlb/run_engine.py):
    "n_draws", "seed", "holdout_cutoff", "n_pre", "n_holdout",
    "line_grid", "alpha_home", "alpha_away", "year_effect_home",
    "year_effect_away", "phase2_single_alpha", "fit_check_single_alpha",
    "fit_check_alpha_lambda", "variance_check", "mc_meta", "k_edge",
    # Conditional (runner/slate-dependent, contract-permitted absences):
    "agreement_vs_moneyline", "agreement_slate",
    # Reference-artifact keys declared in the schema fixture but NOT
    # emitted by this pipeline (market_derived_moneyline*,
    # market_over_*, market_home_cover_*): reserved, not required —
    # requiring them would fail valid production runs.
)

# --- NFL/NHL/NBA JSON key sets (fixtures: <s>_artifact_schemas.json) --------
_NNX_MONEYLINE_KEYS = ("created_utc", "config", "slate_date", "n_games",
                       "games")
_NNX_MONEYLINE_NULLABLE = frozenset({"slate_date"})  # empty slate
_NNX_CALIBRATION_KEYS = ("date", "trained_at", "n_games", "created_utc",
                         "config", "metrics", "calibration",
                         "calibration_buckets", "daily")
_NNX_MATCHUP_KEYS = ("created_utc", "n_games", "games")
_NNX_FEATURE_KEYS = ("created_utc", "feature_set_version", "manifest",
                     "served_columns", "coverage", "fold_geometry", "config")
_NNX_FEATURE_NULLABLE = frozenset({"served_columns"})
_NNX_MODEL_MONITOR_KEYS = ("last_retrained", "last_retrained_note",
                           "next_retrain", "next_retrain_note", "upset_note",
                           "feature_drift", "features_metadata",
                           "feature_coverage", "ensemble", "rolling_brier",
                           "brier_baseline", "brier_baseline_label",
                           "rolling_brier_meta", "version_history",
                           "fold_geometry", "config")
_NNX_MODEL_MONITOR_NULLABLE = frozenset({"last_retrained_note",
                                         "next_retrain_note", "upset_note",
                                         "brier_baseline"})
_NNX_MARKETS_META_KEYS = ("record", "written_utc", "config", "columns",
                          "n_rows", "n_oof", "n_slate", "grids")
_NNX_RUN_ENGINE_MONITOR_KEYS = ("schema", "date", "markets_persisted",
                                "markets_persist_error", "winner_cards",
                                "slate_history", "fit", "market_metrics",
                                "config")
_NNX_RUN_ENGINE_MONITOR_NULLABLE = frozenset({"markets_persist_error"})

# Columns shared by all three NNX market grids (base block, before the
# per-sport spread/totals grids are appended by the engines).
_NNX_MARKETS_BASE_COLS = (
    "kind", "game_id", "gameday", "season", "home_team", "away_team",
    "home_record", "away_record", "p_home_win", "p_away_win",
    "derived_ml", "p_home_win_derived", "p_away_win_derived", "mu_h",
    "mu_a", "mu_margin", "mu_total", "fair_spread", "fair_total",
    "spread_line", "total_line", "has_offer", "p_cover_offered",
    "p_push_offered", "p_over_offered", "p_under_offered",
    "p_push_total_offered", "home_score", "away_score", "total",
    "margin", "p_over_fair", "p_cover_fair", "y_over_fair",
    "y_under_fair", "y_push_fair", "y_cover_fair", "y_push_spread_fair",
    "y_home_win", "decided", "frame_view", "pred_home", "pred_away")


def _nnx_matchup_keys(family_keys: tuple) -> tuple:
    """Matchup JSON adds the matchup family key set (per-sport)."""
    return family_keys


def _register_mlba(reg: dict) -> None:
    """MLB entries (14: 13 primaries + run_engine_markets.meta)."""
    reg["mlb", "todays_games"] = RecordType(
        "frame",
        ("game_id", "game_date", "start_time_utc", "home_team", "away_team",
         "home_record", "away_record", "home_win_prob_model",
         "away_win_prob_model", "edge_home", "edge_away", "sp_name_home",
         "sp_name_away", "sp_era_home", "sp_k9_home", "sp_era_away",
         "sp_k9_away", "venue", "model_pick", "home_win", "home_score",
         "away_score", "total_runs", "game_state", "game_status_detail"),
        frozenset({"sp_name_home", "sp_name_away", "sp_era_home",
                   "sp_k9_home", "sp_era_away", "sp_k9_away", "home_score",
                   "away_score", "total_runs"}),
        ("home_win_prob_model", "away_win_prob_model"),
        description="MLB game cards (frontend required fields)")
    reg["mlb", "power_rankings"] = RecordType(
        "frame", _RANK_COLS, description="MLB power rankings")
    reg["mlb", "calibration"] = RecordType(
        "json", _MLB_CALIBRATION_KEYS, description="MLB calibration")
    reg["mlb", "model_monitor"] = RecordType(
        "json", _MLB_MODEL_MONITOR_KEYS, description="MLB model monitor")
    reg["mlb", "rolling_brier"] = RecordType(
        "json", _MLB_ROLLING_BRIER_KEYS, _MLB_ROLLING_BRIER_NULLABLE,
        description="MLB rolling Brier")
    reg["mlb", "features_metadata"] = RecordType(
        "json", _MLB_FEATURES_METADATA_KEYS,
        description="MLB feature-set metadata")
    reg["mlb", "run_engine_monitor"] = RecordType(
        "json", _MLB_RUN_ENGINE_MONITOR_KEYS, _MLB_RUN_ENGINE_MONITOR_NULLABLE,
        description="MLB run-engine monitor (markets_monitor legacy alias)")
    reg["mlb", "run_engine_markets.meta"] = RecordType(
        "json", _MLB_MARKETS_META_KEYS,
        description="MLB markets .meta.json companion (fit/mc metadata)")
    reg["mlb", "shap_game"] = RecordType(
        "frame", _SHAP_COLS, description="MLB per-game SHAP")
    reg["mlb", "predictions_history"] = RecordType(
        "frame", _PRED_COLS,
        frozenset({"actual_winner"}),
        ("home_win_prob_model", "home_win_prob_model_calibrated"),
        description="MLB predictions history")
    reg["mlb", "feature_drift"] = RecordType(
        "frame",
        ("feature", "current_mean", "baseline_mean", "psi", "psi_adjusted",
         "noise_floor", "mean_shift", "shift_se", "location_shift", "status",
         "weight_pct", "n_baseline", "n_current"),
        description="MLB feature drift")
    reg["mlb", "feature_coverage"] = RecordType(
        "frame",
        ("feature", "window", "n_games", "n_nonnull", "pct_nonnull",
         "n_measured", "pct_measured", "n_default_zero", "status"),
        description="MLB feature coverage")
    reg["mlb", "run_engine_oof"] = RecordType(
        "frame",
        ("game_pk", "game_date", "home_expected_runs", "away_expected_runs",
         "home_score", "away_score"),
        description="MLB run-engine OOF store (retention-exempt)")
    # 78-col market grid; p_* columns probability-checked by convention.
    reg["mlb", "run_engine_markets"] = RecordType(
        "frame",
        ("game_pk", "game_date", "kind", "home_expected_runs",
         "away_expected_runs", "alpha_home", "alpha_away",
         "p_home_win_derived", "p_away_win_derived", "home_score",
         "away_score", "total_runs", "ml_win_prob", "agreement_conflict",
         "p_over_6_5", "p_over_7_0", "p_over_7_5", "p_over_8_0",
         "p_over_8_5", "p_over_9_0", "p_over_9_5", "p_over_10_0",
         "p_over_10_5", "p_over_11_0", "p_over_11_5", "p_over_12_0",
         "p_over_12_5", "p_push_6_5", "p_push_7_0", "p_push_7_5",
         "p_push_8_0", "p_push_8_5", "p_push_9_0", "p_push_9_5",
         "p_push_10_0", "p_push_10_5", "p_push_11_0", "p_push_11_5",
         "p_push_12_0", "p_push_12_5", "p_under_6_5", "p_under_7_0",
         "p_under_7_5", "p_under_8_0", "p_under_8_5", "p_under_9_0",
         "p_under_9_5", "p_under_10_0", "p_under_10_5", "p_under_11_0",
         "p_under_11_5", "p_under_12_0", "p_under_12_5",
         "p_home_cover_0_5", "p_home_cover_1_5", "p_home_cover_2_5",
         "p_home_cover_3_5", "p_rl_1_0_home", "p_rl_1_0_push",
         "p_rl_1_0_away", "p_rl_1_5_home", "p_rl_1_5_push",
         "p_rl_1_5_away", "p_rl_2_0_home", "p_rl_2_0_push",
         "p_rl_2_0_away", "p_rl_2_5_home", "p_rl_2_5_push",
         "p_rl_2_5_away", "p_rl_3_0_home", "p_rl_3_0_push",
         "p_rl_3_0_away", "p_rl_3_5_home", "p_rl_3_5_push",
         "p_rl_3_5_away", "p_rl_4_0_home", "p_rl_4_0_push",
         "p_rl_4_0_away"),
        description="MLB 78-col market grid (k-edge run engine)")


def _total_line_mass_groups(lines: tuple) -> tuple:
    """(p_over_L, p_push_L, p_under_L) triples for each totals line."""
    return tuple((f"p_over_{l}", f"p_push_{l}", f"p_under_{l}")
                 for l in lines)


def _spread_mass_groups(spreads: tuple) -> tuple:
    """(p_home_cover_S, p_push_S) read-only pairs (sum <= 1; the
    remainder is the away cover probability)."""
    return tuple((f"p_home_cover_{s}", f"p_push_{s}") for s in spreads)


def _rl_mass_groups(lines: tuple) -> tuple:
    """(p_rl_L_home, p_rl_L_push, p_rl_L_away) triples (MLB RL grid)."""
    return tuple((f"p_rl_{l}_home", f"p_rl_{l}_push", f"p_rl_{l}_away")
                 for l in lines)


def _register_nnx(sport: str, matchup_family: str, reg: dict) -> None:
    """NFL/NHL/NBA entries (14 each: 10 artifact primaries +
    run_engine_markets.meta + 3 runner-side OOF/fold stores)."""
    reg[sport, "moneyline_v1"] = RecordType(
        "json", _NNX_MONEYLINE_KEYS, _NNX_MONEYLINE_NULLABLE,
        game_card_probability_fields=("home_win_prob_model",
                                      "away_win_prob_model"),
        description=f"{sport} moneyline_v1 game cards")
    reg[sport, "calibration"] = RecordType(
        "json", _NNX_CALIBRATION_KEYS, description=f"{sport} calibration")
    reg[sport, "model_monitor"] = RecordType(
        "json", _NNX_MODEL_MONITOR_KEYS, _NNX_MODEL_MONITOR_NULLABLE,
        description=f"{sport} model monitor")
    reg[sport, "feature_v1"] = RecordType(
        "json", _NNX_FEATURE_KEYS, _NNX_FEATURE_NULLABLE,
        description=f"{sport} feature_v1 manifest")
    reg[sport, matchup_family] = RecordType(
        "json", _nnx_matchup_keys(_NNX_MATCHUP_KEYS),
        description=f"{sport} {matchup_family} panel")
    reg[sport, "run_engine_monitor"] = RecordType(
        "json", _NNX_RUN_ENGINE_MONITOR_KEYS, _NNX_RUN_ENGINE_MONITOR_NULLABLE,
        description=f"{sport} run-engine monitor")
    reg[sport, "run_engine_markets.meta"] = RecordType(
        "json", _NNX_MARKETS_META_KEYS,
        description=f"{sport} markets .meta.json companion")
    reg[sport, "shap_game"] = RecordType(
        "frame", _SHAP_COLS, description=f"{sport} per-game SHAP")
    reg[sport, "predictions_history"] = RecordType(
        "frame", _PRED_COLS, frozenset({"actual_winner"}),
        ("home_win_prob_model", "home_win_prob_model_calibrated"),
        description=f"{sport} predictions history")
    reg[sport, "power_rankings"] = RecordType(
        "frame", _RANK_COLS,
        frozenset({"home_pct", "away_pct"}),
        description=f"{sport} power rankings")
    reg[sport, "run_engine_markets"] = RecordType(
        "frame", _NNX_MARKETS_BASE_COLS, frozenset(), (),
        description=f"{sport} market grid (shared 44-col base block "
        "required; per-sport grid columns are writer-side fixture-"
        "validated; p_* probability convention applies; NaN-filled "
        "absent grid columns are the documented writer behavior)",
        null_tolerant_prob=True)
    # Runner-side OOF/fold stores (models/ dir, retention-exempt).
    # NOTE: the moneyline OOF frame carries game_date in NHL/NBA and
    # gameday in NFL (per-sport ingestion column naming).
    reg[sport, "oof_moneyline"] = RecordType(
        "frame",
        ("game_id", "season", "home_team", "away_team",
         "fold_id", "home_win", "p_ensemble", "p_ensemble_calibrated"),
        frozenset({"p_ensemble_calibrated"}),
        ("home_win", "p_ensemble", "p_ensemble_calibrated"),
        description=f"{sport} OOF moneyline store (p_ensemble_calibrated "
        "is null where calibration was inapplicable; date column is "
        "gameday in NFL, game_date in NHL/NBA)")
    reg[sport, "oof_distribution"] = RecordType(
        "frame",
        ("game_id", "season", "fold_id", "mu_h", "mu_a",
         "home_score", "away_score", "margin", "total", "resid_margin",
         "resid_total"),
        description=f"{sport} OOF score-distribution store (date column is "
        "gameday in NFL, game_date in NHL/NBA)")
    reg[sport, "fold_table"] = RecordType(
        "frame",
        ("fold_id", "n_train", "n_val"),
        description=f"{sport} per-fold reporting table (naming of the "
        "window columns differs NFL vs NHL/NBA; only the stable core "
        "is required here)")


def _register_markets_mass_groups(reg: dict) -> None:
    """Attach read-only mass groups to the market-grid frames.

    MLB totals lines are half-run 6.5..12.5 (p_over/p_push/p_under) plus
    the runline triples; NFL spreads -14..14 integer + half; NHL spreads
    -4..4 + 1.5/5.5/6.5 and totals 5..12; NBA has no p_push groups (no
    ties) so only the probability convention applies.
    """
    mlb = reg["mlb", "run_engine_markets"]
    reg["mlb", "run_engine_markets"] = RecordType(
        record_kind=mlb.record_kind,
        required_fields=mlb.required_fields,
        nullable_fields=mlb.nullable_fields,
        probability_fields=mlb.probability_fields,
        mass_groups=(
            _total_line_mass_groups(("6_5", "7_0", "7_5", "8_0", "8_5",
                                     "9_0", "9_5", "10_0", "10_5", "11_0",
                                     "11_5", "12_0", "12_5"))
            + _rl_mass_groups(("1_0", "1_5", "2_0", "2_5", "3_0", "3_5",
                               "4_0"))),
        description=mlb.description,
        null_tolerant_prob=mlb.null_tolerant_prob)
    nfl = reg["nfl", "run_engine_markets"]
    reg["nfl", "run_engine_markets"] = RecordType(
        record_kind=nfl.record_kind,
        required_fields=nfl.required_fields,
        nullable_fields=nfl.nullable_fields,
        probability_fields=nfl.probability_fields,
        mass_groups=(
            _total_line_mass_groups(tuple(str(i) for i in range(24, 67)))
            + _spread_mass_groups(
                tuple(f"m{i}" for i in range(14, 0, -1))
                + tuple(str(i) for i in range(0, 15)))),
        description=nfl.description,
        null_tolerant_prob=nfl.null_tolerant_prob)
    nhl = reg["nhl", "run_engine_markets"]
    reg["nhl", "run_engine_markets"] = RecordType(
        record_kind=nhl.record_kind,
        required_fields=nhl.required_fields,
        nullable_fields=nhl.nullable_fields,
        probability_fields=nhl.probability_fields,
        mass_groups=(
            _total_line_mass_groups(tuple(str(i) for i in range(5, 13)))
            + _spread_mass_groups(
                tuple(f"m{i}" for i in range(4, 0, -1))
                + ("0", "1", "2", "3", "4"))),
        description=nhl.description,
        null_tolerant_prob=nhl.null_tolerant_prob)


REGISTRY: dict[tuple[str, str], RecordType] = {}
_register_mlba(REGISTRY)
_register_nnx("nfl", "qb_matchup", REGISTRY)
_register_nnx("nhl", "goalie_matchup", REGISTRY)
_register_nnx("nba", "player_matchup", REGISTRY)
_register_markets_mass_groups(REGISTRY)

# Pinned count: 14 MLB + 14 NFL + 14 NHL + 14 NBA (separate .meta
# entries). Pinned by tests/test_contracts.py with an explanatory
# comment; a change here is a registry change and must be reviewed.
EXPECTED_REGISTRY_SIZE = 56
