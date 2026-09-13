"""Tests for core.retention."""

import re
from pathlib import Path

from core.config import DEFAULT_ALLOWLISTED_FAMILIES, default_config
from core.artifacts import (
    RetentionPlan,
    _family_name,
    apply_retention,
    plan_retention,
    retention_cutoff,
    run_production_retention,
)

# ---------------------------------------------------------------------------
# B-004: emission <-> allowlist parity (both directions, all four sports).
# EMISSIONS is the audited inventory of DATED filename prefixes the runners
# write to each sport's sink (evidence: sports/*/runner.py, mlb artifacts.py
# ``sink / f"<prefix>_<date>..."`` call sites). Any new emitted family MUST
# be added here AND to DEFAULT_ALLOWLISTED_FAMILIES in the same commit.
# ---------------------------------------------------------------------------
EMISSIONS: dict[str, tuple[str, ...]] = {
    "mlb": (
        "todays_games", "power_rankings", "calibration", "model_monitor",
        "run_engine_markets", "run_engine_monitor", "shap_game",
        "predictions_history", "feature_drift", "feature_coverage",
        "rolling_brier", "model_history", "model_version_history",
        "features_metadata",
    ),
    "nfl": (
        "nfl_moneyline_v1", "nfl_calibration", "nfl_predictions_history",
        "nfl_power_rankings", "nfl_run_engine_markets", "nfl_run_engine_monitor",
        "nfl_qb_matchup", "nfl_feature_v1", "nfl_model_monitor", "nfl_shap_game",
        "nfl_feature_drift", "nfl_feature_coverage",
    ),
    "nhl": (
        "nhl_moneyline_v1", "nhl_calibration", "nhl_predictions_history",
        "nhl_power_rankings", "nhl_run_engine_markets", "nhl_run_engine_monitor",
        "nhl_goalie_matchup", "nhl_feature_v1", "nhl_model_monitor",
        "nhl_shap_game",
    ),
    "nba": (
        "nba_moneyline_v1", "nba_calibration", "nba_predictions_history",
        "nba_power_rankings", "nba_player_matchup", "nba_feature_v1",
        "nba_model_monitor", "nba_run_engine_markets", "nba_run_engine_monitor",
        "nba_shap_game",
    ),
}


def test_emitted_dated_families_are_all_allowlisted():
    """B-004 (forward): every dated family a runner can emit has an allowlist
    entry — nothing the runners produce is permanently protected."""
    allow = set(DEFAULT_ALLOWLISTED_FAMILIES)
    missing: list[str] = []
    for sport, prefixes in EMISSIONS.items():
        for prefix in prefixes:
            family = _family_name(f"{prefix}_20270105.csv")
            if family not in allow:
                missing.append(f"{sport}: {family}")
    assert not missing, f"emitted families missing from allowlist: {missing}"


def test_allowlisted_families_are_emitted_or_contracts():
    """B-004 (reverse): every allowlist entry is an emitted dated family OR
    a durable-contract family (contract-frozen naming, not a dated emission)."""
    durable = {"markets", "markets_monitor", "model_history",
               "model_version_history", "features_metadata"}
    allow = set(DEFAULT_ALLOWLISTED_FAMILIES)
    emitted = {fam for fams in EMISSIONS.values()
               for fam in (_family_name(f"{p}_20270105.csv") for p in fams)}
    stale = allow - emitted - durable
    assert not stale, (
        "allowlist entries that match no emission and no durable contract: "
        + str(sorted(stale)))


def test_stale_alias_families_removed():
    """B-004 duality cleanup: the stale alias families are gone from the
    allowlist (nothing emits them; sink scan found zero historical files)."""
    allow = set(DEFAULT_ALLOWLISTED_FAMILIES)
    stale_aliases = {"nfl_moneyline_json", "nfl_feature_json", "nfl_markets",
                     "nfl_markets_monitor"}
    assert not (allow & stale_aliases), "stale alias families still allowlisted"


def _mk_sink(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / "sports" / "mlb" / "data_delivery"
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (d / name).write_text(content)
    (d / "models").mkdir(exist_ok=True)
    (d / "models" / "ensemble_latest.joblib").write_bytes(b"m")
    return d


def test_retention_cutoff():
    assert retention_cutoff(20, "20260907") == "20260818"
    assert retention_cutoff(1, "20260907") == "20260906"


def test_plan_candidates_and_protection(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {
        "todays_games_20260815.csv": "old",
        "todays_games_20260907.csv": "new",
        "calibration_20260815.json": "old",
        "model_history.json": "durable",
    })
    plan = plan_retention(sink, cfg, "mlb", reference_date="20260907")
    names = {p.name for p in plan.candidates}
    assert names == {"todays_games_20260815.csv", "calibration_20260815.json"}
    protected = {p.name for p in plan.protected}
    assert "model_history.json" in protected
    assert "ensemble_latest.joblib" in protected
    assert plan.dry_run


def test_apply_retention_deletes_only_candidates(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {
        "todays_games_20260815.csv": "old",
        "todays_games_20260907.csv": "new",
        "model_history.json": "durable",
    })
    plan = apply_retention(sink, cfg, "mlb", reference_date="20260907",
                           execute=True)
    assert {p.name for p in plan.deleted} == {"todays_games_20260815.csv"}
    assert (sink / "todays_games_20260907.csv").exists()
    assert (sink / "model_history.json").exists()
    assert not plan.dry_run


def test_dry_run_deletes_nothing(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {"todays_games_20260815.csv": "old"})
    plan = apply_retention(sink, cfg, "mlb", reference_date="20260907")
    assert plan.candidates
    assert plan.deleted == ()
    assert (sink / "todays_games_20260815.csv").exists()


def test_non_allowlisted_family_protected(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {"not_a_family_20260815.csv": "x"})
    plan = plan_retention(sink, cfg, "mlb", reference_date="20260907")
    assert plan.candidates == ()
    assert any(p.name == "not_a_family_20260815.csv" for p in plan.protected)


def test_missing_sink_is_noop(tmp_path):
    cfg = default_config()
    plan = plan_retention(tmp_path / "nope", cfg, "mlb",
                          reference_date="20260907")
    assert plan.candidates == ()
    assert isinstance(plan, RetentionPlan)


def test_shap_family_name_extraction():
    assert _family_name("shap_game_20260907_WSH@SD.csv") == "shap_game"
    assert _family_name("todays_games_20260907.csv") == "todays_games"
    assert _family_name("model_history.json") == "model_history"
    # B-002 regressions: stamped-SHAP (date + id), .meta.json siblings,
    # and run-engine names must resolve to their allowlist families.
    assert _family_name("nba_shap_game_20270105_0022600001.csv") == "nba_shap_game"
    assert _family_name(
        "nba_run_engine_markets_20270105.meta.json") == "nba_run_engine_markets"
    assert _family_name(
        "nba_run_engine_markets_20270105.json") == "nba_run_engine_markets"
    assert _family_name("nfl_qb_matchup_20260907_KC@BUF.json") == "nfl_qb_matchup"


def test_family_on_allowlist_deletes(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {"shap_game_20260815_X@Y.csv": "old"})
    plan = apply_retention(sink, cfg, "mlb", reference_date="20260907",
                           execute=True)
    assert {p.name for p in plan.deleted} == {"shap_game_20260815_X@Y.csv"}


def test_stamped_shap_and_meta_families_deletable(tmp_path):
    """B-002 + B-004: sport-prefixed stamped-SHAP and .meta.json artifacts
    classify into their (now-allowlisted) families and are ACTUALLY
    deletable — previously they were permanently protected."""
    cfg = default_config()
    sink = _mk_sink(tmp_path, {
        "nba_shap_game_20270105_0022600001.csv": "old",
        "nba_run_engine_markets_20270105.meta.json": "old meta",
        "nba_shap_game_20270130_0022600009.csv": "recent",
    })
    plan = run_production_retention(
        sink, cfg, "mlb", generation_succeeded=True,
        reference_date="20270201")  # cutoff 20270112
    deleted = {p.name for p in plan.deleted}
    assert deleted == {
        "nba_shap_game_20270105_0022600001.csv",
        "nba_run_engine_markets_20270105.meta.json",
    }
    # Within-window stamped artifact survives.
    assert (sink / "nba_shap_game_20270130_0022600009.csv").exists()


def test_within_window_artifact_retained(tmp_path):
    """A 6-day-old artifact is inside the 20-day window: never a candidate."""
    cfg = default_config()
    sink = _mk_sink(tmp_path, {"todays_games_20260901.csv": "recent"})
    plan = apply_retention(sink, cfg, "mlb", reference_date="20260907",
                           execute=True)
    assert plan.candidates == ()
    assert plan.deleted == ()
    assert (sink / "todays_games_20260901.csv").exists()


# ---------------------------------------------------------------------------
# Production retention execution (merged from
# tests/test_retention_production.py — Phase 1 review decision 4:
# retention stays provably dry-run by default; production runners delete
# ONLY after successful artifact generation, and only date-stamped,
# non-exempt frontend artifacts older than 10 days.
# ---------------------------------------------------------------------------


def _mk_sink_prod(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / "sports" / "mlb" / "data_delivery"
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (d / name).write_text(content)
    # Durable stores that must NEVER be deleted.
    (d / "models").mkdir(exist_ok=True)
    (d / "models" / "ensemble_latest.joblib").write_bytes(b"m")
    (d / "store").mkdir(exist_ok=True)
    (d / "store" / "duckdb").mkdir(exist_ok=True)
    (d / "store" / "duckdb" / "games.duckdb").write_bytes(b"d")
    (d / "model_history.json").write_text("[]")
    (d / "features_metadata.json").write_text("{}")
    (d / "study_config.json").write_text("{}")
    return d


def test_default_is_dry_run(tmp_path):
    sink = _mk_sink_prod(tmp_path, {"todays_games_20260815.csv": "old"})
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=False, reference_date="20260907")
    assert plan.dry_run
    assert plan.candidates  # it planned the deletion...
    assert plan.deleted == ()  # ...but deleted nothing
    assert (sink / "todays_games_20260815.csv").exists()


def test_failed_generation_never_deletes(tmp_path):
    """A failed pipeline run must not shrink the sink."""
    sink = _mk_sink_prod(tmp_path, {"todays_games_20260815.csv": "old"})
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=False, reference_date="20260907")
    assert plan.deleted == ()
    assert (sink / "todays_games_20260815.csv").exists()


def test_successful_generation_executes_deletion(tmp_path):
    sink = _mk_sink_prod(tmp_path, {
        "todays_games_20260815.csv": "old",
        "todays_games_20260907.csv": "new",
    })
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=True, reference_date="20260907")
    assert {p.name for p in plan.deleted} == {"todays_games_20260815.csv"}
    assert (sink / "todays_games_20260907.csv").exists()
    assert not plan.dry_run


def test_durable_state_never_deleted_even_on_success(tmp_path):
    sink = _mk_sink_prod(tmp_path, {
        "todays_games_20260815.csv": "old",
        "raw_cache_20260815.json": "raw payload",
        "store_todays_games_20260815.parquet": "store data",
    })
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=True, reference_date="20260907")
    deleted = {p.name for p in plan.deleted}
    assert deleted == {"todays_games_20260815.csv"}
    # Raw caches, stores, contracts, study configs survive.
    assert (sink / "raw_cache_20260815.json").exists()
    assert (sink / "store" / "duckdb" / "games.duckdb").exists()
    assert (sink / "models" / "ensemble_latest.joblib").exists()
    assert (sink / "model_history.json").exists()
    assert (sink / "features_metadata.json").exists()
    assert (sink / "study_config.json").exists()
    # The raw cache was protected (undated family, not on allowlist).
    protected = {p.name for p in plan.protected}
    assert "raw_cache_20260815.json" in protected
    assert "study_config.json" in protected


def test_twenty_day_boundary_exact(tmp_path):
    # B-003 resolved (7.5d): window is 20 days per spec §23.
    """Artifacts exactly 20 days old are kept; 21 days old are deleted."""
    sink = _mk_sink_prod(tmp_path, {
        "todays_games_20260818.csv": "exactly 20 days",
        "todays_games_20260817.csv": "21 days",
    })
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=True, reference_date="20260907")
    assert {p.name for p in plan.deleted} == {"todays_games_20260817.csv"}
    assert (sink / "todays_games_20260818.csv").exists()


def test_direct_execute_still_dry_by_default(tmp_path):
    """apply_retention keeps its provably-dry default signature."""
    sink = _mk_sink_prod(tmp_path, {"todays_games_20260815.csv": "old"})
    plan = apply_retention(sink, default_config(), "mlb",
                           reference_date="20260907")
    assert plan.dry_run
    assert (sink / "todays_games_20260815.csv").exists()
    # And plan_retention can never delete.
    plan2 = plan_retention(sink, default_config(), "mlb",
                           reference_date="20260907")
    assert plan2.deleted == ()
