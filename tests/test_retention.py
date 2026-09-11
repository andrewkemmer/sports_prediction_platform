"""Tests for core.retention."""

from pathlib import Path

from core.config import default_config
from core.retention import (
    RetentionPlan,
    _family_name,
    apply_retention,
    plan_retention,
    retention_cutoff,
    run_production_retention,
)


def _mk_sink(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / "sports" / "mlb" / "data_delivery"
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (d / name).write_text(content)
    (d / "models").mkdir(exist_ok=True)
    (d / "models" / "ensemble_latest.joblib").write_bytes(b"m")
    return d


def test_retention_cutoff():
    assert retention_cutoff(10, "20260907") == "20260828"
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


def test_family_on_allowlist_deletes(tmp_path):
    cfg = default_config()
    sink = _mk_sink(tmp_path, {"shap_game_20260815_X@Y.csv": "old"})
    plan = apply_retention(sink, cfg, "mlb", reference_date="20260907",
                           execute=True)
    assert {p.name for p in plan.deleted} == {"shap_game_20260815_X@Y.csv"}


def test_within_window_artifact_retained(tmp_path):
    """A 6-day-old artifact is inside the 10-day window: never a candidate."""
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


def test_ten_day_boundary_exact(tmp_path):
    # B-003 pending: 7.5d updates the 10-day window to 20; this literal
    # tracks the production value until then.
    """Artifacts exactly 10 days old are kept; 11 days old are deleted."""
    sink = _mk_sink_prod(tmp_path, {
        "todays_games_20260828.csv": "exactly 10 days",
        "todays_games_20260827.csv": "11 days",
    })
    plan = run_production_retention(
        sink, default_config(), "mlb",
        generation_succeeded=True, reference_date="20260907")
    assert {p.name for p in plan.deleted} == {"todays_games_20260827.csv"}
    assert (sink / "todays_games_20260828.csv").exists()


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
