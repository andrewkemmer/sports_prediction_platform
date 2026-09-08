"""Tests for core.retention."""

from pathlib import Path

from core.config import default_config
from core.retention import (
    RetentionPlan,
    _family_name,
    apply_retention,
    plan_retention,
    retention_cutoff,
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
