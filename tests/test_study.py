"""Tests for core.study."""

import pytest

from core.study import StudyConfig, StudyConfigError, load_study


def _valid(**overrides):
    base = dict(
        study_id="exp-001",
        kind="experiment",
        sport="mlb",
        date_start="20260401",
        date_end="20260901",
    )
    base.update(overrides)
    return StudyConfig(**base)


def test_valid_study_roundtrip():
    s = _valid()
    assert s.is_production is False
    d = s.to_dict()
    s2 = StudyConfig.from_dict(d)
    assert s2 == s
    assert StudyConfig.from_json(s.to_json()) == s


def test_production_study_may_push():
    s = _valid(kind="production", may_push_artifacts=True)
    s.validate_push_allowed()  # no raise


def test_experiment_cannot_push():
    s = _valid(kind="experiment", may_push_artifacts=True)
    with pytest.raises(StudyConfigError, match="may not push"):
        s.validate_push_allowed()


def test_production_without_flag_cannot_push():
    s = _valid(kind="production")
    with pytest.raises(StudyConfigError):
        s.validate_push_allowed()


@pytest.mark.parametrize("bad", [
    {"kind": "nope"},
    {"sport": "nhl2"},
    {"date_start": "2026-04-01"},
    {"date_end": "20260301"},  # before start
])
def test_invalid_fields_rejected(bad):
    kwargs = dict(study_id="x", kind="experiment", sport="mlb",
                  date_start="20260401", date_end="20260901")
    kwargs.update(bad)
    with pytest.raises(StudyConfigError):
        StudyConfig(**kwargs)


def test_empty_study_id_rejected():
    with pytest.raises(StudyConfigError, match="study_id"):
        _valid(study_id="")


def test_unknown_fields_rejected():
    with pytest.raises(StudyConfigError, match="unknown study fields"):
        StudyConfig.from_dict({**_valid().to_dict(), "surprise": 1})


def test_missing_field_rejected():
    d = _valid().to_dict()
    del d["sport"]
    with pytest.raises(StudyConfigError, match="missing study field"):
        StudyConfig.from_dict(d)


def test_invalid_json_rejected():
    with pytest.raises(StudyConfigError, match="invalid study JSON"):
        StudyConfig.from_json("{not json")


def test_walk_forward_override():
    from core.config import WalkForwardConfig
    s = _valid(walk_forward=WalkForwardConfig(min_train_games=50,
                                              step_days=2))
    assert s.walk_forward.min_train_games == 50
    assert s.walk_forward.step_days == 2


def test_walk_forward_dict_accepted_in_from_dict():
    s = StudyConfig.from_dict({**_valid().to_dict(),
                               "walk_forward": {"min_train_games": 50}})
    assert s.walk_forward.min_train_games == 50


def test_load_study_from_file(tmp_path):
    p = tmp_path / "study.json"
    p.write_text(_valid().to_json())
    assert load_study(p) == _valid()
