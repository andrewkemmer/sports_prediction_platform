"""Tests for core.config."""

import json

import pytest

from core.config import (
    DEFAULT_SPORT,
    SPORTS,
    DeliveryConfig,
    PlatformConfig,
    PredictionWindowConfig,
    SportConfig,
    WarmupConfig,
    default_config,
    is_unknown_sport,
    load_config,
    normalize_sport_key,
)


def test_default_config_has_all_sports():
    cfg = default_config()
    assert set(cfg.sports) == set(SPORTS)
    assert DEFAULT_SPORT == "mlb"


def test_participant_roles_per_phase0_decisions():
    cfg = default_config()
    assert cfg.sport("mlb").participant_role == "pitcher"
    assert cfg.sport("nfl").participant_role == "qb"
    assert cfg.sport("nhl").participant_role == "goalie"
    assert cfg.sport("nba").participant_role == "top_scorer"


def test_normalize_sport_key_falls_back_to_mlb():
    assert normalize_sport_key("NFL") == "nfl"
    assert normalize_sport_key(" mlb ") == "mlb"
    assert normalize_sport_key(None) == "mlb"
    assert normalize_sport_key("") == "mlb"
    assert normalize_sport_key("hockey") == "mlb"


def test_is_unknown_sport_only_flags_nonempty_unknowns():
    assert is_unknown_sport("nfl") is False
    assert is_unknown_sport("mlb") is False
    assert is_unknown_sport(None) is False
    assert is_unknown_sport("") is False
    assert is_unknown_sport("none") is False
    assert is_unknown_sport("hockey") is True


def test_data_delivery_dir_layout():
    cfg = default_config()
    from pathlib import Path
    d = cfg.data_delivery_dir("mlb", Path("/repo"))
    assert d == Path("/repo") / "sports" / "mlb" / "data_delivery"
    # legacy naming never appears
    assert "backend" not in str(d)


def test_prediction_window_validation():
    with pytest.raises(ValueError):
        PredictionWindowConfig(lookahead_days=8)
    with pytest.raises(ValueError):
        PredictionWindowConfig(slate_cutoff_hour_local=24)


def test_warmup_validation():
    with pytest.raises(ValueError):
        WarmupConfig(min_games_per_team=0)
    with pytest.raises(ValueError):
        WarmupConfig(min_history_days=-1)


def test_load_config_from_dict_overrides():
    cfg = load_config(raw={
        "sports": {
            "mlb": {"lookahead_days": 2, "min_train_games": 500},
            "nfl": {"enabled": False},
        },
        "retention": {"frontend_days": 5},
        "delivery": {"owner": "andrewkemmer", "repo": "sports_prediction_platform"},
        "run_utc_hour": 7,
    })
    assert cfg.sport("mlb").prediction_window.lookahead_days == 2
    assert cfg.sport("mlb").walk_forward.min_train_games == 500
    assert cfg.sports["nfl"].enabled is False
    assert cfg.retention.frontend_days == 5
    assert cfg.delivery.repo_slug == "andrewkemmer/sports_prediction_platform"
    assert cfg.run_utc_hour == 7
    # untouched sport keeps defaults
    assert cfg.sport("nba").participant_role == "top_scorer"


def test_load_config_unknown_sport_rejected():
    with pytest.raises(ValueError, match="unknown sport"):
        load_config(raw={"sports": {"nhl2": {}}})


def test_load_config_from_file(tmp_path):
    p = tmp_path / "platform.json"
    p.write_text(json.dumps({"retention": {"frontend_days": 3}}))
    cfg = load_config(p)
    assert cfg.retention.frontend_days == 3


def test_load_config_bad_root_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="JSON object"):
        load_config(p)


def test_load_config_path_and_raw_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        load_config(tmp_path / "x.json", raw={})


def test_load_config_defaults_when_nothing_given():
    cfg = load_config()
    assert cfg.retention.frontend_days == 20
    assert isinstance(cfg.delivery, DeliveryConfig)
    assert isinstance(cfg, PlatformConfig)


def test_sport_config_rejects_bad_role():
    with pytest.raises(ValueError, match="participant_role"):
        SportConfig(key="mlb", label="MLB", emoji="x",
                    participant_role="center")


def test_platform_config_key_mismatch_rejected():
    sport = SportConfig(key="nfl", label="NFL", emoji="x",
                        participant_role="qb")
    with pytest.raises(ValueError, match="registry key"):
        PlatformConfig(sports={"mlb": sport})


def test_delivery_allowlist_contains_frontend_families():
    d = DeliveryConfig()
    assert "todays_games" in d.allowlisted_families
    assert "shap_game" in d.allowlisted_families
