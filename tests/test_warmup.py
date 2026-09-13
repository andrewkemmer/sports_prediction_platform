"""Tests for core.warmup."""

from core.config import WarmupConfig
from core.study import (
    evaluate_warmup,
    teams_ready_for_display,
    warmup_complete_date,
)


def test_empty_store_not_ready():
    st = evaluate_warmup(sport_warmup=WarmupConfig(), first_game_date=None,
                         run_date="20260907")
    assert not st.ready
    assert "no games" in st.reason


def test_history_days_gate():
    st = evaluate_warmup(sport_warmup=WarmupConfig(min_history_days=30),
                         first_game_date="20260901",
                         run_date="20260907",
                         team_game_counts={"NYY": 50, "BOS": 50})
    assert not st.ready
    assert st.days_of_history == 6
    assert "history 6d" in st.reason


def test_min_games_gate_names_teams():
    st = evaluate_warmup(sport_warmup=WarmupConfig(min_history_days=10),
                         first_game_date="20260801",
                         run_date="20260907",
                         team_game_counts={"NYY": 50, "BOS": 5, "LAD": 9})
    assert not st.ready
    assert st.teams_below_minimum == ("BOS", "LAD")
    assert st.days_of_history == 37


def test_ready_when_all_gates_pass():
    st = evaluate_warmup(sport_warmup=WarmupConfig(min_history_days=10,
                                                   min_games_per_team=5),
                         first_game_date="20260801",
                         run_date="20260907",
                         team_game_counts={"NYY": 50, "BOS": 6})
    assert st.ready
    assert st.reason == "ok"
    assert bool(st) is True


def test_ready_with_no_team_counts():
    st = evaluate_warmup(sport_warmup=WarmupConfig(min_history_days=1),
                         first_game_date="20260901",
                         run_date="20260907")
    assert st.ready


def test_warmup_complete_date():
    assert warmup_complete_date(WarmupConfig(min_history_days=30),
                                "20260901") == "20261001"


def test_teams_ready_for_display():
    ready, below = teams_ready_for_display({"NYY": 12, "BOS": 3}, 10)
    assert ready == ("NYY",)
    assert below == ("BOS",)
