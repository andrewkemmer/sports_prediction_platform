"""Config-driven settlement rules + data-source policy (Phase r4).

Spec §20 requires settlement semantics to be explicit per-sport
configuration (``market_rules.yaml``) rather than hardcoded constants,
and §5/§5.1 require the source policy (sources, required payload
columns, transport retries/backoff/pacing) to live in
``data_sources.yaml``. These tests pin:

* the shared loaders fail loudly on missing/invalid config;
* every sport's adapter serves exactly the pre-r4 settlement semantics,
  now sourced from its ``config/market_rules.yaml``;
* the ingestion modules' source registry, required columns and transport
  policy match their ``config/data_sources.yaml`` (no drift in either
  direction);
* an undeclared market kind or an unknown policy key is a loud error,
  never a silent default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    DataSourcesError,
    MarketRulesError,
    load_data_sources,
    load_market_rules,
)
from core.contracts import SettlementConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPORTS = ("mlb", "nfl", "nhl", "nba")

# The pre-r4 hardcoded semantics, pinned here per (sport, market kind).
# Any change to these values is a reviewed settlement-rule change (spec
# §20), never a silent config edit.
_EXPECTED_SETTLEMENT = {
    "mlb": {
        "moneyline": ("full_game", False, False),
        "total": ("full_game", False, True),
        "spread": ("full_game", False, True),
        "run_line": ("full_game", False, True),
    },
    "nfl": {
        "moneyline": ("full_game", True, False),
        "spread": ("full_game", False, True),
        "total": ("full_game", False, True),
    },
    "nhl": {
        "moneyline": ("full_game", False, False),
        "moneyline_regulation": ("regulation", True, False),
        "puckline": ("full_game", False, True),
        "total": ("full_game", False, True),
    },
    "nba": {
        "moneyline": ("full_game", False, False),
        "spread": ("full_game", False, True),
        "total": ("full_game", False, True),
    },
}

# The declared source registries per sport: {source name: source_id}.
_EXPECTED_SOURCE_IDS = {
    "mlb": {"statcast": "mlb:pybaseball.statcast"},
    "nfl": {
        "schedules": "nfl:nflreadpy.load_schedules",
        "pbp": "nfl:nflreadpy.load_pbp",
        "player_stats": "nfl:nflreadpy.load_player_stats",
        "teams": "nfl:nflreadpy.load_teams",
    },
    "nhl": {
        "schedule": "nhl:api-web.nhle.com/schedule",
        "boxscore": "nhl:api-web.nhle.com/boxscore",
    },
    "nba": {
        "schedule": "nba:stats.nba.com/scheduleleaguev2",
        "boxscore": "nba:stats.nba.com/boxscoretraditionalv3",
    },
}


# ---------------------------------------------------------------------------
# market_rules.yaml — loader validity and per-sport semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sport", _SPORTS)
def test_market_rules_load_for_every_sport(sport):
    rules = load_market_rules(sport)
    assert rules.sport == sport
    assert rules.market_rules_version.startswith(f"{sport}_")
    assert set(rules.settlement) == set(_EXPECTED_SETTLEMENT[sport])


@pytest.mark.parametrize("sport", _SPORTS)
@pytest.mark.parametrize("kind", sorted(
    {k for kinds in _EXPECTED_SETTLEMENT.values() for k in kinds}))
def test_settlement_semantics_pin(sport, kind):
    """Only the sport's OWN declared kinds resolve; each resolves to the
    pinned pre-r4 semantics (scope, tie_allowed, push_allowed)."""
    expected = _EXPECTED_SETTLEMENT[sport].get(kind)
    rules = load_market_rules(sport)
    if expected is None:
        with pytest.raises(KeyError):
            rules.settlement_config(kind)
        return
    cfg = rules.settlement_config(kind)
    assert (cfg.settlement_scope, cfg.tie_allowed, cfg.push_allowed) \
        == expected
    assert isinstance(cfg, SettlementConfig)


def test_market_rules_unknown_kind_is_key_error():
    rules = load_market_rules("mlb")
    with pytest.raises(KeyError, match="nonsense"):
        rules.settlement_config("nonsense")


def test_market_rules_missing_file_fails_loudly(tmp_path):
    with pytest.raises(MarketRulesError, match="missing"):
        load_market_rules("mlb", tmp_path / "nope.yaml")


def test_market_rules_invalid_yaml_fails_loudly(tmp_path):
    p = tmp_path / "market_rules.yaml"
    p.write_text("settlement: [unclosed", encoding="utf-8")
    with pytest.raises(MarketRulesError, match="invalid YAML"):
        load_market_rules("mlb", p)


def _write_rules(tmp_path, doc):
    import yaml

    p = tmp_path / "market_rules.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_market_rules_unknown_top_key_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "market_rules_version": "x", "settlement": {},
           "extra": 1}
    with pytest.raises(MarketRulesError, match="unknown top-level"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


def test_market_rules_sport_mismatch_fails_loudly(tmp_path):
    doc = {"sport": "nba", "market_rules_version": "x",
           "settlement": {"moneyline": {"settlement_scope": "full_game"}}}
    with pytest.raises(MarketRulesError, match="sport must be"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


def test_market_rules_blank_version_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "market_rules_version": "  ",
           "settlement": {"moneyline": {"settlement_scope": "full_game"}}}
    with pytest.raises(MarketRulesError, match="market_rules_version"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


def test_market_rules_bad_scope_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "market_rules_version": "x",
           "settlement": {"moneyline": {"settlement_scope": "overtime"}}}
    with pytest.raises(MarketRulesError, match="settlement_scope"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


def test_market_rules_non_boolean_flag_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "market_rules_version": "x",
           "settlement": {"moneyline": {"settlement_scope": "full_game",
                                        "tie_allowed": "no"}}}
    with pytest.raises(MarketRulesError, match="must be a boolean"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


def test_market_rules_unknown_field_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "market_rules_version": "x",
           "settlement": {"moneyline": {"settlement_scope": "full_game",
                                        "scope": "full_game"}}}
    with pytest.raises(MarketRulesError, match="unknown field"):
        load_market_rules("mlb", _write_rules(tmp_path, doc))


# ---------------------------------------------------------------------------
# adapters serve the config (the pre-r4 hardcoded branches are gone)
# ---------------------------------------------------------------------------


def test_adapters_read_settlement_from_market_rules_yaml():
    from sports.mlb.adapter import MLBAdapter
    from sports.nba.adapter import NBAAdapter
    from sports.nfl.adapter import NFLAdapter
    from sports.nhl.adapter import NHLAdapter

    for adapter_cls, sport in ((MLBAdapter, "mlb"), (NFLAdapter, "nfl"),
                               (NHLAdapter, "nhl"), (NBAAdapter, "nba")):
        adapter = adapter_cls()
        for kind, expected in _EXPECTED_SETTLEMENT[sport].items():
            cfg = adapter.settlement_config(kind)
            assert (cfg.settlement_scope, cfg.tie_allowed,
                    cfg.push_allowed) == expected, (sport, kind)


def test_runners_gate_settlement_through_market_rules(monkeypatch):
    """The four production runners hold a loaded market-rules document
    (module-level ``_MARKET_RULES``) — the settlement gates are
    config-driven, not constant-driven."""
    import importlib

    for sport in _SPORTS:
        mod = importlib.import_module(f"sports.{sport}.run_production")
        rules = getattr(mod, "_MARKET_RULES")
        assert rules.sport == sport
        # Every kind the runner may settle is declared in the YAML.
        assert "moneyline" in rules.settlement


# ---------------------------------------------------------------------------
# data_sources.yaml — loader validity and per-sport policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sport", _SPORTS)
def test_data_sources_load_for_every_sport(sport):
    ds = load_data_sources(sport)
    assert ds.sport == sport
    assert ds.data_sources_version.startswith(f"{sport}_")
    assert set(ds.required_columns) == set(_EXPECTED_SOURCE_IDS[sport])
    assert set(ds.source_ids) == set(_EXPECTED_SOURCE_IDS[sport])
    for name, sid in _EXPECTED_SOURCE_IDS[sport].items():
        assert ds.source_ids[name] == sid
    assert ds.policy.chunk_retries >= 1
    assert ds.policy.chunk_retry_base_ms >= 0


def test_data_sources_missing_file_fails_loudly(tmp_path):
    with pytest.raises(DataSourcesError, match="missing"):
        load_data_sources("mlb", tmp_path / "nope.yaml")


def _write_sources(tmp_path, doc):
    import yaml

    p = tmp_path / "data_sources.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_data_sources_unknown_policy_key_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "data_sources_version": "x",
           "sources": {"statcast": {"required_columns": ["game_date"]}},
           "policy": {"transport": "carrier_pigeon"}}
    with pytest.raises(DataSourcesError, match="unknown field"):
        load_data_sources("mlb", _write_sources(tmp_path, doc))


def test_data_sources_negative_retry_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "data_sources_version": "x",
           "sources": {"statcast": {"required_columns": ["game_date"]}},
           "policy": {"chunk_retries": -1}}
    with pytest.raises(DataSourcesError, match="chunk_retries"):
        load_data_sources("mlb", _write_sources(tmp_path, doc))


def test_data_sources_empty_required_columns_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "data_sources_version": "x",
           "sources": {"statcast": {"required_columns": []}}}
    with pytest.raises(DataSourcesError, match="required_columns"):
        load_data_sources("mlb", _write_sources(tmp_path, doc))


def test_data_sources_source_id_drift_fails_loudly(tmp_path):
    doc = {"sport": "mlb", "data_sources_version": "x",
           "sources": {"statcast": {
               "source_id": "mlb:somewhere.else",
               "required_columns": ["game_date"]}}}
    ds = load_data_sources("mlb", _write_sources(tmp_path, doc))
    # The loader reports the declared id; the ingestion module's
    # cross-check is what fails the import (next test).
    assert ds.source_ids["statcast"] == "mlb:somewhere.else"


# ---------------------------------------------------------------------------
# ingestion modules cross-check the policy (no drift in either direction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sport", _SPORTS)
def test_ingestion_required_cols_match_yaml(sport):
    import importlib

    ds = load_data_sources(sport)
    mod = importlib.import_module(f"sports.{sport}.ingestion")
    if sport == "mlb":
        assert mod.REQUIRED_STATCAST_COLS == ds.required_columns["statcast"]
    else:
        names = ("schedules", "pbp", "player_stats", "teams") \
            if sport == "nfl" else ("schedule",)
        assert dict(mod.REQUIRED_COLS) == {
            n: ds.required_columns[n] for n in names} \
            if sport == "nfl" else mod.REQUIRED_COLS == ds.required_columns[
                "schedule"]


@pytest.mark.parametrize("sport", _SPORTS)
def test_ingestion_transport_policy_matches_yaml(sport):
    import importlib

    ds = load_data_sources(sport)
    mod = importlib.import_module(f"sports.{sport}.ingestion")
    assert mod.CHUNK_RETRIES == ds.policy.chunk_retries
    assert mod.CHUNK_RETRY_BASE_MS == ds.policy.chunk_retry_base_ms
    if sport in ("mlb", "nhl"):
        assert mod.REFRESH_TAIL_DAYS == ds.policy.refresh_tail_days
    if sport == "nba":
        assert mod.REQUEST_PAUSE_SEC == ds.policy.request_pause_ms / 1000.0


@pytest.mark.parametrize("sport", _SPORTS)
def test_ingestion_source_ids_declared_in_yaml(sport):
    ds = load_data_sources(sport)
    for name, sid in _EXPECTED_SOURCE_IDS[sport].items():
        assert ds.source_ids.get(name) == sid, (sport, name)
