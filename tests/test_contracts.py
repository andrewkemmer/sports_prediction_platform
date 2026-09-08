"""Tests for core.contracts."""

from pathlib import Path

import pandas as pd
import pytest

from core.config import default_config
from core.contracts import (
    ArtifactContract,
    ArtifactFamily,
    FeatureContract,
    FeatureSpec,
    ParticipantBox,
    default_artifact_contract,
    load_artifact_contract,
)


def _spec(name="is_home"):
    return FeatureSpec(name=name, summary="s", definition="d", formula="1",
                       source="static", window="n/a", units="binary",
                       direction="n/a", members=("xgboost",))


def test_feature_contract_basics():
    c = FeatureContract(sport="mlb", version="v1",
                        features=(_spec("a"), _spec("b")))
    assert c.feature_names() == ("a", "b")
    assert c.get("a").formula == "1"
    with pytest.raises(KeyError):
        c.get("zzz")


def test_feature_contract_rejects_duplicates_and_empty():
    with pytest.raises(ValueError, match="duplicate"):
        FeatureContract(sport="mlb", version="v1",
                        features=(_spec("a"), _spec("a")))
    with pytest.raises(ValueError, match="non-empty"):
        FeatureContract(sport="mlb", version="v1", features=(_spec(""),))
    with pytest.raises(ValueError, match="must declare"):
        FeatureContract(sport="mlb", version="v1", features=())


def test_feature_metadata_json_shape():
    c = FeatureContract(sport="mlb", version="v1", features=(_spec("is_home"),))
    meta = c.to_metadata_json(generated_for="mlb")
    assert meta["n_features"] == 1
    assert meta["features"]["is_home"]["members"] == ["xgboost"]
    assert meta["warnings"] == []
    assert "categorical_context" in meta


def test_artifact_family_date_parsing():
    f = ArtifactFamily(family="todays_games", pattern="todays_games_*.csv",
                       ext="csv")
    assert f.date_from_name("todays_games_20260907.csv") == "20260907"
    assert f.date_from_name("todays_games_bad.csv") is None
    shap = ArtifactFamily(family="shap_game", pattern="shap_game_*.csv",
                          ext="csv")
    assert shap.date_from_name("shap_game_20260907_WSH@SD.csv") == "20260907"


def test_undated_artifacts_must_be_retention_exempt():
    with pytest.raises(ValueError, match="retention_exempt"):
        ArtifactContract(sport="mlb", families=(
            ArtifactFamily(family="model_history",
                           pattern="model_history.json", ext="json",
                           date_stamped=False),
        ))


def test_default_artifact_contract_mlb():
    c = default_artifact_contract("mlb")
    assert c.sport == "mlb"
    names = {f.family for f in c.families}
    assert "todays_games" in names
    assert "model_history" in names
    # dated families are retention candidates; durable ones are not
    cand = {f.family for f in c.retention_candidates()}
    assert "todays_games" in cand
    assert "model_history" not in cand


def test_default_artifact_contract_sport_prefix():
    c = default_artifact_contract("nfl")
    fam = c.family("todays_games")
    assert fam.pattern == "nfl_todays_games_*.csv"


def test_default_artifact_contract_nhl_nba():
    for s in ("nhl", "nba"):
        c = default_artifact_contract(s)
        assert c.family("markets").pattern.startswith(f"{s}_")


def test_load_artifact_contract_from_file(tmp_path):
    p = tmp_path / "contract.json"
    p.write_text(
        '{"sport": "mlb", "families": [{"family": "todays_games",'
        ' "pattern": "todays_games_*.csv", "ext": "csv"}]}')
    c = load_artifact_contract(p)
    assert c.sport == "mlb"
    assert c.family("todays_games").pattern == "todays_games_*.csv"


def test_participant_box_rules():
    box = ParticipantBox(role="pitcher", name="Yoshinobu Yamamoto",
                         stats=(("ERA", "2.45"), ("K/9", "10.1")))
    assert box.name
    with pytest.raises(ValueError, match="non-empty"):
        ParticipantBox(role="pitcher", name="")
    tbd = ParticipantBox(role="pitcher", name="", is_tbd=True)
    assert tbd.is_tbd


def test_sport_adapter_protocol_is_runtime_checkable():
    from core.contracts import SportAdapter

    class _Adapter:
        sport_key = "mlb"

        def artifact_contract(self):
            return default_artifact_contract("mlb")

        def feature_contract(self):
            return FeatureContract(sport="mlb", version="v1",
                                   features=(_spec(),))

        def normalize_games(self, raw):
            return raw

        def participant_box(self, game_row, side):
            return ParticipantBox(role="pitcher", name="X")

        def market_frame(self, raw):
            return raw

        def market_probs(self, market_row):
            return {"p_home": 0.5, "p_away": 0.5}

        def settlement_config(self, market_kind):
            from core.markets import FULL_GAME_NO_TIE
            return FULL_GAME_NO_TIE

    assert isinstance(_Adapter(), SportAdapter)


def test_sport_adapter_settlement_config_contract():
    """Every adapter exposes per-market-kind settlement configs."""
    from core.contracts import SportAdapter
    from core.markets import (
        FULL_GAME_NO_TIE,
        FULL_GAME_TIE_ALLOWED,
        REGULATION_TIE_ALLOWED,
        SettlementConfig,
    )

    class _Adapter:
        sport_key = "nfl"

        def artifact_contract(self):
            return default_artifact_contract("nfl")

        def feature_contract(self):
            return FeatureContract(sport="nfl", version="v1",
                                   features=(_spec(),))

        def normalize_games(self, raw):
            return raw

        def participant_box(self, game_row, side):
            return ParticipantBox(role="qb", name="X")

        def market_frame(self, raw):
            return raw

        def market_probs(self, market_row):
            return {"p_home": 0.5, "p_away": 0.45, "p_tie": 0.05}

        def settlement_config(self, market_kind):
            return {
                "moneyline": FULL_GAME_TIE_ALLOWED,
                "regulation_moneyline": REGULATION_TIE_ALLOWED,
                "spread": FULL_GAME_TIE_ALLOWED,
                "total": FULL_GAME_NO_TIE,
            }[market_kind]

    adapter = _Adapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.settlement_config("moneyline").tie_allowed
    assert not adapter.settlement_config("total").tie_allowed
    assert isinstance(adapter.settlement_config("spread"), SettlementConfig)


def test_contract_paths_under_new_layout(tmp_path):
    cfg = default_config()
    d = cfg.data_delivery_dir("mlb", tmp_path)
    assert d == tmp_path / "sports" / "mlb" / "data_delivery"
    assert isinstance(d, Path)
