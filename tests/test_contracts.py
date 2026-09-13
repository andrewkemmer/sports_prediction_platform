"""Tests for core.contracts."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import default_config
from core.contracts import (
    FEATURE_SPEC_FIELDS,
    ArtifactContract,
    ArtifactFamily,
    FeatureContract,
    FeatureContractError,
    FeatureSpec,
    FeaturesMetadataError,
    ParticipantBox,
    default_artifact_contract,
    load_artifact_contract,
    require_features_metadata,
    validate_columns_have_metadata,
    validate_metadata_is_reachable,
)
from core.contracts import (
    EXPECTED_REGISTRY_SIZE,
    REGISTRY,
    RecordValidationError,
    validate_record,
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


def test_validate_declared_contracts_against_metadata():
    """The shared interface fails loud in BOTH directions: a declared
    column without metadata, and metadata no declared tuple serves."""
    # passing cases
    validate_columns_have_metadata("sport", "moneyline", ("a", "b"),
                                   {"a", "b"})
    validate_metadata_is_reachable("sport", {"a", "b"}, {"a", "b", "c"})
    # declared-but-undocumented column
    with pytest.raises(FeatureContractError, match="missing registry"):
        validate_columns_have_metadata("sport", "moneyline", ("a", "c"),
                                       {"a", "b"})
    # orphan metadata (parallel/dead registry)
    with pytest.raises(FeatureContractError, match="not reachable"):
        validate_metadata_is_reachable("sport", {"a", "b"}, {"a"})


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
            from core.contracts import FULL_GAME_NO_TIE
            return FULL_GAME_NO_TIE

    assert isinstance(_Adapter(), SportAdapter)


def test_sport_adapter_settlement_config_contract():
    """Every adapter exposes per-market-kind settlement configs."""
    from core.contracts import SportAdapter
    from core.contracts import (
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


# ---------------------------------------------------------------------------
# B-005: pre-write record-type validation registry (hardened rule 10)
# ---------------------------------------------------------------------------

def test_registry_size_pinned():
    """The registry pins 56 entries: 14 per sport. MLB = 13 primaries +
    run_engine_markets.meta; each of NFL/NHL/NBA = 10 artifact primaries
    + run_engine_markets.meta + 3 runner-side OOF/fold stores. Any
    change is a reviewed registry change — never silent growth."""
    assert len(REGISTRY) == EXPECTED_REGISTRY_SIZE == 56
    from collections import Counter
    per_sport = Counter(s for s, _ in REGISTRY)
    assert per_sport == {"mlb": 14, "nfl": 14, "nhl": 14, "nba": 14}


def test_registry_nonempty_required_fields_and_probability_closed():
    """Registry hardening: non-empty required_fields everywhere; every
    declared probability constraint is the closed interval [0, 1]."""
    for (sport, family), spec in sorted(REGISTRY.items()):
        assert spec.required_fields, (
            f"({sport}, {family}): empty required_fields")
        assert spec.record_kind in ("frame", "json")


def test_validate_record_rejects_unknown_family():
    with pytest.raises(RecordValidationError, match="no registry entry"):
        validate_record("mlb", "not_a_family", {})


def test_validate_record_rejects_wrong_record_kind():
    with pytest.raises(RecordValidationError, match="expected DataFrame"):
        validate_record("mlb", "shap_game", {"feature": "x"})
    with pytest.raises(RecordValidationError, match="expected dict"):
        validate_record("mlb", "calibration", pd.DataFrame({"a": [1]}))


def _shap_frame(shap_value=0.1):
    return pd.DataFrame({
        "feature": ["f1"], "shap_value": [shap_value],
        "signed_effect": [1], "perspective_team": ["HOU"]})


def _mlb_calibration_record(**over):
    rec = {k: 1 if k == "n_games" else ([] if k in (
        "calibration_buckets", "daily") else
        ({} if k in ("metrics", "calibration", "league_total",
                     "evening_games_league") else "x"))
        for k in REGISTRY["mlb", "calibration"].required_fields}
    rec.update(over)
    return rec


@pytest.mark.parametrize(
    "sport,family", sorted(REGISTRY.keys()),
    ids=[f"{s}-{f}" for s, f in sorted(REGISTRY.keys())])
def test_validate_record_valid_record_passes(sport, family):
    """Every registry entry accepts a minimal structurally-valid record
    built from its own declared required fields."""
    spec = REGISTRY[sport, family]
    if spec.record_kind == "frame":
        def _pval(c):
            if c in spec.probability_fields:
                return 0.5
            if c.startswith("p_push_"):
                return 0.1
            if c.startswith(("p_over_", "p_under_")):
                return 0.45
            if c.endswith(("_home", "_away")) and c.startswith("p_rl_"):
                return 0.3
            if c.startswith("p_rl_") and c.endswith("_push"):
                return 0.4
            if c.startswith(("p_home_cover_", "p_rl_")):
                return 0.45
            if c.startswith("p_"):
                return 0.5
            return 1
        rec = pd.DataFrame({c: [_pval(c)] for c in spec.required_fields})
    else:
        rec = {}
        for k in spec.required_fields:
            if k in spec.probability_fields:
                rec[k] = 0.5
            elif k.endswith("_utc") or k in ("trained_at", "date"):
                rec[k] = "2026-09-07T00:00:00Z"
            elif k == "games":
                # moneyline_v1/matchup families embed per-game cards; a
                # minimal card with contract-permitted nulls is valid.
                rec[k] = []
            elif k in ("n_games", "n_draws", "n_pre", "n_holdout",
                       "n_points", "n_games_total", "excluded_sparse_days",
                       "n_games_in_series", "n_features", "seed",
                       "columns", "n_rows", "n_oof", "n_slate"):
                rec[k] = 1
            else:
                rec[k] = {} if k in (
                    "metrics", "config", "calibration", "games",
                    "manifest", "coverage", "fold_geometry", "fit",
                    "market_metrics", "winner_cards", "slate_history",
                    "rolling_brier", "features_metadata", "ensemble",
                    "series", "categorical_context", "warnings",
                    "calibration_buckets", "daily", "line_grid",
                    "alpha_home", "alpha_away", "year_effect_home",
                    "year_effect_away", "phase2_single_alpha",
                    "fit_check_single_alpha", "fit_check_alpha_lambda",
                    "variance_check", "mc_meta", "k_edge",
                    "agreement_vs_moneyline", "agreement_slate",
                    "drift_summary", "feature_drift", "feature_coverage",
                    "brier_baseline", "brier_baseline_label",
                    "rolling_brier_meta", "run_engine", "model_history",
                    "version_history", "last_retrained", "next_retrain",
                    "last_retrained_note", "next_retrain_note",
                    "upset_note", "map_scope_note", "source_column",
                    "calibrator_is_identity", "history_mean_brier",
                    "generated_for", "feature_set_version",
                    "served_columns", "grids", "record", "written_utc",
                    "slate_date", "markets_persisted",
                    "markets_persist_error", "schema", "phase1") else "x"
    validate_record(sport, family, rec)


@pytest.mark.parametrize(
    "sport,family", sorted(REGISTRY.keys()),
    ids=[f"{s}-{f}" for s, f in sorted(REGISTRY.keys())])
def test_validate_record_missing_required_field_raises(sport, family):
    """Every registry entry rejects a record missing one required field
    (zero-byte discipline lives in the writer-level tests)."""
    spec = REGISTRY[sport, family]
    if spec.record_kind == "frame":
        cols = [c for c in spec.required_fields[1:]]
        rec = pd.DataFrame({c: [1] for c in cols}) if cols else pd.DataFrame()
    else:
        rec = {k: "x" for k in spec.required_fields[1:]}
    with pytest.raises(RecordValidationError, match="missing required"):
        validate_record(sport, family, rec)


def test_validate_record_probability_range_enforced():
    """Out-of-range probabilities are rejected (closed [0, 1]) on both
    declared probability fields and the p_* column convention."""
    validate_record("mlb", "shap_game", _shap_frame(0.1))  # passes
    bad = pd.DataFrame({
        "game_id": ["g"], "game_date": ["2026-09-07"],
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [5.0], "away_score": [3.0], "home_win": [1.0],
        "home_win_prob_model": [1.2],
        "home_win_prob_model_calibrated": [0.5],
        "model_pick": ["A"], "actual_winner": ["A"], "correct": [True],
    })
    with pytest.raises(RecordValidationError, match="outside"):
        validate_record("mlb", "predictions_history", bad)


def test_validate_record_nan_inf_rejected_json():
    rec = _mlb_calibration_record(metrics=float("nan"))
    with pytest.raises(RecordValidationError, match="NaN/inf"):
        validate_record("mlb", "calibration", rec)
    rec2 = _mlb_calibration_record(metrics=float("inf"))
    with pytest.raises(RecordValidationError, match="NaN/inf"):
        validate_record("mlb", "calibration", rec2)


def test_validate_record_none_only_where_nullable():
    """None is rejected unless the contract declares the key nullable;
    nullable keys accept it (e.g. markets_persist_error on success)."""
    rec = _mlb_calibration_record(metrics=None)
    with pytest.raises(RecordValidationError, match="None not permitted"):
        validate_record("mlb", "calibration", rec)
    mon = {k: "x" for k in REGISTRY["mlb", "run_engine_monitor"]
           .required_fields}
    mon["markets_persist_error"] = None
    validate_record("mlb", "run_engine_monitor", mon)


def test_validate_record_mass_group_read_only():
    """Mass groups are read-only: a triple summing outside [0, 1+eps]
    fails; the validator never renormalizes."""
    from sports.mlb.artifacts import MARKET_COLUMNS_V3, persist_markets
    from sports.mlb.models.market.run_engine import TOTAL_LINE_GRID
    grid = {}
    for c in MARKET_COLUMNS_V3:
        if c.startswith(("p_over_", "p_under_")):
            grid[c] = [0.45, 0.4]
        elif c.startswith("p_push_"):
            grid[c] = [0.1, 0.2]
        elif c.startswith(("p_home_cover_", "p_rl_")):
            grid[c] = [0.4, 0.35]
        elif c in ("kind", "game_date", "agreement_conflict"):
            grid[c] = ["oof", "oof"] if c == "kind" else ["2026-09-07"] * 2 \
                if c == "game_date" else [False, False]
        else:
            grid[c] = [1, 2] if c == "game_pk" else [4.0] * 2 \
                if "expected_runs" in c or c in ("home_score", "away_score",
                                                 "total_runs") \
                else [0.55, 0.45]
    frame = pd.DataFrame(grid)
    frame.loc[0, "p_under_" + str(TOTAL_LINE_GRID[0]).replace(".", "_")] = 0.5
    with pytest.raises(RecordValidationError, match="mass"):
        persist_markets(frame, "20260907", {}, out_dir=".")


# ---------------------------------------------------------------------------
# features_metadata canonical shape (shared interface, all four sports)
# ---------------------------------------------------------------------------
# The per-sport METADATA MODELS stay per-sport on purpose (MLB keeps its
# FeatureEntry records, NFL/NHL/NBA keep their ``_SPEC_DEFS`` mapping); what
# is shared and enforced here is the SERIALIZED document every model-monitor
# writer embeds and the durable ``features_metadata`` artifact carries.

@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_require_features_metadata_accepts_every_sport_contract(sport):
    """Each sport's frozen registry contract serializes to a document the
    shared validator accepts: one entry per moneyline feature, each carrying
    the full FeatureSpec field set."""
    module = importlib.import_module(f"sports.{sport}.features.registry")
    meta = module.build_feature_contract().to_metadata_json("2026-09-01")
    assert require_features_metadata(sport, meta) is meta
    assert set(meta["features"]) == set(module.MONEYLINE_FEATURE_COLS)
    for name, entry in meta["features"].items():
        assert set(entry) == set(FEATURE_SPEC_FIELDS)
        assert entry["name"] == name


def test_require_features_metadata_rejects_stub_and_partial():
    """The exact stub the NNX model-monitor writers used to emit (an empty
    per-feature mapping, because their only caller passes cov=[]) is rejected
    rather than written — as is a partial or wrapper-less document."""
    with pytest.raises(FeaturesMetadataError, match="no per-feature entries"):
        require_features_metadata("nfl", {"generated_for": "", "n_features": 0,
                                          "warnings": [], "features": {}})
    with pytest.raises(FeaturesMetadataError, match="missing wrapper key"):
        require_features_metadata("nfl", {"features": {"a": {}}})
    with pytest.raises(FeaturesMetadataError, match="missing FeatureSpec"):
        require_features_metadata("nfl", {"generated_for": "", "n_features": 1,
                                          "warnings": [],
                                          "features": {"a": {"name": "a"}}})
    with pytest.raises(FeaturesMetadataError, match="declares name"):
        require_features_metadata("nfl", {
            "generated_for": "", "n_features": 1, "warnings": [],
            "features": {"a": {f: "" for f in FEATURE_SPEC_FIELDS}}})


# ---------------------------------------------------------------------------
# Artifact-family <-> pre-write record-contract alignment
# ---------------------------------------------------------------------------
# The platform carries two independent declarations of what a sport writes:
# ``ArtifactFamily`` (core.contracts: emitted file family + pattern + ext) and
# ``RecordType`` (core.record_validation: the pre-write record contract, keyed
# by the EXACT emitted family name, carrying ``record_kind``). The two tables
# below are the reviewed remainder after the family names and types were
# aligned. They are asserted as EXACT sets: a new unmapped family, a dropped
# registry entry, or a changed record kind fails here instead of drifting.

#: Artifact family -> record family, for the only families whose NAMES differ.
#: Both exist because the frontend globs the legacy allowlist alias while the
#: record contract is keyed by the exact emitted family (see the module
#: docstring of core/record_validation.py: "MLB emits run_engine_markets /
#: run_engine_monitor while the legacy allowlist keys markets/markets_monitor
#: resolve to those same files via the frontend globs").
ARTIFACT_TO_RECORD_FAMILY = {
    "markets": "run_engine_markets",
    "markets_monitor": "run_engine_monitor",
}

#: Artifact-contract families with deliberately NO record contract.
_ARTIFACT_FAMILY_WITHOUT_RECORD_CONTRACT = {
    # Reserved durable names — zero production writers today (documented in
    # core/record_validation.py; a future writer must add a registry entry).
    "mlb": {"model_history", "model_version_history"},
    "nfl": {"nfl_model_history", "nfl_model_version_history",
            "nfl_features_metadata"},
    "nhl": {"nhl_model_history", "nhl_model_version_history",
            "nhl_features_metadata"},
    "nba": {"nba_model_history", "nba_model_version_history",
            "nba_features_metadata"},
}
#: NNX game cards / drift / coverage are NOT standalone records: the cards ship
#: in ``<sport>_moneyline_v1_*.json`` and drift/coverage are file-embedded
#: fields of ``<sport>_model_monitor.json`` (core/record_validation.py
#: docstring). Their artifact-contract families are the legacy standalone
#: aliases, which is why no record contract is keyed by them.
for _s in ("nfl", "nhl", "nba"):
    _ARTIFACT_FAMILY_WITHOUT_RECORD_CONTRACT[_s] |= {
        "todays_games", "feature_drift", "feature_coverage"}

#: Registry families with deliberately NO artifact-contract family.
_RECORD_FAMILY_WITHOUT_ARTIFACT_FAMILY = {
    # run_engine_markets.meta is the companion of run_engine_markets (same
    # family name for retention, separate record schema).
    "mlb": {"run_engine_markets.meta"},
    "nfl": {"feature_v1", "fold_table", "moneyline_v1", "oof_distribution",
            "oof_moneyline", "qb_matchup", "run_engine_markets.meta"},
    "nhl": {"feature_v1", "fold_table", "goalie_matchup", "moneyline_v1",
            "oof_distribution", "oof_moneyline", "run_engine_markets.meta"},
    "nba": {"feature_v1", "fold_table", "moneyline_v1", "oof_distribution",
            "oof_moneyline", "player_matchup", "run_engine_markets.meta"},
}

_EXT_TO_RECORD_KIND = {"csv": "frame", "json": "json"}


@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_artifact_families_align_with_record_contracts(sport):
    """Both directions of the artifact-family <-> record-contract mapping are
    pinned: every family resolves or is an explicit reviewed exception, and
    every family that DOES resolve agrees on type (``.csv`` frame vs
    ``.json`` record)."""
    contract = default_artifact_contract(sport)
    by_family = {af.family: af for af in contract.families}

    unmapped = set()
    for af in contract.families:
        record_family = ARTIFACT_TO_RECORD_FAMILY.get(af.family, af.family)
        spec = REGISTRY.get((sport, record_family))
        if spec is None:
            unmapped.add(af.family)
            continue
        assert spec.record_kind == _EXT_TO_RECORD_KIND[af.ext], (
            f"{sport}/{af.family}: artifact ext {af.ext!r} vs record kind "
            f"{spec.record_kind!r}")
    assert unmapped == _ARTIFACT_FAMILY_WITHOUT_RECORD_CONTRACT[sport]

    covered = set(ARTIFACT_TO_RECORD_FAMILY.values()) | set(by_family)
    registry_families = {f for (sp, f) in REGISTRY if sp == sport}
    assert registry_families - covered \
        == _RECORD_FAMILY_WITHOUT_ARTIFACT_FAMILY[sport]
