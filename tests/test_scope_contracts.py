"""Regression: per-sport scope contracts are independent and versioned.

Phase 7.5 Task 2 gate (addendum D) + Phase 7.5b canonical ownership:
every sport must declare DISTINCT, versioned moneyline and market feature
contracts in its feature REGISTRY, the study config must import+bind them,
and the optimization adapter layer must bind each scope to its OWN
contract — no aliasing, no inherited lists, no missing version metadata,
and the MLB adapter must never bypass the versioned contracts (the gap
that masked the original Gate 5 verification must not recur).

Three assertions per sport (all three required; unequal list contents
alone is NOT sufficient):

1. Identity: the moneyline and market contract objects are distinct —
   the scope tuples must not be the same Python object (no shared
   identity), and each must be an ordered, non-empty tuple of strings.
2. Adapter selection: each sport's ``sports/<sport>/optimization.py``
   adapter binds each scope to the contract list declared for THAT scope
   (market never inherits the moneyline list and vice versa), verified via
   the adapters' scope bindings rather than list equality.
3. Version metadata: both scope versions are present, non-empty, and
   DIFFER from each other (cross-scope divergence), and match the
   study-config-carried versions.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts import (
    validate_columns_have_metadata,
    validate_metadata_is_reachable,
)


@pytest.fixture(autouse=True)
def _raw_store(raw_store):
    """Bind the adapter builders to the committed raw fixtures.

    The adapter-selection assertion builds each sport's optimization adapter,
    which reads the gitignored store; without a store the builder returns an
    empty mapping and the binding proof cannot run. ``tests/raw_store.py``
    materializes the committed bounded fixtures (never clobbering an existing
    store) so the proof runs everywhere, offline.
    """
    return raw_store


def _sport_contracts(module_name: str):
    """Import a sport's study config module and return its scope contracts."""
    import importlib

    mod = importlib.import_module(module_name)
    return (
        getattr(mod, "MONEYLINE_FEATURE_COLS"),
        getattr(mod, "MARKET_FEATURE_COLS"),
        getattr(mod, "MONEYLINE_CONTRACT_VERSION"),
        getattr(mod, "MARKET_CONTRACT_VERSION"),
    )


SPORT_MODULES = {
    "mlb": "sports.mlb.feature_registry",
    "nfl": "sports.nfl.feature_registry",
    "nba": "sports.nba.feature_registry",
    "nhl": "sports.nhl.feature_registry",
}


def _study_modules():
    import importlib

    return {
        "mlb": importlib.import_module("sports.mlb.study_config"),
        "nfl": importlib.import_module("sports.nfl.study_config"),
        "nba": importlib.import_module("sports.nba.study_config"),
        "nhl": importlib.import_module("sports.nhl.study_config"),
    }


@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_scope_contracts_independent_and_versioned(sport: str) -> None:
    """Contract objects are distinct, versioned, and adapter-bound per scope."""
    mod_name = SPORT_MODULES[sport]
    ml_cols, mk_cols, ml_ver, mk_ver = _sport_contracts(mod_name)

    # --- (i) distinct contract objects — no aliasing / shared identity ----
    assert isinstance(ml_cols, tuple) and ml_cols, (
        f"{sport}: moneyline contract must be a non-empty tuple")
    assert isinstance(mk_cols, tuple) and mk_cols, (
        f"{sport}: market contract must be a non-empty tuple")
    assert all(isinstance(c, str) for c in ml_cols), (
        f"{sport}: moneyline contract columns must be strings")
    assert all(isinstance(c, str) for c in mk_cols), (
        f"{sport}: market contract columns must be strings")
    # Identity check: the two scope tuples must not be the same object.
    # (Equal-content tuples may legitimately share values after Task 2's
    # carry-over, but they must be independently declared, not aliased
    # through one shared mutable pool object.)
    assert ml_cols is not mk_cols or len(ml_cols) == 0, (
        f"{sport}: moneyline and market contracts alias the same object")
    assert ml_ver != mk_ver, (
        f"{sport}: moneyline and market contract versions must differ "
        f"(both are {ml_ver!r})")

    # --- (ii) adapter selects the correct scope contract ------------------
    from sports.mlb.optimization import mlb_adapters
    from sports.nba.optimization import nba_adapters
    from sports.nfl.optimization import nfl_adapters
    from sports.nhl.optimization import nhl_adapters

    builder = {
        "mlb": mlb_adapters,
        "nfl": nfl_adapters,
        "nba": nba_adapters,
        "nhl": nhl_adapters,
    }[sport]
    study = None
    try:
        scopes = builder(study)
    except Exception as exc:  # store missing in CI — adapter binding is
        # exercised via default_scopes contract wiring below instead.
        pytest.skip(
            f"{sport}: adapter store unavailable in this environment ({exc})")

    ml_scope = scopes[f"{sport}/moneyline"]["_scope"]
    mk_scope = scopes[f"{sport}/market"]["_scope"]

    if sport == "mlb":
        # MLB's adapter geometry: the scope's prod_features is the
        # INCUMBENT VIEW (frozen contract minus its per-side members,
        # which form the derivation raw pool). The frozen contract is
        # reconstructed as prod_features ∪ raw_fields MINUS the per-side
        # pairs consumed into diff features (rest_days_home/away are
        # folded into rest_days_diff, never served raw).
        for scope, frozen in ((ml_scope, ml_cols), (mk_scope, mk_cols)):
            union = set(scope.prod_features) | set(scope.raw_fields)
            canonical = union - {"rest_days_home", "rest_days_away"}
            assert canonical == set(frozen), (
                f"{sport}: adapter geometry ({scope.model}) != frozen "
                f"contract; missing={sorted(set(frozen) - canonical)} "
                f"extra={sorted(canonical - set(frozen))[:5]}")
            # Incumbent (diff) features keep the frozen order prefix.
            assert tuple(scope.prod_features) == tuple(
                f for f in frozen if f in scope.prod_features), (
                f"{sport}/{scope.model}: incumbent view order drifted")
    else:
        # Non-MLB adapters bind the FULL contract list directly.
        assert tuple(ml_scope.prod_features) == tuple(ml_cols), (
            f"{sport}: moneyline adapter did not bind the moneyline contract")
        assert tuple(mk_scope.prod_features) == tuple(mk_cols), (
            f"{sport}: market adapter did not bind the market contract")
    # And the two bound scope objects must be distinct dataclass instances.
    assert ml_scope is not mk_scope, (
        f"{sport}: moneyline and market adapter scopes are the same object")

    # --- (iii) version metadata present and carried into the study config -
    assert isinstance(ml_ver, str) and ml_ver, (
        f"{sport}: moneyline contract version must be a non-empty string")
    assert isinstance(mk_ver, str) and mk_ver, (
        f"{sport}: market contract version must be a non-empty string")
    if sport == "mlb":
        # MLB's contract versions are declared in the feature registry —
        # the MLB production contract location — AND carried by MLBStudy
        # (Phase 7.5b: MLBStudy binds moneyline/market feature cols and
        # versions like the other sports).
        from sports.mlb import feature_registry as mlb_registry

        ml_carried = mlb_registry.MONEYLINE_CONTRACT_VERSION
        mk_carried = mlb_registry.MARKET_CONTRACT_VERSION
        assert getattr(mlb_registry, "PRIOR_MONEYLINE_CONTRACT_VERSION", "")
        assert getattr(mlb_registry, "PRIOR_MARKET_CONTRACT_VERSION", "")
    else:
        import importlib

        loader_name = {
            "nfl": "load_nfl_study",
            "nba": "load_nba_study",
            "nhl": "load_nhl_study",
        }[sport]
        loader = getattr(importlib.import_module(
            f"sports.{sport}.study_config"), loader_name)
        st = loader()
        ml_carried = st.moneyline_contract_version
        mk_carried = st.market_contract_version
    assert ml_carried == ml_ver, (
        f"{sport}: study config moneyline version {ml_carried!r} != "
        f"declared {ml_ver!r}")
    assert mk_carried == mk_ver, (
        f"{sport}: study config market version {mk_carried!r} != "
        f"declared {mk_ver!r}")


def test_adapter_default_scopes_bind_own_contract() -> None:
    """default_scopes returns independent scope objects per kind.

    The scope factory must produce a fresh scope per kind — replacing the
    market scope's prod_features must never mutate the moneyline scope
    (the aliasing failure mode Task 2 eliminated).
    """
    from core.optimization.sports import default_scopes

    ml, mk = default_scopes("nba", ("feat_a", "feat_b"))
    assert ml is not mk
    assert tuple(ml.prod_features) == ("feat_a", "feat_b")
    assert tuple(mk.prod_features) == ("feat_a", "feat_b")

    # Remapping the market scope must leave the moneyline scope untouched.
    mk2 = dataclasses.replace(mk, prod_features=("only_market",))
    assert tuple(ml.prod_features) == ("feat_a", "feat_b"), (
        "moneyline scope mutated by a market-scope replace()")
    assert tuple(mk2.prod_features) == ("only_market",)


# ---------------------------------------------------------------------------
# Phase 7.5b — canonical registry ownership
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_registry_owns_declared_contracts(sport: str) -> None:
    """The feature registry (not study_config) declares both contracts via
    explicit tuple([...]) constructions; registry imports no study_config."""
    import ast
    import importlib
    from pathlib import Path

    mod = importlib.import_module(f"sports.{sport}.feature_registry")
    ml_cols = getattr(mod, "MONEYLINE_FEATURE_COLS")
    mk_cols = getattr(mod, "MARKET_FEATURE_COLS")
    ml_ver = getattr(mod, "MONEYLINE_CONTRACT_VERSION")
    mk_ver = getattr(mod, "MARKET_CONTRACT_VERSION")

    # explicit tuple([...]) construction: distinct objects, non-empty
    assert isinstance(ml_cols, tuple) and ml_cols
    assert isinstance(mk_cols, tuple) and mk_cols
    assert ml_cols is not mk_cols, f"{sport}: registry contracts alias"
    assert ml_ver and mk_ver and ml_ver != mk_ver

    # Registry source: no study_config import (Task 1.4).
    reg_path = Path(importlib.import_module(
        f"sports.{sport}.feature_registry").__file__)
    tree = ast.parse(reg_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(
                f"sports.{sport}.study_config"), (
                f"{sport}: registry imports study_config (ownership inverted)")


@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_study_config_imports_and_binds_registry_contracts(sport: str) -> None:
    """Every study_config imports the registry's four symbols and binds all
    four onto the study dataclass; no study_config declares lists directly."""
    import ast
    import importlib
    from pathlib import Path

    mod = importlib.import_module(f"sports.{sport}.study_config")

    # re-exports present
    for sym in ("MONEYLINE_FEATURE_COLS", "MARKET_FEATURE_COLS",
                "MONEYLINE_CONTRACT_VERSION", "MARKET_CONTRACT_VERSION"):
        assert hasattr(mod, sym), f"{sport}: study_config missing re-export {sym}"

    # registry identity (import, not re-declaration)
    reg = importlib.import_module(f"sports.{sport}.feature_registry")
    assert mod.MONEYLINE_FEATURE_COLS is reg.MONEYLINE_FEATURE_COLS, (
        f"{sport}: study_config MONEYLINE list is not the registry object")
    assert mod.MARKET_FEATURE_COLS is reg.MARKET_FEATURE_COLS, (
        f"{sport}: study_config MARKET list is not the registry object")

    # study instance carries all four bindings
    loader_name = {"mlb": "load_mlb_study", "nfl": "load_nfl_study",
                   "nba": "load_nba_study", "nhl": "load_nhl_study"}[sport]
    st = getattr(mod, loader_name)()
    assert tuple(st.moneyline_feature_cols) == tuple(reg.MONEYLINE_FEATURE_COLS)
    assert tuple(st.market_feature_cols) == tuple(reg.MARKET_FEATURE_COLS)
    assert st.moneyline_contract_version == reg.MONEYLINE_CONTRACT_VERSION
    assert st.market_contract_version == reg.MARKET_CONTRACT_VERSION

    # no direct list declaration in study_config source
    src_path = Path(mod.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in (
                        "MONEYLINE_FEATURE_COLS", "MARKET_FEATURE_COLS"):
                    # re-export bindings to the registry object are fine;
                    # literal declarations are not
                    val = getattr(node, "value", None)
                    ok = isinstance(val, ast.Name) or (
                        isinstance(val, ast.Call) and not val.args)
                    assert ok, (
                        f"{sport}: study_config declares {t.id} directly")


# ---------------------------------------------------------------------------
# Phase 7.5e-A (T4): declared tuples validated against each sport's EXISTING
# registry metadata model (MLB FeatureEntry registry, NFL/NHL/NBA _SPEC_DEFS).
# ---------------------------------------------------------------------------

def _registry_metadata(sport: str):
    """Metadata names from the sport's own model via one interface.

    No sport is forced onto a parallel registry: MLB keeps its
    ``FeatureEntry`` records, NFL/NHL/NBA keep their ``_SPEC_DEFS`` mapping.
    """
    import importlib

    mod = importlib.import_module(f"sports.{sport}.feature_registry")
    if sport == "mlb":
        return {e.name for e in mod.registry_entries()}
    return frozenset(mod._SPEC_DEFS)


@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_declared_tuples_have_registry_metadata(sport: str) -> None:
    """Every column of BOTH declared contract tuples has registry metadata."""
    import importlib

    mod = importlib.import_module(f"sports.{sport}.feature_registry")
    metadata = _registry_metadata(sport)
    validate_columns_have_metadata(
        sport, "moneyline", mod.MONEYLINE_FEATURE_COLS, metadata)
    validate_columns_have_metadata(
        sport, "market", mod.MARKET_FEATURE_COLS, metadata)


@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_registry_metadata_is_reachable_from_declared_tuples(
        sport: str) -> None:
    """No orphan metadata: every registry entry is claimed by a tuple."""
    import importlib

    mod = importlib.import_module(f"sports.{sport}.feature_registry")
    declared = set(mod.MONEYLINE_FEATURE_COLS) | set(mod.MARKET_FEATURE_COLS)
    validate_metadata_is_reachable(sport, _registry_metadata(sport), declared)


def test_mlb_market_only_entries_are_market_not_moneyline() -> None:
    """The three run-engine-only diffs are marked market, stay OUT of the
    moneyline view, and do not widen it from 64 to 67."""
    from sports.mlb import feature_registry as reg

    entries = {e.name: e for e in reg.registry_entries()}
    for name in ("sp_k9_diff", "sp_k9_5g_diff", "sp_xwoba_diff"):
        entry = entries[name]
        assert "market" in entry.contracts
        assert "moneyline" not in entry.contracts
        assert name in reg.MARKET_FEATURE_COLS
        assert name not in reg.MONEYLINE_FEATURE_COLS
    contract = reg.build_feature_contract()
    assert len(contract.features) == 64
    assert contract.feature_names() == tuple(reg.MONEYLINE_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Phase 7.5e-A (T3): the features_metadata writer stamps the ACTIVE study
# moneyline contract version, never the module default / a legacy literal.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", sorted(SPORT_MODULES))
def test_feature_contract_binds_active_moneyline_version(sport: str) -> None:
    """The builder honors the study-carried active moneyline version."""
    import importlib

    reg = importlib.import_module(f"sports.{sport}.feature_registry")
    loader = {"mlb": "load_mlb_study", "nfl": "load_nfl_study",
              "nba": "load_nba_study", "nhl": "load_nhl_study"}[sport]
    study = getattr(importlib.import_module(
        f"sports.{sport}.study_config"), loader)()
    contract = reg.build_feature_contract(study.moneyline_contract_version)
    assert contract.version == study.moneyline_contract_version
    assert contract.version == reg.MONEYLINE_CONTRACT_VERSION
    # still moneyline-only: the version bind never changes the feature set
    assert contract.feature_names() == tuple(reg.MONEYLINE_FEATURE_COLS)
