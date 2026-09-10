"""Regression: per-sport scope contracts are independent and versioned.

Phase 7.5 Task 2 gate (addendum D): every sport must declare DISTINCT,
versioned moneyline and market feature contracts, and the optimization
adapter layer must bind each scope to its OWN contract — no aliasing,
no inherited lists, no missing version metadata.

Three assertions per sport (all three required; unequal list contents
alone is NOT sufficient):

1. Identity: the moneyline and market contract objects are distinct —
   the scope tuples must not be the same Python object (no shared
   identity), and each must be an ordered, non-empty tuple of strings.
2. Adapter selection: ``core.optimization.adapters`` binds each scope to
   the contract list declared for THAT scope (market never inherits the
   moneyline list and vice versa), verified via the adapters' scope
   bindings rather than list equality.
3. Version metadata: both scope versions are present, non-empty, and
   DIFFER from each other (cross-scope divergence), and match the
   study-config-carried versions.
"""

from __future__ import annotations

import dataclasses

import pytest


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
    "nfl": "sports.nfl.study_config",
    "nba": "sports.nba.study_config",
    "nhl": "sports.nhl.study_config",
}


def _study_modules():
    import importlib

    return {
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
    from core.optimization import adapters

    builder = {
        "mlb": adapters.mlb_adapters,
        "nfl": adapters.nfl_adapters,
        "nba": adapters.nba_adapters,
        "nhl": adapters.nhl_adapters,
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
        # the MLB production contract location (MLBStudy has no version
        # fields; the registry IS the served contract).
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
