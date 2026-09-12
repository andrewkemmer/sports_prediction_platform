"""Phase 7 optimizer harness tests.

Covers: candidate-catalog determinism, pairwise-only interaction
restriction, PIT safety (no current/future feature on slate rows),
metric/gate logic against a synthetic incumbent, and the rejection path
with the no-auto-promotion invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization import (
    Bounds,
    OptimizationConfig,
    build_candidates,
    candidate_catalog_fingerprint,
    default_scopes,
    evaluate_gates,
    leakage_audit,
    model_scope,
    pit_safe_matrix,
    pooled_metrics,
    period_stability,
    recommend,
    regression_metrics,
    regression_period_stability,
    run_optimization,
    slate_available_columns,
)
from core.optimization.candidates import (
    diff_candidates,
    interaction_candidates,
    rolling_candidates,
    derive_column,
    spec_from_name,
)
from core.optimization.models import ModelScope
from core.optimization.runner import ScopeRunner

RAW = ("elo_home", "elo_away", "rest_days_home", "rest_days_away",
       "win_pct_home", "win_pct_away")
PROD = ("elo_diff", "win_pct_diff", "rest_days_diff")


# ---------------------------------------------------------------------------
# Candidate catalog determinism
# ---------------------------------------------------------------------------
def test_catalog_determinism_same_inputs_same_fingerprint():
    c1, w1 = build_candidates(RAW, Bounds(), incumbent=PROD)
    c2, w2 = build_candidates(RAW, Bounds(), incumbent=PROD)
    assert candidate_catalog_fingerprint(c1) == candidate_catalog_fingerprint(c2)
    assert [c.name for c in c1] == [c.name for c in c2]
    assert w1 == w2


def test_catalog_determinism_independent_of_input_order():
    shuffled = tuple(reversed(RAW))
    c1, _ = build_candidates(RAW, Bounds(), incumbent=PROD)
    c2, _ = build_candidates(shuffled, Bounds(), incumbent=PROD)
    assert candidate_catalog_fingerprint(c1) == candidate_catalog_fingerprint(c2)


def test_catalog_changes_when_catalog_changes():
    c1, _ = build_candidates(RAW, Bounds(), incumbent=PROD)
    c2, _ = build_candidates(RAW + ("sp_era_home",), Bounds(), incumbent=PROD)
    assert candidate_catalog_fingerprint(c1) != candidate_catalog_fingerprint(c2)


def test_horizon_and_model_candidate_bounds():
    b = Bounds()
    assert 5 <= len(b.horizon_choices) <= 10
    assert 5 <= len(b.model_names) <= 8


def test_bounds_reject_out_of_policy_values():
    with pytest.raises(ValueError):
        Bounds(max_candidate_features=2_000)      # > 1,000 frozen cap
    with pytest.raises(ValueError):
        Bounds(horizon_choices=(3,))              # < 5 horizons
    with pytest.raises(ValueError):
        Bounds(model_names=("xgboost",))          # < 5 models
    with pytest.raises(ValueError):
        Bounds(interaction_order=3)               # pairwise only, frozen


# ---------------------------------------------------------------------------
# Pairwise-only interaction restriction
# ---------------------------------------------------------------------------
def test_interactions_are_pairwise_only():
    inters = interaction_candidates(("a", "b", "c", "d"))
    for c in inters:
        parts = c.name.split("__x__")
        assert len(parts) == 2          # exactly two factors
        assert len(c.params) == 1
        # partner must be a raw field, never another interaction
        assert "__x__" not in c.params[0]


def test_diff_candidates_twin_checked_against_incumbent():
    """A diff whose twin production diff is already served must NOT be
    regenerated as a candidate; a genuinely new pair must be."""
    cands = diff_candidates(RAW, incumbent=PROD)
    names = [c.name for c in cands]
    assert not any(n.startswith("elo_") for n in names)
    assert not any(n.startswith("win_pct_") for n in names)
    assert not any(n.startswith("rest_days_") for n in names)
    # A new pair (not served in production) IS generated.
    new_cands = diff_candidates(RAW + ("pace_home", "pace_away"),
                                incumbent=PROD)
    assert "pace_home_minus_away" in [c.name for c in new_cands]


def test_diff_candidates_skip_incomplete_pairs():
    cands = diff_candidates(("elo_home",), incumbent=())
    assert cands == []  # no fabricated diff without its away twin


def test_rolling_candidates_are_per_team_side_columns():
    rolls = rolling_candidates(RAW, (3, 5))
    # Candidate names are per side-pair stem with the window.
    assert "elo__rollmean3_diff" in [c.name for c in rolls]
    assert all(c.base_field.endswith("_home") for c in rolls)
    assert "elo_diff__rollmean3_diff" not in [c.name for c in rolls]


def test_spec_from_name_roundtrip():
    cands, _ = build_candidates(RAW, Bounds(), incumbent=PROD)
    for c in cands:
        if c.derivation == "incumbent":
            assert spec_from_name(c.name, RAW) is None
            continue
        back = spec_from_name(c.name, RAW)
        assert back is not None
        assert back.name == c.name and back.derivation == c.derivation
        assert back.base_field == c.base_field
        assert back.params == c.params


def test_derive_column_diff_matches_production_values():
    df = pd.DataFrame({
        "game_date": pd.date_range("2025-01-01", periods=4),
        "home_team": ["A", "B", "A", "B"],
        "away_team": ["B", "A", "B", "A"],
        "pace_home": [10.0, 12.0, 14.0, 16.0],
        "pace_away": [8.0, 9.0, 10.0, 11.0],
    })
    from core.optimization.candidates import CandidateSpec

    spec = CandidateSpec("pace_home_minus_away", "diff", "pace_home", ())
    out = derive_column(df, spec)
    assert out.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_derive_column_team_rolling_is_strictly_prior():
    """Per-team rolling must use each team's OWN strictly-prior events —
    the current row's value never enters its own feature."""
    df = pd.DataFrame({
        "game_date": pd.date_range("2025-01-01", periods=4),
        "home_team": ["A", "B", "A", "B"],
        "away_team": ["B", "A", "B", "A"],
        "elo_home": [100.0, 200.0, 110.0, 210.0],
        "elo_away": [200.0, 100.0, 210.0, 110.0],
    })
    from core.optimization.candidates import CandidateSpec

    spec = CandidateSpec("elo__rollmean3_diff", "rolling", "elo_home", (3,))
    out = derive_column(df, spec)
    # Each team's event stream unpivots BOTH twin columns (the home
    # slot's value belongs to the home team, the away slot's to the
    # away team). Team A's stream: (row0,100),(row1,100),(row2,110),
    # (row3,110); team B's: (row0,200),(row1,200),(row2,210),(row3,210).
    # shift(1) then window-3 mean per team, then HOME team's rolled mean
    # − AWAY team's rolled mean:
    # row1: home=B prior 200, away=A prior 100 → +100;
    # row2: home=A prior mean(100,100)=100, away=B prior mean(200,200)=200
    #       → −100;
    # row3: home=B prior mean(200,200,210)=203.33, away=A prior
    #       mean(100,100,110)=103.33 → +100.
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 100.0
    assert out.iloc[2] == -100.0
    assert abs(out.iloc[3] - 100.0) < 0.01


def test_no_higher_order_products_in_full_catalog():
    cands, _ = build_candidates(RAW, Bounds(max_candidate_features=1_000),
                                incumbent=PROD)
    for c in cands:
        if c.derivation == "interaction":
            assert c.name.count("__x__") == 1
            assert "__x__" not in c.base_field
            assert all("__x__" not in str(p) for p in c.params)


def test_interaction_budget_truncates_deterministically():
    small = Bounds(max_candidate_features=25)
    c_small, w_small = build_candidates(RAW, small, incumbent=PROD)
    assert len(c_small) <= 25
    c_again, w_again = build_candidates(RAW, small, incumbent=PROD)
    assert w_small == w_again  # withheld list is also deterministic
    assert len(w_small) > 0    # the bound actually bound something


def test_leakage_audit_flags_outcome_column_as_base():
    cands, _ = build_candidates(("elo_home", "elo_away"), Bounds(),
                                incumbent=PROD)
    from core.optimization.candidates import CandidateSpec

    bad = list(cands) + [CandidateSpec("cheat", "rolling", "home_win", (3,))]
    report = leakage_audit(bad, ("elo_home", "elo_away"))
    assert not report.ok
    assert any("home_win" in v for v in report.violations)


def test_leakage_audit_flags_unknown_base_field():
    from core.optimization.candidates import CandidateSpec

    bad = [CandidateSpec("ghost", "rolling", "not_a_real_field", (5,))]
    report = leakage_audit(bad, ("elo_home",))
    assert not report.ok
    assert any("not_a_real_field" in v for v in report.violations)


def test_leakage_audit_passes_clean_catalog():
    cands, _ = build_candidates(RAW, Bounds(), incumbent=PROD)
    report = leakage_audit(cands, RAW)
    assert report.ok, report.violations


# ---------------------------------------------------------------------------
# PIT safety on concrete frames
# ---------------------------------------------------------------------------
def _feature_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame({
        "game_id": [f"G{i}" for i in range(n)],
        "elo_diff": rng.normal(0, 20, n),
        "win_pct_diff": rng.normal(0, .15, n),
        "rest_days_diff": rng.integers(-3, 4, n).astype(float),
        "home_win": (rng.random(n) < .54).astype(float),
        "margin": rng.normal(2.5, 12, n),
        "total": rng.normal(225, 18, n),
    })


def test_pit_flags_identity_copy_of_outcome():
    df = _feature_frame()
    df["cheat_total"] = df["total"]  # leaked copy
    ok, violations = pit_safe_matrix(df, pd.Series(False, index=df.index))
    assert not ok
    assert any("cheat_total" in v for v in violations)


def test_pit_flags_outcome_on_slate_rows():
    df = _feature_frame()
    slate = pd.Series([False] * 100 + [True] * 20, index=df.index)
    df.loc[df.index[100:], "home_win"] = 1.0  # undecided rows with outcomes
    ok, violations = pit_safe_matrix(df, slate)
    assert not ok
    assert any("slate" in v for v in violations)


def test_pit_passes_honest_features():
    df = _feature_frame()
    slate = pd.Series([False] * 100 + [True] * 20, index=df.index)
    df.loc[df.index[100:], ("home_win", "margin", "total")] = np.nan
    ok, violations = pit_safe_matrix(df, slate)
    assert ok, violations


def test_slate_availability_split():
    slate = pd.DataFrame({"elo_home": [1.0], "elo_away": [1.0],
                          "rest_days_home": [2.0],
                          "rest_days_away": [1.0], "win_pct_away": [0.5]})
    cands, _ = build_candidates(RAW, Bounds(), incumbent=PROD)
    available, unavailable = slate_available_columns(cands, slate)
    # rolling rest_days pair (both bases on slate) → available; rolls
    # referencing absent bases → unavailable. All three diff twins are
    # already served (PROD), so no diff candidates exist at all.
    assert "rest_days__rollmean3_diff" in available
    assert not any("_home_minus_away" in a for a in available)
    assert any("win_pct__rollmean3_diff" in u for u in unavailable)


def test_regression_metrics_values():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["n"] == 3
    assert m["rmse"] > m["mae"] >= 0.0
    assert abs(m["bias"] - (-2.0 / 3.0)) < 1e-9


def test_regression_period_stability_flags_late_degradation():
    st = regression_period_stability([1.0] * 6 + [2.0] * 6)
    assert st["delta"] > 0.5


# ---------------------------------------------------------------------------
# Metrics + gate logic (synthetic incumbent)
# ---------------------------------------------------------------------------
def _metrics_for(probs, outcomes) -> dict:
    m = pooled_metrics(probs, outcomes)
    folds = [float(f) for f in np.array_split(
        np.asarray(probs), 6) for f in [m["logloss"]]]  # flat proxy
    return m


def test_pooled_metrics_values():
    p = [0.9] * 50 + [0.2] * 50
    y = [1.0] * 50 + [0.0] * 50
    m = pooled_metrics(p, y)
    assert m["logloss"] < 0.25
    assert m["auc"] == 1.0
    assert m["brier"] < 0.05
    assert 0.0 <= m["ece"] <= 1.0


def test_pooled_metrics_ece_zero_when_perfectly_calibrated():
    # 100 rows at p=0.7 with exactly 70% wins → ECE ~0 for that single bin.
    rng = np.random.default_rng(3)
    p = np.full(100, 0.7)
    y = (rng.random(100) < 0.7).astype(float)
    # force the empirical rate to exactly 0.7
    y[:70] = 1.0
    y[70:] = 0.0
    m = pooled_metrics(p, y)
    assert m["ece"] < 0.02


def test_pooled_metrics_handles_nan_rows():
    p = [0.8, float("nan"), 0.3]
    y = [1.0, 1.0, 0.0]
    m = pooled_metrics(p, y)
    assert m["n"] == 2


def test_period_stability_flags_late_degradation():
    early = [0.60] * 6
    late = [0.75] * 6
    st = period_stability(early + late)
    assert st["delta"] > 0.10
    # Against the default 0.020 tolerance this must fail the gate.
    g = evaluate_gates(
        "unstable", _cand(stability=st),
        _incumbent(), stability_max_delta=0.020)
    assert not g.gates["stability"]


def _cand(**over) -> dict:
    base = {
        "pooled": {"logloss": 0.6200, "auc": 0.7000, "brier": 0.2150,
                   "ece": 0.0120, "n": 1000},
        "sealed": {"logloss": 0.6250, "auc": 0.6950, "brier": 0.2200,
                   "ece": 0.0130, "n": 200},
        "fold_logloss": [0.62] * 9,
        "stability": {"delta": 0.001},
        "coverage": 0.98,
        "leakage_ok": True,
        "deterministic": True,
        "slate_available": True,
    }
    base.update(over)
    return base


def _incumbent() -> dict:
    return _cand()


def test_gate_recommends_better_candidate():
    cand = _cand(pooled={"logloss": 0.6100, "auc": 0.7100, "brier": 0.2100,
                         "ece": 0.0110, "n": 1000})
    g = evaluate_gates("better", cand, _incumbent())
    assert g.recommended
    assert all(g.gates.values())
    assert g.deltas["pooled_logloss"] < 0


def test_gate_rejects_logloss_regression():
    cand = _cand(pooled={"logloss": 0.6400, "auc": 0.7000, "brier": 0.2150,
                         "ece": 0.0120, "n": 1000})
    g = evaluate_gates("worse", cand, _incumbent())
    assert not g.recommended
    assert not g.gates["pooled_logloss"]


def test_gate_rejects_coverage_and_leakage():
    g = evaluate_gates("lowcov", _cand(coverage=0.80), _incumbent())
    assert not g.recommended and not g.gates["coverage"]

    g = evaluate_gates("leaker", _cand(leakage_ok=False), _incumbent())
    assert not g.recommended and not g.gates["leakage"]


def test_gate_rejects_calibration_regression():
    cand = _cand(pooled={"logloss": 0.6150, "auc": 0.7050, "brier": 0.2120,
                         "ece": 0.0500, "n": 1000})
    g = evaluate_gates("miscal", cand, _incumbent())
    assert not g.recommended and not g.gates["ece"]


# ---------------------------------------------------------------------------
# Rejection path + no-auto-promotion invariant
# ---------------------------------------------------------------------------
def test_recommend_returns_none_when_all_rejected():
    results = [
        evaluate_gates("a", _cand(coverage=0.5), _incumbent()),
        evaluate_gates("b", _cand(leakage_ok=False), _incumbent()),
    ]
    assert all(not r.recommended for r in results)
    assert recommend(results) is None


def test_recommend_picks_lowest_logloss_among_passing():
    passing = evaluate_gates("good", _cand(pooled={
        "logloss": 0.6050, "auc": 0.7120, "brier": 0.2080, "ece": 0.0100,
        "n": 1000}), _incumbent())
    better = evaluate_gates("best", _cand(pooled={
        "logloss": 0.6000, "auc": 0.7150, "brier": 0.2050, "ece": 0.0090,
        "n": 1000}), _incumbent())
    assert recommend([passing, better]).candidate == "best"


def test_scope_runner_never_mutates_production_lists():
    """No-auto-promotion invariant: running a scope leaves the config's
    incumbent feature tuple and the study feature list untouched."""
    prod = ("elo_diff", "win_pct_diff")
    scope = ModelScope(sport="nba", model="moneyline",
                       raw_fields=RAW, prod_features=prod, target="home_win")
    cfg = OptimizationConfig(bounds=Bounds(max_seconds=30.0),
                             scopes=(scope,))

    df = _feature_frame()
    df["elo_home"] = df["elo_diff"]
    df["elo_away"] = -df["elo_diff"]
    df["rest_days_home"] = 2.0
    df["rest_days_away"] = 1.0
    df["win_pct_home"] = 0.55
    df["win_pct_away"] = 0.45

    def matrix(frame, feats):
        return frame.reindex(columns=list(feats)).astype(float)

    adapter = {
        "load_decided": lambda: df,
        "load_slate": lambda: df.head(0),
        "build_folds": lambda d: [
            (list(range(0, 60)), list(range(60, 80))),
            (list(range(0, 80)), list(range(80, 100))),
        ],
        "matrix": matrix,
        "member_factory": lambda seed: {
            "fit": lambda X, y: None,
            "predict": lambda X: np.full(len(X), 0.54),
        },
        "raw_columns": lambda d: RAW,
        "slate_check": lambda s, f: True,
        "leakage_extra": True,
    }
    before_prod = tuple(prod)
    before_study = tuple(cfg.scopes[0].prod_features)
    out = ScopeRunner(scope, cfg, adapter).run()
    assert tuple(prod) == before_prod
    assert tuple(cfg.scopes[0].prod_features) == before_study
    # The winner (if any) is only a recommendation object — nothing wrote.
    assert "winner" in out and ("results" in out)


def test_run_optimization_defers_scope_without_adapter():
    scope = model_scope("nba", "moneyline", PROD, "home_win")
    cfg = OptimizationConfig(bounds=Bounds(max_seconds=10.0),
                             scopes=(scope,))
    out = run_optimization(cfg, adapters={})
    assert out["nba/moneyline"]["deferred"] is True


def test_registry_adapters_run_end_to_end_and_write_nothing(monkeypatch,
                                                            tmp_path):
    """WS6/B-009: the composition root builds the REAL per-sport adapters
    and the shared runner executes them end-to-end.

    Constructing adapters through ``SPORT_ADAPTER_BUILDERS`` and running
    the real ``run_optimization(config, adapters)`` is required evidence —
    import-only coverage is insufficient. The synthetic store is created
    first; its contents are snapshotted, the optimization runs, and the
    store must be byte-identical afterwards (no artifact write, no store
    mutation, no new files)."""
    import sports.nba.optimization as nba_optimization
    from sports.optimization_adapters import SPORT_ADAPTER_BUILDERS
    from tests.nba_fixtures import make_schedule

    store = tmp_path / "store"
    raw = store / "nba" / "raw"
    raw.mkdir(parents=True)
    make_schedule(start_season=2014, n_seasons=4, weeks_per_season=10) \
        .to_parquet(raw / "schedule.parquet", index=False)
    monkeypatch.setattr(nba_optimization, "STORE", store)

    def snapshot() -> dict:
        return {p.relative_to(store).as_posix(): p.read_bytes()
                for p in sorted(store.rglob("*")) if p.is_file()}

    before = snapshot()
    assert before, "the synthetic store must exist before the run"

    adapters = SPORT_ADAPTER_BUILDERS["nba"](None)
    assert set(adapters) == {"nba/moneyline", "nba/market"}
    key = "nba/moneyline"
    scope = adapters[key].pop("_scope")
    cfg = OptimizationConfig(bounds=Bounds(max_seconds=60.0, sweep_cap=4),
                             scopes=(scope,))
    out = run_optimization(cfg, {key: adapters[key]})

    result = out[key]
    assert result.get("deferred") is not True, "registry adapter must bind"
    assert "winner" in result and "results" in result
    assert result["n_candidate_sets"] > 0
    # No artifact write / no store mutation after the run.
    assert snapshot() == before


def test_default_scopes_are_independent_per_model():
    ml, mk = default_scopes("nba", PROD)
    assert ml.model == "moneyline" and mk.model == "market"
    assert ml.target == "home_win" and mk.target == "margin"
    # Both scopes declare their own raw pool from the durable catalog
    # (run-engine distribution raws absent from the decided frame are
    # NOT fabricated; the mapping exists for future frame-present raws).
    assert set(ml.raw_fields) == set(mk.raw_fields)
    # independent optimization: separate models/targets, shared incumbent
    assert ml.model != mk.model and ml.target != mk.target


def test_sweep_cap_bounds_feature_sets_deterministically():
    """The sweep cap truncates the candidate additions to a stable prefix
    of the deterministic priority order — bounded AND reproducible."""
    scope = model_scope("nba", "moneyline", PROD, "home_win")
    cfg = OptimizationConfig(bounds=Bounds(max_seconds=60.0, sweep_cap=4),
                             scopes=(scope,))

    def _run() -> dict:
        df = _feature_frame()
        df["elo_home"] = df["elo_diff"]
        df["elo_away"] = -df["elo_diff"]
        df["rest_days_home"] = 2.0
        df["rest_days_away"] = 1.0
        df["win_pct_home"] = 0.55
        df["win_pct_away"] = 0.45
        adapter = {
            "load_decided": lambda: df,
            "load_slate": lambda: df.head(0),
            "build_folds": lambda d: [
                (list(range(0, 60)), list(range(60, 80))),
                (list(range(0, 80)), list(range(80, 100))),
            ],
            "matrix": lambda f, feats: f.reindex(
                columns=list(feats)).astype(float),
            "member_factory": lambda seed: {
                "fit": lambda X, y: None,
                "predict": lambda X: np.full(len(X), 0.54),
            },
            "raw_columns": lambda d: RAW,
            "slate_check": lambda s, f: True,
            "leakage_extra": True,
        }
        return ScopeRunner(scope, cfg, adapter).run()

    out1, out2 = _run(), _run()
    assert out1["n_candidate_sets"] == 4
    assert [r.candidate for r in out1["results"]] == \
        [r.candidate for r in out2["results"]]
    assert out1["catalog_fingerprint"] == out2["catalog_fingerprint"]
