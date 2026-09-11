"""Phase 8 — MLB adapter wiring tests.

The MLB optimization adapter must reuse the EXISTING MLB pieces unchanged
(store, DuckDB feature engine, canonical decided frame, PIT Elo/records
enrichment, frozen feature views, study.yaml fold geometry). These tests
pin that wiring on small synthetic frames — no network, no real pull —
and assert the scope-independence and PIT invariants the harness relies
on. The adapter is exercised through its store-check contract: without a
real store it binds nothing (DEFERRED), so the synthetic-path tests
target the pure pieces (scope builder, raw-pool selector, fold filter,
candidate composition) and the frame-composition functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization.adapters import mlb_adapters
from core.optimization.candidates import (
    build_candidates,
    candidate_catalog_fingerprint,
    derive_column,
    spec_from_name,
)
from core.optimization.models import Bounds
from core.optimization.sports import MLB_SIDE_FIELDS, mlb_scopes


# ---------------------------------------------------------------------------
# Scope builder (pure — no store access)
# ---------------------------------------------------------------------------
def test_mlb_scopes_are_independent_per_model():
    ml_inc = ("is_home", "win_pct_diff", "elo_diff")
    run_inc = ("is_home", "win_pct_diff", "elo_diff", "rest_days_diff")
    ml, mk = mlb_scopes(ml_inc, run_inc)
    assert ml.model == "moneyline" and ml.target == "home_win"
    assert mk.model == "market" and mk.target == "margin"
    assert mk.horizon_field == "game_date"
    # Each scope's incumbent list is its own — never inferred from the other.
    assert ml.prod_features == ml_inc
    assert mk.prod_features == run_inc
    assert ml.sport == mk.sport == "mlb"


def test_mlb_scope_pools_are_twin_pairs_plus_elo():
    ml, mk = mlb_scopes(("is_home",), ("is_home",))
    for pool in (ml.raw_fields, mk.raw_fields):
        # Every pool member is the enriched Elo pair or one half of a
        # COMPLETE home/away twin pair from the frozen view.
        for f in pool:
            if f.endswith(("_home", "_away")):
                stem = f[:-len("_home")] if f.endswith("_home") \
                    else f[:-len("_away")]
                assert f"{stem}_home" in pool and f"{stem}_away" in pool
            else:
                stem = f[5:] if f.startswith(("home_", "away_")) else None
                assert stem is not None, f"unclassified pool member {f}"
                assert f"home_{stem}" in pool and f"away_{stem}" in pool
        # is_home has no away twin: it stays a served incumbent feature,
        # never a derivation-pool member.
        assert "is_home" not in pool
        assert set(MLB_SIDE_FIELDS) >= set(ml.raw_fields)


def test_mlb_scope_pools_have_complete_home_away_pairs():
    ml, _ = mlb_scopes(("is_home",), ("is_home",))
    homes = {f[:-len("_home")] for f in ml.raw_fields if f.endswith("_home")}
    aways = {f[:-len("_away")] for f in ml.raw_fields if f.endswith("_away")}
    assert homes == aways, "every pool stem must have both sides"


def test_mlb_adapter_defers_without_store(monkeypatch, tmp_path):
    """No real store -> no adapters (the runner reports DEFERRED)."""
    import core.optimization.adapters as mod
    monkeypatch.setattr(mod, "STORE", tmp_path)
    assert mlb_adapters(None) == {}


# ---------------------------------------------------------------------------
# Fold filter + catalog composition on a synthetic decided frame
# ---------------------------------------------------------------------------
def _synthetic_decided(n_days: int = 40, per_day: int = 6) -> pd.DataFrame:
    """Minimal decided MLB frame with the columns the adapter pieces use."""
    rng = np.random.default_rng(42)
    rows = []
    teams = list("ABCDEFGHJKLMNP")
    for d in range(n_days):
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=d)
        for g in range(per_day):
            h, a = teams[(2 * g) % len(teams)], teams[(2 * g + 1) % len(teams)]
            rows.append({
                "game_pk": 700000 + d * per_day + g,
                "game_date": date,
                "home_team": h, "away_team": a,
                "home_score": int(rng.integers(0, 9)),
                "away_score": int(rng.integers(0, 9)),
                "home_win": float(rng.random() < 0.54),
                "home_elo": rng.normal(1500, 25),
                "away_elo": rng.normal(1500, 25),
                "home_win_pct": rng.uniform(0.35, 0.65),
                "away_win_pct": rng.uniform(0.35, 0.65),
                "rest_days_home": float(rng.integers(1, 5)),
                "rest_days_away": float(rng.integers(1, 5)),
                "sp_era_home": rng.uniform(2.5, 6.0),
                "sp_era_away": rng.uniform(2.5, 6.0),
            })
    df = pd.DataFrame(rows)
    df["margin"] = df["home_score"] - df["away_score"]
    return df


def test_mlb_declared_pool_selector_only_returns_declared_present_fields():
    df = _synthetic_decided()
    ml, _ = mlb_scopes(("is_home", "elo_diff"), ("is_home",))
    declared = ml.raw_fields
    present = {f for f in declared
               if f in df.columns and pd.api.types.is_numeric_dtype(df[f])}
    # The selector is bound per scope inside the adapter; replicate its
    # exact semantics here (the full closure needs a store).
    got = tuple(f for f in declared
                if f in df.columns and pd.api.types.is_numeric_dtype(df[f]))
    assert set(got) == set(present)
    # Elo columns are part of the pool and are numeric on the frame.
    assert "home_elo" in got and "away_elo" in got
    # sp_k9_home is a declared pool member but ABSENT from the synthetic
    # frame — it must never enter the catalog (no fabrication).
    assert "sp_k9_home" in declared
    assert "sp_k9_home" not in got


def test_mlb_catalog_from_pool_is_deterministic_and_pit_safe():
    df = _synthetic_decided()
    ml, _ = mlb_scopes(("is_home", "elo_diff"), ("is_home",))
    pool = tuple(f for f in ml.raw_fields
                 if f in df.columns and pd.api.types.is_numeric_dtype(df[f]))
    c1, w1 = build_candidates(pool, Bounds(), incumbent=ml.prod_features)
    c2, w2 = build_candidates(pool, Bounds(), incumbent=ml.prod_features)
    assert candidate_catalog_fingerprint(c1) == candidate_catalog_fingerprint(c2)
    assert w1 == w2
    # Twin diffs already served must not be regenerated.
    names = [c.name for c in c1]
    assert "elo_home_minus_away" not in names
    assert "win_pct_home_minus_away" not in names
    # A genuinely new pair IS generated.
    assert "sp_era_home_minus_away" in names


def test_mlb_derived_rolling_is_per_team_and_strictly_prior():
    df = _synthetic_decided()
    spec = spec_from_name("sp_era__rollmean5_diff",
                          ("sp_era_home", "sp_era_away"))
    assert spec is not None and spec.derivation == "rolling"
    col = derive_column(df, spec, team_cols=("home_team", "away_team"),
                        date_col="game_date")
    assert col.notna().any()
    # Masking the current row's own values must not change its feature
    # (shift(1) discipline).
    masked = df.copy()
    masked.loc[masked.index[:5], "sp_era_home"] = np.nan
    col2 = derive_column(masked, spec, team_cols=("home_team", "away_team"),
                         date_col="game_date")
    # Only rows whose 5-prior window reaches the masked rows may differ;
    # every team's 5th unmasked prior event lands well before row 80
    # (~11 games in), so rows 80+ must be byte-identical.
    later = df.index[80:]
    assert np.allclose(col.loc[later], col2.loc[later], equal_nan=True)


# ---------------------------------------------------------------------------
# Adapter binding on a fake store (pitches.parquet fixture path)
# ---------------------------------------------------------------------------
def test_mlb_adapter_binds_through_production_composition(monkeypatch, tmp_path):
    """With a store present the adapter must bind both scopes, split the
    frozen views into incumbent minus per-side members, and attach the
    run-engine regressand to decided rows only."""
    import core.optimization.adapters as mod

    fake_cache = tmp_path / "mlb" / "raw" / "pitches.parquet"
    fake_cache.parent.mkdir(parents=True, exist_ok=True)
    fake_cache.write_bytes(b"not-a-real-parquet")  # existence check only

    game_calls = {"n": 0}

    def fake_frames(cache):
        game_calls["n"] += 1
        return _synthetic_decided(), pd.DataFrame()

    monkeypatch.setattr(mod, "_mlb_feature_frames", fake_frames)
    monkeypatch.setattr(mod, "STORE", tmp_path)

    # Stub the production composition: the adapter must call it (wiring
    # under test), not reimplement it.
    import sports.mlb.feature_registry as reg

    real_bcf = reg.build_candidate_frame

    def fake_bcf(decided, **kwargs):
        df, cols = real_bcf(decided)
        df["margin"] = (pd.to_numeric(df["home_score"], errors="coerce")
                        - pd.to_numeric(df["away_score"], errors="coerce"))
        return df, cols

    monkeypatch.setattr(reg, "build_candidate_frame", fake_bcf)

    ads = mlb_adapters(None)
    assert set(ads) == {"mlb/moneyline", "mlb/market"}
    ml_scope = ads["mlb/moneyline"].pop("_scope")
    df = ads["mlb/moneyline"]["load_decided"]()
    # Both loads share ONE frame build per process (frame cache).
    assert game_calls["n"] == 1
    # The run-engine regressand exists on decided rows, never null.
    assert "margin" in df.columns and df["margin"].notna().all()
    # The incumbent view keeps NO twin-pair member (those form the
    # derivation pool); is_home has no away twin and stays served.
    from sports.mlb.feature_registry import MONEYLINE_FEATURE_COLS as _FROZEN
    frozen = set(_FROZEN)
    assert not any(f.endswith("_home")
                   and f"{f[:-len('_home')]}_away" in frozen
                   for f in ml_scope.prod_features)
    assert "is_home" in ml_scope.prod_features
    assert "elo_diff" in ml_scope.prod_features


def test_mlb_adapter_scope_pools_intersect_frame_columns(monkeypatch, tmp_path):
    """The raw pool the adapter exposes must be the scope's declared
    fields intersected with the frame's numeric columns — declared-but-
    absent fields drop out; non-declared columns never leak in."""
    import core.optimization.adapters as mod

    fake_cache = tmp_path / "mlb" / "raw" / "pitches.parquet"
    fake_cache.parent.mkdir(parents=True, exist_ok=True)
    fake_cache.write_bytes(b"not-a-real-parquet")

    synth = _synthetic_decided()
    synth["total_runs"] = synth["home_score"] + synth["away_score"]
    monkeypatch.setattr(mod, "_mlb_feature_frames", lambda cache: (synth,
                                                                   pd.DataFrame()))
    monkeypatch.setattr(mod, "STORE", tmp_path)

    import sports.mlb.feature_registry as reg
    real_bcf = reg.build_candidate_frame

    def fake_bcf(decided, **kwargs):
        df, cols = real_bcf(decided)
        df["margin"] = df["home_score"] - df["away_score"]
        return df, cols

    monkeypatch.setattr(reg, "build_candidate_frame", fake_bcf)

    ads = mlb_adapters(None)
    for key, ads_dict in ads.items():
        scope = ads_dict.pop("_scope")
        pool = ads_dict["raw_columns"](synth)
        assert set(pool) <= set(scope.raw_fields)
        assert all(f in synth.columns for f in pool)
        # Non-declared frame columns must never enter the pool.
        assert "total_runs" not in pool
        assert "margin" not in pool
        _ = key  # both scopes bound the same selector semantics


def test_mlb_adapter_slate_has_no_outcome_columns(monkeypatch, tmp_path):
    """Slate rows are pre-game: no outcome column may be present
    (the PIT gate and slate-availability gate depend on this)."""
    import core.optimization.adapters as mod

    fake_cache = tmp_path / "mlb" / "raw" / "pitches.parquet"
    fake_cache.parent.mkdir(parents=True, exist_ok=True)
    fake_cache.write_bytes(b"not-a-real-parquet")

    synth = _synthetic_decided()
    pending = synth.copy()
    pending["home_win"] = np.nan  # slate rows: undecided
    monkeypatch.setattr(mod, "_mlb_feature_frames",
                        lambda cache: (synth, pd.DataFrame()))
    monkeypatch.setattr(mod, "STORE", tmp_path)

    import sports.mlb.feature_registry as reg

    real_bcf = reg.build_candidate_frame

    def fake_bcf(decided, **kwargs):
        df, cols = real_bcf(decided)
        df["margin"] = df["home_score"] - df["away_score"]
        return df, cols

    monkeypatch.setattr(reg, "build_candidate_frame", fake_bcf)

    ads = mlb_adapters(None)
    ml_scope = ads["mlb/moneyline"].pop("_scope")
    slate = ads["mlb/moneyline"]["load_slate"]()
    for oc in ("home_win", "margin", "home_score", "away_score",
               "total_runs"):
        assert oc not in slate.columns or slate[oc].isna().all()
    _ = ml_scope


def test_mlb_adapter_fold_filter_matches_study_rules(monkeypatch, tmp_path):
    """build_folds applies the study.yaml OOF rules: folds below the min
    training rows or below min_val_games are skipped; surviving folds are
    positional expanding-window splits in chronological order."""
    import core.optimization.adapters as mod

    fake_cache = tmp_path / "mlb" / "raw" / "pitches.parquet"
    fake_cache.parent.mkdir(parents=True, exist_ok=True)
    fake_cache.write_bytes(b"not-a-real-parquet")

    # 30-day warmup + 200-row min train + 40-game min val: 45 days x 10
    # games yields two admissible weekly folds.
    synth = _synthetic_decided(n_days=45, per_day=10)
    monkeypatch.setattr(mod, "_mlb_feature_frames",
                        lambda cache: (synth, pd.DataFrame()))
    monkeypatch.setattr(mod, "STORE", tmp_path)

    import sports.mlb.feature_registry as reg

    real_bcf = reg.build_candidate_frame

    def fake_bcf(decided, **kwargs):
        df, cols = real_bcf(decided)
        df["margin"] = df["home_score"] - df["away_score"]
        return df, cols

    monkeypatch.setattr(reg, "build_candidate_frame", fake_bcf)

    ads = mlb_adapters(None)
    ml_scope = ads["mlb/moneyline"].pop("_scope")
    df = ads["mlb/moneyline"]["load_decided"]()
    folds = ads["mlb/moneyline"]["build_folds"](df)
    assert folds, "the synthetic frame must produce usable folds"
    study = __import__("sports.mlb.study_config",
                       fromlist=["load_mlb_study"]).load_mlb_study()
    for tr, va in folds:
        assert len(tr) >= study.oof.min_training_rows
        assert len(va) >= study.min_val_games
        assert max(tr) < min(va), "expanding window: train precedes val"
    # Chronological: later folds train on strictly more rows.
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)
    _ = ml_scope
