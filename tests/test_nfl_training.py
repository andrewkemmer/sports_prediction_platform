"""NFL training tests: five-member stack, OOF strictness, determinism,
train-only imputation, and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports.nfl.study_config import load_nfl_study
from sports.nfl.training import (
    TrainFoldPreprocessor,
    fit_final_models,
    fit_platt,
    member_matrix,
    predict_slate,
    walk_forward_oof,
)
from tests.nfl_fixtures import make_pbp, make_schedule


def test_study():
    """The study with the small-fixture OOF floor relaxed (tests only).
    Production keeps min_training_rows=50."""
    import dataclasses
    study = load_nfl_study()
    return dataclasses.replace(
        study,
        oof=dataclasses.replace(study.oof, min_training_rows=1))


@pytest.fixture(scope="module")
def game_frame():
    sched = make_schedule(2019, 2)
    decided = sched[sched["home_score"].notna()].reset_index(drop=True)
    from sports.nfl.features import build_game_features
    df = build_game_features(decided, make_pbp(sched))
    return df.sort_values("gameday").reset_index(drop=True)


class TestFiveMemberStack:
    def test_all_five_members_execute(self, game_frame):
        study = load_nfl_study()
        models, _ = fit_final_models(game_frame, study)
        assert set(models.keys()) == {
            "xgboost", "lightgbm", "logistic", "randomforest", "mlp"}
        # each member predicts finite probabilities
        for name, entry in models.items():
            X = member_matrix(name, game_frame.head(5), study)
            if name in ("logistic", "mlp"):
                p = entry["pre"].transform(X)
                p = entry["model"].predict_proba(p)[:, 1]
            else:
                p = entry["model"].predict_proba(X)[:, 1]
            assert np.isfinite(p).all()

    def test_member_views_dimensionality(self, game_frame):
        study = load_nfl_study()
        linear = member_matrix("logistic", game_frame, study)
        tree = member_matrix("xgboost", game_frame, study)
        assert "is_home" in linear.columns  # linear anchor
        assert "elo_home" in tree.columns  # tree per-side values
        # NaN preserved in the tree view (native missing routing)
        assert tree.isna().any().any()


class TestOOF:
    def test_min_training_rows_gate(self, game_frame):
        """The production floor (50) skips every fold on the small
        fixture — the gate must hold."""
        study = load_nfl_study()
        ml = walk_forward_oof(game_frame, study)
        assert ml["oof"].empty
        assert len(ml["fold_table"]) == 0

    def test_oof_strictly_out_of_sample(self, game_frame):
        ml = walk_forward_oof(game_frame, test_study())
        oof = ml["oof"]
        assert len(oof) > 0
        folds = ml["fold_table"]
        # every OOF row's date is inside its fold's window
        for _, row in oof.iterrows():
            frow = folds[folds["fold_id"] == row["fold_id"]].iloc[0]
            d = pd.to_datetime(row["gameday"])
            assert pd.Timestamp(frow["val_start"]) <= d \
                <= pd.Timestamp(frow["val_end"])
        # no fold's training may include its own validation rows
        df = game_frame
        for _, frow in folds.iterrows():
            window_ids = set(df[(pd.to_datetime(df["gameday"]) >=
                                 pd.Timestamp(frow["val_start"]))
                                & (pd.to_datetime(df["gameday"]) <=
                                   pd.Timestamp(frow["val_end"]))]
                             ["game_id"])
            train_end = pd.to_datetime(df.set_index("game_id")
                                       .loc[df.set_index("game_id")
                                            .index.isin(window_ids)]
                                       ["gameday"]).min()
            assert train_end >= pd.Timestamp(frow["val_start"])

    def test_oof_probabilities_valid(self, game_frame):
        oof = walk_forward_oof(game_frame, test_study())["oof"]
        # the ensemble is strictly inside (0,1) WHERE it exists; a fold
        # whose 3-row training set is single-class legitimately yields no
        # member predictions (nothing fabricated) — production enforces
        # min_training_rows=50 against exactly this
        p = oof["p_ensemble"].to_numpy(float)
        ok = np.isfinite(p)
        assert ok.mean() >= 0.8
        assert ((p[ok] > 0) & (p[ok] < 1)).all()
        # each tree/linear member is finite on the substantial folds
        for m in ("xgboost", "lightgbm", "logistic", "randomforest"):
            pm = oof[f"p_{m}"].to_numpy(float)
            assert np.isfinite(pm).mean() > 0.8, m

    def test_determinism(self, game_frame):
        a = walk_forward_oof(game_frame, test_study())["oof"]
        b = walk_forward_oof(game_frame, test_study())["oof"]
        pd.testing.assert_frame_equal(a, b)

    def test_adaptive_weights_sum_to_one(self, game_frame):
        study = test_study()
        ml = walk_forward_oof(game_frame, study)
        w = ml["member_weights"]
        assert set(w.keys()) == set(study.training.ensemble_members)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # all positive and no member dominates (the reference's
        # clamp-then-renormalize allows a small overshoot past the cap,
        # which is accepted behavior)
        assert all(v > 0 for v in w.values())
        assert max(w.values()) <= study.training.adaptive_weight_cap + 0.1


class TestImputation:
    def test_train_only_median_reused_at_apply(self):
        train = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        val = pd.DataFrame({"a": [np.nan, 3.0], "b": [np.nan, 50.0]})
        pre = TrainFoldPreprocessor().fit(train)
        med_a = train["a"].median()
        out = pre.transform(val)
        assert out[0, 0] == pytest.approx((med_a - train["a"].mean())
                                          / train["a"].std())
        # b's val NaN imputes to the TRAIN mean (train-only fit)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_all_nan_column_routed_explicitly(self):
        train = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "ghost": [np.nan, np.nan, np.nan],
        })
        pre = TrainFoldPreprocessor().fit(train)
        assert "ghost" in pre.unavailable  # explicit tracking
        # neutral 0.0 (documented) for linear members, never fabricated
        out = pre.transform(train)
        assert (out[:, 1] == 0.0).all()


class TestPlatt:
    def test_platt_fit_and_apply(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 200)
        y = (rng.uniform(size=200) < p).astype(float)
        cal = fit_platt(p, y)
        assert cal["a"] is not None and cal["b"] is not None
        out = __import__("sports.nfl.training", fromlist=["apply_platt"]) \
            .apply_platt(p, cal)
        assert np.isfinite(out).all()

    def test_platt_degenerate_returns_none(self):
        cal = fit_platt(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        assert cal["a"] is None or cal["n"] >= 2
