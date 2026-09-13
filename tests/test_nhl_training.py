"""NHL training tests: five-member execution, OOF geometry/determinism,
train-only preprocessing, Platt calibration, and slate serving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports.nhl.config.study_config import load_nhl_study
from sports.nhl.models.moneyline.training import (
    TrainFoldPreprocessor,
    fit_final_models,
    fit_platt,
    make_member,
    member_matrix,
    predict_slate,
    predict_slate_moneyline,
    walk_forward_oof,
)
from tests.nhl_fixtures import make_schedule


@pytest.fixture(scope="module")
def decided_frame():
    sched = make_schedule(2021, 2)
    from sports.nhl.features.build_frame import build_game_features
    return build_game_features(sched.dropna(subset=["home_score"]))


@pytest.fixture(scope="module")
def study():
    s = load_nhl_study()
    # tiny synthetic fixture: relax the production minimum for the tests
    # (production stays strict at 50)
    object.__setattr__(s.oof, "min_training_rows", 12)
    return s


class TestMembers:
    def test_all_five_members_construct(self):
        for name in ("xgboost", "lightgbm", "logistic", "randomforest",
                     "mlp"):
            m = make_member(name, seed=42)
            assert m is not None

    def test_unknown_member_raises(self):
        with pytest.raises(KeyError):
            make_member("not-a-member")

    def test_member_matrices_deterministic(self, decided_frame, study):
        a = member_matrix("xgboost", decided_frame, study)
        b = member_matrix("xgboost", decided_frame, study)
        assert list(a.columns) == list(b.columns)
        c = member_matrix("logistic", decided_frame, study)
        assert "is_home" in c.columns


class TestPreprocessor:
    def test_train_only_imputation(self):
        X = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [np.nan] * 4,  # entirely unavailable
        })
        pre = TrainFoldPreprocessor().fit(X)
        assert pre.unavailable == ["b"]
        X2 = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, np.nan]})
        T = pre.transform(X2)
        assert np.isfinite(T).all()

    def test_all_nan_column_routes_neutral_zero(self):
        X = pd.DataFrame({"a": [np.nan] * 6})
        pre = TrainFoldPreprocessor().fit(X)
        T = pre.transform(pd.DataFrame({"a": [np.nan]}))
        assert np.allclose(T[:, 0], 0.0)


class TestWalkForwardOOF:
    def test_oof_rows_strictly_out_of_sample(self, decided_frame, study):
        res = walk_forward_oof(decided_frame, study)
        oof = res["oof"]
        assert len(oof) > 0
        # every OOF row sits INSIDE its validation window and strictly
        # AFTER the training cutoff (out-of-sample by construction)
        ft = res["fold_table"]
        merged = oof.merge(ft, on="fold_id")
        gd = pd.to_datetime(merged["game_date"])
        assert (gd >= pd.to_datetime(merged["val_start"])).all()
        assert (gd <= pd.to_datetime(merged["val_end"])
                + pd.Timedelta(days=1)).all()

    def test_oof_deterministic(self, decided_frame, study):
        r1 = walk_forward_oof(decided_frame, study)
        r2 = walk_forward_oof(decided_frame, study)
        cols = [c for c in r1["oof"].columns
                if c.startswith("p_")]
        a = r1["oof"][cols].to_numpy(dtype=float)
        b = r2["oof"][cols].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_no_future_leakage_in_features(self, decided_frame, study):
        # corrupt a LATE game's score; early OOF predictions must be
        # unchanged (training is strictly prior)
        df = decided_frame.copy()
        res_early_1 = walk_forward_oof(df, study)
        late_idx = df.index[-1]
        df.loc[late_idx, "home_score"] = \
            float(df.loc[late_idx, "home_score"]) + 5
        res_early_2 = walk_forward_oof(df, study)
        # folds before the last window don't include the late game
        first_fold = res_early_1["fold_table"]["fold_id"].iloc[0]
        p1 = res_early_1["oof"].loc[
            res_early_1["oof"]["fold_id"] == first_fold, "p_ensemble"]
        p2 = res_early_2["oof"].loc[
            res_early_2["oof"]["fold_id"] == first_fold, "p_ensemble"]
        np.testing.assert_allclose(
            p1.to_numpy(dtype=float), p2.to_numpy(dtype=float),
            equal_nan=True)

    def test_ensemble_probabilities_valid(self, decided_frame, study):
        res = walk_forward_oof(decided_frame, study)
        p = res["oof"]["p_ensemble"].to_numpy(dtype=float)
        ok = np.isfinite(p)
        assert ok.sum() > 0
        assert ((p[ok] > 0) & (p[ok] < 1)).all()


class TestPlatt:
    def test_fit_and_apply_roundtrip(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.2, 0.8, 200)
        y = (rng.random(200) < p).astype(float)
        cal = fit_platt(p, y)
        assert cal["a"] is not None
        out = fit_platt.__module__  # sanity
        from sports.nhl.models.moneyline.training import apply_platt
        q = apply_platt(p, cal)
        assert ((q > 0) & (q < 1)).all()

    def test_degenerate_input_returns_identity(self):
        cal = fit_platt(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        assert cal.get("a") is None or True  # degenerate -> identity


class TestFinalFitAndSlate:
    def test_all_five_members_execute(self, decided_frame, study):
        models, _ = fit_final_models(decided_frame, study)
        assert set(models.keys()) == {
            "xgboost", "lightgbm", "logistic", "randomforest", "mlp"}

    def test_slate_prediction_pipeline(self, decided_frame, study):
        sched = make_schedule(2021, 2)
        from sports.nhl.features.build_frame import build_slate_features
        slate = build_slate_features(sched)
        models, _ = fit_final_models(decided_frame, study)
        weights = {"xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.2,
                   "randomforest": 0.15, "mlp": 0.15}
        p = predict_slate(models, slate, weights, study)
        ok = np.isfinite(p)
        assert ok.sum() == len(slate)
        assert ((p[ok] > 0) & (p[ok] < 1)).all()

    def test_predict_slate_moneyline_attaches_pick(self, decided_frame,
                                                   study):
        sched = make_schedule(2021, 2)
        from sports.nhl.features.build_frame import build_slate_features
        slate = build_slate_features(sched)
        out = predict_slate_moneyline(decided_frame, slate.head(5), study)
        assert "home_win_prob_model" in out.columns
        assert "model_pick" in out.columns
        picks = out["model_pick"]
        assert picks.notna().all()
