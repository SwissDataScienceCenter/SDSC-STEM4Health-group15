import numpy as np
import pandas as pd
import pytest

from bioreactor_decoding.dataset import RAW_PATH, TARGETS, load_tables, load_template
from bioreactor_decoding.features import build_features, feature_columns
from bioreactor_decoding.modeling.models import FoldChangeRegressor, MedianFoldChange, mape
from bioreactor_decoding.modeling.predict import fill_template
from bioreactor_decoding.modeling import hybrid, mechanistic
from bioreactor_decoding.modeling.hybrid_v3 import HybridV3, input_tables
from bioreactor_decoding.modeling.train import MODELS, holdout_check, labelled_data

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not available")


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_test_batches_have_only_t0_measurements(tables):
    samples, _ = tables
    test = samples[samples["is_test"]]
    assert test["batch_id"].nunique() == 12
    measured = test[TARGETS].notna().all(axis=1)
    assert np.allclose(test.loc[measured, "elapsed_time"], 0.0)
    assert (test.loc[~measured, "to_predict"]).all()


def test_frozen_split_roles(tables):
    samples, _ = tables
    roles = samples.groupby("batch_id")["role"].agg(["nunique", "first"])
    assert (roles["nunique"] == 1).all()
    assert roles["first"].value_counts().to_dict() == {"dev": 39, "test": 12, "holdout": 10}
    dev_folds = samples.loc[samples["role"] == "dev"].groupby("batch_id")["fold"].first()
    assert set(dev_folds) == {0, 1, 2, 3, 4}


def test_template_matches_to_predict_rows(tables):
    samples, _ = tables
    tp = samples.loc[samples["to_predict"], ["batch_id", "elapsed_time"]].copy()
    tp[TARGETS] = 1.0
    filled = fill_template(load_template(), tp)
    assert len(filled) == 156 and (filled[TARGETS] == 1.0).all().all()


@pytest.mark.parametrize("use_future_feeds", [True, False])
def test_features_ignore_measurements_after_t0(tables, use_future_feeds):
    """Scrambling every measurement after t = 0 must not change any feature."""
    samples, feeds = tables
    scrambled = samples.copy()
    later = scrambled["elapsed_time"] > 0
    scrambled.loc[later, TARGETS] = np.random.default_rng(0).random((later.sum(), len(TARGETS)))
    a = build_features(samples, feeds, use_future_feeds)
    b = build_features(scrambled, feeds, use_future_feeds)
    pd.testing.assert_frame_equal(a, b)


def test_features_of_a_batch_ignore_other_batches(tables):
    samples, feeds = tables
    one = samples["batch_id"] == "Batch1"
    full = build_features(samples, feeds)
    alone = build_features(samples[one], feeds[feeds["batch_id"] == "Batch1"])
    pd.testing.assert_frame_equal(full.loc[alone.index], alone)


def test_estimators_predict_positive_and_recover_constant_fold_change():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"k": np.tile(np.arange(1, 4), 5)})
    for c in TARGETS:
        X[f"log0_{c}"] = rng.normal(size=len(X))
    Y = pd.DataFrame({c: np.exp(X[f"log0_{c}"] + 0.5 * X["k"]) for c in TARGETS})
    pred = MedianFoldChange().fit(X, Y).predict(X)
    assert mape(Y["vcd"], pred["vcd"]) < 1e-8

    from sklearn.linear_model import LinearRegression

    pred = FoldChangeRegressor(LinearRegression()).fit(X, Y).predict(X)
    assert (pred > 0).all().all() and mape(Y["tcd"], pred["tcd"]) < 1e-6


def test_feature_columns_exclude_meta(tables):
    samples, feeds = tables
    cols = feature_columns(build_features(samples, feeds))
    assert "batch_id" not in cols and "elapsed_time" not in cols


CHEAP_STAGE1 = hybrid.HybridODEConfig(mech_starts=1, seeds=1, mech_max_nfev=3, net_max_nfev=3)


def _cheap_hybrid_v3():
    """Same code path as hybrid_v3, with a stage 1 that is fitted only a few iterations."""
    return HybridV3(inner_folds=2, n_bags=2, stage1=CHEAP_STAGE1)


@pytest.mark.parametrize("model_name", ["null_constant", "null_per_day", "xgb_raw", "xgb_log_fold_change", "null_xgb_ensemble", "hybrid_v3"])
def test_holdout_never_used_for_fitting(model_name):
    """The model behind the holdout figures is fitted without any holdout data."""
    X, Y, split = labelled_data()
    model = _cheap_hybrid_v3() if model_name == "hybrid_v3" else MODELS[model_name]()
    holdout = split["role"] == "holdout"
    n = int(holdout.sum())
    rng = np.random.default_rng(0)
    pred, _ = holdout_check(model, X, Y, split)

    # 1. Holdout labels never reach the model: replacing them changes no prediction.
    Ys = Y.copy()
    Ys.loc[holdout] = rng.random((n, Y.shape[1])) * 100
    pd.testing.assert_frame_equal(holdout_check(model, X, Ys, split)[0], pred)

    # 2. Holdout features are only used to predict their own row, never to fit:
    #    shuffling them across holdout rows just shuffles the predictions.
    perm = rng.permutation(n)
    Xs = X.copy()
    Xs.loc[holdout] = X.loc[holdout].to_numpy()[perm]
    np.testing.assert_allclose(holdout_check(model, Xs, Y, split)[0].to_numpy(), pred.to_numpy()[perm])


def test_v3_inputs_ignore_raw_measurements_after_t0(tables):
    """hybrid_v3 reads only inputs from the raw file: the ODE inputs carry no later measurement."""
    samples, feeds = input_tables()
    assert samples.loc[samples["elapsed_time"] > 0, TARGETS].isna().all().all()
    raw, _ = tables
    ids = sorted(raw.loc[raw["role"] == "dev", "batch_id"].unique())[:5]
    scrambled = raw.copy()
    later = scrambled["elapsed_time"] > 0
    scrambled.loc[later, TARGETS] = np.random.default_rng(0).random((later.sum(), len(TARGETS)))
    a = mechanistic.build_batches(samples, feeds, ids)
    b = mechanistic.build_batches(scrambled, feeds, ids)
    for name in ["y0", "v0", "feeds", "vsnap"]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_hybrid_ode_with_zero_network_is_the_mechanistic_model(tables):
    """With the output layer at zero the hybrid ODE reduces to the mechanistic one (up to the
    fade-out of lactate consumption at low lactate, which only the hybrid has)."""
    samples, feeds = input_tables()
    ids = sorted(samples.loc[samples["role"] == "dev", "batch_id"].unique())[:5]
    bs = mechanistic.build_batches(samples, feeds, ids)
    spec = hybrid.HybridODEConfig().spec()
    theta = spec.init(0)  # random input layer, zero output layer
    a = hybrid.simulate(bs, mechanistic.P0, spec, theta)
    b = mechanistic.simulate(bs, mechanistic.P0)
    np.testing.assert_allclose(a[..., [0, 1, 4]], b[..., [0, 1, 4]], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a[..., 5], b[..., 5], atol=0.01)


def test_v3_features_of_a_batch_ignore_other_batches(tables):
    """Once fitted, hybrid_v3's stage-2 features of a batch do not depend on other batches."""
    X, _, split = labelled_data()
    est = HybridV3()
    est.schedule_ = {c: X.groupby(np.rint(X["t"]).astype(int))["t"].mean() for c in
                     ["fb_next", "fb_prev", "fc_next", "fc_prev", "V"]}  # any fitted values
    est.sample_volume_ = 0.05
    s1 = np.ones((len(X), len(TARGETS)))
    full = est._rows(X, s1)
    one = X["batch_id"] == X["batch_id"].iat[0]
    alone = est._rows(X[one], s1[one.to_numpy()])
    pd.testing.assert_frame_equal(full.loc[alone.index], alone)

