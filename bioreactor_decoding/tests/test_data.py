import numpy as np
import pandas as pd
import pytest

from bioreactor_decoding.dataset import RAW_PATH, TARGETS, load_tables, load_template
from bioreactor_decoding.features import build_features, feature_columns
from bioreactor_decoding.modeling.models import FoldChangeRegressor, MedianFoldChange, mape
from bioreactor_decoding.modeling.predict import fill_template
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


@pytest.mark.parametrize("model_name", ["null_constant", "null_per_day", "xgb_raw", "xgb_log_fold_change"])
def test_holdout_never_used_for_fitting(model_name):
    """The model behind the holdout figures is fitted without any holdout data."""
    X, Y, split = labelled_data()
    model = MODELS[model_name]()
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
