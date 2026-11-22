"""The ported team work reproduces what is on its branches.

anomaly.py must give the numbers printed in notebooks/anomaly_v7.ipynb and
anomaly_option_a.ipynb on branch anomalies_detection, and ported Option A the CV and submission committed on branch
hybrid_v3_tr. Skipped without the raw data or the imported inputs (data/external, DVC).
"""

import numpy as np
import pandas as pd
import pytest

from bioreactor_decoding import anomaly
from bioreactor_decoding.config import PROJ_ROOT
from bioreactor_decoding.dataset import RAW_PATH, TARGETS
from bioreactor_decoding.modeling import option_a
from bioreactor_decoding.modeling.train import labelled_data

pytestmark = [
    pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not available"),
    pytest.mark.skipif(not option_a.INPUTS.exists(), reason="data/external not pulled"),
]


@pytest.fixture(scope="module")
def result():
    return anomaly.run("hybrid_v7")


@pytest.mark.skipif(not anomaly.OPTION_A_OOF.exists(), reason="run dvc repro option_a first")
def test_anomaly_on_option_a_matches_its_notebook():
    """Values as printed in notebooks/anomaly_option_a.ipynb (same analysis, Option A's
    residuals); the out-of-fold file comes from our option_a stage, not the branch."""
    r = anomaly.run("option_a")
    assert {v: round(x, 2) for v, x in r["mape"].items()} == {
        "vcd": 6.9, "tcd": 6.82, "substrate": 7.24, "byproduct": 8.25,
    }  # fmt: skip
    assert f"{r['R'].std(0).mean():.3f}" == "0.097"
    thr = r["THR"]
    assert (f"{thr[0]:.2f}", f"{thr[1]:.2f}", f"{thr[2]:.1f}") == ("1.75", "5.87", "124.0")
    assert int(r["batch"]["flagged"].sum()) == 6
    dev = r["dev"]
    assert list(zip(dev["batch"], dev["type"], dev["variable"], dev["days"])) == [
        ("Batch46", "persistent", "substrate", "5-7"),
        ("Batch42", "persistent", "substrate", "6-10"),
        ("Batch50", "persistent", "substrate", "4-6"),
        ("Batch34", "spike", "byproduct", "13"),
        ("Batch29", "spike", "byproduct", "13"),
        ("Batch3", "spike", "vcd", "8"),
    ]
    assert dev["strict level"].tolist() == ["yes", "no", "no", "yes", "yes", "no"]
    assert r["tab"].loc[("no fault", "")].round(2).tolist() == [0.12, 0.12, 0.06]


def test_anomaly_matches_the_original_notebook(result):
    """Values as printed in the original notebook's outputs."""
    mape = result["mape"]
    assert {v: round(x, 2) for v, x in mape.items()} == {
        "vcd": 6.72, "tcd": 6.68, "substrate": 7.19, "byproduct": 8.04,
    }  # fmt: skip
    assert round(np.mean(list(mape.values())), 2) == 7.16
    assert f"{result['R'].std(0).mean():.3f}" == "0.096"
    thr = result["THR"]
    assert (f"{thr[0]:.2f}", f"{thr[1]:.2f}", f"{thr[2]:.1f}") == ("1.79", "6.69", "128.3")
    assert int(result["batch"]["flagged"].sum()) == 6

    dev = result["dev"]
    assert list(zip(dev["batch"], dev["type"], dev["variable"], dev["days"])) == [
        ("Batch46", "persistent", "substrate", "4-7"),
        ("Batch4", "persistent", "byproduct", "2-6"),
        ("Batch50", "persistent", "substrate", "4-6"),
        ("Batch34", "spike", "byproduct", "13"),
        ("Batch29", "spike", "byproduct", "13"),
    ]
    assert dev["strict level"].tolist() == ["yes", "no", "no", "yes", "yes"]

    no_fault = result["tab"].loc[("no fault", "")].round(2).tolist()
    assert no_fault == [0.12, 0.10, 0.06]
    assert result["inputs"].loc[result["inputs"]["flag"] == "yes", "batch"].tolist() == [
        "Batch54", "Batch5",
    ]  # fmt: skip
    assert result["top3"] == {"Batch23": 3.78, "Batch44": 3.36, "Batch49": 2.61}


def test_ported_option_a_reproduces_its_branch(tmp_path, monkeypatch):
    """CV as committed in outputs/option_a/cv_score.csv, and the submitted file."""
    monkeypatch.setattr(option_a, "WORK", tmp_path)
    option_a.run_option_a()
    cv = pd.read_csv(tmp_path / "cv_score.csv").iloc[0].to_dict()
    assert cv == {"vcd": 6.902, "tcd": 6.816, "substrate": 7.235, "byproduct": 8.249,
                  "leaderboard_score": 7.3}  # fmt: skip
    read = {"dtype": {"batch_id": str}}
    run = pd.read_csv(tmp_path / "holdout_predictions.csv", **read)
    submitted = pd.read_csv(option_a.SUBMITTED, **read)
    assert run["batch_id"].equals(submitted["batch_id"])
    np.testing.assert_allclose(run[TARGETS], submitted[TARGETS], rtol=1e-12)


def test_option_a_holdout_rows_match_the_holdout_split():
    _, Y, split = labelled_data(use_future_feeds=False)
    hold = split["role"] == "holdout"
    perfect = split.loc[hold, ["batch_id"]].assign(
        t_key=split.loc[hold, "elapsed_time"].to_numpy(),
        **{f"f|{v}": Y.loc[hold, v].to_numpy() for v in TARGETS},
    )
    out, scores = option_a.holdout_from_oof(perfect)
    assert len(out) == hold.sum() and scores["score"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="no out-of-fold prediction"):
        option_a.holdout_from_oof(perfect.iloc[1:])


def test_option_a_holdout_predictions_have_no_measured_values():
    out = pd.read_csv(PROJ_ROOT / "reports" / "holdout" / "option_a.csv")
    assert list(out.columns) == ["batch_id", "elapsed_time", *TARGETS]
