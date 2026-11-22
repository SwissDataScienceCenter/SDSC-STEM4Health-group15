"""The ported team work reproduces what is on its branches.

anomaly.py must give the numbers printed in notebooks/anomaly_v7.ipynb on branch
anomalies_detection, and the Option A stage must read its results from commit option_a.ref.
Skipped without the raw data, or without the pinned commits (a clone without branch
hybrid_v3_tr).
"""

import subprocess

import numpy as np
import pandas as pd
import pytest

from bioreactor_decoding import anomaly
from bioreactor_decoding.config import PROJ_ROOT, load_params
from bioreactor_decoding.dataset import RAW_PATH, TARGETS
from bioreactor_decoding.modeling import external_option_a
from bioreactor_decoding.modeling.train import labelled_data


def _has_commit(ref: str) -> bool:
    return (
        subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=PROJ_ROOT).returncode
        == 0
    )


PARAMS = load_params()
pytestmark = [
    pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not available"),
    pytest.mark.skipif(
        not (_has_commit(PARAMS["hybrid_v7"]["ref"]) and _has_commit(PARAMS["option_a"]["ref"])),
        reason="pinned team commits not available",
    ),
]


@pytest.fixture(scope="module")
def result():
    return anomaly.run()


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


def test_option_a_leaderboard_is_read_from_its_commit():
    lb = external_option_a.leaderboard_result(PARAMS["option_a"]["ref"])
    assert lb == {"vcd": 6.568, "tcd": 6.41, "substrate": 6.546, "byproduct": 7.716, "score": 6.81}


def test_option_a_holdout_rows_match_the_holdout_split():
    _, Y, split = labelled_data(use_future_feeds=False)
    hold = split["role"] == "holdout"
    perfect = split.loc[hold, ["batch_id"]].assign(
        t_key=split.loc[hold, "elapsed_time"].to_numpy(),
        **{f"f|{v}": Y.loc[hold, v].to_numpy() for v in TARGETS},
    )
    out, scores = external_option_a.holdout_from_oof(perfect)
    assert len(out) == hold.sum() and scores["score"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="no out-of-fold prediction"):
        external_option_a.holdout_from_oof(perfect.iloc[1:])


def test_option_a_holdout_predictions_have_no_measured_values():
    out = pd.read_csv(PROJ_ROOT / "reports" / "holdout" / "option_a.csv")
    assert list(out.columns) == ["batch_id", "elapsed_time", *TARGETS]
