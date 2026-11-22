"""Option A, the team's final model (ported from branch hybrid_v3_tr; code in option_a_src/).

Option A is a simplified hybrid v7: one mechanistic core, the sensor-conditioned particle
Monod core of v7, and two learned views on it, the hybrid v3 recipe and a functional-PCA
trajectory model. They are blended geometrically with weights per variable and per day
segment (days 1-6 / 7-13), fitted on out-of-fold predictions. Glucose is then moved to the
MAPE-optimal point inside the bolus-rule interval, and TCD >= VCD is enforced.

This stage runs Tobias' option_a.main() as ported, on the raw data and the cached core and
member outputs imported into data/external/hybrid_v3_tr (see option_a_src/__init__.py), and
collects its results:

    python -m bioreactor_decoding.modeling.option_a
-> data/processed/submission_option_a.csv   the submission (the registered artifact)
   reports/holdout/option_a.csv    its out-of-fold predictions of our 10 holdout batches
   reports/option_a/blend_weights.csv   weights per variable and day segment
   reports/metrics/option_a.json   7-fold CV, leaderboard score, holdout MAPE (out-of-fold),
                                   and how far this run is from the file that was submitted

Protocol: Option A is validated with 7 GroupKFold folds over all 49 labelled batches, not our
5 frozen dev folds, so its CV is called cv7_mape. Each of our holdout batches lies in one of
those folds, and its out-of-fold prediction comes from a model that never saw that batch.
That model was fitted on about 42 batches, though, including the other holdout batches, while
ours are fitted on the 39 dev batches. The fold-wise choices were also made on those folds.
So holdout_mape_oof7 is a fair out-of-sample error but not the same protocol as holdout_mape.
It is named differently for that reason.
"""

import json
import shutil
import sys

from loguru import logger
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import (
    EXTERNAL_DATA_DIR,
    HOLDOUT_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from bioreactor_decoding.dataset import TARGETS, load_template
from bioreactor_decoding.modeling.models import challenge_score
from bioreactor_decoding.modeling.predict import fill_template
from bioreactor_decoding.modeling.train import labelled_data

app = typer.Typer()
SRC = PROJ_ROOT / "bioreactor_decoding" / "modeling" / "option_a_src"
INPUTS = EXTERNAL_DATA_DIR / "hybrid_v3_tr" / "outputs"
SUBMITTED = INPUTS / "option_a" / "leaderboard_submission.csv"  # the file that was submitted
LEADERBOARD = PROJ_ROOT / "references" / "option_a_leaderboard.json"
WORK = INTERIM_DATA_DIR / "option_a"  # option_a.py's own output files


def run_option_a():
    """Tobias' option_a.main(), writing its files to WORK instead of the inputs folder."""
    sys.path.insert(0, str(SRC))
    import option_a  # the ported script; imports its modules from SRC

    option_a.OUT = WORK
    option_a.main()


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    return fill_template(load_template(), pred[["batch_id", "elapsed_time", *TARGETS]])


def holdout_from_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Out-of-fold predictions ('f|' columns, after the finishing steps) of our holdout rows."""
    _, Y, split = labelled_data(use_future_feeds=False)
    holdout = split["role"] == "holdout"
    rows = split.loc[holdout, ["batch_id", "elapsed_time"]].assign(
        t_key=lambda d: d["elapsed_time"].round(6)
    )
    pred = oof.assign(t_key=oof["t_key"].round(6))[
        ["batch_id", "t_key", *[f"f|{v}" for v in TARGETS]]
    ]
    merged = rows.merge(pred, on=["batch_id", "t_key"], how="left", validate="one_to_one")
    if merged.filter(like="f|").isna().any().any():
        raise ValueError("Option A has no out-of-fold prediction for some holdout rows")
    out = merged[["batch_id", "elapsed_time"]].copy()
    out[TARGETS] = merged[[f"f|{v}" for v in TARGETS]].to_numpy()
    return out, challenge_score(Y[holdout].reset_index(drop=True), out)


@app.command()
def main():
    run_option_a()

    read = {"dtype": {"batch_id": str}}
    submission = to_submission(pd.read_csv(WORK / "holdout_predictions.csv", **read))
    submitted = to_submission(pd.read_csv(SUBMITTED, **read))
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(PROCESSED_DATA_DIR / "submission_option_a.csv", index=False)

    holdout, holdout_scores = holdout_from_oof(pd.read_csv(WORK / "oof_predictions.csv", **read))
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(HOLDOUT_DIR / "option_a.csv", index=False)

    rel = 100 * (submission[TARGETS] - submitted[TARGETS]).abs() / submitted[TARGETS]
    cv = pd.read_csv(WORK / "cv_score.csv").iloc[0]
    metrics = {
        "cv7_mape": {
            **{v: float(cv[v]) for v in TARGETS},
            "score": float(cv["leaderboard_score"]),  # his name for the mean of the 4 MAPEs
        },
        "leaderboard_mape": json.loads(LEADERBOARD.read_text())["leaderboard_mape"],
        "holdout_mape_oof7": {k: round(v, 4) for k, v in holdout_scores.items()},
        "rerun_vs_submitted_pct": {
            "median": float(np.median(rel.to_numpy())),
            "max": float(rel.to_numpy().max()),
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "option_a.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (REPORTS_DIR / "option_a").mkdir(parents=True, exist_ok=True)
    shutil.copy(WORK / "blend_weights.csv", REPORTS_DIR / "option_a" / "blend_weights.csv")
    logger.success(f"option_a: {json.dumps(metrics)}")


if __name__ == "__main__":
    app()
