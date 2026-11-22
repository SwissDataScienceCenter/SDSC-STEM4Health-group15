"""Option A, the team's final model (branch hybrid_v3_tr): run unchanged at a pinned commit.

Option A is a simplified hybrid v7 (src/option_a.py at the commit in params.yaml, option_a.ref):
one mechanistic core, the sensor-conditioned particle Monod core of v7, and two learned views
on it, the hybrid v3 recipe and a functional-PCA trajectory model. They are blended
geometrically with weights per variable and per day segment (days 1-6 / 7-13), fitted on
out-of-fold predictions. Glucose is then moved to the MAPE-optimal point inside the bolus-rule
interval, and TCD >= VCD is enforced. It drops three of v7's four cores and the TCD ratio model.

The commit adds option_a.py on top of the v7 commit and changes nothing that v7 uses, so the
two share every script and cached core output.

    python -m bioreactor_decoding.modeling.external_option_a
-> data/processed/submission_option_a.csv            rerun of src/option_a.py at the commit
   data/processed/submission_option_a_submitted.csv  the file that was submitted (as committed;
                                                     the rerun overwrites it in the worktree)
   reports/holdout/option_a.csv    its out-of-fold predictions of our 10 holdout batches
   reports/option_a/blend_weights.csv   weights per variable and day segment
   reports/metrics/option_a.json   7-fold CV, leaderboard score, holdout MAPE (out-of-fold),
                                   and how far the rerun is from the submitted file

Protocol: like v7, Option A is validated with 7 GroupKFold folds over all 49 labelled batches,
not our 5 frozen dev folds, so its CV is called cv7_mape. Each of our holdout batches lies in
one of those folds, and its out-of-fold prediction comes from a model that never saw that
batch. That model was fitted on about 42 batches, though, including the other holdout
batches, while ours are fitted on the 39 dev batches. The fold-wise choices were also made on
those folds. So holdout_mape_oof7 is a fair out-of-sample error but not the same protocol as
holdout_mape. It is named differently for that reason.

The leaderboard score is not in a results file at the commit. It is the printed output of
notebooks/Leaderboard_template.ipynb, which submitted outputs/option_a/leaderboard_submission.csv.
"""

import ast
import io
import json
import shutil
import subprocess
import sys

from loguru import logger
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import (
    HOLDOUT_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from bioreactor_decoding.dataset import TARGETS
from bioreactor_decoding.modeling import external
from bioreactor_decoding.modeling.models import challenge_score
from bioreactor_decoding.modeling.train import labelled_data

app = typer.Typer()
WORKTREE = external.EXTERNAL_DIR / "option_a_tr"
OUT = "outputs/option_a"
LEADERBOARD_NOTEBOOK = "notebooks/Leaderboard_template.ipynb"


def leaderboard_result(ref: str) -> dict[str, float]:
    """Leaderboard MAPE of the Option A submission, from the notebook output at `ref`."""
    nb = json.loads(external.git_show(ref, LEADERBOARD_NOTEBOOK))
    sources = ["".join(c["source"]) for c in nb["cells"]]
    if not any(f"{OUT}/leaderboard_submission.csv" in s for s in sources):
        raise ValueError(f"{LEADERBOARD_NOTEBOOK} at {ref} did not submit {OUT}")
    for cell in nb["cells"]:
        if "submit(" in "".join(cell["source"]) and cell.get("outputs"):
            text = "".join(cell["outputs"][0]["data"]["text/plain"])
            lb = ast.literal_eval(text)
            return {
                **{v: float(lb[f"{v}_mape"]) for v in TARGETS},
                "score": float(lb["mean_mape"]),
            }
    raise ValueError(f"No leaderboard result in {LEADERBOARD_NOTEBOOK} at {ref}")


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
    ref = str(load_params()["option_a"]["ref"])
    external.checkout(ref, WORKTREE)
    logger.info(f"Running src/option_a.py at {ref}")
    subprocess.run([sys.executable, "src/option_a.py"], cwd=WORKTREE, check=True)

    read = {"dtype": {"batch_id": str}}
    rerun = external.to_submission(pd.read_csv(WORKTREE / OUT / "holdout_predictions.csv", **read))
    committed = external.git_show(ref, f"{OUT}/leaderboard_submission.csv")
    submitted = external.to_submission(pd.read_csv(io.StringIO(committed), **read))
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rerun.to_csv(PROCESSED_DATA_DIR / "submission_option_a.csv", index=False)
    submitted.to_csv(PROCESSED_DATA_DIR / "submission_option_a_submitted.csv", index=False)

    oof = pd.read_csv(WORKTREE / OUT / "oof_predictions.csv", **read)
    holdout, holdout_scores = holdout_from_oof(oof)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(HOLDOUT_DIR / "option_a.csv", index=False)

    rel = 100 * (rerun[TARGETS] - submitted[TARGETS]).abs() / submitted[TARGETS]
    cv = pd.read_csv(WORKTREE / OUT / "cv_score.csv").iloc[0]
    metrics = {
        "cv7_mape": {
            **{v: float(cv[v]) for v in TARGETS},
            "score": float(cv["leaderboard_score"]),  # his name for the mean of the 4 MAPEs
        },
        "leaderboard_mape": leaderboard_result(ref),
        "holdout_mape_oof7": {k: round(v, 4) for k, v in holdout_scores.items()},
        "rerun_vs_submitted_pct": {
            "median": float(np.median(rel.to_numpy())),
            "max": float(rel.to_numpy().max()),
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "option_a.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (REPORTS_DIR / "option_a").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        WORKTREE / OUT / "blend_weights.csv", REPORTS_DIR / "option_a" / "blend_weights.csv"
    )
    logger.success(f"option_a at {ref}: {json.dumps(metrics)}")


if __name__ == "__main__":
    app()
