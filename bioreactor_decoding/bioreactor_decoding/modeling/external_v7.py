"""Hybrid v7 (Tobias Ruff, branch hybrid_v3_tr): run his code at a pinned commit.

v7 is a 4-model geometric blend on top of hybrid_v3, with sensor-assimilated mechanistic
cores (see docs/mlops.md, "Hybrid v7"). Its code (about 2,000 lines in 10 scripts) chains on
cached outputs of its cores, so instead of a rewrite it is run unchanged from a git worktree
of the commit in params.yaml (hybrid_v7.ref). The commit hash is the provenance: DVC reruns
this stage when it changes.

    python -m bioreactor_decoding.modeling.external_v7
-> data/processed/submission_hybrid_v7.csv        rerun of src/hybrid_v7.py at the commit
   data/processed/submission_hybrid_v7_submitted.csv   the file that was submitted (committed
                                                       by Tobias; scored 6.752)
   reports/metrics/hybrid_v7.json                  his 7-fold CV, the leaderboard score, and
                                                   how far the rerun is from the submitted file
   reports/hybrid_v7/blend_weights.csv             weights per variable and day segment

His CV protocol differs from ours (7 GroupKFold folds over all 49 labelled batches, not the 5
frozen dev folds), so the metric is called cv7_mape and is not comparable with cv_mape.
"""

import json
import shutil
import subprocess
import sys

from loguru import logger
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import (
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from bioreactor_decoding.dataset import TARGETS, load_template
from bioreactor_decoding.modeling.predict import fill_template

app = typer.Typer()
WORKTREE = PROJ_ROOT / "external" / "hybrid_v7_tr"
OUT = "outputs/hybrid_v7"


def _git(*args, cwd=PROJ_ROOT):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def checkout(ref: str):
    """Worktree of the team repository at `ref`, reset to the committed state."""
    if not WORKTREE.exists():
        WORKTREE.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(WORKTREE), ref)
    _git("checkout", "--detach", "--force", ref, cwd=WORKTREE)
    _git("reset", "--hard", "--quiet", cwd=WORKTREE)
    data = WORKTREE / "input-data" / "Project 1 - DTB"  # where his code expects the data
    data.parent.mkdir(exist_ok=True)
    if data.is_symlink() or data.exists():
        data.unlink()
    data.symlink_to(RAW_DATA_DIR, target_is_directory=True)


def to_submission(path) -> pd.DataFrame:
    pred = pd.read_csv(WORKTREE / path, dtype={"batch_id": str})
    return fill_template(load_template(), pred[["batch_id", "elapsed_time", *TARGETS]])


@app.command()
def main():
    ref = str(load_params()["hybrid_v7"]["ref"])
    checkout(ref)
    logger.info(f"Running src/hybrid_v7.py at {ref}")
    subprocess.run([sys.executable, "src/hybrid_v7.py"], cwd=WORKTREE, check=True)

    rerun = to_submission(f"{OUT}/holdout_predictions.csv")
    submitted = to_submission(f"{OUT}/leaderboard_submission.csv")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rerun.to_csv(PROCESSED_DATA_DIR / "submission_hybrid_v7.csv", index=False)
    submitted.to_csv(PROCESSED_DATA_DIR / "submission_hybrid_v7_submitted.csv", index=False)

    rel = 100 * (rerun[TARGETS] - submitted[TARGETS]).abs() / submitted[TARGETS]
    cv = pd.read_csv(WORKTREE / OUT / "cv_score.csv").iloc[0]
    lb = pd.read_csv(WORKTREE / OUT / "leaderboard_result.csv").iloc[0]
    metrics = {
        "cv7_mape": {
            **{v: float(cv[v]) for v in TARGETS},
            "score": float(cv["leaderboard_score"]),
        },
        "leaderboard_mape": {**{v: float(lb[v]) for v in TARGETS}, "score": float(lb["mean"])},
        "rerun_vs_submitted_pct": {
            "median": float(np.median(rel.to_numpy())),
            "max": float(rel.to_numpy().max()),
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "hybrid_v7.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (REPORTS_DIR / "hybrid_v7").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        WORKTREE / OUT / "blend_weights.csv", REPORTS_DIR / "hybrid_v7" / "blend_weights.csv"
    )
    logger.success(f"hybrid_v7 at {ref}: {json.dumps(metrics)}")


if __name__ == "__main__":
    app()
