"""Run team code that lives on another branch, unchanged, at a pinned commit.

Used by the stages that wrap models we did not rewrite (external_v7.py, external_option_a.py).
Each one gets its own git worktree under external/ (ignored by git and DVC), checked out at the
commit in params.yaml and reset to its committed state, with the challenge data linked to where
that code expects it. The commit hash is the provenance: DVC reruns a stage when it changes.
"""

from pathlib import Path
import subprocess

import pandas as pd

from bioreactor_decoding.config import PROJ_ROOT, RAW_DATA_DIR
from bioreactor_decoding.dataset import TARGETS, load_template
from bioreactor_decoding.modeling.predict import fill_template

EXTERNAL_DIR = PROJ_ROOT / "external"


def _git(*args, cwd=PROJ_ROOT):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def checkout(ref: str, worktree: Path):
    """Worktree of the team repository at `ref`, reset to the committed state."""
    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(worktree), ref)
    _git("checkout", "--detach", "--force", ref, cwd=worktree)
    _git("reset", "--hard", "--quiet", cwd=worktree)
    data = worktree / "input-data" / "Project 1 - DTB"  # where the team code expects the data
    data.parent.mkdir(exist_ok=True)
    if data.is_symlink() or data.exists():
        data.unlink()
    data.symlink_to(RAW_DATA_DIR, target_is_directory=True)


def git_show(ref: str, path: str) -> str:
    """Content of `path` (relative to the repository root) as committed at `ref`.

    Use this, not the worktree, for files the team committed that a rerun overwrites.
    """
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=PROJ_ROOT, check=True, capture_output=True, text=True
    ).stdout


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    """Team predictions of the test batches, in the format of the submission template."""
    return fill_template(load_template(), pred[["batch_id", "elapsed_time", *TARGETS]])
