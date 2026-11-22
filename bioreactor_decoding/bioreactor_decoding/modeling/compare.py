"""Per-fold comparison of CV scores against a baseline: the decision rule for experiments.

Both runs use the same frozen dev folds, so the per-fold difference is far less noisy than
the spread of scores across folds. A candidate counts as better when its CV MAPE is lower on
average AND in at least `min_folds` of the 5 folds.

    python -m bioreactor_decoding.modeling.compare hybrid_lite
        workspace reports/cv/hybrid_lite.csv vs. the same model at git HEAD
    python -m bioreactor_decoding.modeling.compare hybrid_blend --baseline hybrid_v3
        a new model vs. the current best model at HEAD

Only CV scores are compared; the holdout plays no part in the decision.
"""

import io
from pathlib import Path
import subprocess

import pandas as pd
import typer

from bioreactor_decoding.config import CV_DIR, PROJ_ROOT
from bioreactor_decoding.dataset import TARGETS

app = typer.Typer()
COLS = [*TARGETS, "score"]


def read_cv_at(ref: str, model: str) -> pd.DataFrame:
    """reports/cv/<model>.csv as committed at git `ref`."""
    rel = CV_DIR.relative_to(PROJ_ROOT) / f"{model}.csv"
    text = subprocess.check_output(["git", "show", f"{ref}:./{rel}"], cwd=PROJ_ROOT, text=True)
    return pd.read_csv(io.StringIO(text)).set_index("fold")[COLS]


def compare(candidate: pd.DataFrame, baseline: pd.DataFrame, min_folds: int = 4) -> dict:
    """Per-fold difference (candidate - baseline; negative = better) and the verdict."""
    diff = candidate[COLS] - baseline[COLS]
    improved = int((diff["score"] < 0).sum())
    mean = float(diff["score"].mean())
    return {
        "diff": diff,
        "mean_change": mean,
        "sd_change": float(diff["score"].std()),
        "folds_improved": improved,
        "n_folds": len(diff),
        "better": mean < 0 and improved >= min_folds,
    }


@app.command()
def main(
    model: str,
    baseline: str | None = None,
    ref: str = "HEAD",
    min_folds: int = 4,
    baseline_file: Path | None = None,
):
    """Compare the workspace CV scores of MODEL with BASELINE (default: MODEL) at git REF,
    or with a saved CV table (--baseline-file, e.g. a model not committed yet)."""
    cand = pd.read_csv(CV_DIR / f"{model}.csv").set_index("fold")
    base_name = baseline or model
    if baseline_file is not None:
        base = pd.read_csv(baseline_file).set_index("fold")[COLS]
        base_name, ref = baseline_file.name, "file"
    else:
        base = read_cv_at(ref, base_name)
    r = compare(cand, base, min_folds)
    print(f"{model} (workspace) vs. {base_name} ({ref}): change in CV MAPE per fold")
    print(r["diff"].T.round(3).to_string())
    print(
        f"mean change {r['mean_change']:+.3f} (sd over folds {r['sd_change']:.3f}), "
        f"better in {r['folds_improved']}/{r['n_folds']} folds "
        f"-> {'BETTER' if r['better'] else 'not better'}"
        f" (rule: mean < 0 and better in >= {min_folds} folds)"
    )
    print(f"CV MAPE: {cand['score'].mean():.3f} vs. {base['score'].mean():.3f}")


if __name__ == "__main__":
    app()
