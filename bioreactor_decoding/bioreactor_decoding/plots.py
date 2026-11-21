"""Holdout reality-check plots: actual holdout trajectories vs. a model fitted on dev only.

Every figure is a grid of variables (rows, shared x axis) x holdout batches (columns): the
actual measurements in blue, the model's prediction for that batch in black. Models are
fitted via `holdout_check`, which uses dev batches only.
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import FIGURES_DIR
from bioreactor_decoding.dataset import TARGETS, load_tables
from bioreactor_decoding.modeling.models import ConstantNullModel, PerDayNullModel
from bioreactor_decoding.modeling.train import holdout_check, labelled_data, make_xgb_raw

app = typer.Typer()

LABELS = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}
ACTUAL, PREDICTED = "#2a78d6", "black"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def _batch_number(batch_id: str) -> int:
    return int("".join(filter(str.isdigit, batch_id)) or 0)


def _style_axis(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)


def plot_holdout_per_batch(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    scores: dict[str, float],
    title: str,
    step: bool = False,
) -> plt.Figure:
    """actual: batch_id, elapsed_time, TARGETS (incl. t = 0); predicted: same columns, t > 0.

    With step=True each prediction is drawn as a step one day wide, centred on its time.
    """
    batches = sorted(actual["batch_id"].unique(), key=_batch_number)
    fig, axes = plt.subplots(
        len(TARGETS),
        len(batches),
        sharex=True,
        sharey="row",
        figsize=(2.1 * len(batches) + 1, 9),
        layout="constrained",
        squeeze=False,
    )
    for j, b in enumerate(batches):
        act = actual[actual["batch_id"] == b].sort_values("elapsed_time")
        pred = predicted[predicted["batch_id"] == b].sort_values("elapsed_time")
        axes[0, j].set_title(b, fontsize=9, color=INK)
        for i, c in enumerate(TARGETS):
            ax = axes[i, j]
            if step:
                t = pred["elapsed_time"].to_numpy()
                edges = np.r_[t[0] - 0.5, (t[1:] + t[:-1]) / 2, t[-1] + 0.5]
                ax.stairs(pred[c], edges, baseline=None, color=PREDICTED, lw=3)
            else:
                ax.plot(pred["elapsed_time"], pred[c], color=PREDICTED, lw=3)
            ax.plot(act["elapsed_time"], act[c], color=ACTUAL, lw=1.2, marker="o", ms=2.5)
            _style_axis(ax)
    for i, c in enumerate(TARGETS):
        # set limits once per row, after everything is drawn, so no batch gets clipped
        top = max(actual[c].max(), predicted[c].max())
        axes[i, 0].set_ylim(0, 1.05 * top)
        axes[i, 0].set_ylabel(LABELS[c], color=INK, fontsize=9)
    axes[-1, 0].set_xticks([0, 4, 8, 12])
    fig.supxlabel("Elapsed time [days]", color=INK, fontsize=10)
    handles = [
        plt.Line2D([], [], color=ACTUAL, lw=1.2, marker="o", ms=2.5, label="Actual"),
        plt.Line2D([], [], color=PREDICTED, lw=3, label="Predicted"),
    ]
    fig.legend(handles=handles, loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    per_variable = ", ".join(f"{LABELS[c]}: {scores[c]:.1f}%" for c in TARGETS)
    fig.suptitle(
        f"{title}   ·   holdout MAPE  {per_variable}, average {scores['score']:.1f}%",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    return fig


def holdout_figure(model, title: str, output_path: Path, step: bool = False):
    """Fit `model` on dev batches only, predict the holdout batches, and plot the grid."""
    X, Y, split = labelled_data()
    pred, scores = holdout_check(model, X, Y, split)
    pred[["batch_id", "elapsed_time"]] = split.loc[pred.index, ["batch_id", "elapsed_time"]]

    samples, _ = load_tables()
    actual = samples.loc[samples["role"] == "holdout", ["batch_id", "elapsed_time", *TARGETS]]
    fig = plot_holdout_per_batch(actual, pred, scores, title, step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.success(f"{title}: holdout score {scores['score']:.1f}% -> {output_path}")


@app.command()
def constant_null_model(output_path: Path = FIGURES_DIR / "null_model_constant.png"):
    """Constant null model (per-variable mean over all times and dev batches)."""
    holdout_figure(ConstantNullModel(), "Constant null model vs. holdout batches", output_path)


@app.command()
def per_day_null_model(output_path: Path = FIGURES_DIR / "null_model_per_day.png"):
    """Per-day null model (per-variable mean of each sampling day over dev batches)."""
    holdout_figure(
        PerDayNullModel(), "Per-day null model vs. holdout batches", output_path, step=True
    )


@app.command()
def xgb_raw(output_path: Path = FIGURES_DIR / "xgb_raw.png"):
    """Straight-up XGBoost (raw features -> raw targets)."""
    holdout_figure(make_xgb_raw(), "Straight-up XGBoost vs. holdout batches", output_path)


@app.command()
def all_figures():
    """Regenerate every holdout figure."""
    constant_null_model()
    per_day_null_model()
    xgb_raw()


if __name__ == "__main__":
    app()
