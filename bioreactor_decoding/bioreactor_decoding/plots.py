"""Holdout reality-check plots: observed holdout trajectories vs. a model fitted on dev only."""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import FIGURES_DIR
from bioreactor_decoding.dataset import TARGETS, load_tables
from bioreactor_decoding.modeling.models import (
    ConstantNullModel,
    PerDayNullModel,
    challenge_score,
)
from bioreactor_decoding.modeling.train import labelled_data, make_xgb_raw

app = typer.Typer()

LABELS = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}
# Validated categorical palette (8 hues, fixed order). Batches beyond 8 reuse the hues with a
# dashed line, so every batch keeps a unique (colour, line style) pair.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def batch_styles(batch_ids) -> dict[str, dict]:
    ordered = sorted(batch_ids, key=lambda b: int("".join(filter(str.isdigit, b)) or 0))
    return {
        b: {"color": SERIES[i % len(SERIES)], "linestyle": "-" if i < len(SERIES) else "--"}
        for i, b in enumerate(ordered)
    }


def plot_holdout(
    observed: pd.DataFrame,
    reference: pd.DataFrame,
    reference_label: str,
    scores: dict[str, float],
    title: str,
    step: bool = False,
) -> plt.Figure:
    """observed: batch_id, elapsed_time, TARGETS (holdout, incl. t = 0).
    reference: elapsed_time, TARGETS - one model trajectory drawn as a thick black line;
    with step=True each value is drawn as a step one day wide, centred on its time."""
    styles = batch_styles(observed["batch_id"].unique())
    fig, axes = plt.subplots(len(TARGETS), 1, sharex=True, figsize=(8, 10), layout="constrained")

    for ax, c in zip(axes, TARGETS):
        for b, style in styles.items():
            g = observed[observed["batch_id"] == b].sort_values("elapsed_time")
            ax.plot(g["elapsed_time"], g[c], lw=1.2, marker="o", ms=2.5, label=b, **style)
        ref_style = {"color": "black", "lw": 3.5, "label": reference_label}
        if step:
            t = reference["elapsed_time"].to_numpy()
            ax.stairs(reference[c], np.r_[t - 0.5, t[-1] + 0.5], baseline=None, **ref_style)
        else:
            ax.plot(reference["elapsed_time"], reference[c], **ref_style)
        ax.set_ylabel(LABELS[c], color=INK)
        ax.set_title(
            f"{LABELS[c]}   ·   holdout MAPE {scores[c]:.1f}%",
            loc="left",
            fontsize=10,
            color=INK,
        )
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK)

    axes[-1].set_xlabel("Elapsed time [days]", color=INK)
    axes[-1].set_xlim(left=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside right center",
        frameon=False,
        fontsize=9,
        title="Holdout batch",
        title_fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        f"{title}   ·   mean holdout MAPE {scores['score']:.1f}%",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
    )
    return fig


def _style_axis(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)


def plot_holdout_per_batch(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    model_label: str,
    scores: dict[str, float],
    title: str,
) -> plt.Figure:
    """Grid of variables (rows, shared x) x holdout batches (columns).

    For models whose prediction differs per batch: each cell shows that batch's observed series
    (thin, batch colour) and the model's prediction for it (thick black).
    observed: batch_id, elapsed_time, TARGETS (incl. t = 0); predicted: same columns, t > 0.
    """
    styles = batch_styles(observed["batch_id"].unique())
    fig, axes = plt.subplots(
        len(TARGETS),
        len(styles),
        sharex=True,
        sharey="row",
        figsize=(2.1 * len(styles) + 1, 9),
        layout="constrained",
        squeeze=False,
    )
    for j, (b, style) in enumerate(styles.items()):
        obs = observed[observed["batch_id"] == b].sort_values("elapsed_time")
        pred = predicted[predicted["batch_id"] == b].sort_values("elapsed_time")
        axes[0, j].set_title(b, fontsize=9, color=INK)
        for i, c in enumerate(TARGETS):
            ax = axes[i, j]
            ax.plot(pred["elapsed_time"], pred[c], color="black", lw=3, label=model_label)
            ax.plot(obs["elapsed_time"], obs[c], lw=1.2, marker="o", ms=2.5, label="Observed",
                    **style)  # fmt: skip
            _style_axis(ax)
    for i, c in enumerate(TARGETS):
        # set limits once per row, after everything is drawn, so no batch gets clipped
        top = max(observed[c].max(), predicted[c].max())
        axes[i, 0].set_ylim(0, 1.05 * top)
        axes[i, 0].set_ylabel(f"{LABELS[c]}\nMAPE {scores[c]:.1f}%", color=INK, fontsize=9)
    axes[-1, 0].set_xticks([0, 4, 8, 12])
    fig.supxlabel("Elapsed time [days]", color=INK, fontsize=10)
    handles = [
        plt.Line2D([], [], color="black", lw=3, label=model_label),
        plt.Line2D([], [], color=INK_MUTED, lw=1.2, marker="o", ms=2.5,
                   label="Observed (batch colour)"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"{title}   ·   mean holdout MAPE {scores['score']:.1f}%",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    return fig


@app.command()
def xgb_raw(output_path: Path = FIGURES_DIR / "xgb_raw.png"):
    """Straight-up XGBoost (raw features -> raw targets), fitted on dev, vs. holdout batches."""
    X, Y, split = labelled_data()
    dev, holdout = split["role"] == "dev", split["role"] == "holdout"
    model = make_xgb_raw().fit(X[dev], Y[dev])
    pred = model.predict(X[holdout])
    scores = challenge_score(Y[holdout], pred)
    n_nonpos = int((pred <= 0).to_numpy().sum())
    if n_nonpos:
        logger.warning(f"{n_nonpos} non-positive holdout predictions")
    pred[["batch_id", "elapsed_time"]] = split.loc[holdout, ["batch_id", "elapsed_time"]]

    samples, _ = load_tables()
    observed = samples.loc[samples["role"] == "holdout", ["batch_id", "elapsed_time", *TARGETS]]
    fig = plot_holdout_per_batch(
        observed, pred, "XGBoost (fit on dev)", scores, "Straight-up XGBoost vs. holdout batches"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.success(f"XGBoost raw: holdout score {scores['score']:.1f}% -> {output_path}")


def _null_figure(model, reference_fn, label: str, title: str, output_path: Path, step: bool):
    """Fit `model` on dev, score it on holdout, and plot it against the holdout batches."""
    X, Y, split = labelled_data()
    dev, holdout = split["role"] == "dev", split["role"] == "holdout"
    model.fit(X[dev], Y[dev])
    scores = challenge_score(Y[holdout], model.predict(X[holdout]))

    samples, _ = load_tables()
    observed = samples.loc[samples["role"] == "holdout", ["batch_id", "elapsed_time", *TARGETS]]
    fig = plot_holdout(observed, reference_fn(model, observed), label, scores, title, step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.success(f"{title}: holdout score {scores['score']:.1f}% -> {output_path}")


@app.command()
def constant_null_model(output_path: Path = FIGURES_DIR / "null_model_constant.png"):
    """Constant null model (per-variable mean over all times and dev batches) vs. holdout."""

    def reference(model, observed):
        t = np.array([0.0, observed["elapsed_time"].max()])
        return pd.DataFrame({"elapsed_time": t, **{c: model.means_[c] for c in TARGETS}})

    _null_figure(
        ConstantNullModel(),
        reference,
        "Constant null model\n(mean of dev batches)",
        "Constant null model vs. holdout batches",
        output_path,
        step=False,
    )


@app.command()
def per_day_null_model(output_path: Path = FIGURES_DIR / "null_model_per_day.png"):
    """Per-day null model (per-variable mean of each sampling day over dev batches) vs. holdout."""

    def reference(model, observed):
        return model.means_.rename_axis("elapsed_time").reset_index()

    _null_figure(
        PerDayNullModel(),
        reference,
        "Per-day null model\n(daily mean of dev batches)",
        "Per-day null model vs. holdout batches",
        output_path,
        step=True,
    )


@app.command()
def all_figures():
    """Regenerate every holdout figure."""
    constant_null_model()
    per_day_null_model()
    xgb_raw()


if __name__ == "__main__":
    app()
