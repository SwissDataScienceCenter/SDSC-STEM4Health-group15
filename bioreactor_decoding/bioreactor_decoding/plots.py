"""Holdout reality-check figures, drawn from saved holdout predictions (no refitting).

Inputs are reports/holdout/<model>.csv, written by bioreactor_decoding.modeling.evaluate from
the model fitted on dev batches only. Every figure is a grid of variables (rows, shared x axis)
x holdout batches (columns).
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import FIGURES_DIR, HOLDOUT_DIR
from bioreactor_decoding.dataset import TARGETS, load_tables
from bioreactor_decoding.modeling.models import challenge_score

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


# model -> (figure file, title, draw the prediction as day-wide steps)
FIGURES = {
    "null_constant": ("null_model_constant.png", "Constant null model", False),
    "null_per_day": ("null_model_per_day.png", "Per-day null model", True),
    "xgb_raw": ("xgb_raw.png", "Straight-up XGBoost", False),
    "xgb_log_fold_change": ("xgb_log_fold_change.png", "XGBoost on log fold change", False),
    "null_xgb_ensemble": ("null_xgb_ensemble.png", "Null + XGBoost 50/50 ensemble", False),
    "hybrid_v3": ("hybrid_v3.png", "Hybrid v3", False),
    "hybrid_lite": ("hybrid_lite.png", "Hybrid lite (v3 stage 2, mechanistic stage 1)", False),
}


def load_holdout(model: str, holdout_dir: Path = HOLDOUT_DIR):
    """(actual, predicted, scores) for one model.

    Predictions come from reports/holdout/<model>.csv; the measured values are joined from the
    raw data (they are not stored in reports/, which is in git).
    """
    pred = pd.read_csv(holdout_dir / f"{model}.csv")
    samples, _ = load_tables()
    meas = samples.loc[samples["role"] == "holdout", ["batch_id", "elapsed_time", *TARGETS]]
    key = ["batch_id", "t_key"]
    merged = pred.assign(t_key=pred["elapsed_time"].round(6)).merge(
        meas.assign(t_key=meas["elapsed_time"].round(6)).drop(columns="elapsed_time"),
        on=key, how="left", suffixes=("_pred", ""), validate="one_to_one",
    )  # fmt: skip
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Holdout predictions of {model} do not match the holdout samples")
    actual = merged[["batch_id", "elapsed_time", *TARGETS]]
    predicted = merged[["batch_id", "elapsed_time"]].join(
        merged[[f"{c}_pred" for c in TARGETS]].set_axis(TARGETS, axis=1)
    )
    return actual, predicted, challenge_score(actual, predicted)


def _save(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


@app.command()
def holdout(model: str, output_dir: Path = FIGURES_DIR):
    """Actual vs. predicted on the holdout batches for MODEL."""
    filename, title, step = FIGURES[model]
    actual, predicted, scores = load_holdout(model)
    samples, _ = load_tables()  # add the given t = 0 values to the actual traces
    t0 = samples.loc[(samples["role"] == "holdout") & (samples["elapsed_time"] == 0)]
    actual = pd.concat([t0[["batch_id", "elapsed_time", *TARGETS]], actual], ignore_index=True)
    fig = plot_holdout_per_batch(actual, predicted, scores, f"{title} vs. holdout batches", step)
    _save(fig, output_dir / filename)
    logger.success(f"{model}: holdout score {scores['score']:.1f}% -> {output_dir / filename}")


# Categorical slots 1-4 of the validated palette (fixed order); one fixed colour per model.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PE_LIM = 15.0


def plot_pe_per_batch(pe: dict[str, pd.DataFrame], scores: dict[str, dict], title: str):
    """Grid of variables (rows) x holdout batches (columns) with one percent-error trace per model.

    pe[name]: batch_id, elapsed_time, TARGETS holding signed percent errors,
    100 * (predicted - actual) / actual (positive = overprediction).
    The y axis is shared and fixed to [-PE_LIM, PE_LIM] with a zero line; errors beyond the
    limits are drawn as a triangle on the top/bottom edge so they stay visible.
    """
    first = next(iter(pe.values()))
    batches = sorted(first["batch_id"].unique(), key=_batch_number)
    fig, axes = plt.subplots(
        len(TARGETS),
        len(batches),
        sharex=True,
        sharey=True,
        figsize=(2.1 * len(batches) + 1, 9),
        layout="constrained",
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axhline(0, color=INK_MUTED, lw=1)
    for (name, err), color in zip(pe.items(), MODEL_COLORS):
        for j, b in enumerate(batches):
            e = err[err["batch_id"] == b].sort_values("elapsed_time")
            for i, c in enumerate(TARGETS):
                ax = axes[i, j]
                ax.plot(e["elapsed_time"], e[c], color=color, lw=1.6, marker="o", ms=2.5)
                for sign, marker in ((1, "^"), (-1, "v")):
                    off = e[sign * e[c] > PE_LIM]
                    (marks,) = ax.plot(off["elapsed_time"], np.full(len(off), sign * PE_LIM),
                                       marker, color=color, ms=6, clip_on=False)  # fmt: skip
                    marks.set_in_layout(False)  # drawn on the edge; must not squeeze the layout
    for j, b in enumerate(batches):
        axes[0, j].set_title(b, fontsize=9, color=INK)
        for i in range(len(TARGETS)):
            _style_axis(axes[i, j])
    axes[0, 0].set_ylim(-PE_LIM, PE_LIM)
    axes[0, 0].set_yticks([-15, -10, -5, 0, 5, 10, 15])
    for i, c in enumerate(TARGETS):
        axes[i, 0].set_ylabel(f"{LABELS[c]}\nPE [%]", color=INK, fontsize=9)
    axes[-1, 0].set_xticks([0, 4, 8, 12])
    fig.supxlabel(
        "Elapsed time [days]      ·      PE = (predicted − actual) / actual;  "
        "above 0 = overprediction, below 0 = underprediction",
        color=INK,
        fontsize=10,
    )
    handles = [
        plt.Line2D([], [], color=color, lw=1.6, marker="o", ms=2.5,
                   label=f"{name}: {scores[name]['score']:.1f}%")
        for name, color in zip(pe, MODEL_COLORS)
    ]  # fmt: skip
    handles.append(
        plt.Line2D([], [], color=INK_MUTED, marker="^", ls="none", ms=6,
                   label=f"|PE| > {PE_LIM:.0f}% (off scale)")
    )  # fmt: skip
    fig.legend(
        handles=handles,
        loc="outside upper right",
        ncols=len(handles),
        frameon=False,
        fontsize=9,
        title="Model: holdout MAPE (average of the 4 variables)",
        title_fontsize=9,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    return fig


# model -> legend name, in the fixed palette order
PE_MODELS = {
    "xgb_raw": "XGBoost",
    "null_per_day": "Per-day null",
    "null_xgb_ensemble": "50/50 ensemble",
    "hybrid_v3": "Hybrid v3",
}


@app.command()
def pe_all_models(output_path: Path = FIGURES_DIR / "PE_all_models.png"):
    """Signed percent error of each model on the holdout batches (models fitted on dev)."""
    pe, scores = {}, {}
    for model, name in PE_MODELS.items():
        actual, predicted, scores[name] = load_holdout(model)
        err = 100 * (predicted[TARGETS] - actual[TARGETS]) / actual[TARGETS]
        pe[name] = pd.concat([actual[["batch_id", "elapsed_time"]], err], axis=1)
        n_off = int((err.abs() > PE_LIM).to_numpy().sum())
        logger.info(f"{name}: MAPE {scores[name]['score']:.1f}%, {n_off} points beyond ±{PE_LIM}%")
    fig = plot_pe_per_batch(pe, scores, "Percent error on holdout batches")
    _save(fig, output_path)
    logger.success(f"PE comparison -> {output_path}")


@app.command()
def all_figures():
    """Regenerate every holdout figure."""
    for model in FIGURES:
        holdout(model)
    pe_all_models()


if __name__ == "__main__":
    app()
