"""Plots for understanding the fed-batch bioreactor data.

The remaining file is an event log. Measurement rows carry VCD, TCD, substrate,
byproduct, and volume. Feed rows carry one feed each. Twelve holdout batches
keep only the t = 0 measurement; their later target values are hidden, while
volume and all four feeds stay observed.

Run from the repo root:

    python src/plot_eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "input-data" / "Project 1 - DTB"
OUT_DIR = ROOT / "outputs" / "eda"

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

TRAIN = "#4C78A8"
HOLDOUT = "#E45756"
MEDIAN = "#1F4E79"
FEED_COLORS = {
    "feed_substrate": "#F58518",
    "feed_b": "#54A24B",
    "feed_c": "#B279A2",
    "feed_d": "#72B7B2",
}
TEST_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#EECA3B", "#9D755D",
    "#FF9DA6", "#1F4E79", "#88D27A", "#499894",
]
FEED_LABELS = {
    "feed_substrate": "Substrate feed",
    "feed_b": "Feed B",
    "feed_c": "Feed C",
    "feed_d": "Feed D",
}
TARGET_LABELS = {
    "vcd": "Viable cell density (VCD)",
    "tcd": "Total cell density (TCD)",
    "substrate": "Substrate",
    "byproduct": "Byproduct",
    "volume": "Volume",
    "viability": "Viability (VCD / TCD)",
    "nonviable": "Non-viable cells (TCD − VCD)",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    remaining = pd.read_csv(DATA_DIR / "bioreaction_remaining.csv")
    template = pd.read_csv(DATA_DIR / "bioreaction_template.csv")
    remaining["batch_id"] = remaining["batch_id"].astype(str)
    template["batch_id"] = template["batch_id"].astype(str)
    holdout_ids = set(template["batch_id"])

    remaining = remaining.sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    remaining["role"] = np.where(remaining["batch_id"].isin(holdout_ids), "holdout", "train")

    measured = remaining.dropna(subset=TARGETS).copy()
    measured["k"] = measured.groupby("batch_id").cumcount()
    measured["viability"] = measured["vcd"] / measured["tcd"]
    measured["nonviable"] = measured["tcd"] - measured["vcd"]

    volume = remaining.dropna(subset=["volume"]).copy()
    volume["k"] = volume.groupby("batch_id").cumcount()
    return remaining, measured, volume


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "figure.dpi": 120,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _day_positions(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("k")["elapsed_time"].median()


def _band(ax: plt.Axes, frame: pd.DataFrame, column: str, color: str) -> None:
    grouped = frame.groupby("k")[column]
    low = grouped.quantile(0.10)
    mid = grouped.quantile(0.50)
    high = grouped.quantile(0.90)
    time = _day_positions(frame)
    ax.fill_between(time, low, high, color=color, alpha=0.18, linewidth=0, label="10th–90th pct")
    ax.plot(time, mid, color=MEDIAN, linewidth=2.0, label="Median")


def _faint_lines(ax: plt.Axes, frame: pd.DataFrame, column: str, color: str, alpha: float = 0.18) -> None:
    for _, group in frame.groupby("batch_id"):
        ax.plot(group["elapsed_time"], group[column], color=color, alpha=alpha, linewidth=0.7)


def _feed_event_lines(ax: plt.Axes, remaining: pd.DataFrame) -> None:
    """Thin vertical marks at each feed addition, behind the trajectories."""
    for feed in FEEDS:
        times = remaining.loc[remaining[feed].notna(), "elapsed_time"].to_numpy()
        if len(times) == 0:
            continue
        ax.vlines(
            times,
            0.0,
            1.0,
            transform=ax.get_xaxis_transform(),
            colors=FEED_COLORS[feed],
            linewidth=0.5,
            alpha=0.10,
            zorder=1,
        )


def plot_distribution_by_day(measured: pd.DataFrame, volume: pd.DataFrame) -> None:
    """Spread of each series at the daily samples the model has to predict."""
    panels = [
        (measured, "vcd"),
        (measured, "tcd"),
        (measured, "viability"),
        (measured, "substrate"),
        (measured, "byproduct"),
        (volume, "volume"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, constrained_layout=True)
    for ax, (frame, column) in zip(axes.ravel(), panels):
        train = frame[frame["role"] == "train"]
        holdout = frame[frame["role"] == "holdout"]
        positions = _day_positions(train)
        series = [train.loc[train["k"] == k, column].to_numpy() for k in positions.index]
        boxes = ax.boxplot(
            series,
            positions=positions.to_numpy(),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "#4a4a4a"},
            capprops={"color": "#4a4a4a"},
            boxprops={"facecolor": TRAIN, "edgecolor": TRAIN, "alpha": 0.35},
        )
        del boxes
        for k, position in positions.items():
            values = train.loc[train["k"] == k, column]
            ax.scatter(
                np.full(len(values), position),
                values,
                s=8,
                color=TRAIN,
                alpha=0.35,
                zorder=3,
            )
        if column != "volume" and not holdout.empty:
            ax.scatter(
                np.zeros(len(holdout)),
                holdout[column],
                s=18,
                color=HOLDOUT,
                zorder=4,
                label="Holdout, t = 0",
            )
        ax.set_title(TARGET_LABELS[column])
        ax.set_xlim(-0.6, 13.6)
    for ax in axes[1, :]:
        ax.set_xlabel("Elapsed time (days)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Distribution at each daily sample. Day 0 has all 61 runs; later days have the 49 fully observed runs.",
        fontsize=12,
    )
    save(fig, "01_distribution_by_day.png")


def plot_trajectories(remaining: pd.DataFrame, measured: pd.DataFrame, volume: pd.DataFrame) -> None:
    panels = [
        (measured, "vcd", False),
        (measured, "tcd", False),
        (measured, "viability", False),
        (measured, "substrate", False),
        (measured, "byproduct", False),
        (volume, "volume", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, constrained_layout=True)
    for ax, (frame, column, holdout_known) in zip(axes.ravel(), panels):
        train = frame[frame["role"] == "train"]
        holdout = frame[frame["role"] == "holdout"]
        _feed_event_lines(ax, remaining)
        _faint_lines(ax, train, column, TRAIN)
        _band(ax, train, column, TRAIN)
        if holdout_known:
            _faint_lines(ax, holdout, column, HOLDOUT, alpha=0.55)
        elif not holdout.empty:
            ax.scatter(
                holdout["elapsed_time"],
                holdout[column],
                s=16,
                color=HOLDOUT,
                zorder=4,
            )
        ax.set_title(TARGET_LABELS[column])
    for ax in axes[1, :]:
        ax.set_xlabel("Elapsed time (days)")
    trajectory_handles = [
        Line2D([0], [0], color=TRAIN, alpha=0.5, label="Training run"),
        Line2D([0], [0], color=MEDIAN, linewidth=2, label="Training median"),
        Patch(facecolor=TRAIN, alpha=0.25, label="Training 10th–90th"),
        Line2D([0], [0], color=HOLDOUT, marker="o", linestyle="none", label="Holdout (targets only at t = 0)"),
    ]
    feed_handles = [
        Line2D([0], [0], color=FEED_COLORS[feed], linewidth=1.6, label=FEED_LABELS[feed])
        for feed in FEEDS
    ]
    # The legend packs down each column, so zip keeps feeds on the second row.
    handles = [item for pair in zip(trajectory_handles, feed_handles) for item in pair]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Batch trajectories. Volume stays observed on the holdout runs (red); the other targets do not.")
    save(fig, "02_trajectories.png")


def _example_batch_ids(measured: pd.DataFrame) -> list[str]:
    train = measured[measured["role"] == "train"]
    peak = train.groupby("batch_id")["vcd"].max().sort_values()
    return [peak.index[0], peak.index[len(peak) // 2], peak.index[-1]]


def plot_example_batches(remaining: pd.DataFrame, measured: pd.DataFrame) -> None:
    batch_ids = _example_batch_ids(measured)
    peak = measured.groupby("batch_id")["vcd"].max()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12.5, 9.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.85, 0.7]},
    )
    row_specs = [
        [("vcd", TRAIN), ("tcd", "#F2A200")],
        [("substrate", "#54A24B"), ("byproduct", "#B279A2")],
        [("volume", "#4C78A8")],
    ]
    for col, batch_id in enumerate(batch_ids):
        batch = remaining[remaining["batch_id"] == batch_id]
        observed = measured[measured["batch_id"] == batch_id]
        for row, specs in enumerate(row_specs):
            ax = axes[row, col]
            for column, color in specs:
                source = observed if column != "volume" else batch.dropna(subset=["volume"])
                ax.plot(source["elapsed_time"], source[column], color=color, marker="o", markersize=3.5, label=TARGET_LABELS[column])
            if row == 0:
                ax.set_title(f"{batch_id}   peak VCD {peak.loc[batch_id]:.1f}")
            ax.legend(frameon=False, fontsize=8, loc="best")
        feed_ax = axes[3, col]
        for level, feed in enumerate(FEEDS):
            events = batch.dropna(subset=[feed])
            scale = remaining[feed].max()
            sizes = 18 + 70 * (events[feed] / scale)
            feed_ax.scatter(
                events["elapsed_time"],
                np.full(len(events), level),
                s=sizes,
                color=FEED_COLORS[feed],
                zorder=3,
            )
        feed_ax.set_yticks(range(len(FEEDS)), [FEED_LABELS[feed] for feed in FEEDS])
        feed_ax.set_ylim(-0.6, len(FEEDS) - 0.4)
        feed_ax.set_xlabel("Elapsed time (days)")
        feed_ax.grid(axis="y")
    fig.suptitle("Three training runs spanning low, middle, and high peak VCD. Marker size on the feed row scales with amount.")
    save(fig, "03_example_batches.png")


def plot_feed_schedule(remaining: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, feed in zip(axes.ravel(), FEEDS):
        events = remaining.dropna(subset=[feed])
        for role, color in (("train", TRAIN), ("holdout", HOLDOUT)):
            part = events[events["role"] == role]
            ax.scatter(
                part["elapsed_time"],
                part[feed],
                s=16,
                alpha=0.45 if role == "train" else 0.8,
                color=color,
                label=role,
            )
        ax.set_title(FEED_LABELS[feed])
        ax.set_xlabel("Elapsed time (days)")
        ax.set_ylabel("Amount per addition")
        ax.legend(frameon=False)
    fig.suptitle("When each feed is added, and how large each addition is. Holdout feeds are fully observed.")
    save(fig, "04_feed_schedule.png")


def _rates(measured: pd.DataFrame) -> pd.DataFrame:
    train = measured[measured["role"] == "train"].sort_values(["batch_id", "elapsed_time"]).copy()
    grouped = train.groupby("batch_id")
    train["dt"] = grouped["elapsed_time"].diff()
    for column in ["vcd", "tcd", "substrate", "byproduct"]:
        train[f"d_{column}"] = grouped[column].diff() / train["dt"]
    train["substrate_at_start"] = grouped["substrate"].shift(1)
    train["vcd_at_start"] = grouped["vcd"].shift(1)
    return train.dropna(subset=["dt"])


def plot_kinetics(measured: pd.DataFrame) -> None:
    rates = _rates(measured)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    time_panels = [
        (axes[0, 0], "d_vcd", "Change in VCD per day"),
        (axes[0, 1], "d_substrate", "Change in substrate per day"),
        (axes[1, 0], "d_byproduct", "Change in byproduct per day"),
    ]
    for ax, column, title in time_panels:
        _faint_lines(ax, rates, column, TRAIN, alpha=0.12)
        _band(ax, rates, column, TRAIN)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Elapsed time (days)")
    ax = axes[1, 1]
    points = ax.scatter(
        rates["substrate_at_start"],
        rates["d_vcd"],
        c=rates["elapsed_time"],
        cmap="viridis",
        s=14,
        alpha=0.75,
    )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Substrate at the start of the interval")
    ax.set_ylabel("Change in VCD per day")
    ax.set_title("Growth rate against substrate")
    fig.colorbar(points, ax=ax, label="Elapsed time (days)", fraction=0.046)
    fig.suptitle("Day-to-day rates on the 49 training runs. Positive substrate change means a feed exceeded consumption.")
    save(fig, "05_kinetics.png")


def plot_relationships(measured: pd.DataFrame) -> None:
    train = measured[measured["role"] == "train"]
    pairs = [
        ("vcd", "tcd", "TCD tracks VCD; the gap is the non-viable fraction"),
        ("elapsed_time", "viability", "Viability declines slowly as non-viable cells accumulate"),
        ("vcd", "substrate", "Substrate is high while VCD is still low, then stays low"),
        ("elapsed_time", "byproduct", "Byproduct rises early and falls after day 6"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    last = None
    for ax, (x_col, y_col, title) in zip(axes.ravel(), pairs):
        last = ax.scatter(
            train[x_col],
            train[y_col],
            c=train["elapsed_time"],
            cmap="viridis",
            s=16,
            alpha=0.8,
        )
        ax.set_xlabel(TARGET_LABELS.get(x_col, "Elapsed time (days)"))
        ax.set_ylabel(TARGET_LABELS[y_col])
        ax.set_title(title, fontsize=10)
    fig.colorbar(last, ax=axes, label="Elapsed time (days)", fraction=0.03, pad=0.02)
    fig.suptitle("Joint distributions on training measurements")
    save(fig, "06_relationships.png")


def _cumulative_feeds(remaining: pd.DataFrame) -> pd.DataFrame:
    totals = remaining.groupby(["batch_id", "role"], sort=False)[FEEDS].sum()
    counts = remaining.groupby(["batch_id", "role"], sort=False)[FEEDS].count()
    counts = counts.rename(columns={feed: f"n_{feed}" for feed in FEEDS})
    return totals.join(counts).reset_index()


def _annotate_correlation(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.04, 0.96, f"r = {r:.2f}", transform=ax.transAxes, va="top")


def plot_batch_differences(remaining: pd.DataFrame, measured: pd.DataFrame) -> None:
    train_measured = measured[measured["role"] == "train"]
    peak = train_measured.groupby("batch_id")["vcd"].max().rename("peak_vcd")
    feeds = _cumulative_feeds(remaining)
    train = feeds[feeds["role"] == "train"].join(peak, on="batch_id")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    specs = [
        ("feed_b", "Cumulative feed B"),
        ("feed_c", "Cumulative feed C"),
        ("n_feed_c", "Number of feed C additions"),
        ("feed_substrate", "Cumulative substrate feed"),
    ]
    for ax, (column, label) in zip(axes.ravel(), specs):
        ax.scatter(train[column], train["peak_vcd"], s=28, color=TRAIN, alpha=0.8)
        _annotate_correlation(ax, train[column], train["peak_vcd"])
        ax.set_xlabel(label)
        ax.set_ylabel("Peak VCD")
    fig.suptitle("What differs between training runs, and how that lines up with peak viable cell density")
    save(fig, "07_batch_differences.png")


def _strip(ax: plt.Axes, train: pd.Series, holdout: pd.Series, rng: np.random.Generator) -> None:
    for x, values, color in ((0, train, TRAIN), (1, holdout, HOLDOUT)):
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=16, color=color, alpha=0.8, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [values.median(), values.median()], color="black", linewidth=1.4, zorder=4)
    ax.set_xticks([0, 1], [f"Train ({len(train)})", f"Holdout ({len(holdout)})"])
    ax.set_xlim(-0.6, 1.6)


def plot_holdout_comparison(remaining: pd.DataFrame, measured: pd.DataFrame, volume: pd.DataFrame) -> None:
    t0 = measured[measured["k"] == 0]
    feeds = _cumulative_feeds(remaining)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)

    initial = ["vcd", "tcd", "substrate", "byproduct"]
    for ax, column in zip(axes[0], initial):
        _strip(
            ax,
            t0.loc[t0["role"] == "train", column],
            t0.loc[t0["role"] == "holdout", column],
            rng,
        )
        ax.set_title(f"t = 0 {TARGET_LABELS[column]}")

    for ax, feed in zip(axes[1], FEEDS):
        _strip(
            ax,
            feeds.loc[feeds["role"] == "train", feed],
            feeds.loc[feeds["role"] == "holdout", feed],
            rng,
        )
        ax.set_title(f"Total {FEED_LABELS[feed]}")

    fig.suptitle("Holdout inputs against the training runs. Black bars are medians. Targets after t = 0 are hidden; these inputs are not.")
    save(fig, "08_holdout_vs_train.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    train_volume = volume[volume["role"] == "train"]
    holdout_volume = volume[volume["role"] == "holdout"]
    _faint_lines(ax, train_volume, "volume", TRAIN)
    _band(ax, train_volume, "volume", TRAIN)
    _faint_lines(ax, holdout_volume, "volume", HOLDOUT, alpha=0.7)
    ax.set_xlabel("Elapsed time (days)")
    ax.set_ylabel("Volume")
    ax.set_title("Volume is recorded for every run, including the 12 holdout batches")
    handles = [
        Line2D([0], [0], color=TRAIN, alpha=0.6, label="Training run"),
        Line2D([0], [0], color=MEDIAN, linewidth=2, label="Training median"),
        Line2D([0], [0], color=HOLDOUT, label="Holdout run"),
    ]
    ax.legend(handles=handles, frameon=False)
    save(fig, "09_volume_holdout.png")


def plot_test_given_values(remaining: pd.DataFrame) -> None:
    """Every value supplied for the 12 test batches.

    That is the t = 0 measurement, volume at all 14 samples, and every feed
    addition. Later target values are absent and are not drawn.
    """
    test = remaining[remaining["role"] == "holdout"].copy()
    batches = sorted(test["batch_id"].unique(), key=lambda name: int(name.replace("Batch", "")))
    colors = dict(zip(batches, TEST_COLORS))
    short = {batch: batch.replace("Batch", "") for batch in batches}
    state_columns = ["vcd", "tcd", "substrate", "byproduct", "volume"]
    state_labels = ["VCD", "TCD", "Substrate", "Byproduct", "Volume"]

    measured0 = test.dropna(subset=TARGETS).set_index("batch_id").loc[batches]
    raw = measured0[state_columns].to_numpy(dtype=float)
    shade = np.zeros_like(raw)
    for column in range(raw.shape[1]):
        lo, hi = raw[:, column].min(), raw[:, column].max()
        shade[:, column] = 0.5 if hi == lo else (raw[:, column] - lo) / (hi - lo)

    fig = plt.figure(figsize=(13.6, 9.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.08)
    ax_state = fig.add_subplot(top[0, 0])
    volume_grid = top[0, 1].subgridspec(3, 4, wspace=0.08, hspace=0.28)
    feed_grid = outer[1].subgridspec(1, 4, wspace=0.08)

    ax_state.imshow(shade, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax_state.set_xticks(range(len(state_labels)), state_labels)
    ax_state.set_yticks(range(len(batches)), [short[batch] for batch in batches])
    ax_state.set_ylabel("Batch")
    ax_state.set_xlabel("Shading ranks batches within a column")
    ax_state.set_title("Starting measurement, the only targets provided")
    ax_state.tick_params(length=0)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax_state.text(
                j,
                i,
                f"{raw[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shade[i, j] > 0.62 else "#102A43",
            )

    volume = test.dropna(subset=["volume"])
    volume_axes: list[plt.Axes] = []
    first: plt.Axes | None = None
    for index, batch in enumerate(batches):
        ax = fig.add_subplot(volume_grid[index // 4, index % 4], sharex=first, sharey=first)
        first = first or ax
        volume_axes.append(ax)
        series = volume[volume["batch_id"] == batch].sort_values("elapsed_time")
        ax.plot(
            series["elapsed_time"],
            series["volume"],
            color=colors[batch],
            marker="o",
            markersize=2.6,
            linewidth=1.2,
        )
        ax.set_title(short[batch], color=colors[batch], fontsize=10, pad=2)
        ax.tick_params(labelsize=7)
        if index % 4 != 0:
            ax.tick_params(labelleft=False)
        if index < 8:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Days")
        if index % 4 == 0:
            ax.set_ylabel("Volume")
    volume_axes[0].set_ylim(10.0, 17.6)
    fig.suptitle(
        "Test data as given: t = 0 measurement, all 14 volumes, and every feed addition\n"
        "VCD, TCD, substrate, and byproduct after t = 0 are not in the file",
        fontsize=12,
    )

    y_pos = {batch: index for index, batch in enumerate(batches)}
    for index, (ax, feed) in enumerate(zip([fig.add_subplot(feed_grid[0, i]) for i in range(4)], FEEDS)):
        events = test.dropna(subset=[feed])
        amounts = events[feed].to_numpy()
        size = 14 + 170 * np.sqrt(amounts / amounts.max())
        ax.scatter(
            events["elapsed_time"],
            events["batch_id"].map(y_pos),
            s=size,
            color=FEED_COLORS[feed],
            alpha=0.9,
            linewidths=0,
        )
        ax.set_ylim(len(batches) - 0.5, -0.5)
        ax.set_xlim(-0.3, 13.4)
        ax.set_title(
            f"{FEED_LABELS[feed]}\n{len(events)} shots, {amounts.min():.3g}–{amounts.max():.3g}",
            fontsize=10,
        )
        ax.set_xlabel("Days")
        ax.set_yticks(range(len(batches)), [short[batch] for batch in batches])
        if index == 0:
            ax.set_ylabel("Batch")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=7)

    save(fig, "10_test_given_values.png")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remaining, measured, volume = load()
    plot_distribution_by_day(measured, volume)
    plot_trajectories(remaining, measured, volume)
    plot_example_batches(remaining, measured)
    plot_feed_schedule(remaining)
    plot_kinetics(measured)
    plot_relationships(measured)
    plot_batch_differences(remaining, measured)
    plot_holdout_comparison(remaining, measured, volume)
    plot_test_given_values(remaining)
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
