"""Summarize distributions in the Project 1 bioreaction input data.

The remaining file is an event log, not a dense time grid. Each row is one
event for a batch: a full measurement, a volume-only timestamp that still
needs predictions, or a single feed addition. Feed columns never share a
row with the measured targets.

Run from the repo root:

    python src/explore_input_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "input-data" / "Project 1 - DTB"
OUT_DIR = ROOT / "outputs" / "exploration"

REMAINING_PATH = DATA_DIR / "bioreaction_remaining.csv"
TEMPLATE_PATH = DATA_DIR / "bioreaction_template.csv"

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining = pd.read_csv(REMAINING_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    remaining["batch_id"] = remaining["batch_id"].astype(str)
    template["batch_id"] = template["batch_id"].astype(str)
    return remaining, template


def row_kind(frame: pd.DataFrame) -> pd.Series:
    """Label each remaining-file row by which event it records."""
    kind = pd.Series("other", index=frame.index, dtype="object")
    measured = frame[TARGETS].notna().all(axis=1)
    kind.loc[measured] = "measurement"
    kind.loc[frame["to_predict"]] = "to_predict"
    for feed in FEEDS:
        kind.loc[frame[feed].notna()] = feed
    return kind


def distribution_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Distribution of observed (non-null) values. Missing cells are not zeros."""
    rows = []
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        quantiles = values.quantile(QUANTILES)
        rows.append(
            {
                "column": column,
                "n_observed": int(values.shape[0]),
                "n_missing": int(frame[column].isna().sum()),
                "missing_pct": 100.0 * frame[column].isna().mean(),
                "mean": values.mean(),
                "std": values.std(),
                "skew": values.skew(),
                "min": values.min(),
                **{f"p{int(q * 100):02d}": quantiles.loc[q] for q in QUANTILES},
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def batch_summary(remaining: pd.DataFrame, holdout_ids: set[str]) -> pd.DataFrame:
    summary = remaining.groupby("batch_id", sort=False).agg(
        n_rows=("elapsed_time", "size"),
        t_min=("elapsed_time", "min"),
        t_max=("elapsed_time", "max"),
        n_measurements=("vcd", "count"),
        n_to_predict=("to_predict", "sum"),
        **{f"n_{feed}": (feed, "count") for feed in FEEDS},
    )
    summary["role"] = np.where(summary.index.isin(holdout_ids), "holdout", "train")
    summary["n_to_predict"] = summary["n_to_predict"].astype(int)
    return summary


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save_histogram(frame: pd.DataFrame, columns: list[str], path: Path, title: str) -> None:
    n_cols = len(columns)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, column in zip(axes.ravel(), columns):
        values = frame[column].dropna()
        ax.hist(values, bins=30, color="#3d5a80", edgecolor="white")
        ax.axvline(values.median(), color="#ee6c4d", linestyle="--", label="median")
        ax.set_title(f"{column}  n={len(values)}")
        ax.set_xlabel(column)
        ax.set_ylabel("rows")
        ax.legend(frameon=False)
    for ax in axes.ravel()[n_cols:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_trajectories(remaining: pd.DataFrame, holdout_ids: set[str], path: Path) -> None:
    measured = remaining[remaining[TARGETS].notna().all(axis=1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, column in zip(axes.ravel(), TARGETS):
        for batch_id, group in measured.groupby("batch_id"):
            holdout = batch_id in holdout_ids
            ax.plot(
                group["elapsed_time"],
                group[column],
                color="#ee6c4d" if holdout else "#98c1d9",
                alpha=1.0 if holdout else 0.45,
                linewidth=1.4 if holdout else 0.8,
            )
        ax.set_title(column)
        ax.set_xlabel("elapsed time (days)")
        ax.set_ylabel(column)
    fig.suptitle("Observed targets over time (holdout batches in orange, t = 0 only)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_correlation(measured: pd.DataFrame, path: Path) -> None:
    columns = TARGETS + ["volume", "elapsed_time"]
    corr = measured[columns].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Correlation on measurement rows")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_row_mix(remaining: pd.DataFrame, path: Path) -> None:
    counts = (
        remaining.groupby(["batch_id", "row_kind"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    order = ["measurement", "to_predict", *FEEDS]
    counts = counts.reindex(columns=[c for c in order if c in counts.columns])
    counts["_key"] = counts.index.str.replace("Batch", "", regex=False).astype(int)
    counts = counts.sort_values("_key").drop(columns="_key")

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.9, colormap="tab20")
    ax.set_xlabel("batch")
    ax.set_ylabel("rows")
    ax.set_title("Event mix per batch")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    remaining, template = load()
    remaining = remaining.copy()
    remaining["row_kind"] = row_kind(remaining)
    holdout_ids = set(template["batch_id"])
    measured = remaining[remaining["row_kind"] == "measurement"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Files")
    print(f"remaining: {REMAINING_PATH.relative_to(ROOT)}  {remaining.shape[0]} rows, {remaining.shape[1] - 1} columns")
    print(f"template:  {TEMPLATE_PATH.relative_to(ROOT)}  {template.shape[0]} rows (prediction placeholders)")
    print(f"batches in remaining: {remaining['batch_id'].nunique()}")
    print(f"holdout batches (also listed in the template): {len(holdout_ids)}")
    print("template target columns are all zeros; they are not observations.")

    print_section("Row types")
    print(
        "Each row records exactly one event. Measurement rows carry vcd, tcd, "
        "substrate, byproduct, and volume together. to_predict rows carry volume "
        "only. Each feed row carries a single feed column."
    )
    kind_counts = remaining["row_kind"].value_counts()
    print(kind_counts.to_string())
    unexpected = kind_counts.index.difference(
        ["measurement", "to_predict", *FEEDS]
    )
    if len(unexpected):
        print(f"unexpected row kinds: {sorted(unexpected)}")

    print_section("Observed distributions (nulls excluded)")
    observed_cols = ["elapsed_time", *TARGETS, "volume", *FEEDS]
    distributions = distribution_table(remaining, observed_cols)
    print(distributions.round(4).to_string())
    distributions.to_csv(OUT_DIR / "distributions.csv")

    print_section("Train vs holdout")
    batches = batch_summary(remaining, holdout_ids)
    role_counts = batches["role"].value_counts()
    print(role_counts.to_string())
    print()
    print(
        batches.groupby("role")[["n_rows", "n_measurements", "n_to_predict", "t_max"]]
        .agg(["min", "median", "max"])
        .round(3)
        .to_string()
    )
    print()
    print("Holdout batches keep the t = 0 measurement and hide the later 13 target rows.")
    print(
        "Sorted holdout ids:",
        ", ".join(sorted(holdout_ids, key=lambda name: int(name.replace("Batch", "")))),
    )
    batches.to_csv(OUT_DIR / "batch_summary.csv")

    print_section("Target spread across training batches")
    train_measured = measured[~measured["batch_id"].isin(holdout_ids)]
    per_batch = train_measured.groupby("batch_id")[TARGETS].mean()
    spread = per_batch.agg(["mean", "std", "min", "median", "max"]).T
    print("Mean of each target within a batch, then the spread of those means:")
    print(spread.round(3).to_string())
    per_batch.to_csv(OUT_DIR / "train_batch_target_means.csv")

    print_section("Correlation on measurement rows")
    corr = measured[TARGETS + ["volume", "elapsed_time"]].corr()
    print(corr.round(3).to_string())
    corr.to_csv(OUT_DIR / "measurement_correlation.csv")

    save_histogram(
        measured,
        TARGETS,
        OUT_DIR / "target_histograms.png",
        "Observed target distributions (measurement rows only)",
    )
    save_histogram(
        remaining,
        FEEDS,
        OUT_DIR / "feed_histograms.png",
        "Feed-event distributions (one column per event row)",
    )
    save_histogram(
        remaining,
        ["volume", "elapsed_time", "feed_b", "feed_c"],
        OUT_DIR / "volume_and_common_feeds.png",
        "Volume, elapsed time, and the two most frequent feeds",
    )
    save_trajectories(remaining, holdout_ids, OUT_DIR / "targets_over_time.png")
    save_correlation(measured, OUT_DIR / "correlation.png")
    save_row_mix(remaining, OUT_DIR / "event_mix_per_batch.png")

    print_section("Plots")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
