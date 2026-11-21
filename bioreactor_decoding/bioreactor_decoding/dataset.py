"""Load the raw long-format CSV and split it into a sample table and a feed-event table.

Raw rows are either a *sample* (``volume`` set, measurements set unless ``to_predict``)
or a single *feed event* (exactly one ``feed_*`` column set).
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from bioreactor_decoding.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

RAW_PATH = RAW_DATA_DIR / "bioreaction_remaining.csv"
TEMPLATE_PATH = RAW_DATA_DIR / "bioreaction_template.csv"
# Frozen team split (from branch 3_ML_models, outputs/splits.csv): one row per batch with
# role in {dev, holdout, test} and fold 0-4 for dev batches (-1 otherwise).
#   dev     -> model selection with 5-fold CV (train/val)
#   holdout -> final reality check only; never used to choose between models
#   test    -> the 12 batches graded by the organisers (no labels)
SPLITS_PATH = PROJ_ROOT / "references" / "splits.csv"

app = typer.Typer()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return raw.sort_values(["batch_id", "elapsed_time"], kind="stable").reset_index(drop=True)


def test_batch_ids(raw: pd.DataFrame) -> set[str]:
    """Batches with any row to predict; these have measurements only at t = 0."""
    return set(raw.loc[raw["to_predict"], "batch_id"])


def split_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (samples, feeds).

    samples: one row per sampling time with batch_id, elapsed_time, volume, targets,
             to_predict, is_test.
    feeds:   long format with batch_id, elapsed_time, feed, amount.
    """
    is_sample = raw["volume"].notna()
    n_feed_cols = raw[FEEDS].notna().sum(axis=1)
    if (is_sample & (n_feed_cols > 0)).any() or ((~is_sample) & (n_feed_cols != 1)).any():
        raise ValueError("Unexpected row type: rows must be a sample or exactly one feed event")

    test_ids = test_batch_ids(raw)
    samples = raw.loc[is_sample, ["batch_id", "elapsed_time", "volume", *TARGETS, "to_predict"]]
    samples = samples.assign(is_test=samples["batch_id"].isin(test_ids)).reset_index(drop=True)

    feeds = (
        raw.loc[~is_sample, ["batch_id", "elapsed_time", *FEEDS]]
        .melt(id_vars=["batch_id", "elapsed_time"], var_name="feed", value_name="amount")
        .dropna(subset=["amount"])
        .sort_values(["batch_id", "elapsed_time"], kind="stable")
        .reset_index(drop=True)
    )
    return samples, feeds


def load_splits(path: Path = SPLITS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)[["batch_id", "role", "fold"]]


def attach_split(samples: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Add each batch's role and fold; check the split agrees with the data's test batches."""
    out = samples.merge(splits, on="batch_id", how="left", validate="many_to_one")
    out.index = samples.index
    if out["role"].isna().any():
        missing = sorted(out.loc[out["role"].isna(), "batch_id"].unique())
        raise ValueError(f"Batches missing from splits: {missing}")
    if not (out["is_test"] == (out["role"] == "test")).all():
        raise ValueError("Split 'test' role does not match the batches with to_predict rows")
    return out


def load_tables(
    path: Path = RAW_PATH, splits_path: Path | None = SPLITS_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, feeds); samples carry role/fold columns when a splits file is given."""
    samples, feeds = split_tables(load_raw(path))
    if splits_path is not None:
        samples = attach_split(samples, load_splits(splits_path))
    return samples, feeds


def load_template(path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    """Template with elapsed_time kept as the original string so it is written back unchanged."""
    return pd.read_csv(path, dtype={"elapsed_time": str})


@app.command()
def main(
    input_path: Path = RAW_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
):
    samples, feeds = load_tables(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "samples.csv", index=False)
    feeds.to_csv(output_dir / "feeds.csv", index=False)
    n_test = samples.loc[samples["is_test"], "batch_id"].nunique()
    logger.success(
        f"{samples['batch_id'].nunique()} batches ({n_test} test), "
        f"{len(samples)} sample rows, {len(feeds)} feed events -> {output_dir}"
    )


if __name__ == "__main__":
    app()
