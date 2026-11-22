"""Holdout traces (actual and predicted) of the models shown in the presentation videos.

    python presentation/export_traces.py
-> presentation/data/<model>.csv: batch_id, t, <target>_actual, <target>_pred, one row per
   sample of the 10 holdout batches, day 0 included (the given starting values).

Predictions come from reports/holdout/<model>.csv (the model fitted on the dev batches only).
The files contain measured values, so presentation/data/ is not in git.
"""

from pathlib import Path

import pandas as pd

from bioreactor_decoding.dataset import TARGETS, load_tables
from bioreactor_decoding.plots import load_holdout

MODELS = ["mechanistic_lite", "ml_sensors", "hybrid_blend"]
OUT = Path(__file__).resolve().parent / "data"


def traces(model: str) -> pd.DataFrame:
    actual, predicted, _ = load_holdout(model)
    df = actual.rename(columns={c: f"{c}_actual" for c in TARGETS}).join(
        predicted[TARGETS].rename(columns={c: f"{c}_pred" for c in TARGETS})
    )
    samples, _ = load_tables()
    day0 = samples.loc[(samples["role"] == "holdout") & (samples["elapsed_time"] == 0)]
    day0 = day0[["batch_id", "elapsed_time", *TARGETS]]
    for c in TARGETS:  # both traces start from the given day-0 value
        day0[f"{c}_actual"] = day0[c]
        day0[f"{c}_pred"] = day0[c]
    df = pd.concat([day0.drop(columns=TARGETS), df], ignore_index=True)
    return df.rename(columns={"elapsed_time": "t"}).sort_values(["batch_id", "t"])


def feeds() -> pd.DataFrame:
    """Feed events of the holdout batches: batch_id, t, feed, amount, and rel = amount
    relative to the largest dose of that feed in the data (scales differ ~500-fold)."""
    samples, feed_events = load_tables()
    largest = feed_events.groupby("feed")["amount"].max()
    holdout = set(samples.loc[samples["role"] == "holdout", "batch_id"])
    f = feed_events.loc[feed_events["batch_id"].isin(holdout)].copy()
    f["rel"] = f["amount"] / f["feed"].map(largest)
    return f.rename(columns={"elapsed_time": "t"})[["batch_id", "t", "feed", "amount", "rel"]]


def main():
    OUT.mkdir(exist_ok=True)
    for m in MODELS:
        traces(m).to_csv(OUT / f"{m}.csv", index=False)
        print("wrote", OUT / f"{m}.csv")
    (OUT / "feeds").mkdir(exist_ok=True)
    feeds().to_csv(OUT / "feeds" / "feeds.csv", index=False)
    print("wrote", OUT / "feeds" / "feeds.csv")


if __name__ == "__main__":
    main()
