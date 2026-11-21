"""Holdout reality check from a saved model: no refitting.

Loads models/{model}_dev.pkl (fitted on dev batches only by `train`), predicts the holdout
batches and writes
  * reports/metrics/{model}.json  - holdout MAPE per variable and the average (DVC metrics);
  * reports/holdout/{model}.csv   - holdout predictions (no measured values: those are challenge
                                    data and stay out of git; plots.py joins them from data/raw).
The holdout is only ever *scored* here: nothing is fitted, and model selection uses CV scores.
"""

import json
from pathlib import Path
import time

import joblib
from loguru import logger
import pandas as pd
import typer

from bioreactor_decoding.config import HOLDOUT_DIR, METRICS_DIR, MODELS_DIR
from bioreactor_decoding.dataset import TARGETS
from bioreactor_decoding.modeling.models import challenge_score
from bioreactor_decoding.modeling.train import labelled_data

app = typer.Typer()


def holdout_predictions(bundle: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predictions of a saved dev-fitted bundle on the holdout rows, and their scores."""
    # The model never sees Y here; Y is only used to score the predictions.
    X, Y, split = labelled_data(bundle["use_future_feeds"])
    holdout = split["role"] == "holdout"
    if set(split.loc[holdout, "batch_id"]) & set(bundle.get("fitted_batches", [])):
        raise ValueError("Model was fitted on holdout batches; use the *_dev.pkl bundle")
    cols = (
        bundle["features"]
        if "batch_id" in bundle["features"]
        else ["batch_id", *bundle["features"]]
    )
    t0 = time.perf_counter()
    pred = bundle["model"].predict(X.loc[holdout, cols])
    seconds = time.perf_counter() - t0
    scores = challenge_score(Y[holdout], pred)
    out = split.loc[holdout, ["batch_id", "elapsed_time"]].copy()
    out[TARGETS] = pred[TARGETS].to_numpy()
    return out, {**scores, "predict_seconds": seconds}


@app.command()
def main(
    model: str = "xgb_raw",
    metrics_dir: Path = METRICS_DIR,
    predictions_dir: Path = HOLDOUT_DIR,
):
    bundle = joblib.load(MODELS_DIR / f"{model}_dev.pkl")
    out, scores = holdout_predictions(bundle)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    seconds = scores.pop("predict_seconds")
    n_batches = out["batch_id"].nunique()
    metrics = {
        "holdout_mape": {k: round(v, 4) for k, v in scores.items()},
        "footprint": {
            "predict_seconds": round(seconds, 3),
            "predict_seconds_per_batch": round(seconds / n_batches, 4),
        },
    }
    (metrics_dir / f"{model}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    out.to_csv(predictions_dir / f"{model}.csv", index=False)
    logger.success(
        f"{model}: holdout MAPE {scores['score']:.2f}% -> {metrics_dir / f'{model}.json'}"
    )


if __name__ == "__main__":
    app()
