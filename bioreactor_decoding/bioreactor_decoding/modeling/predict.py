"""Predict the test batches and fill the submission template without changing its format.

python -m bioreactor_decoding.modeling.predict --model hybrid_v3
-> data/processed/submission_hybrid_v3.csv
"""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import typer

from bioreactor_decoding.config import MODELS_DIR, PROCESSED_DATA_DIR
from bioreactor_decoding.dataset import (
    RAW_PATH,
    TARGETS,
    TEMPLATE_PATH,
    load_tables,
    load_template,
)
from bioreactor_decoding.features import build_features

app = typer.Typer()


def fill_template(template, pred_rows):
    """Align predictions to the template on batch_id + sorted time (times differ by <1 ms)."""
    tmpl = template.copy()
    tmpl[TARGETS] = tmpl[TARGETS].astype(float)  # template ships integer zeros
    t_float = tmpl["elapsed_time"].astype(float)
    order = tmpl.assign(_t=t_float).sort_values(["batch_id", "_t"]).index
    pred = pred_rows.sort_values(["batch_id", "elapsed_time"])
    if len(order) != len(pred) or not (
        (tmpl.loc[order, "batch_id"].to_numpy() == pred["batch_id"].to_numpy()).all()
        and np.allclose(t_float.loc[order].to_numpy(), pred["elapsed_time"].to_numpy(), atol=1e-6)
    ):
        raise ValueError("Predictions do not align with the template rows")
    tmpl.loc[order, TARGETS] = pred[TARGETS].to_numpy()
    return tmpl


@app.command()
def main(
    model: str = "xgb_raw",
    model_path: Path | None = None,
    raw_path: Path = RAW_PATH,
    template_path: Path = TEMPLATE_PATH,
    output_path: Path | None = None,
):
    """Fill the template with MODEL's predictions (models/MODEL_all.pkl, all 49 labelled
    batches) -> data/processed/submission_MODEL.csv."""
    model_path = model_path or MODELS_DIR / f"{model}_all.pkl"
    output_path = output_path or PROCESSED_DATA_DIR / f"submission_{model}.csv"
    bundle = joblib.load(model_path)
    samples, feeds = load_tables(raw_path)
    feats = build_features(samples, feeds, bundle["use_future_feeds"])
    feats = feats.loc[samples.loc[feats.index, "to_predict"]]

    cols = bundle["features"]
    if "batch_id" not in cols:  # bundles saved before X carried batch_id
        cols = ["batch_id", *cols]
    pred = bundle["model"].predict(feats[cols])
    pred[["batch_id", "elapsed_time"]] = feats[["batch_id", "elapsed_time"]]

    submission = fill_template(load_template(template_path), pred)
    if (submission[TARGETS] <= 0).any().any():
        raise ValueError("Non-positive predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.success(f"{len(submission)} rows -> {output_path}")


if __name__ == "__main__":
    app()
