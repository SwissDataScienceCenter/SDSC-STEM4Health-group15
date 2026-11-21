"""Cross-validation on the frozen team split, and final fits.

Validation protocol (see docs/intro.md, "Validation strategy"; split in references/splits.csv):
  * dev (39 batches): 5-fold CV using the frozen `fold` column -> all model selection happens here;
  * holdout (10 batches): final reality check only, never used to choose between models
    (see bioreactor_decoding.plots / evaluate_holdout);
  * test (12 batches): no labels; predicted for the submission.
Every row of a batch is in one fold. A validation batch is predicted from x(0), sampling times,
volume and feeds only, exactly like a test batch. Estimators are cloned per fold, so anything
fitted is fitted on the training fold only.

Two final models are saved:
  * fitted on dev only      -> for the holdout reality check;
  * fitted on dev + holdout -> for the submission (uses all 49 labelled batches).
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.base import clone
import typer
from xgboost import XGBRegressor

from bioreactor_decoding.config import MODELS_DIR, REPORTS_DIR
from bioreactor_decoding.dataset import RAW_PATH, TARGETS, load_tables
from bioreactor_decoding.features import build_features, feature_columns
from bioreactor_decoding.modeling.models import (
    ConstantNullModel,
    FoldChangeRegressor,
    MedianFoldChange,
    PerDayNullModel,
    challenge_score,
)

app = typer.Typer()


def _xgb(random_state: int = 0) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=random_state,
    )


def make_xgb(random_state: int = 0) -> FoldChangeRegressor:
    """XGBoost on log fold change vs. t = 0 (positive by construction, relative-error loss)."""
    return FoldChangeRegressor(_xgb(random_state))


MODELS = {
    "null_constant": ConstantNullModel,
    "null_per_day": PerDayNullModel,
    "median_fold_change": MedianFoldChange,
    "xgb": make_xgb,
}


def labelled_data(use_future_feeds: bool = True, raw_path: Path = RAW_PATH):
    """Features X, targets Y and split info (batch_id, role, fold) for dev + holdout rows."""
    samples, feeds = load_tables(raw_path)
    feats = build_features(samples, feeds, use_future_feeds)
    split = samples.loc[feats.index, ["batch_id", "elapsed_time", "role", "fold"]]
    keep = split["role"].isin(["dev", "holdout"])
    X = feats.loc[keep, feature_columns(feats)]
    Y = samples.loc[X.index, TARGETS]
    return X, Y, split.loc[keep]


def cross_validate(models: dict, X, Y, split) -> pd.DataFrame:
    """Score each model on the frozen dev folds. Holdout rows are never touched."""
    dev = split["role"] == "dev"
    rows = []
    for fold in sorted(split.loc[dev, "fold"].unique()):
        tr, va = dev & (split["fold"] != fold), dev & (split["fold"] == fold)
        assert set(split.loc[tr, "batch_id"]).isdisjoint(split.loc[va, "batch_id"])
        for name, model in models.items():
            m = clone(model).fit(X[tr], Y[tr])
            rows.append({"model": name, "fold": fold, **challenge_score(Y[va], m.predict(X[va]))})
    return pd.DataFrame(rows)


@app.command()
def main(
    model: str = "xgb",
    use_future_feeds: bool = True,
    cv_path: Path = REPORTS_DIR / "cv_scores.csv",
):
    X, Y, split = labelled_data(use_future_feeds)
    dev = split["role"] == "dev"
    logger.info(
        f"dev: {split.loc[dev, 'batch_id'].nunique()} batches, "
        f"holdout: {split.loc[~dev, 'batch_id'].nunique()} batches (not used in CV)"
    )

    cv = cross_validate({name: make() for name, make in MODELS.items()}, X, Y, split)
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(cv_path, index=False)
    summary = cv.groupby("model", sort=False)[[*TARGETS, "score"]].agg(["mean", "std"]).round(2)
    logger.info(f"CV MAPE (%) on frozen dev folds:\n{summary.to_string()}")

    for scope, rows in {"dev": dev, "all": split["role"].isin(["dev", "holdout"])}.items():
        fitted = MODELS[model]().fit(X[rows], Y[rows])
        path = MODELS_DIR / f"{model}_{scope}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": fitted, "use_future_feeds": use_future_feeds, "features": list(X.columns)},
            path,
        )
        logger.success(f"{model} fitted on {scope} batches -> {path}")


if __name__ == "__main__":
    app()
