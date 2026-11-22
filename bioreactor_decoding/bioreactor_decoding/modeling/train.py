"""Cross-validation on the frozen team split, and final fits, for one model.

Validation protocol (see docs/intro.md, "Validation strategy"; split in references/splits.csv):
  * dev (39 batches): 5-fold CV using the frozen `fold` column -> all model selection happens here;
  * holdout (10 batches): final reality check only, never used to choose between models
    (scored by bioreactor_decoding.modeling.evaluate);
  * test (12 batches): no labels; predicted for the submission.
Every row of a batch is in one fold. A validation batch is predicted from x(0), sampling times,
volume and feeds only, exactly like a test batch. Estimators are cloned per fold, so anything
fitted is fitted on the training fold only.

Outputs of `main --model NAME`:
  * reports/cv/NAME.csv            CV MAPE per fold and variable;
  * reports/metrics/cv_NAME.json   mean CV MAPE (DVC metrics);
  * reports/metrics/footprint_NAME.json   time, memory and size of training (DVC metrics);
  * models/NAME_dev.pkl            fitted on dev only      -> holdout reality check;
  * models/NAME_all.pkl            fitted on dev + holdout -> submission (all 49 labelled batches).
Hyperparameters come from params.yaml.
"""

import json
import os
from pathlib import Path
import resource
import time

import joblib
from loguru import logger
import pandas as pd
from sklearn.base import clone
import typer
from xgboost import XGBRegressor

from bioreactor_decoding.config import CV_DIR, METRICS_DIR, MODELS_DIR, load_params
from bioreactor_decoding.dataset import RAW_PATH, TARGETS, load_tables
from bioreactor_decoding.features import build_features, feature_columns
from bioreactor_decoding.modeling.hybrid import HybridODEConfig
from bioreactor_decoding.modeling.hybrid_v3 import HybridV3
from bioreactor_decoding.modeling.lite import LiteODEConfig
from bioreactor_decoding.modeling.models import (
    AverageEnsemble,
    ConstantNullModel,
    FoldChangeRegressor,
    MedianFoldChange,
    PerDayNullModel,
    PerTargetRegressor,
    challenge_score,
)

app = typer.Typer()


def _seed(random_state: int | None) -> int:
    return load_params().get("seed", 0) if random_state is None else random_state


def _xgb(random_state: int | None = None) -> XGBRegressor:
    return XGBRegressor(
        **load_params().get("xgb", {}),
        objective="reg:squarederror",
        n_jobs=4,
        random_state=_seed(random_state),
    )


def make_xgb_raw(random_state: int | None = None) -> PerTargetRegressor:
    """Straight-up XGBoost: raw features -> raw target values, one model per target."""
    return PerTargetRegressor(_xgb(random_state))


def make_xgb(random_state: int | None = None) -> FoldChangeRegressor:
    """XGBoost on log fold change vs. t = 0 (positive by construction, relative-error loss)."""
    return FoldChangeRegressor(_xgb(random_state))


def make_null_xgb_ensemble(random_state: int | None = None) -> AverageEnsemble:
    """50/50 average of the per-day null model and raw XGBoost."""
    return AverageEnsemble([PerDayNullModel(), make_xgb_raw(random_state)])


def _hybrid(section: str, stage1_cls, random_state):
    p = dict(load_params().get(section, {}))
    stage1 = p.pop("stage1", {})
    if "outputs" in stage1:
        stage1 = {**stage1, "outputs": tuple(stage1["outputs"])}
    return HybridV3(**p, stage1=stage1_cls(**stage1), random_state=_seed(random_state))


def make_hybrid_v3(random_state: int | None = None) -> HybridV3:
    """The team's hybrid v3 (hybrid ODE + soft sensors + 50/50 blend with boosting)."""
    return _hybrid("hybrid_v3", HybridODEConfig, random_state)


def make_hybrid_lite(random_state: int | None = None) -> HybridV3:
    """Hybrid v3's stage 2 on a light stage 1: a mechanistic ODE without neural nets."""
    return _hybrid("hybrid_lite", LiteODEConfig, random_state)


def make_hybrid_blend(random_state: int | None = None) -> AverageEnsemble:
    """Average of hybrid_v3 and hybrid_lite: same stage 2, different stage 1."""
    return AverageEnsemble([make_hybrid_v3(random_state), make_hybrid_lite(random_state)])


MODELS = {
    "null_constant": ConstantNullModel,
    "null_per_day": PerDayNullModel,
    "median_fold_change": MedianFoldChange,
    "xgb_raw": make_xgb_raw,
    "xgb_log_fold_change": make_xgb,
    "null_xgb_ensemble": make_null_xgb_ensemble,
    "hybrid_v3": make_hybrid_v3,
    "hybrid_lite": make_hybrid_lite,
    "hybrid_blend": make_hybrid_blend,
}


def labelled_data(use_future_feeds: bool = True, raw_path: Path = RAW_PATH):
    """Features X, targets Y and split info (batch_id, role, fold) for dev + holdout rows.

    X carries `batch_id` next to the features: models that need a batch's whole feed and volume
    log (hybrid_v3) look it up by batch; the other estimators drop it before fitting.
    """
    samples, feeds = load_tables(raw_path)
    feats = build_features(samples, feeds, use_future_feeds)
    split = samples.loc[feats.index, ["batch_id", "elapsed_time", "role", "fold"]]
    keep = split["role"].isin(["dev", "holdout"])
    X = feats.loc[keep, ["batch_id", *feature_columns(feats)]]
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


def holdout_check(model, X, Y, split) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit a fresh clone of `model` on dev batches only and score it on the holdout batches.

    The model never sees holdout rows (features or labels): it is fitted on the dev rows
    alone, and features are built per batch with no cross-batch statistics.
    """
    dev, holdout = split["role"] == "dev", split["role"] == "holdout"
    if set(split.loc[dev, "batch_id"]) & set(split.loc[holdout, "batch_id"]):
        raise ValueError("A batch is in both dev and holdout")
    fitted = clone(model).fit(X[dev], Y[dev])
    pred = fitted.predict(X[holdout])
    return pred, challenge_score(Y[holdout], pred)


@app.command()
def main(model: str = "xgb_raw", use_future_feeds: bool | None = None):
    """Cross-validate MODEL on the dev folds, then fit it on dev and on dev + holdout."""
    if use_future_feeds is None:
        use_future_feeds = load_params().get("features", {}).get("use_future_feeds", True)
    X, Y, split = labelled_data(use_future_feeds)
    dev = split["role"] == "dev"
    logger.info(
        f"{model}: dev {split.loc[dev, 'batch_id'].nunique()} batches, "
        f"holdout {split.loc[~dev, 'batch_id'].nunique()} batches (not used in CV)"
    )

    t0 = time.perf_counter()
    cv = cross_validate({model: MODELS[model]()}, X, Y, split)
    cv_seconds = time.perf_counter() - t0
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(CV_DIR / f"{model}.csv", index=False)
    mean = cv[[*TARGETS, "score"]].mean()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {"cv_mape": {k: round(float(v), 4) for k, v in mean.items()}}
    (METRICS_DIR / f"cv_{model}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info(f"{model}: CV MAPE {mean['score']:.2f}% (sd over folds {cv['score'].std():.2f})")

    footprint = {"cv_seconds": round(cv_seconds, 1), "cpu_count": os.cpu_count()}
    for scope, rows in {"dev": dev, "all": split["role"].isin(["dev", "holdout"])}.items():
        t0 = time.perf_counter()
        fitted = MODELS[model]().fit(X[rows], Y[rows])
        footprint[f"fit_{scope}_seconds"] = round(time.perf_counter() - t0, 1)
        if hasattr(fitted, "stage1_from_cache_"):  # a cached stage 1 makes the fit look cheap
            footprint[f"fit_{scope}_stage1_from_cache"] = bool(fitted.stage1_from_cache_)
        path = MODELS_DIR / f"{model}_{scope}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": fitted,
                "use_future_feeds": use_future_feeds,
                "features": list(X.columns),
                "fitted_batches": sorted(split.loc[rows, "batch_id"].unique()),
            },
            path,
        )
        footprint[f"model_{scope}_mb"] = round(path.stat().st_size / 1e6, 3)
        logger.success(f"{model} fitted on {scope} batches -> {path}")

    # Peak resident memory of this process and of finished worker processes (ru_maxrss: KiB).
    footprint["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    footprint["peak_rss_children_mb"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1
    )
    (METRICS_DIR / f"footprint_{model}.json").write_text(
        json.dumps({"footprint": footprint}, indent=2) + "\n"
    )
    logger.info(f"{model}: footprint {footprint}")


if __name__ == "__main__":
    app()
