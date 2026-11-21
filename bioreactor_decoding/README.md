# bioreactor_decoding

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Develop a model that captures the key dynamics of batch-fed culture runs in a bioreactor, learns from historical dat, and proves its predictive power on unseen runs.

## Quick start

```bash
make create_environment && make requirements   # .venv from uv.lock
make test                                      # leakage + holdout-isolation tests
make pipeline                                  # dvc repro: CV, fits, holdout check, figures, submissions
make metrics                                   # CV and holdout MAPE of every model
```

Docs: [problem and data](docs/intro.md) · [pipeline, models, MLOps rules](docs/mlops.md)

## Project Organization

```
├── Makefile           <- make test / pipeline / metrics / lint
├── dvc.yaml           <- the pipeline: train -> evaluate -> submission, figures (docs/mlops.md)
├── dvc.lock           <- hashes of every stage's inputs and outputs (what produced each result)
├── params.yaml        <- hyperparameters and the list of models the pipeline runs
├── pyproject.toml     <- package metadata, dependencies, ruff config
├── uv.lock            <- exact dependency versions
├── data
│   ├── raw            <- challenge data (not in git; versioned by DVC via data/raw.dvc)
│   └── processed      <- submission_<model>.csv (pipeline output)
├── docs               <- intro.md (problem, data, validation), mlops.md (pipeline, models)
├── models             <- <model>_dev.pkl / <model>_all.pkl, cache/ (not in git)
├── references         <- splits.csv: the frozen dev / holdout / test split
├── reports
│   ├── cv             <- CV MAPE per fold, per model
│   ├── metrics        <- CV and holdout MAPE (DVC metrics)
│   ├── holdout        <- holdout predictions (measured values stay in data/raw)
│   └── figures        <- holdout figures
├── tests              <- leakage, holdout-isolation and porting tests
└── bioreactor_decoding   <- source code
    ├── config.py      <- paths, params.yaml loader
    ├── dataset.py     <- load raw data, split into samples / feeds, attach the frozen split
    ├── features.py    <- leak-free per-batch features
    ├── plots.py       <- holdout figures from saved predictions
    └── modeling
        ├── models.py       <- metric, null models, XGBoost wrappers, ensembles
        ├── mechanistic.py  <- Sartorius-style ODE (numba) and its fitting
        ├── hybrid.py       <- hybrid ODE: neural nets correcting growth and lactate rates
        ├── hybrid_v3.py    <- hybrid v3: soft sensors, two branches, blend (one estimator)
        ├── lite.py         <- light stage 1 for hybrid_lite: mechanistic ODE, staged fit
        ├── train.py        <- CV on the dev folds and final fits, per model
        ├── evaluate.py     <- holdout check from the saved dev model
        └── predict.py      <- fill the submission template
```

--------

