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

Videos: [presentation/README.md](presentation/README.md)

Docs: [problem and data](docs/intro.md) · [pipeline, models, MLOps rules](docs/mlops.md) · [experiment log](docs/experiments.md)

## Model registry

Model versions are git tags managed with [GTO](https://dvc.org/doc/gto), DVC's model registry
(list on GitHub: [Tags](https://github.com/SwissDataScienceCenter/SDSC-STEM4Health-group15/tags)).
A tag such as `bioreactor_decoding/dvc.yaml=hybrid_v7@v1.0.0` marks the commit whose code,
`params.yaml` and `dvc.lock` produced that version, so `git checkout <tag> && make pipeline`
rebuilds it. The models are declared under `artifacts:` in [dvc.yaml](dvc.yaml).

```bash
# from the git root, one level above this project
gto show                                            # every registered model, latest version
gto describe bioreactor_decoding/dvc.yaml:hybrid_v7 # commit, path, description
```

| Model | Latest | What it is |
|---|---|---|
| `hybrid_v7` | v1.0.0 | Team's leaderboard model (6.752 MAPE); run from a pinned commit, the registered file is the submitted one |
| `hybrid_blend` | v1.0.0 | Average of `hybrid_v3` and `hybrid_lite`; best in-house protocol (CV 7.95, holdout 6.99) |
| `hybrid_v3`, `hybrid_lite` | v1.1.0 | Mechanistic ODE + learned correction, full and light stage 1 |
| `xgb_raw`, `xgb_log_fold_change`, `null_xgb_ensemble` | v1.0.0 | XGBoost baselines (the ensemble averages `xgb_raw` with `null_per_day`) |
| `null_constant`, `null_per_day` | v1.0.0 | Null models: the bar every model must beat |

Scores of all models, registered or not (`ml_sensors`, `mechanistic_lite`, `hybrid_blend_ml`):
`make metrics`. Versioning rules (when to bump major/minor/patch, what earns a version):
[docs/mlops.md](docs/mlops.md#versioning-models).

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
        ├── external_v7.py  <- runs Tobias' hybrid v7 at a pinned commit (DVC stage hybrid_v7)
        ├── ml_sensors.py   <- ridge + XGBoost with soft sensors (ml_sensors)
        ├── lite.py         <- light stage 1 for hybrid_lite: mechanistic ODE, staged fit
        ├── train.py        <- CV on the dev folds and final fits, per model
        ├── evaluate.py     <- holdout check from the saved dev model
        ├── compare.py      <- per-fold CV comparison: the decision rule for experiments
        └── predict.py      <- fill the submission template
```

--------

