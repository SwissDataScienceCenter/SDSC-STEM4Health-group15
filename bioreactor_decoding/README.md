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
| **`option_a`** | v1.0.0, stage **`prod`** | **The final model**: one sensor-assimilated Monod core with two learned views (leaderboard 6.81); ported from branch `hybrid_v3_tr`, reproduces its submission exactly |
| `hybrid_v7` | v1.0.0 | Team's best leaderboard score (6.752 MAPE), 4-core stack; run from a pinned commit, the registered file is the submitted one |
| `hybrid_blend` | v1.0.0 | Average of `hybrid_v3` and `hybrid_lite`; best in-house protocol (CV 7.95, holdout 6.99) |
| `hybrid_v3`, `hybrid_lite` | v1.1.0 | Mechanistic ODE + learned correction, full and light stage 1 |
| `xgb_raw`, `xgb_log_fold_change`, `null_xgb_ensemble` | v1.0.0 | XGBoost baselines (the ensemble averages `xgb_raw` with `null_per_day`) |
| `null_constant`, `null_per_day` | v1.0.0 | Null models: the bar every model must beat |

Scores of all models, registered or not (`ml_sensors`, `mechanistic_lite`, `hybrid_blend_ml`):
`make metrics`. Versioning rules (when to bump major/minor/patch, what earns a version):
[docs/mlops.md](docs/mlops.md#versioning-models).

### Dependency on branch `hybrid_v3_tr`

Two models still need branch `hybrid_v3_tr` to rerun (their registered files and results are
in this repository either way). **Keep that branch on GitHub.**

| Stage | Needs from `hybrid_v3_tr` | How |
|---|---|---|
| `hybrid_v7`, `anomaly@hybrid_v7` | Tobias' code at commit `18810698` (`hybrid_v7.ref` in `params.yaml`) | the stage checks it out as a git worktree in `external/` |
| `option_a`, `anomaly@option_a` | 8 cached core and member outputs at commit `863e751`, which its code reads (they cannot be rebuilt here) | `make external-data` copies them into `data/external/` |

Every other model depends only on the raw data. On a fresh clone: copy the challenge data
into `data/raw/`, run `make external-data`, then `make pipeline`.

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
│   ├── external       <- Option A's cached core and member outputs from hybrid_v3_tr (DVC)
│   └── processed      <- submission_<model>.csv (pipeline output)
├── docs               <- intro.md (problem, data, validation), mlops.md (pipeline, models)
├── models             <- <model>_dev.pkl / <model>_all.pkl, cache/ (not in git)
├── notebooks          <- anomaly_v7.ipynb: narrative of the anomaly analysis (code in anomaly.py)
├── references         <- splits.csv: the frozen dev / holdout / test split
├── reports
│   ├── cv             <- CV MAPE per fold, per model
│   ├── metrics        <- CV and holdout MAPE (DVC metrics)
│   ├── holdout        <- holdout predictions (measured values stay in data/raw)
│   ├── anomaly        <- anomaly screen of the labelled batches (scores, flags, rates)
│   └── figures        <- holdout figures
├── tests              <- leakage, holdout-isolation and porting tests
└── bioreactor_decoding   <- source code
    ├── config.py      <- paths, params.yaml loader
    ├── dataset.py     <- load raw data, split into samples / feeds, attach the frozen split
    ├── features.py    <- leak-free per-batch features
    ├── plots.py       <- holdout figures from saved predictions
    ├── anomaly.py     <- anomaly detection on hybrid v7's residuals (DVC stage anomaly)
    └── modeling
        ├── models.py       <- metric, null models, XGBoost wrappers, ensembles
        ├── mechanistic.py  <- Sartorius-style ODE (numba) and its fitting
        ├── hybrid.py       <- hybrid ODE: neural nets correcting growth and lactate rates
        ├── hybrid_v3.py    <- hybrid v3: soft sensors, two branches, blend (one estimator)
        ├── external.py     <- runs hybrid v7's code from its branch at a pinned commit
        ├── external_v7.py  <- runs Tobias' hybrid v7 at a pinned commit (DVC stage hybrid_v7)
        ├── option_a.py     <- Option A, the final model (DVC stage option_a)
        ├── option_a_src/   <- Option A's code, ported unchanged from branch hybrid_v3_tr
        ├── ml_sensors.py   <- ridge + XGBoost with soft sensors (ml_sensors)
        ├── lite.py         <- light stage 1 for hybrid_lite: mechanistic ODE, staged fit
        ├── train.py        <- CV on the dev folds and final fits, per model
        ├── evaluate.py     <- holdout check from the saved dev model
        ├── compare.py      <- per-fold CV comparison: the decision rule for experiments
        └── predict.py      <- fill the submission template
```

--------

