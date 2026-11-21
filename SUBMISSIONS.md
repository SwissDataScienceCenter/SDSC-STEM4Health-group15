# Submissions log

One entry per leaderboard submission. Fewer submissions rank higher on ties, so we submit rarely and only from committed code.
Metric: MAPE (%) per variable (vcd, tcd, substrate, byproduct); final = average of the 4, lower is better.

## ML-1: ridge + XGBoost blend (not submitted yet)

| | |
|---|---|
| Date prepared | 2026-11-21 |
| Status | **ready, NOT submitted** |
| Code | `notebooks/05_final_model.ipynb`, commit `31564de` (branch `3_ML_models`) |
| File | `outputs/ml/submission_ml_blend.csv` (156 rows, no NaN, all values > 0, template format; regenerate by running the notebook, not committed) |
| Leaderboard score | - (not submitted) |

**Model.** Geometric average of ridge and XGBoost (Philip's settings). Label = `log(value / per-day mean curve)`, prediction = mean curve x exp(model output). Features: day, day-0 values, volume0, and per feed the amount added so far, in the last 24 h, and right after the sample (`next_*`). Fit on all 49 labelled batches.

**Validation (frozen split `outputs/splits.csv`).**

| | vcd | tcd | substrate | byproduct | average |
|---|---|---|---|---|---|
| 5-fold CV on dev (39 batches), mean +/- std over folds | 7.17 | 7.07 | 11.15 | 11.57 | **9.24 +/- 0.87** |
| Same CV, per-day mean curve only | 8.63 | 8.66 | 11.97 | 12.90 | 10.54 +/- 0.89 |
| Holdout (10 batches), fit on dev, opened once | 6.25 | 6.25 | 8.58 | 10.35 | **7.86** |
| Holdout, per-day mean curve only | 9.21 | 9.24 | 10.63 | 8.39 | 9.37 |

**Notes / caveats.**
- The holdout was opened once, for this model only; no changes were made afterwards.
- On the holdout, byproduct is *worse* than the mean curve (10.35 vs 8.39) while the other three are better. Not investigated (10 batches only; Batch34 has a suspicious last-day byproduct value).
- `next_*` features (feed dosed right after each sample) reflect the operator's reaction to the measurement. Allowed here because test feed schedules are given, but not available at day 0 in real life. Gain in CV was small (9.40 -> 9.22 for the blend).
- Batch54 (lowest day-0 substrate) was checked by eye: predictions follow the normal shape and stay within the range of the training batches.
