# v6: 10 more ideas on top of the v5 stack

Every score uses the same 7-fold cross-validation over whole batches, with stack weights fitted on the other folds only. The full log is in `trials.csv`.

**Result: CV 7.366 → 7.244 (VCD 6.81, TCD 6.74, glucose 7.25, lactate 8.18).** The CV target of 7 was not reached. The v5 submission scored 7.137 on the leaderboard against 7.366 in CV. The v6 holdout predictions are in `outputs/stack_v6/holdout_predictions.csv` and have not been submitted.

| # | Idea | Score | Kept |
|---|---|---|---|
| 1 | Weight search that can drop members completely (the old search kept every weight ≥ 0.04) | 7.357 | yes |
| 2 | Hybrid and data-driven branches of every core as separate members | 7.414 | no |
| 3 | Finer time segments for the weights (1, 3 or 4 segments) | 7.48–7.52 | no |
| 4 | **Two-population core** (`src/core_twopop.py`): growing cells switch to a non-growing, lactate-consuming state. Stage 1: 13.34; v3 on it: 7.765. | 7.342 | yes |
| 5 | Bootstrap-bagged Monod core (5 fits on resampled batches) | 7.327 | no |
| 6 | Interval-aware glucose posterior instead of a clip | 7.323 | yes |
| 7 | TCD = stacked VCD × modelled TCD/VCD ratio | 7.331 | yes |
| 8 | **Consensus stage 1** (`src/core_avg.py`): geometric mean of six cores' stage-1 forecasts. Stage 1: 10.33. | 7.321 | yes |
| 9 | Data-driven branch with 30 instead of 10 bags in every core | 7.302 | yes |
| 10 | **Sensor-conditioned particle core** (`src/core_assim.py`): 400 per-batch parameter draws weighted by the exact bolus-rule glucose intervals and the base volume. Stage 1: 12.80 → 11.58; v3 on it: **7.585**, the best single model. | 7.290 | yes |
| | All kept ideas combined | **7.244** | |

Ideas 1–9 are scored without the finishing steps of ideas 6 and 7, except where noted in `trials.csv`.

The largest gain came from idea 10. It conditions the mechanistic model on each batch's logged sensors rather than correcting its forecast afterwards. That was the "sensors inside the ODE" idea from the earlier list.

Holdout predictions: `python src/stack_v6.py` (needs `core_twopop`, `core_avg` and `core_assim` to have run first).
