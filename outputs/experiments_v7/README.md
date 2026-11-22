# v7: improving hybrid_v3 with at most 4 models in the blend

All scores come from the same 7-fold cross-validation over whole batches as hybrid_v3. Blend weights are fitted on the other folds only. The full log is in `trials.csv`. The final model is in `src/hybrid_v7.py`, and its holdout predictions are in `outputs/hybrid_v7/holdout_predictions.csv` (not submitted).

**Result: 7.674 (hybrid_v3) → 7.187 with 4 models (VCD 6.77, TCD 6.71, glucose 7.17, lactate 8.10).** The target of 7 was not reached in CV. The 17-model v6 stack scored 7.244, so the 4-model blend is both smaller and better.

| # | Trial | Score |
|---|---|---|
| 0 | hybrid_v3 (1 model) | 7.674 |
| 1 | Best k-model blend, members chosen inside each fold from 21 candidates: k = 1 / 2 / 3 / 4 | 7.558 / 7.382 / 7.396 / 7.366 |
| 2 | Assimilation-core variants (`src/core_assim2.py`): more per-batch parameters, t=0 cell-count factor, noise levels. Stage 1 plateaus at 11.53–11.75; v3 on them scores 7.565–7.576. | 7.352 (4-model) |
| 3 | Posterior parameters of the assimilation as v3 features: data-driven branch only / plus cells / everywhere | 7.550 / 7.604 / 7.608 (single model) |
| 4 | Fixed, reasoned 4-model sets, with a finish that uses only the 4 models' own outputs | **7.187** – 7.483 |
| 5 | Same sets with 30-bag data-driven branches | 7.191 / 7.204 |
| 6 | Sensitivity: choice of trajectory-model member | 7.187 – 7.227 |
| 7 | Sensor-assimilated two-population core (stage 1: 13.34 → 12.11) swapped into the blend | 7.242 – 7.402 |

Final 4 models (`src/hybrid_v7.py`):
1. v3 on the sensor-conditioned Monod core, with its posterior parameters as data-driven features (glucose).
2. A trajectory model on the assimilation core (glucose, lactate late, cells).
3. hybrid_v3 (cells and lactate on days 1–6).
4. v3 on the two-population core (cells and lactate on days 7–13).

Finish: glucose is set to the MAPE-optimal point inside the bolus-rule interval. TCD = VCD × a ratio model whose inputs are the 4 models' own ratios.

Caveat: the choice among six fixed 4-model sets was made on these folds. The spread in trial 6 suggests about 0.03 of optimism.
