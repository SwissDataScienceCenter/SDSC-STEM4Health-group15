# Stategies that improved the hybrid model

All scores: 7-fold cross-validation over the 49 training batches, holding out whole batches. Each held-out batch is predicted from its t=0 values plus its feed and volume log only, and scored with the leaderboard metric (mean of the four MAPEs). Lower is better.

| # | Strategy | Why it helps | Score |
|---|---|---|---|
| 1 | Sartorius-style mechanistic ODE (cells, lysis, biomaterial, glucose, lactate overflow), fitted on squared error | Baseline | 12.52 |
| 2 | Fit on relative error instead of squared error | Matches the MAPE metric | 12.02 |
| 3 | **Hybrid:** a small neural net corrects the growth and lactate rates inside the ODE | Fixes early growth and the lactate shift | 9.78 |
| 4 | Cell correction from t=0 state and feed-B deviations | Batches that start alike grow alike; the t=0 assay noise gets shrunk | 9.35 |
| 5 | Substrate soft sensor: glucose bolus size → glucose level before it | The bolus is a top-up, so a big bolus means low glucose | 9.16 |
| 6 | MAPE-optimal scaling per variable and day (weighted median) | The metric penalises over-prediction more | 9.01 |
| 7 | 50/50 blend with a data-driven gradient-boosting model | The two models make different errors | 8.54 |
| 8 | **Volume-balance sensor:** extra volume on days 2–5 = unlogged pH-control base, which tracks lactate production | Biggest lactate gain (byproduct 11.3 → 9.3) | 7.98 |
| 9 | Bolus-decision features: day-11, day-12 and early (day 7–9) glucose boluses | Reveal glucose on days 4–13 and which batches had more cells | 7.90 |
| 10 | Lactate data-driven model uses lactate sensors only; bagging over batches | Less noise | 7.84 |
| 11 | Stacking: corrections also see the stage-1 prediction | Learns when to trust the ODE | 7.75 |
| 12 | Lactate correction smooth across days (varying coefficients) | Shares strength across days | **7.67** |

Final per variable: VCD 6.97, TCD 6.95, substrate 8.02, byproduct 8.76. Code: `src/hybrid_model.py` (stage 1, steps 1–3) and `src/hybrid_v3.py` (steps 4–12).

## Ideas that failed to improvee
- Extra ensemble members: the second hybrid variant, and a similarity model for substrate and lactate.
- A boosted-tree lactate correction.
- Learned blend weights, and scaling after the blend.
- 10 instead of 4 networks in stage 1.
- Feed C dose patterns, timing jitter, late base addition, and a feed-B-aware bolus sensor.

## Caveats
- **Selection on the same folds:** steps 9–12 were chosen by comparing options on the same folds, so 7.67 is slightly optimistic. Expect about ±1 on any 12 holdout batches.
- **Soft sensors read the full log:** they use the whole feed and volume log, including events after the prediction time. That is allowed here, because holdouts come with the full log, but it would not work as a real-time forecaster.
- **Noise floor:** each batch's VCD/TCD carries about 6–7% assay noise. Even an oracle knowing each batch's average error would only reach about 6.2.
