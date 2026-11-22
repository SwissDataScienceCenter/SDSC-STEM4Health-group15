# Strategies that improved the hybrid model

All scores: 7-fold cross-validation over the 49 training batches, holding out whole batches. Each held-out batch is predicted from its t=0 values plus its feed and volume log only, and scored with the leaderboard metric (mean of the four MAPEs). Lower is better.

| # | Strategy | Why it helps | Score |
|---|---|---|---|
| 1 | Sartorius-style mechanistic ODE (cells, lysis, biomaterial, glucose, lactate overflow), fitted on squared error | Baseline | 12.52 |
| 2 | Fit on relative error instead of squared error | Matches the MAPE metric | 12.02 |
| 3 | **Hybrid:** a small neural net corrects the growth and lactate rates inside the ODE | Fixes early growth and the lactate shift | 9.78 |
| 4 | Cell correction from t=0 state and feed-B deviations | Batches that start alike grow alike; the t=0 assay noise gets shrunk | 9.35 |
| 5 | Substrate soft sensor: glucose bolus size → glucose just before the bolus | A bigger top-up means glucose was further below the refill target; the fit turns bolus size into that level | 9.16 |
| 6 | MAPE-optimal scaling per variable and day (weighted median) | The metric penalises over-prediction more | 9.01 |
| 7 | 50/50 blend with a data-driven gradient-boosting model | The two models make different errors | 8.54 |
| 8 | **Volume-balance sensor:** extra volume on days 2–5 = unlogged pH-control base | Base volume is the acid the controller neutralised, so it is turned into a lactate correction (byproduct 11.3 → 9.3) | 7.98 |
| 9 | Bolus-decision features: day-11, day-12 and early (day 7–9) glucose boluses | The same bolus sizes, reused on the surrounding days, fill in glucose on days 4–13 and flag batches with more cells | 7.90 |
| 10 | Lactate data-driven model uses lactate sensors only; bagging over batches | Less noise | 7.84 |
| 11 | Stacking: corrections also see the stage-1 prediction | Learns when to trust the ODE | 7.75 |
| 12 | Lactate correction smooth across days (varying coefficients) | Shares strength across days | **7.67** |

Final per variable: VCD 6.97, TCD 6.95, substrate 8.02, byproduct 8.76. Code: `src/hybrid_model.py` (stage 1, steps 1–3) and `src/hybrid_v3.py` (steps 4–12).

The step-7 blend is the geometric mean, `sqrt(hybrid × tree)`, which is an equal-weight average of the two forecasts in log space. A factor of two high and a factor of two low cancel, matching a percent-error metric. Learned blend weights did not help. Total cell density is then raised to the viable-cell prediction when it falls below it.

## How the sensors become a glucose or a lactate number

Neither sensor measures the concentration. Each is a logged action whose size depends on the concentration, and the mapping from that size onto the concentration is fitted on the training batches, where both are known. The quantity that is fitted is the correction `log(measured / stage-1 prediction)`, so the sensor moves the ODE forecast rather than replacing it. On a holdout batch only the logged action is available; the fitted line supplies the concentration.

**Glucose, from the bolus.** Feed substrate is a top-up: it is added when glucose has fallen, and a larger dose brings a lower culture back up to the same target. The signal is the bolus volume divided by the culture volume at that moment (`A`). On training batches the fit is, separately for the day-11 bolus, the day-12 bolus and any earlier bolus,

`log(measured glucose / stage-1 glucose) = a + b·A + c·log(stage-1 glucose)`.

`b` comes out negative, so a bigger bolus pulls the glucose prediction down, and by about the amount that bolus size differed from the usual one. Step 5 applies this only to the sample immediately before the bolus. Step 9 keeps the three bolus sizes as batch features (`A11`, `A12`, `A_early`) and lets a separate per-day fit use them on the neighbouring days: the day-11 bolus informs glucose on days 9–12, the day-12 bolus informs days 11–13, and an early bolus (days 7–9) marks the batches that had already run low, which are also the ones with more cells.

**Lactate, from the volume that no feed explains.** At each sample the logged volume is `start + every logged feed − sample taken out`. After putting the sample volume back (~0.05 mL per sample, read off the steady late-culture decline), volume on days 2–5 is still ~0.18 mL higher than the feeds account for. That extra is base the pH controller added once the buffer was spent, and it is not in the feed log. Lactate is the acid that base is neutralising, so the extra volume is a proxy for how much lactate the batch has produced (within a day, `r = +0.56` with lactate and `r = −0.50` with glucose consumed). The features are that extra volume accumulated by day 3, by day 4 and by day 5, plus the total once addition has stopped (the day-6-to-13 plateau minus the day-1-to-2 baseline). On training batches a smooth-in-day fit converts those millilitres into the lactate correction: each base feature has an effect that changes gradually with culture day, and each day keeps its own intercept. From day 6 on, the running total is replaced by the finished total, because no further base is added. The same total base is also one feature in the glucose fit, since the batches that needed more base had also consumed more glucose; the bolus remains the main glucose signal.

## Ideas that failed to improve
- Extra ensemble members: the second hybrid variant, and a similarity model for substrate and lactate.
- A boosted-tree lactate correction.
- Learned blend weights, and scaling after the blend.
- 10 instead of 4 networks in stage 1.
- Feed C dose patterns, timing jitter, late base addition, and a feed-B-aware bolus sensor.
- Glucose-uptake correction inside the stage-1 network (`--outputs mu,glc,lac`). Stage 1 goes 9.78 → 9.69 (substrate 10.82 → 10.08, byproduct 11.94 → 12.81). Scored through v3 the blend is 7.66 against 7.67: substrate 7.95, byproduct 8.68, VCD 7.06, TCD 6.97. Runs: `outputs/hybrid_glc`, `outputs/hybrid_v3_glc`.

## Next ideas to try
Ordered by expected value. Expected gains are estimates, not measurements.

1. **Similarity member for cells.** Predict each batch's VCD/TCD error curve from the most similar training batches at t=0. An exploratory run reached about 7.60 (VCD 6.85, TCD 6.82); it still needs one full pipeline run to confirm.
2. **Put the sensors inside the ODE.** Treat the base-addition volume as a measured lactate-production flux and the bolus as a glucose top-up to a target, rather than correcting afterwards. Cell, glucose and lactate predictions would then stay mutually consistent; today each variable is corrected separately.
3. **Batch-specific parameters, re-simulated.** Estimate a few per-batch parameters (growth rate, lactate yield) from the sensors and t=0, then re-run the ODE. This is the mixed-effects version of point 2, and no CHO paper has done it yet.
4. **Nested cross-validation for the stage-2 choices.** This doesn't improve the model; it gives an honest score by removing the optimism from step 9–12 selection.
5. **Calibrate against the leaderboard.** Submit once to see how the 12 holdout batches compare with the cross-validation estimate before tuning further.
6. **Interpretability (not score).** Distil the network's learned growth and lactate corrections into simple formulas (symbolic regression) for the presentation.

Realistic ceiling: an oracle batch-level scale leaves 6.24, so the next steps are worth roughly 0.1–0.4 points, not another big jump.

## Caveats
- **Selection on the same folds:** steps 9–12 were chosen by comparing options on the same folds, so 7.67 is slightly optimistic. Expect about ±1 on any 12 holdout batches.
- **Soft sensors read the full log:** they use the whole feed and volume log, including events after the prediction time. That is allowed here, because holdouts come with the full log, but it would not work as a real-time forecaster.
- **Oracle floor:** for each batch and assay, multiply the whole trajectory by `c = exp(mean(log(measured / predicted)))`, using that batch's own measurements. This is not a deployable model. On the current out-of-fold predictions it leaves VCD 5.62, TCD 5.49, substrate 6.71, byproduct 7.14, and a leaderboard score of 6.24. A missed batch-wide scale is about one point of the current 7.67. The rest is point-to-point scatter around a level that is already right. There are no replicate assays, so this does not separate instrument noise from real day-to-day biology.
