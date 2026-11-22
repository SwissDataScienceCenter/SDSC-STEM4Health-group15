# Experiment log

Every idea is run as a DVC experiment (`dvc exp run -n <name> --set-param ...`) and judged
by `python -m bioreactor_decoding.modeling.compare` against a baseline on the same 5 dev
folds. **Rule: adopt only if the CV MAPE is lower on average and in at least 4 of the 5 folds.**
The holdout plays no part in any decision; it is looked at once, for the final candidate.
Negative results are kept here on purpose: they say where the remaining error is not.

## Round 1 (2026-11-22): towards 7% MAPE

Baseline: `hybrid_v3` CV 8.05, `hybrid_lite` CV 8.06 (v1.0.0 of both).

| # | Idea | Model | CV change | Folds better | Verdict |
|---|---|---|---|---|---|
| 1 | Blend `hybrid_v3` and `hybrid_lite` (`hybrid_blend`) | vs. hybrid_v3 | −0.05 | 4/5 | adopt |
| 2 | Similarity member for VCD/TCD, weight 1/3 | hybrid_lite | −0.07 | 5/5 | adopt |
| 2 | (same) | hybrid_v3 | −0.03 | 4/5 | adopt |
| 2 | (same, in both members) | hybrid_blend | −0.05 | 4/5 | adopt |
| 3 | Feed-B soft sensor (batch mean log-dose deviation) | hybrid_lite | +0.03 | 2/5 | reject |
| 3 | Feed-B sensor + similarity | hybrid_lite | −0.04 (similarity alone: −0.07) | 3/5 | reject |
| 4 | Branch blend weight 0.35 (vs. 0.5) | hybrid_lite + sim. | +0.09 | 2/5 | reject |
| 4 | Branch blend weight 0.65 | hybrid_lite + sim. | −0.02 | 3/5 | reject |
| 4 | Similarity weight 0.5 (vs. 1/3) | hybrid_lite | +0.01 | 2/5 | reject |
| 4 | 30 bootstrap bags (vs. 10) | hybrid_lite + sim. | −0.03 | 3/5 | reject |

**Result:** `hybrid_blend` with the similarity member in both models: **CV 7.95**
(from 8.05), holdout 6.99 (looked at once, after the decision).

### Diagnostics (out-of-fold predictions of `hybrid_lite` on the dev batches)

- **Where the error is.** The batch offset is the constant part of a batch's error relative
  to the prediction (e.g. predicted 9% too low on every day), not its day-0 values, which are
  already inputs. If each batch's offset were known, the CV MAPE would fall from 8.06 to 6.40. Getting under 7% on CV therefore
  means explaining most of these batch offsets from the inputs.
- **The inputs do not explain them.** The batch offset of VCD correlates 0.02 with the feed-B
  dose deviation and +0.13 with the day-0 values. Raw doses do track raw VCD (batch-level
  correlation 0.55), but stage 2 already uses that information.
- **Day-0 shrinkage** (pulling the noisy t = 0 values toward the population) was not run: the
  correlation is positive, i.e. the model already regresses to the mean, so shrinking further
  would make it worse.

**Conclusion.** Under 7% on CV looks out of reach with signals derived from the inputs. The
remaining error is mostly per-batch offsets that the inputs do not predict, plus measurement
noise (floor about 6.2). Further gains are likely to be a few hundredths per idea.
