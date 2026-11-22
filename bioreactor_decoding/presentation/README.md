# Presentation videos (Manim)

Three videos of the holdout batches, one per approach, with the same fixed layout: four
stacked axes (VCD, TCD, substrate, byproduct) sharing the day axis, actual (blue, dots) and
predicted (orange) growing together, batch number and its MAPE in the legend.
Under the axes, a feeds strip shows the model inputs: one row per feed (B, C, D, substrate),
one tick per dose, flashing as it happens. Tick height is the dose relative to that feed's
largest dose in the data (the feeds' scales differ ~500-fold); every logged dose is shown,
since the log has a row only when a dose was given (no zero amounts, so no threshold).

| Video | Model | What it is |
|---|---|---|
| `mechanistic_lite.mp4` | `mechanistic_lite` | Pure mechanistic: the ODE alone |
| `ml_sensors.mp4` | `ml_sensors` | Pure ML: ridge + XGBoost with soft sensors |
| `option_a.mp4` | `option_a` | Hybrid, the final model (Option A): one sensor-assimilated ODE core + learned corrections |

Predictions are those in `reports/holdout/`, so every batch shown is unseen by the model that
predicted it. For the mechanistic and ML models that is the model fitted on the dev batches
only. Option A has no such model: it was validated with its own 7 folds over all 49 labelled
batches, and its videos show its out-of-fold predictions, each from a model fitted on about 42
batches that never saw that batch. Its MAPE is therefore close to, but not the same protocol
as, the other two (`docs/mlops.md`, "Option A"). The earlier hybrid video,
`hybrid_blend.mp4`, is no longer rendered; `hybrid_blend` traces are still exported for the
slides.

## Hero video (for the end of the talk)

`hero.py` (`make hero-video`): one 42 s shot at 1080p60 →
`media/videos/hero/1080p60/hero.mp4`, plus `media/images/hero/hero_end_card.png` as a still
fallback.

1. The hybrid (Option A, the final model) forecasts holdout batch 10 from day 0 and the feed plan (actual vs. predicted).
2. The panels morph into percent error: the actual curve flattens into the zero line.
3. The mechanistic and ML error curves join; per-batch MAPEs in the legend.
4. Batches 17 (harder) and 34 (ML matches the hybrid), all three models at once.
5. End card: MAPE over all 10 holdout batches, read from `reports/metrics/` (Option A's is
   its out-of-fold MAPE, `holdout_mape_oof7`, noted on the card).

Batches are typical, not the best: Option A's MAPE on 10 equals its holdout mean (6.6). Colours: actual off-white,
models as in the report (mechanistic orange, ML blue, hybrid aqua); feeds are neutral grey
here so that colour means "model" only. Error axes span ±30% (beyond: triangles).

## Percent-error videos

`holdout_errors.py` (`make error-videos`) shows the percent error, 100 × (predicted − actual)
/ actual, instead of the values, on one shared y range (−20% to +20%, line at 0; errors
beyond are drawn at the edge with ▲/▼). One approach is added per video:

| Video | Static | Evolving |
|---|---|---|
| `errors_1.mp4` | – | mechanistic |
| `errors_2.mp4` | mechanistic | machine learning |
| `errors_3.mp4` | mechanistic, machine learning | hybrid (Option A) |

Colours follow the model: mechanistic orange, ML blue, hybrid off-white (aqua is feed B).
The legend shows each model's MAPE for the batch; the feeds strip is the same as above.

## Setup (once per machine)

[Manim Community](https://www.manim.community/) renders without a display. Its Python
packages are in the uv dependency group `presentation`; two of them (manimpango, pycairo)
build from source against cairo and pango, which come from a small conda environment:

```bash
micromamba create -y -r /home/renku/work/.micromamba -f presentation/environment.yml
make presentation-env        # uv sync --group presentation, built against that environment
```

## Render

```bash
make videos                  # export traces + render all three at 1080p60 -> presentation/media/
MODEL=ml_sensors N_BATCHES=1 .venv/bin/manim -ql --media_dir presentation/media \
    presentation/holdout_traces.py HoldoutTraces   # quick draft, first batch only
```

`presentation/data/` and `presentation/media/` hold measured values and are not in git.
