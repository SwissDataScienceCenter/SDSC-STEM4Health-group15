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
| `hybrid_blend.mp4` | `hybrid_blend` | Hybrid: ODE + data-driven correction |

Predictions are those of the models fitted on the dev batches only (`reports/holdout/`), so
every batch shown is unseen. Hybrid v7 is not among them: it has no predictions for our
holdout batches (it was validated with its own folds).

## Percent-error videos

`holdout_errors.py` (`make error-videos`) shows the percent error, 100 × (predicted − actual)
/ actual, instead of the values, on one shared y range (−20% to +20%, line at 0; errors
beyond are drawn at the edge with ▲/▼). One approach is added per video:

| Video | Static | Evolving |
|---|---|---|
| `errors_1.mp4` | – | mechanistic |
| `errors_2.mp4` | mechanistic | machine learning |
| `errors_3.mp4` | mechanistic, machine learning | hybrid |

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
