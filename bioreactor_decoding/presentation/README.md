# Presentation videos (Manim)

Three videos of the holdout batches, one per approach, with the same fixed layout: four
stacked axes (VCD, TCD, substrate, byproduct) sharing the day axis, actual (blue, dots) and
predicted (orange) growing together, batch number and its MAPE in the legend.

| Video | Model | What it is |
|---|---|---|
| `mechanistic_lite.mp4` | `mechanistic_lite` | Pure mechanistic: the ODE alone |
| `ml_sensors.mp4` | `ml_sensors` | Pure ML: ridge + XGBoost with soft sensors |
| `hybrid_blend.mp4` | `hybrid_blend` | Hybrid: ODE + data-driven correction |

Predictions are those of the models fitted on the dev batches only (`reports/holdout/`), so
every batch shown is unseen. Hybrid v7 is not among them: it has no predictions for our
holdout batches (it was validated with its own folds).

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
