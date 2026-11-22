"""Manim scene: actual vs. predicted on the holdout batches, all four variables at once.

    MODEL=hybrid_blend manim -qh presentation/holdout_traces.py HoldoutTraces
(or `make videos`). Reads presentation/data/<MODEL>.csv (export_traces.py).

Fixed elements: four stacked axes sharing the day axis (VCD, TCD, substrate, byproduct) and
a legend. For each holdout batch, both traces grow together from the given day-0 values;
the batch number and its MAPE are shown in the legend. The y ranges are the same in every
video (computed over all exported models), so the three videos can be compared directly.
"""

import os
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    config,
    linear,
)
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
MODEL = os.environ.get("MODEL", "hybrid_blend")
N_BATCHES = int(os.environ.get("N_BATCHES", "0"))  # > 0: only the first N (drafts)
TITLES = {
    "mechanistic_lite": "Pure mechanistic: ODE model",
    "ml_sensors": "Pure machine learning: ridge + XGBoost",
    "hybrid_blend": "Hybrid: ODE + data-driven correction",
}
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
LABELS = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}

# Dark palette, validated with the dataviz validator (CVD and contrast checks pass).
SURFACE, TEXT, TEXT_2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
ACTUAL, PRED = "#3987e5", "#d95926"
DAYS = 14
GROW_SECONDS, HOLD_SECONDS = 4.0, 1.2

config.background_color = SURFACE
FONT = "DejaVu Sans"


def txt(text, font_size, color):
    return Text(text, font=FONT, font_size=font_size, color=color)


def y_max() -> dict:
    """Common y range per variable over every exported model (actual and predicted)."""
    frames = [pd.read_csv(p) for p in sorted(DATA.glob("*.csv"))]
    allv = pd.concat(frames)
    return {c: 1.08 * allv[[f"{c}_actual", f"{c}_pred"]].to_numpy().max() for c in TARGETS}


def nice_step(top: float) -> float:
    raw = top / 3
    mag = 10 ** np.floor(np.log10(raw))
    return min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)


def mape(df: pd.DataFrame) -> float:
    d = df[df["t"] > 0]
    return float(
        np.mean([np.mean(np.abs(d[f"{c}_pred"] / d[f"{c}_actual"] - 1)) for c in TARGETS]) * 100
    )


def partial_points(ax, t, y, T):
    """Points of the polyline (t, y) up to time T, with the end interpolated at T."""
    keep = t <= T
    ts, ys = list(t[keep]), list(y[keep])
    if T > t[0] and T < t[-1] and not np.isclose(ts[-1], T):
        ts.append(T)
        ys.append(float(np.interp(T, t, y)))
    return [ax.c2p(a, b) for a, b in zip(ts, ys)]


class HoldoutTraces(Scene):
    def construct(self):
        df = pd.read_csv(DATA / f"{MODEL}.csv")
        tops = y_max()

        # --- fixed elements: title, legend, four axes with a shared day axis -------------
        title = txt(TITLES.get(MODEL, MODEL), font_size=30, color=TEXT)
        title.to_corner(UP + LEFT, buff=0.35)
        sub = txt("Holdout batches, never seen in training", font_size=18, color=TEXT_2)
        sub.next_to(title, DOWN, aligned_edge=LEFT, buff=0.14)

        def key(color, label, dots):
            line = Line(LEFT * 0.3, RIGHT * 0.3, color=color, stroke_width=4)
            mark = VGroup(line)
            if dots:
                mark.add(Dot(line.get_center(), radius=0.07, color=color))
            return VGroup(mark, txt(label, font_size=20, color=TEXT)).arrange(RIGHT, buff=0.15)

        legend = VGroup(key(ACTUAL, "Actual", True), key(PRED, "Predicted", False))
        legend.arrange(RIGHT, buff=0.5).to_edge(RIGHT, buff=0.4).match_y(sub)

        axes = {}
        height, gap, top_y = 1.25, 0.28, 2.1
        for i, c in enumerate(TARGETS):
            step = nice_step(tops[c])
            ax = Axes(
                x_range=[0, DAYS, 1],
                y_range=[0, tops[c], step],
                x_length=11.2,
                y_length=height,
                tips=False,
                axis_config={"color": GRID, "stroke_width": 2, "include_ticks": True,
                             "tick_size": 0.04},
            )  # fmt: skip
            ax.move_to([0.95, top_y - i * (height + gap), 0])
            ylab = txt(LABELS[c], font_size=20, color=TEXT).next_to(ax, LEFT, buff=0.7)
            ticks = VGroup(*[
                txt(f"{v:g}", font_size=15, color=TEXT_2).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                for v in np.arange(0, tops[c] + 1e-9, step)[1:]
            ])  # fmt: skip
            self.add(ax, ylab, ticks)
            axes[c] = ax
        bottom = axes[TARGETS[-1]]
        days = VGroup(*[
            txt(str(d), font_size=15, color=TEXT_2).next_to(bottom.c2p(d, 0), DOWN, buff=0.12)
            for d in range(0, DAYS + 1, 2)
        ])  # fmt: skip
        xlab = txt("Culture day", font_size=20, color=TEXT_2).next_to(days, DOWN, buff=0.1)
        self.add(title, sub, legend, days, xlab)

        batch_label = txt("", font_size=24, color=TEXT)

        # --- one pass per holdout batch ---------------------------------------------------
        groups = list(df.groupby("batch_id", sort=False))
        for b, g in groups[:N_BATCHES] if N_BATCHES else groups:
            g = g.sort_values("t")
            t = g["t"].to_numpy()
            T = ValueTracker(0.0)
            moving = VGroup()
            for c in TARGETS:
                ax = axes[c]
                ya, yp = g[f"{c}_actual"].to_numpy(), g[f"{c}_pred"].to_numpy()

                def curve(ax=ax, y=ya, color=ACTUAL, width=3, t=t, T=T):
                    pts = partial_points(ax, t, y, T.get_value())
                    m = VMobject(stroke_color=color, stroke_width=width)
                    if len(pts) >= 2:
                        m.set_points_as_corners(pts)
                    else:
                        m.set_points_as_corners([pts[0], pts[0]])
                    return m

                def dots(ax=ax, y=ya, t=t, T=T):
                    return VGroup(*[
                        Dot(ax.c2p(a, v), radius=0.05, color=ACTUAL)
                        for a, v in zip(t, y) if a <= T.get_value() + 1e-9
                    ])  # fmt: skip

                moving.add(
                    always_redraw(lambda f=curve, ax=ax, y=ya: f(ax, y, ACTUAL, 3)),
                    always_redraw(dots),
                    always_redraw(lambda f=curve, ax=ax, y=yp: f(ax, y, PRED, 4)),
                )
            number = b.replace("Batch", "")
            new_label = txt(f"Batch {number}", font_size=24, color=TEXT)
            new_label.to_edge(RIGHT, buff=0.4).match_y(title)
            score = txt(f"MAPE {mape(g):.1f}%", font_size=20, color=TEXT_2)
            score.next_to(new_label, LEFT, buff=0.35)

            self.add(moving)
            self.play(FadeOut(batch_label), FadeIn(new_label), run_time=0.4)
            batch_label = new_label
            self.play(T.animate.set_value(t[-1]), run_time=GROW_SECONDS, rate_func=linear)
            self.play(FadeIn(score), run_time=0.3)
            self.wait(HOLD_SECONDS)
            self.play(FadeOut(moving), FadeOut(score), run_time=0.4)
        self.play(FadeOut(batch_label), run_time=0.3)
        self.wait(0.3)
