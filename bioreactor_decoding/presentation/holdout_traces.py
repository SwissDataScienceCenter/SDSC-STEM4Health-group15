"""Manim scene: actual vs. predicted on the holdout batches, all four variables at once.

    MODEL=option_a manim -qh presentation/holdout_traces.py HoldoutTraces
(or `make videos`). Reads presentation/data/<MODEL>.csv (export_traces.py).

Fixed elements: four stacked axes sharing the day axis (VCD, TCD, substrate, byproduct) and
a legend, plus a feeds strip under the axes: one row per feed (B, C, D, substrate), one tick
per dose with its height proportional to the amount (relative to that feed's largest dose,
since the feeds' scales differ ~500-fold). For each holdout batch, both traces grow together
from the given day-0 values and each feed tick appears, with a short flash, as time passes
it; the batch number and its MAPE are shown in the legend. The y ranges are the same in every
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
MODEL = os.environ.get("MODEL", "option_a")
N_BATCHES = int(os.environ.get("N_BATCHES", "0"))  # > 0: only the first N (drafts)
TITLES = {
    "mechanistic_lite": "Pure mechanistic: ODE model",
    "ml_sensors": "Pure machine learning: ridge + XGBoost",
    "hybrid_blend": "Hybrid: ODE + data-driven correction",
    "option_a": "Hybrid (Option A): ODE + learned correction",
}
# option_a: out-of-fold predictions from its own 7-fold CV (see export_traces.py)
SUBTITLES = {"option_a": "Holdout batches, each predicted by a model that never saw it"}
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
LABELS = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}

# Dark palette, validated with the dataviz validator (CVD and contrast checks pass).
SURFACE, TEXT, TEXT_2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
ACTUAL, PRED = "#3987e5", "#d95926"
# Feeds: palette slots 3-5 and 7 (slot 6 fails CVD next to aqua/orange). Adjacent rows pass
# CVD and normal-vision checks; B and D (aqua/magenta, close for deutan) are not adjacent and
# every row is labelled, so position carries the type too.
FEEDS = {"feed_b": ("Feed B", "#199e70"), "feed_c": ("Feed C", "#c98500"),
         "feed_d": ("Feed D", "#d55181"), "feed_substrate": ("Substrate", "#9085e9")}  # fmt: skip
FLASH_DAYS = 1.0  # a new dose flashes for ~0.3 s (the sweep runs ~3.3 days per second)
DAYS = 14
GROW_SECONDS, HOLD_SECONDS = 4.0, 1.2

config.background_color = SURFACE
FONT = "DejaVu Sans"


def txt(text, font_size, color):
    return Text(text, font=FONT, font_size=font_size, color=color)


def y_max() -> dict:
    """Common y range per variable over every exported model (actual and predicted)."""
    frames = [pd.read_csv(p) for p in sorted(DATA.glob("*.csv"))]  # feeds are in data/feeds/
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
        subtitle = SUBTITLES.get(MODEL, "Holdout batches, never seen in training")
        sub = txt(subtitle, font_size=18, color=TEXT_2)
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
        height, gap, top_y = 1.05, 0.22, 2.2
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

        # Feeds strip: one row per feed under the bottom axis, sharing its day axis.
        row_h, strip_top = 0.2, bottom.get_bottom()[1] - 0.18
        x0, x1 = bottom.c2p(0, 0)[0], bottom.c2p(DAYS, 0)[0]
        rows = {}
        for i, (name, (label, color)) in enumerate(FEEDS.items()):
            base = strip_top - (i + 1) * row_h
            rows[name] = base
            self.add(Line([x0, base, 0], [x1, base, 0], color=GRID, stroke_width=1))
            swatch = Dot(radius=0.045, color=color)
            lab = txt(label, font_size=13, color=TEXT_2)
            key_ = VGroup(lab, swatch).arrange(RIGHT, buff=0.1)
            key_.move_to([x0 - 0.2 - key_.width / 2, base + row_h * 0.35, 0])
            self.add(key_)
        strip_bottom = strip_top - len(FEEDS) * row_h
        feeds_title = txt("Feeds", font_size=20, color=TEXT)
        feeds_title.next_to([x0, (strip_top + strip_bottom) / 2, 0], LEFT, buff=1.55)
        self.add(feeds_title)
        feed_events = pd.read_csv(DATA / "feeds" / "feeds.csv")

        days = VGroup(*[
            txt(str(d), font_size=15, color=TEXT_2)
            .move_to([bottom.c2p(d, 0)[0], strip_bottom - 0.16, 0])
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

                def error_label(ax=ax, ya=ya, yp=yp, t=t, T=T):
                    """Percent error (predicted vs. actual) at the leading edge, above it.

                    Between samples both lines are linear, so the error shown is the one
                    between the drawn lines; at each sample it is the true point error.
                    """
                    now = min(T.get_value(), t[-1])
                    a, p = np.interp(now, t, ya), np.interp(now, t, yp)
                    err = 100 * (p - a) / a
                    label = txt("0.0%" if abs(err) < 0.05 else f"{err:+.1f}%", 16, TEXT)
                    top = ax.c2p(now, max(a, p))
                    if top[1] + 0.12 + label.height > ax.get_top()[1]:  # no room above
                        label.next_to(ax.c2p(now, min(a, p)), DOWN, buff=0.12)
                    else:
                        label.next_to(top, UP, buff=0.12)
                    return label

                moving.add(
                    always_redraw(lambda f=curve, ax=ax, y=ya: f(ax, y, ACTUAL, 3)),
                    always_redraw(dots),
                    always_redraw(lambda f=curve, ax=ax, y=yp: f(ax, y, PRED, 4)),
                    always_redraw(error_label),
                )
            fe = feed_events.loc[feed_events["batch_id"] == b]

            def feed_ticks(fe=fe, T=T):
                now = T.get_value()
                ticks = VGroup()
                for name, te, rel in zip(fe["feed"], fe["t"], fe["rel"]):
                    if te > now:
                        continue
                    x, base = bottom.c2p(te, 0)[0], rows[name]
                    h = max(rel, 0.12) * row_h * 0.85
                    age = (now - te) / FLASH_DAYS
                    flash = max(0.0, 1.0 - age)  # 1 when the dose happens, 0 after FLASH_DAYS
                    color = FEEDS[name][1]
                    ticks.add(Line([x, base, 0], [x, base + h, 0], color=color,
                                   stroke_width=3 + 5 * flash))  # fmt: skip
                    if flash > 0:
                        ticks.add(Dot([x, base + h, 0], radius=0.03 + 0.07 * flash, color=color,
                                      fill_opacity=flash))  # fmt: skip
                return ticks

            moving.add(always_redraw(feed_ticks))
            number = b.replace("Batch", "")
            new_label = txt(f"Batch {number}", font_size=24, color=TEXT)
            new_label.to_edge(RIGHT, buff=0.4).match_y(title)
            score = txt(f"MAPE {mape(g):.1f}%", font_size=20, color=TEXT_2)
            score.next_to(new_label, LEFT, buff=0.35)

            self.add(moving)
            self.play(FadeOut(batch_label), FadeIn(new_label), run_time=0.4)
            batch_label = new_label
            self.play(T.animate.set_value(t[-1]), run_time=GROW_SECONDS, rate_func=linear)
            # let the last flash finish (the traces stop at the last sample anyway)
            self.play(FadeIn(score), T.animate.set_value(t[-1] + FLASH_DAYS), run_time=0.3)
            self.wait(HOLD_SECONDS)
            self.play(FadeOut(moving), FadeOut(score), run_time=0.4)
        self.play(FadeOut(batch_label), run_time=0.3)
        self.wait(0.3)
