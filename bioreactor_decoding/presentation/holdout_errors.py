"""Manim scene: percent error on the holdout batches, one approach added per video.

    STAGE=1 manim -qh presentation/holdout_errors.py ErrorTraces   # mechanistic evolving
    STAGE=2 ...   mechanistic static, ML evolving
    STAGE=3 ...   mechanistic and ML static, hybrid evolving
(or `make error-videos`). Reads presentation/data/ (export_traces.py).

Same layout as holdout_traces.py, but each panel shows the percent error
100 * (predicted - actual) / actual on one shared y range (-20% to +20%, line at 0).
Errors beyond the range are drawn at the edge, with a triangle pointing off-scale at each
such sample. Below the panels, the same feeds strip as in the trace videos.
"""

import os

from holdout_traces import (
    DATA,
    DAYS,
    FEEDS,
    FLASH_DAYS,
    GRID,
    GROW_SECONDS,
    HOLD_SECONDS,
    LABELS,
    N_BATCHES,
    TARGETS,
    TEXT,
    TEXT_2,
    txt,
)
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Triangle,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)
import numpy as np
import pandas as pd

STAGE = int(os.environ.get("STAGE", "1"))
YLIM = 20.0
# Colour follows the model in every video: mechanistic orange and ML blue (as in the report);
# the hybrid is a bright neutral because aqua, its report colour, is feed B in the strip here.
MODELS = [
    ("mechanistic_lite", "Mechanistic", "#d95926"),
    ("ml_sensors", "Machine learning", "#3987e5"),
    ("hybrid_blend", "Hybrid", "#f0efec"),
]
TITLES = {
    1: "Pure mechanistic model: percent error",
    2: "Machine learning vs. mechanistic: percent error",
    3: "Hybrid vs. mechanistic and ML: percent error",
}


def errors(model: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{model}.csv")
    out = d[["batch_id", "t"]].copy()
    for c in TARGETS:
        out[c] = 100 * (d[f"{c}_pred"] / d[f"{c}_actual"] - 1)
    return out


def batch_mape(e: pd.DataFrame) -> float:
    return float(e.loc[e["t"] > 0, TARGETS].abs().to_numpy().mean())


def polyline(ax, t, y, T, color, width, opacity):
    """Error trace up to time T, clipped to the y range."""
    keep = t <= T
    ts, ys = list(t[keep]), list(y[keep])
    if t[0] < T < t[-1] and not np.isclose(ts[-1], T):
        ts.append(T)
        ys.append(float(np.interp(T, t, y)))
    pts = [ax.c2p(a, float(np.clip(v, -YLIM, YLIM))) for a, v in zip(ts, ys)]
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(pts if len(pts) >= 2 else [pts[0], pts[0]])
    return m


def markers(ax, t, y, T, color, opacity, with_dots):
    """Dots at the samples; off-scale samples get a triangle at the edge pointing outwards."""
    g = VGroup()
    for a, v in zip(t, y):
        if a > T + 1e-9:
            continue
        if abs(v) > YLIM:
            tri = Triangle(color=color, fill_color=color, fill_opacity=opacity,
                           stroke_width=0).scale(0.07)  # fmt: skip
            if v < 0:
                tri.rotate(PI)
            g.add(tri.move_to(ax.c2p(a, np.sign(v) * YLIM)))
        elif with_dots:
            g.add(Dot(ax.c2p(a, v), radius=0.045, color=color, fill_opacity=opacity))
    return g


class ErrorTraces(Scene):
    def construct(self):
        shown = MODELS[:STAGE]  # the last one evolves, the others are static
        errs = {m: errors(m) for m, _, _ in shown}

        # --- fixed elements -----------------------------------------------------------------
        title = txt(TITLES[STAGE], font_size=28, color=TEXT).to_corner(UP + LEFT, buff=0.35)
        sub = txt("Error = 100 × (predicted − actual) / actual on unseen holdout batches;"
                  " ▲▼ mark errors beyond ±20%", font_size=16, color=TEXT_2)  # fmt: skip
        sub.next_to(title, DOWN, aligned_edge=LEFT, buff=0.14)

        keys, value_slots = VGroup(), {}
        for i, (m, name, color) in enumerate(shown):
            static = i < len(shown) - 1
            line = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=2.5 if static else 4,
                        stroke_opacity=0.55 if static else 1)  # fmt: skip
            label = txt(name, font_size=18, color=TEXT)
            slot = txt("00.0%", font_size=18, color=TEXT_2).set_opacity(0)  # reserves room
            key = VGroup(line, label, slot).arrange(RIGHT, buff=0.12)
            keys.add(key)
            value_slots[m] = slot
        keys.arrange(RIGHT, buff=0.45).next_to(sub, DOWN, aligned_edge=LEFT, buff=0.16)
        mape_note = txt("(MAPE of this batch)", font_size=14, color=TEXT_2)
        mape_note.next_to(keys, RIGHT, buff=0.35)

        axes = {}
        height, gap, top_y = 0.98, 0.2, 1.95
        for i, c in enumerate(TARGETS):
            ax = Axes(
                x_range=[0, DAYS, 1],
                y_range=[-YLIM, YLIM, 10],
                x_length=11.2,
                y_length=height,
                tips=False,
                axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False},
                x_axis_config={"color": TEXT_2, "stroke_width": 1.5},  # the 0 line
            )  # fmt: skip
            ax.move_to([0.95, top_y - i * (height + gap), 0])
            for v in (-10, 10):  # faint guides
                self.add(Line(ax.c2p(0, v), ax.c2p(DAYS, v), color=GRID, stroke_width=1))
            ylab = txt(LABELS[c], font_size=20, color=TEXT).next_to(ax, LEFT, buff=0.8)
            ticks = VGroup(*[
                txt(s, font_size=14, color=TEXT_2).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                for v, s in ((-10, "−10%"), (0, "0"), (10, "+10%"))
            ])  # fmt: skip
            self.add(ax, ylab, ticks)
            axes[c] = ax
        bottom = axes[TARGETS[-1]]

        # Feeds strip, as in holdout_traces.py.
        row_h, strip_top = 0.2, bottom.get_bottom()[1] - 0.18
        x0, x1 = bottom.c2p(0, 0)[0], bottom.c2p(DAYS, 0)[0]
        rows = {}
        for i, (name, (label, color)) in enumerate(FEEDS.items()):
            base = strip_top - (i + 1) * row_h
            rows[name] = base
            self.add(Line([x0, base, 0], [x1, base, 0], color=GRID, stroke_width=1))
            key_ = VGroup(txt(label, font_size=13, color=TEXT_2), Dot(radius=0.045, color=color))
            key_.arrange(RIGHT, buff=0.1)
            key_.move_to([x0 - 0.2 - key_.width / 2, base + row_h * 0.35, 0])
            self.add(key_)
        strip_bottom = strip_top - len(FEEDS) * row_h
        feeds_title = txt("Feeds", font_size=20, color=TEXT)
        feeds_title.next_to([x0, (strip_top + strip_bottom) / 2, 0], LEFT, buff=1.55)
        feed_events = pd.read_csv(DATA / "feeds" / "feeds.csv")
        days = VGroup(*[
            txt(str(d), font_size=15, color=TEXT_2)
            .move_to([bottom.c2p(d, 0)[0], strip_bottom - 0.16, 0])
            for d in range(0, DAYS + 1, 2)
        ])  # fmt: skip
        xlab = txt("Culture day", font_size=20, color=TEXT_2).next_to(days, DOWN, buff=0.1)
        self.add(title, sub, keys, mape_note, feeds_title, days, xlab)

        batch_label = txt("", font_size=24, color=TEXT)
        evolving, static = shown[-1][0], shown[:-1]

        # --- one pass per holdout batch ---------------------------------------------------
        batches = list(dict.fromkeys(errs[evolving]["batch_id"]))
        for b in batches[:N_BATCHES] if N_BATCHES else batches:
            T = ValueTracker(0.0)
            frozen, moving = VGroup(), VGroup()
            for m, _, color in shown:
                e = errs[m].loc[errs[m]["batch_id"] == b].sort_values("t")
                t = e["t"].to_numpy()
                for c in TARGETS:
                    ax, y = axes[c], e[c].to_numpy()
                    if m != evolving:  # the whole batch at once, dimmed
                        frozen.add(polyline(ax, t, y, t[-1], color, 2.5, 0.55),
                                   markers(ax, t, y, t[-1], color, 0.55, False))  # fmt: skip
                    else:
                        moving.add(
                            always_redraw(lambda ax=ax, t=t, y=y, color=color, T=T:
                                          polyline(ax, t, y, T.get_value(), color, 4, 1)),
                            always_redraw(lambda ax=ax, t=t, y=y, color=color, T=T:
                                          markers(ax, t, y, T.get_value(), color, 1, True)),
                        )  # fmt: skip

            last_t = errs[evolving].loc[errs[evolving]["batch_id"] == b, "t"].max()
            fe = feed_events.loc[feed_events["batch_id"] == b]

            def feed_ticks(fe=fe, T=T):
                now = T.get_value()
                ticks = VGroup()
                for name, te, rel in zip(fe["feed"], fe["t"], fe["rel"]):
                    if te > now:
                        continue
                    x, base = bottom.c2p(te, 0)[0], rows[name]
                    h = max(rel, 0.12) * row_h * 0.85
                    flash = max(0.0, 1.0 - (now - te) / FLASH_DAYS)
                    color = FEEDS[name][1]
                    ticks.add(Line([x, base, 0], [x, base + h, 0], color=color,
                                   stroke_width=3 + 5 * flash))  # fmt: skip
                    if flash > 0:
                        ticks.add(Dot([x, base + h, 0], radius=0.03 + 0.07 * flash, color=color,
                                      fill_opacity=flash))  # fmt: skip
                return ticks

            moving.add(always_redraw(feed_ticks))

            values = {}
            for m, _, _ in shown:
                e = errs[m].loc[errs[m]["batch_id"] == b]
                v = txt(f"{batch_mape(e):.1f}%", font_size=18, color=TEXT_2)
                values[m] = v.move_to(value_slots[m], aligned_edge=LEFT)
            new_label = txt(f"Batch {b.replace('Batch', '')}", font_size=24, color=TEXT)
            new_label.to_edge(RIGHT, buff=0.35).match_y(title)

            self.add(moving)
            self.play(FadeOut(batch_label), FadeIn(new_label), FadeIn(frozen),
                      *[FadeIn(values[m]) for m, _, _ in static], run_time=0.4)  # fmt: skip
            batch_label = new_label
            self.play(T.animate.set_value(last_t), run_time=GROW_SECONDS, rate_func=linear)
            self.play(FadeIn(values[evolving]), T.animate.set_value(last_t + FLASH_DAYS),
                      run_time=0.3)  # fmt: skip
            self.wait(HOLD_SECONDS)
            self.play(FadeOut(moving), FadeOut(frozen), *[FadeOut(v) for v in values.values()],
                      run_time=0.4)  # fmt: skip
        self.play(FadeOut(batch_label), run_time=0.3)
        self.wait(0.3)
