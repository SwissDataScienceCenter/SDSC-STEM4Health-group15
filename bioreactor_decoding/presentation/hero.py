"""The hero video for the end of the presentation (about 40 s, `make hero-video`).

    manim -qh presentation/hero.py Hero

One continuous shot, built for a projector:
  1. Behaviour: the hybrid forecasts holdout batch 10 from its day-0 values and feed plan;
     actual (off-white, dots) and predicted (aqua) grow together, feeds tick in below.
  2. The morph: each panel turns into percent error. The actual curve flattens into the zero
     line, the prediction bends into its error curve, the value axis becomes +-30%.
  3. Comparison: the mechanistic (orange) and ML (blue) error curves join; MAPEs in the legend.
  4. Batches 17 and 34 in error view, all three models at once. In 34, ML matches the hybrid.
  5. End card: MAPE over all 10 holdout batches (from reports/metrics).

Batches: 10 (typical: hybrid 6.4, near the holdout mean of 7.0), 17 (harder) and 34 (ML as
good as the hybrid), not the best ones. Colours: the models as in the report (validated on
this surface); feeds are neutral here so that colour means "model" only.
"""

import json

from holdout_traces import DATA, DAYS, FEEDS, GRID, LABELS, TARGETS, TEXT, TEXT_2, txt, y_max
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    Line,
    ReplacementTransform,
    Scene,
    Transform,
    Triangle,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
    smooth,
)
import numpy as np
import pandas as pd

METRICS = DATA.parent.parent / "reports" / "metrics"
BATCHES = ["Batch10", "Batch17", "Batch34"]
YLIM = 30.0
TRUTH = "#f0efec"
MODELS = {  # name, colour (report colours; all pairs pass the CVD and contrast checks)
    "mechanistic_lite": ("Mechanistic", "#d95926"),
    "ml_sensors": ("Machine learning", "#3987e5"),
    "hybrid_blend": ("Hybrid", "#199e70"),
}
HYBRID = "hybrid_blend"
FEED_TICK = "#8f8e88"
GROW, FLASH_DAYS = 5.0, 1.0


def load(model):
    d = pd.read_csv(DATA / f"{model}.csv")
    for c in TARGETS:
        d[f"{c}_err"] = 100 * (d[f"{c}_pred"] / d[f"{c}_actual"] - 1)
    return d


def batch_mape(d):
    d = d[d["t"] > 0]
    return float(np.mean([d[f"{c}_err"].abs().mean() for c in TARGETS]))


def line_through(points, color, width, opacity=1.0):
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(points if len(points) >= 2 else [points[0], points[0]])
    return m


def upto(t, y, T):
    """(t, y) up to time T, the end interpolated at T."""
    keep = t <= T
    ts, ys = list(t[keep]), list(y[keep])
    if t[0] < T < t[-1] and not np.isclose(ts[-1], T):
        ts.append(T)
        ys.append(float(np.interp(T, t, y)))
    return np.array(ts), np.array(ys)


def err_points(ax, t, e):
    return [ax.c2p(a, float(np.clip(v, -YLIM, YLIM))) for a, v in zip(t, e)]


def off_scale(ax, t, e, color, T=np.inf):
    """Triangles at the edge for samples beyond +-YLIM, pointing off-scale."""
    g = VGroup()
    for a, v in zip(t, e):
        if a <= T + 1e-9 and abs(v) > YLIM:
            tri = Triangle(color=color, fill_color=color, fill_opacity=1, stroke_width=0)
            tri.scale(0.08)
            if v < 0:
                tri.rotate(PI)
            g.add(tri.move_to(ax.c2p(a, np.sign(v) * YLIM)))
    return g


class Hero(Scene):
    def construct(self):
        data = {m: load(m) for m in MODELS}
        tops = y_max()
        feeds = pd.read_csv(DATA / "feeds" / "feeds.csv")

        # --- frame: title, caption, panels (value and error axes in the same place) -------
        title = txt("From day 0 and the feed plan to 13 days of culture", 30, TEXT)
        title.to_corner(UP + LEFT, buff=0.35)
        caption = txt("Hybrid model, holdout batch never seen in training", 22, TEXT_2)
        caption.next_to(title, DOWN, aligned_edge=LEFT, buff=0.15)

        height, gap, top_y, x_len, x_mid = 0.95, 0.2, 1.75, 11.0, 1.05
        val_axes, err_axes, val_ticks, err_ticks, labels = {}, {}, VGroup(), VGroup(), VGroup()
        for i, c in enumerate(TARGETS):
            y = top_y - i * (height + gap)
            raw = tops[c] / 3
            mag = 10 ** np.floor(np.log10(raw))
            step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
            va = Axes(
                x_range=[0, DAYS, 1],
                y_range=[0, tops[c], step],
                x_length=x_len,
                y_length=height,
                tips=False,
                axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False},
            )
            ea = Axes(x_range=[0, DAYS, 1], y_range=[-YLIM, YLIM, 10], x_length=x_len,
                      y_length=height, tips=False,
                      axis_config={"color": GRID, "stroke_width": 2, "include_ticks": False},
                      x_axis_config={"color": TRUTH, "stroke_width": 2})  # fmt: skip
            va.move_to([x_mid, y, 0])
            ea.move_to([x_mid, y, 0])
            val_axes[c], err_axes[c] = va, ea
            val_ticks.add(*[
                txt(f"{v:g}", 18, TEXT_2).next_to(va.c2p(0, v), LEFT, buff=0.12)
                for v in np.arange(0, tops[c] + 1e-9, step)[1:]
            ])  # fmt: skip
            err_ticks.add(*[
                txt(s, 17, TEXT_2).next_to(ea.c2p(0, v), LEFT, buff=0.12)
                for v, s in ((-20, "−20%"), (0, "0"), (20, "+20%"))
            ])  # fmt: skip
            err_ticks.add(*[
                Line(ea.c2p(0, v), ea.c2p(DAYS, v), color=GRID, stroke_width=1)
                for v in (-20, -10, 10, 20)
            ])  # fmt: skip
            labels.add(txt(LABELS[c], 24, TEXT).next_to(va, LEFT, buff=0.95))
        bottom = val_axes[TARGETS[-1]]

        # feeds strip: neutral ticks, one labelled row per feed
        row_h, strip_top = 0.19, bottom.get_bottom()[1] - 0.16
        x0, x1 = bottom.c2p(0, 0)[0], bottom.c2p(DAYS, 0)[0]
        rows, strip = {}, VGroup()
        for i, (name, (label, _)) in enumerate(FEEDS.items()):
            base = strip_top - (i + 1) * row_h
            rows[name] = base
            strip.add(Line([x0, base, 0], [x1, base, 0], color=GRID, stroke_width=1))
            lab = txt(label, 15, TEXT_2)
            strip.add(lab.move_to([x0 - 0.15 - lab.width / 2, base + row_h * 0.4, 0]))
        strip_bottom = strip_top - len(FEEDS) * row_h
        strip.add(txt("Feeds", 22, TEXT).next_to([x0, (strip_top + strip_bottom) / 2, 0],
                                                LEFT, buff=1.55))  # fmt: skip
        days = VGroup(*[
            txt(str(d), 17, TEXT_2).move_to([bottom.c2p(d, 0)[0], strip_bottom - 0.17, 0])
            for d in range(0, DAYS + 1, 2)
        ])  # fmt: skip
        days.add(txt("Culture day", 20, TEXT_2).next_to(days, DOWN, buff=0.08))

        def feed_ticks(b, T):
            fe = feeds.loc[feeds["batch_id"] == b]

            def draw():
                now, g = T.get_value(), VGroup()
                for name, te, rel in zip(fe["feed"], fe["t"], fe["rel"]):
                    if te > now:
                        continue
                    x, base = bottom.c2p(te, 0)[0], rows[name]
                    h = max(rel, 0.12) * row_h * 0.85
                    flash = max(0.0, 1.0 - (now - te) / FLASH_DAYS)
                    g.add(Line([x, base, 0], [x, base + h, 0], stroke_width=3 + 5 * flash,
                               color=TRUTH if flash > 0.5 else FEED_TICK))  # fmt: skip
                return g

            return always_redraw(draw)

        def batch_tag(b):
            tag = txt(f"Batch {b.replace('Batch', '')}", 28, TEXT)
            return tag.to_edge(RIGHT, buff=0.4).match_y(title)

        # --- 1. behaviour: hybrid on batch 10 ----------------------------------------------
        b = BATCHES[0]
        g = {m: d.loc[d["batch_id"] == b].sort_values("t") for m, d in data.items()}
        t = g[HYBRID]["t"].to_numpy()
        T = ValueTracker(0.0)
        key_actual = VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=TRUTH, stroke_width=3),
                            Dot(radius=0.07, color=TRUTH))  # fmt: skip
        legend1 = VGroup(
            VGroup(key_actual, txt("Actual", 22, TEXT)).arrange(RIGHT, buff=0.15),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=MODELS[HYBRID][1], stroke_width=5),
                   txt("Hybrid prediction", 22, TEXT)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.5)  # fmt: skip
        legend1.next_to(caption, DOWN, aligned_edge=LEFT, buff=0.18)
        tag = batch_tag(b)

        self.add(*val_axes.values(), val_ticks, labels, strip, days)
        self.play(FadeIn(title), FadeIn(caption), FadeIn(legend1), FadeIn(tag), run_time=0.8)

        growing = VGroup()
        for c in TARGETS:
            va = val_axes[c]
            ya, yp = g[HYBRID][f"{c}_actual"].to_numpy(), g[HYBRID][f"{c}_pred"].to_numpy()
            growing.add(
                always_redraw(lambda va=va, ya=ya: line_through(
                    [va.c2p(a, v) for a, v in zip(*upto(t, ya, T.get_value()))], TRUTH, 3)),
                always_redraw(lambda va=va, ya=ya: VGroup(*[
                    Dot(va.c2p(a, v), radius=0.05, color=TRUTH)
                    for a, v in zip(t, ya) if a <= T.get_value() + 1e-9])),
                always_redraw(lambda va=va, yp=yp: line_through(
                    [va.c2p(a, v) for a, v in zip(*upto(t, yp, T.get_value()))],
                    MODELS[HYBRID][1], 5)),
            )  # fmt: skip
        ticks = feed_ticks(b, T)
        self.add(growing, ticks)
        self.play(T.animate.set_value(t[-1]), run_time=GROW, rate_func=linear)
        self.play(T.animate.set_value(t[-1] + FLASH_DAYS), run_time=0.3)
        self.wait(1.2)

        # freeze into static curves that can morph
        actual_lines, actual_dots, pred_lines = VGroup(), VGroup(), VGroup()
        for c in TARGETS:
            va = val_axes[c]
            ya, yp = g[HYBRID][f"{c}_actual"].to_numpy(), g[HYBRID][f"{c}_pred"].to_numpy()
            actual_lines.add(line_through([va.c2p(a, v) for a, v in zip(t, ya)], TRUTH, 3))
            actual_dots.add(*[Dot(va.c2p(a, v), radius=0.05, color=TRUTH) for a, v in zip(t, ya)])
            pred_lines.add(line_through([va.c2p(a, v) for a, v in zip(t, yp)],
                                        MODELS[HYBRID][1], 5))  # fmt: skip
        self.remove(growing)
        self.add(actual_lines, actual_dots, pred_lines)

        # --- 2. the morph into percent error -------------------------------------------------
        caption2 = txt("Measured against the truth: percent error, actual = 0",
                       22, TEXT_2).move_to(caption, aligned_edge=LEFT)  # fmt: skip
        zero_lines, zero_dots, err_lines, err_marks = VGroup(), VGroup(), VGroup(), VGroup()
        for c in TARGETS:
            ea = err_axes[c]
            e = g[HYBRID][f"{c}_err"].to_numpy()
            zero_lines.add(line_through([ea.c2p(a, 0) for a in t], TRUTH, 3))
            zero_dots.add(*[Dot(ea.c2p(a, 0), radius=0.05, color=TRUTH) for a in t])
            err_lines.add(line_through(err_points(ea, t, e), MODELS[HYBRID][1], 5))
            err_marks.add(off_scale(ea, t, e, MODELS[HYBRID][1]))
        self.play(FadeTransform(caption, caption2), FadeOut(val_ticks), run_time=0.6)
        self.play(
            *[ReplacementTransform(val_axes[c], err_axes[c]) for c in TARGETS],
            Transform(actual_lines, zero_lines),
            Transform(actual_dots, zero_dots),
            Transform(pred_lines, err_lines),
            run_time=3.0, rate_func=smooth,
        )  # fmt: skip
        self.play(FadeIn(err_ticks), FadeIn(err_marks), FadeOut(actual_lines), run_time=0.6)

        # --- 3. comparison: mechanistic and ML join -----------------------------------------
        def legend_with(values):
            items = []
            for m, (name, color) in MODELS.items():
                key = Line(LEFT * 0.3, RIGHT * 0.3, color=color, stroke_width=5)
                items.append(
                    VGroup(key, txt(name, 22, TEXT), txt(values[m], 22, TEXT_2)).arrange(
                        RIGHT, buff=0.15
                    )
                )
            lg = VGroup(*items).arrange(RIGHT, buff=0.5)
            return lg.next_to(caption, DOWN, aligned_edge=LEFT, buff=0.18)

        mapes = {m: f"{batch_mape(g[m]):.1f}%" for m in MODELS}
        legend2 = legend_with(mapes)
        caption3 = txt("Percent error of three approaches (MAPE of this batch)", 22, TEXT_2)
        caption3.move_to(caption, aligned_edge=LEFT)
        others = VGroup()
        for m in ("mechanistic_lite", "ml_sensors"):
            for c in TARGETS:
                ea, e = err_axes[c], g[m][f"{c}_err"].to_numpy()
                others.add(line_through(err_points(ea, t, e), MODELS[m][1], 3.5),
                           off_scale(ea, t, e, MODELS[m][1]))  # fmt: skip
        self.play(FadeOut(legend1), FadeTransform(caption2, caption3), run_time=0.5)
        self.play(Create(others), FadeIn(legend2), run_time=2.5)
        self.add(pred_lines, err_marks)  # hybrid on top
        self.wait(2.5)
        self.play(FadeOut(others), FadeOut(pred_lines), FadeOut(err_marks),
                  FadeOut(actual_dots), FadeOut(ticks), FadeOut(legend2), run_time=0.5)  # fmt: skip

        # --- 4. two more batches, error view, all models at once ----------------------------
        notes = {"Batch17": "A harder batch",
                 "Batch34": "Here machine learning matches the hybrid"}  # fmt: skip
        current_caption = caption3
        for b in BATCHES[1:]:
            g = {m: d.loc[d["batch_id"] == b].sort_values("t") for m, d in data.items()}
            t = g[HYBRID]["t"].to_numpy()
            T = ValueTracker(0.0)
            moving = VGroup()
            for m in ("mechanistic_lite", "ml_sensors", HYBRID):  # hybrid drawn last, on top
                color, width = MODELS[m][1], 5 if m == HYBRID else 3.5
                for c in TARGETS:
                    ea, e = err_axes[c], g[m][f"{c}_err"].to_numpy()
                    moving.add(
                        always_redraw(lambda ea=ea, e=e, color=color, width=width, t=t, T=T:
                                      line_through(err_points(ea, *upto(t, e, T.get_value())),
                                                   color, width)),
                        always_redraw(lambda ea=ea, e=e, color=color, t=t, T=T:
                                      off_scale(ea, t, e, color, T.get_value())),
                    )  # fmt: skip
            moving.add(feed_ticks(b, T))
            new_tag = batch_tag(b)
            new_caption = txt(notes[b], 22, TEXT_2).move_to(caption, aligned_edge=LEFT)
            legend = legend_with({m: f"{batch_mape(g[m]):.1f}%" for m in MODELS})
            self.add(moving)
            self.play(FadeTransform(tag, new_tag), FadeTransform(current_caption, new_caption),
                      run_time=0.5)  # fmt: skip
            tag, current_caption = new_tag, new_caption
            self.play(T.animate.set_value(t[-1]), run_time=GROW, rate_func=linear)
            self.play(T.animate.set_value(t[-1] + FLASH_DAYS), FadeIn(legend), run_time=0.4)
            self.wait(2.5)
            self.play(FadeOut(moving), FadeOut(legend), run_time=0.5)

        # --- 5. end card --------------------------------------------------------------------
        scores = {m: json.loads((METRICS / f"{m}.json").read_text())["holdout_mape"]["score"]
                  for m in MODELS}  # fmt: skip
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
        head = txt("All 10 holdout batches", 34, TEXT).move_to(UP * 2.0)
        rows_ = VGroup()
        for k, (m, (name, color)) in enumerate(MODELS.items()):
            y, bar_len = 0.9 - 0.85 * k, scores[m] * 0.45
            name_t = txt(name, 28, TEXT).move_to([-2.3, y, 0], aligned_edge=RIGHT)
            bar = Line([-2.0, y, 0], [-2.0 + bar_len, y, 0], color=color, stroke_width=28)
            value = txt(f"{scores[m]:.1f}%", 28, TEXT).next_to(bar, RIGHT, buff=0.25)
            rows_.add(VGroup(name_t, bar, value))
        foot = txt("Mean absolute percent error (MAPE), lower is better", 22, TEXT_2)
        foot.move_to(DOWN * 1.9)
        self.play(FadeIn(head), run_time=0.5)
        for r in rows_:
            self.play(FadeIn(r[0]), Create(r[1]), FadeIn(r[2]), run_time=0.7)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(3.0)
