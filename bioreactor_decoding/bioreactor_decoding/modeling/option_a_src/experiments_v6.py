"""v6 experiments: 10 more ideas on top of the v5 stack (CV 7.366, leaderboard 7.137).

Same 7-fold grouped CV over whole batches. Stack weights are always fitted on
the other folds only (nested). Every trial is appended to
outputs/experiments_v6/trials.csv.

    python src/experiments_v6.py IDEA [IDEA ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_v5 as x  # noqa: E402
import experiments_v5_ideas as ideas  # noqa: E402
import hybrid_v3 as h3  # noqa: E402

ROOT = h3.ROOT
CV = ROOT / "outputs" / "stack_v5" / "cv"
OUT = ROOT / "outputs" / "experiments_v6"
TARGETS = h3.TARGETS
x.OUT = CV  # cached per-core parts live here
BASE_SRCS = ["states", "glc", "mech", "monod", "msens", "states_fp", "glc_fp", "mech_fp", "monod_fp", "msens_fp", "raw"]
SEG2 = [range(1, 7), range(7, 14)]


def log_trial(idea, s, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trials.csv"
    row = pd.DataFrame([{"idea": idea, **{k: round(float(v), 3) for k, v in s.items()}, "note": note}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{idea:44s} " + "  ".join(f"{v}={s[v]:.3f}" for v in TARGETS)
          + f"  SCORE={s['leaderboard_score']:.3f}  {note}", flush=True)


# ------------------------------------------------------------ stacking
def weights_old(L, y):
    return ideas._nested_weights(np.exp(L), y, None)


def weights_simplex(L, y, steps=(0.1, 0.05, 0.02, 0.01, 0.005), n_iter=300):
    """Coordinate search on the simplex that can move a weight all the way to 0."""
    k = L.shape[1]
    w = np.full(k, 1.0 / k)

    def err(w):
        return np.mean(np.abs(np.exp(L @ w) / y - 1))

    best = err(w)
    for step in steps:
        for _ in range(n_iter):
            improved = False
            for i in range(k):
                for j in range(k):
                    if i == j or w[j] <= 0:
                        continue
                    d = min(step, w[j])
                    w2 = w.copy()
                    w2[i] += d
                    w2[j] -= d
                    e = err(w2)
                    if e < best - 1e-10:
                        best, w, improved = e, w2, True
            if not improved:
                break
    return w


def stack(mem, srcs, segments=SEG2, wfun=weights_simplex, rows=None, clip=True, return_w=False):
    """Nested stacking: per variable and day segment, weights fitted on other folds."""
    o = mem[["batch_id", "t_key", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
    day = o.day.to_numpy()
    W = {}
    for v in TARGETS:
        L = np.column_stack([np.log(mem[f"{s}|{v}"].to_numpy()) for s in srcs])
        y = mem["obs_" + v].to_numpy()
        out = np.zeros(len(o))
        for f in range(h3.N_FOLDS):
            for si, seg in enumerate(segments):
                sm_ = np.isin(day, list(seg))
                tr, te = (o.fold != f).to_numpy() & sm_, (o.fold == f).to_numpy() & sm_
                w = wfun(L[tr], y[tr])
                W.setdefault((v, si), []).append(w)
                out[te] = np.exp(L[te] @ w)
        o[f"st|{v}"] = out
    o = x.floor_tcd(o, "st")
    if clip:
        o = x.idea_clip(rows, o, src="st", dst="st")
    if return_w:
        return o, {k: np.mean(v, 0) for k, v in W.items()}
    return o


def load():
    mem = pd.read_csv(CV / "members_oof.csv", dtype={"batch_id": str})
    rows, events = x.build()  # states core rows (for g_lo / g_hi)
    assert (rows.batch_id.to_numpy() == mem.batch_id.to_numpy()).all()
    return mem, rows, events


def add_parts(mem, core, pref, cols=("hyb", "dd", "fin", "fps")):
    p = pd.read_csv(CV / f"parts_{core}.csv", dtype={"batch_id": str})
    assert (p.batch_id.to_numpy() == mem.batch_id.to_numpy()).all()
    for c in cols:
        for v in TARGETS:
            name = {"fin": pref, "fps": f"{pref}_fp"}.get(c, f"{pref}_{c}")
            mem[f"{name}|{v}"] = p[f"{c}|{v}"].to_numpy()
    return mem


def main(argv):
    t0 = time.time()
    mem, rows, events = load()
    import experiments_v6_ideas as iv6  # noqa: E402
    for name in argv:
        getattr(iv6, name)(mem, rows, events)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
