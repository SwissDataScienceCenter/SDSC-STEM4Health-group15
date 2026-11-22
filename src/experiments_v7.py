"""v7: new line of trials from hybrid_v3 with at most 4 models in the final
geometric blend. Same 7-fold grouped CV over whole batches as hybrid_v3 (7.674).
Member selection and blend weights are fitted on the other folds only.
Every trial is appended to outputs/experiments_v7/trials.csv.

    python src/experiments_v7.py IDEA [IDEA ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_v5 as x  # noqa: E402
import experiments_v6 as e  # noqa: E402
import experiments_v6_ideas as iv6  # noqa: E402
import hybrid_v3 as h3  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "experiments_v7"
CV = e.CV
TARGETS = h3.TARGETS
MAX_MODELS = 4


def log_trial(idea, s, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trials.csv"
    row = pd.DataFrame([{"idea": idea, **{k: round(float(v), 3) for k, v in s.items()}, "note": note}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{idea:46s} " + "  ".join(f"{v}={s[v]:.3f}" for v in TARGETS)
          + f"  SCORE={s['leaderboard_score']:.3f}  {note}", flush=True)


def _blend_err(L, y, w):
    return np.mean(np.abs(np.exp(L @ w) / y - 1))


def select_models(mem, cands, tr, k=MAX_MODELS):
    """Greedy forward selection of <= k models (same set for all variables),
    scored by the mean over variables of the in-fold stacked MAPE."""
    day = mem.day.to_numpy()
    chosen = []
    best_total = np.inf
    while len(chosen) < k:
        best = None
        for c in cands:
            if c in chosen:
                continue
            s = chosen + [c]
            tot = 0.0
            for v in TARGETS:
                L = np.column_stack([np.log(mem[f"{q}|{v}"].to_numpy()) for q in s])
                y = mem["obs_" + v].to_numpy()
                for seg in e.SEG2:
                    m = tr & np.isin(day, list(seg))
                    w = e.weights_simplex(L[m], y[m], steps=(0.1, 0.05, 0.02))
                    tot += _blend_err(L[m], y[m], w) * m.sum()
            if best is None or tot < best[0]:
                best = (tot, c)
        if best[0] >= best_total - 1e-9:
            break
        best_total = best[0]
        chosen.append(best[1])
    return chosen


def stack_k(mem, cands, rows, k=MAX_MODELS, finish=True):
    """Nested: per fold, select <= k models and fit their weights on the other folds."""
    o = mem[["batch_id", "t_key", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
    day = o.day.to_numpy()
    picks = []
    for v in TARGETS:
        o[f"st|{v}"] = np.nan
    for f in range(h3.N_FOLDS):
        tr, te = (o.fold != f).to_numpy(), (o.fold == f).to_numpy()
        s = select_models(mem, cands, tr, k)
        picks.append(s)
        for v in TARGETS:
            L = np.column_stack([np.log(mem[f"{q}|{v}"].to_numpy()) for q in s])
            y = mem["obs_" + v].to_numpy()
            for seg in e.SEG2:
                mt, me_ = tr & np.isin(day, list(seg)), te & np.isin(day, list(seg))
                w = e.weights_simplex(L[mt], y[mt])
                o.loc[me_, f"st|{v}"] = np.exp(L[me_] @ w)
    o = x.floor_tcd(o, "st")
    if finish:
        o = iv6.finish_all(mem, o, rows)
        return o, picks, "f"
    return o, picks, "st"


def main(argv):
    t0 = time.time()
    mem = pd.read_csv(CV / "members_oof_v6h.csv", dtype={"batch_id": str})
    rows, events = x.build()
    import experiments_v7_ideas as iv7  # noqa: E402
    for name in argv:
        getattr(iv7, name)(mem, rows, events)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
