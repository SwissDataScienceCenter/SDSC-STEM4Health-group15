"""Generic runner for alternative mechanistic cores (experiments_v6).

Fits a core module (PARAMS/P0/LB/UB/observe) with 7-fold grouped CV, then on
all 49 training batches, and writes
    outputs/<name>/predictions_vs_measured.csv   (held-out CV + in-sample, v3 format)
    outputs/<name>/holdout_predictions.csv       (12 holdout batches, template format)

Options
    --core monod|twopop     structure
    --bags B                bootstrap-bag the core: fit on B batch resamples, average predictions
    --loss soft_l1|linear   least-squares loss on relative errors
    --seed S                multi-start seed
    --starts K              multi-starts per fit

    python src/core_runner.py --core twopop --name core_twopop
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sartorius_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = sm.TARGETS
MODULES = {"monod": "core_monod", "twopop": "core_twopop"}


def fit(mod, bs, seed=0, starts=3, loss="soft_l1", max_nfev=400, p_init=None):
    y = bs.obs[TARGETS].to_numpy()

    def resid(q):
        pred = mod.observe(bs, np.exp(q), bs.obs)
        return ((np.maximum(pred, 1e-6) - y) / y).ravel()

    rng = np.random.default_rng(seed)
    best = None
    base = np.log(mod.P0 if p_init is None else p_init)
    for s in range(starts):
        q0 = np.clip(base + (0 if s == 0 else rng.normal(0, 0.4, len(base))), mod.LB + 1e-6, mod.UB - 1e-6)
        try:
            r = least_squares(resid, q0, bounds=(mod.LB, mod.UB), loss=loss, f_scale=0.1, max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return np.exp(best.x)


def _job(args):
    core, train, tests, seed, starts, loss, bags = args
    os.environ["NUMBA_NUM_THREADS"] = "1"
    mod = importlib.import_module(MODULES[core])
    rng = np.random.default_rng(seed)
    preds = [[] for _ in tests]
    n = len(train.ids)
    for b in range(bags):
        sub = train if bags == 1 else train.subset(np.sort(rng.choice(n, n, replace=True)))
        p = fit(mod, sub, seed + b, starts, loss)
        for i, (bs, rows) in enumerate(tests):
            preds[i].append(mod.observe(bs, p, rows))
    return [np.mean(pp, axis=0) for pp in preds]


def frame(bs, pred, model, split):
    base = bs.obs[["batch_idx", "elapsed_time"]].copy()
    base["batch_id"] = [bs.ids[i] for i in base.batch_idx]
    parts = []
    for j, t in enumerate(TARGETS):
        part = base.copy()
        part["variable"], part["observed"], part["predicted"] = t, bs.obs[t].to_numpy(), pred[:, j]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).drop(columns="batch_idx")
    out["model"], out["split"] = model, split
    out["day"] = np.rint(out.elapsed_time).astype(int)
    out["error"] = out.predicted - out.observed
    out["abs_pct_error"] = 100.0 * out.error.abs() / out.observed.abs()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="twopop")
    ap.add_argument("--name", required=True)
    ap.add_argument("--bags", type=int, default=1)
    ap.add_argument("--loss", default="soft_l1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", type=int, default=3)
    a = ap.parse_args()
    out_dir = ROOT / "outputs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    holdout = sm.build_batches(events, holdout_ids)
    idx = np.arange(len(train_ids))
    folds = list(GroupKFold(7).split(idx, groups=idx))
    jobs = []
    for tr, te in folds:
        test = train.subset(te)
        jobs.append((a.core, train.subset(tr), [(test, test.obs)], a.seed, a.starts, a.loss, a.bags))
    jobs.append((a.core, train, [(train, train.obs), (holdout, holdout.to_predict)], a.seed, a.starts, a.loss, a.bags))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_job, jobs))
    frames = [frame(train.subset(te), r[0], a.name, "held-out (CV)") for (tr, te), r in zip(folds, res[:-1])]
    frames.append(frame(train, res[-1][0], a.name, "in-sample fit"))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out_dir / "predictions_vs_measured.csv", index=False)

    tp = holdout.to_predict
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in tp.batch_idx], "t_key": tp.elapsed_time.round(6)})
    for j, t in enumerate(TARGETS):
        sub[t] = res[-1][1][:, j]
    sub = (template[["batch_id", "elapsed_time"]].assign(t_key=template.elapsed_time.round(6))
           .merge(sub, on=["batch_id", "t_key"], how="left").drop(columns="t_key"))
    assert sub[TARGETS].notna().all().all()
    sub.to_csv(out_dir / "holdout_predictions.csv", index=False)
    m = preds[preds.split == "held-out (CV)"].groupby("variable").abs_pct_error.mean()[TARGETS]
    print(a.name, "held-out MAPE:", m.round(2).to_dict(), "mean", round(m.mean(), 2))


if __name__ == "__main__":
    main()
