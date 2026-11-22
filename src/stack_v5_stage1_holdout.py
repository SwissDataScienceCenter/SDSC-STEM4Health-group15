"""Stage-1 holdout predictions for the cores used in the v5 stack that were not
saved before: pure mechanistic (Sartorius-style) and the two Monod cores.
All are fitted on the 49 training batches only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sartorius_model as sm  # noqa: E402
import core_monod as cm  # noqa: E402
import core_monod_sens as cms  # noqa: E402
import hybrid_v3 as h3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stack_v5"


def write(name, holdout, pred, template):
    tp = holdout.to_predict
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in tp.batch_idx], "t_key": tp.elapsed_time.round(6)})
    for j, t in enumerate(sm.TARGETS):
        sub[t] = pred[:, j]
    sub = (template[["batch_id", "elapsed_time"]].assign(t_key=template.elapsed_time.round(6))
           .merge(sub, on=["batch_id", "t_key"], how="left").drop(columns="t_key"))
    assert sub[sm.TARGETS].notna().all().all()
    d = OUT / f"stage1_{name}"
    d.mkdir(parents=True, exist_ok=True)
    sub.to_csv(d / "holdout_predictions.csv", index=False)
    print(name, "written", sub[sm.TARGETS].mean().round(3).to_dict(), flush=True)


def main(which):
    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    holdout = sm.build_batches(events, holdout_ids)
    tp = holdout.to_predict
    if "monod" in which:
        p = pd.read_csv(ROOT / "outputs" / "core_monod" / "params_full.csv", index_col=0).iloc[:, 0].to_numpy()
        write("monod", holdout, cm.observe(holdout, p, tp), template)
    if "monod_sens" in which:
        sb, _, _ = h3.sensor_features(events.assign(batch_id=events.batch_id.astype(str)))
        Ztr = sb.loc[train_ids, cms.ZCOLS].to_numpy(float)
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
        p_init = pd.read_csv(ROOT / "outputs" / "core_monod" / "params_full.csv", index_col=0).iloc[:, 0].to_numpy()
        p, beta = cms.fit(train, (Ztr - mu) / sd, 0, p_init=p_init)
        Zho = (sb.loc[holdout_ids, cms.ZCOLS].to_numpy(float) - mu) / sd
        write("monod_sens", holdout, cms.observe(holdout, p, tp, beta, Zho), template)
    if "mech" in which:
        fits = sm.fit_many({"full": train}, n_starts=6, workers=6, max_nfev=300, objective="relative", loss="soft_l1")
        p = fits["full"][0]
        traj = sm.simulate(holdout, p)
        write("mech", holdout, sm.observe(traj, tp.batch_idx, tp.step), template)


if __name__ == "__main__":
    main(sys.argv[1:])
