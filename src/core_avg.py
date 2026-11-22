"""Consensus stage 1 (experiments_v6, idea 8): geometric mean of the stage-1
forecasts of six cores (hybrid ODE, hybrid ODE + glucose net, pure mechanistic
ODE, Monod, sensor-informed Monod, two-population). Written in the v3 stage-1
format so hybrid_v3 / experiments_v5 can use it as a core."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OOF = [(O / "hybrid", "states"), (O / "hybrid_glc", "states"), (O / "hybrid", "mechanistic"),
       (O / "core_monod", "monod"), (O / "core_monod_sens", "monod_sens"), (O / "core_twopop", "core_twopop")]
HOLD = [O / "hybrid/holdout_predictions.csv", O / "hybrid_glc/holdout_predictions.csv",
        O / "stack_v5/stage1_mech/holdout_predictions.csv", O / "stack_v5/stage1_monod/holdout_predictions.csv",
        O / "stack_v5/stage1_monod_sens/holdout_predictions.csv", O / "core_twopop/holdout_predictions.csv"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]


def main():
    key = ["batch_id", "variable", "t_key"]
    frames = []
    for i, (d, m) in enumerate(OOF):
        p = pd.read_csv(d / "predictions_vs_measured.csv", dtype={"batch_id": str})
        p = p[(p.split == "held-out (CV)") & (p.model == m)].copy()
        p["t_key"] = p.elapsed_time.round(6)
        p = p.set_index(key)
        frames.append(np.log(p.predicted.rename(f"p{i}")))
        if i == 0:
            base = p[["elapsed_time", "observed", "day"]]
    P = pd.concat(frames, axis=1)
    assert P.notna().all().all() and len(P) == len(base)
    out = base.copy()
    out["predicted"] = np.exp(P.mean(axis=1))
    out = out.reset_index().drop(columns="t_key")
    out["model"], out["split"] = "avg", "held-out (CV)"
    out["error"] = out.predicted - out.observed
    out["abs_pct_error"] = 100 * out.error.abs() / out.observed
    (O / "core_avg").mkdir(exist_ok=True)
    out.to_csv(O / "core_avg/predictions_vs_measured.csv", index=False)
    print("consensus stage-1 held-out MAPE:", out.groupby("variable").abs_pct_error.mean()[TARGETS].round(2).to_dict(),
          "mean", round(out.groupby("variable").abs_pct_error.mean().mean(), 2))
    H = [pd.read_csv(p, dtype={"batch_id": str}) for p in HOLD]
    h = H[0][["batch_id", "elapsed_time"]].copy()
    for hh in H:
        assert (hh.batch_id == h.batch_id).all()
    for v in TARGETS:
        h[v] = np.exp(np.mean([np.log(hh[v].to_numpy()) for hh in H], axis=0))
    h.to_csv(O / "core_avg/holdout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
