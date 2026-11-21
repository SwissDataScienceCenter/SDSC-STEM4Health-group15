"""Fixed batch-level split shared by the team -> outputs/splits.csv

- 12 hidden test batches: the challenge's own test set (role = 'test')
- 10 of the 49 measured batches: lock-box hold-out, touched once at the end (role = 'holdout')
- 39 remaining: development set with 5 grouped CV folds (role = 'dev', fold 0-4)
Splitting is by whole batch (never by rows) and stratified on day-0 VCD,
the input most correlated with the outcome, so every part covers the same range.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bioreactor.data import load, ROOT  # noqa: E402

SEED = 42
df, _, train_ids, test_ids = load()
rng = np.random.default_rng(SEED)
ic = df[df.elapsed_time == 0].set_index("batch_id").loc[train_ids, "vcd"].sort_values()

# hold-out: one random batch from each block of ~5 consecutive day-0 VCD values
blocks = np.array_split(ic.index.values, 10)
holdout = [rng.choice(b) for b in blocks]
dev = ic.drop(holdout)
# dev folds: deal batches round-robin over 5 folds in VCD order, shuffled inside each block of 5
folds = {}
for blk in np.array_split(dev.index.values, int(np.ceil(len(dev) / 5))):
    for f, b in zip(rng.permutation(5), blk):
        folds[b] = int(f)

rows = ([(b, "test", -1) for b in test_ids] + [(b, "holdout", -1) for b in holdout]
        + [(b, "dev", folds[b]) for b in dev.index])
s = pd.DataFrame(rows, columns=["batch_id", "role", "fold"])
s.to_csv(ROOT / "outputs" / "splits.csv", index=False)
print(s.role.value_counts().to_string())
print("holdout:", sorted(holdout, key=lambda x: int(x[5:])))
print("fold sizes:", s[s.role == "dev"].fold.value_counts().sort_index().to_dict())
