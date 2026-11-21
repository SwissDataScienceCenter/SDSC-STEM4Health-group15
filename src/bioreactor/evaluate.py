"""Shared evaluation metrics."""
import numpy as np
import pandas as pd

from .data import STATES


def metrics(arr, preds, eps=1e-6):
    """Per state: RMSE, NRMSE (= RMSE / mean observed), R2, MAPE [%]; row 'all' = pooled MAPE."""
    Y = np.vstack(arr.obs_y); Yh = np.vstack(preds)
    rmse = np.sqrt(np.mean((Yh - Y) ** 2, axis=0))
    r2 = 1 - np.sum((Yh - Y) ** 2, axis=0) / np.sum((Y - Y.mean(0)) ** 2, axis=0)
    ape = np.abs(Yh - Y) / np.maximum(np.abs(Y), eps)
    out = pd.DataFrame({"RMSE": rmse, "NRMSE": rmse / Y.mean(0), "R2": r2,
                        "MAPE_%": 100 * ape.mean(axis=0)}, index=STATES)
    out.loc["all"] = [np.nan, np.nan, np.nan, 100 * ape.mean()]
    return out.round(3)


def mape_row(arr, preds):
    """MAPE % per state + pooled, as a flat dict (for comparison tables)."""
    m = metrics(arr, preds)["MAPE_%"]
    return m.to_dict()
