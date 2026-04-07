from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("product", group_keys=False)["mid_price"]
    for h in horizons:
        out[f"fwd_ret_{h}"] = g.shift(-h) / out["mid_price"] - 1
    return out


def _hit_rate(signal: pd.Series, target: pd.Series) -> float:
    aligned = pd.concat([signal, target], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float((np.sign(aligned.iloc[:, 0]) == np.sign(aligned.iloc[:, 1])).mean())


def score_signals(df: pd.DataFrame, feature_cols: list[str], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        for h in horizons:
            target_col = f"fwd_ret_{h}"
            if feature == target_col:
                continue
            sample = df[[feature, target_col]].dropna()
            if sample.empty:
                rows.append({"feature": feature, "horizon": h, "ic": np.nan, "hit_rate": np.nan, "samples": 0})
                continue
            signal = sample.iloc[:, 0]
            target = sample.iloc[:, 1]
            ic = float(signal.corr(target))
            hr = _hit_rate(signal, target)
            rows.append({"feature": feature, "horizon": h, "ic": ic, "hit_rate": hr, "samples": int(len(sample))})
    out = pd.DataFrame(rows)
    out["abs_ic"] = out["ic"].abs()
    return out.sort_values(["abs_ic", "hit_rate"], ascending=[False, False]).reset_index(drop=True)


def bucketed_forward_returns(df: pd.DataFrame, feature: str, horizon: int, buckets: int = 5) -> pd.DataFrame:
    target_col = f"fwd_ret_{horizon}"
    sample = df[[feature, target_col]].dropna().copy()
    if sample.empty:
        return pd.DataFrame(columns=["bucket", "avg_fwd_return"])
    sample["bucket"] = pd.qcut(sample[feature], q=buckets, labels=False, duplicates="drop")
    return sample.groupby("bucket", as_index=False)[target_col].mean().rename(columns={target_col: "avg_fwd_return"})
