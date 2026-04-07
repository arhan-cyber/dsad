from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("product", group_keys=False)

    out["ret_1"] = g["mid_price"].pct_change()
    out["mom_5"] = g["mid_price"].pct_change(5)
    out["mom_20"] = g["mid_price"].pct_change(20)

    out["sma_5"] = g["mid_price"].transform(lambda s: s.rolling(5).mean())
    out["sma_20"] = g["mid_price"].transform(lambda s: s.rolling(20).mean())
    out["ema_8"] = g["mid_price"].transform(lambda s: s.ewm(span=8, adjust=False).mean())
    out["ema_21"] = g["mid_price"].transform(lambda s: s.ewm(span=21, adjust=False).mean())
    out["sma_cross_5_20"] = out["sma_5"] - out["sma_20"]
    out["ema_cross_8_21"] = out["ema_8"] - out["ema_21"]

    roll_mean = g["mid_price"].transform(lambda s: s.rolling(20).mean())
    roll_std = g["mid_price"].transform(lambda s: s.rolling(20).std())
    out["zscore_20"] = _safe_div(out["mid_price"] - roll_mean, roll_std)

    out["vol_10"] = g["ret_1"].transform(lambda s: s.rolling(10).std())
    out["vol_50"] = g["ret_1"].transform(lambda s: s.rolling(50).std())

    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    out["rel_spread"] = _safe_div(out["spread"], out["mid_price"])

    bid_depth = out[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1)
    ask_depth = out[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1)
    out["bid_depth_l1_l3"] = bid_depth
    out["ask_depth_l1_l3"] = ask_depth
    out["l1_imbalance"] = _safe_div(out["bid_volume_1"] - out["ask_volume_1"], out["bid_volume_1"] + out["ask_volume_1"])
    out["depth_imbalance_l1_l3"] = _safe_div(bid_depth - ask_depth, bid_depth + ask_depth)

    out["microprice"] = _safe_div(
        (out["ask_price_1"] * out["bid_volume_1"]) + (out["bid_price_1"] * out["ask_volume_1"]),
        out["bid_volume_1"] + out["ask_volume_1"],
    )
    out["microprice_edge"] = out["microprice"] - out["mid_price"]

    out["book_slope_proxy"] = _safe_div(out["ask_price_3"] - out["bid_price_3"], bid_depth + ask_depth)
    out["dt"] = g["timestamp"].diff()
    out["quote_update_intensity"] = _safe_div(1, out["dt"])

    return out


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(
        [
            "day",
            "timestamp",
            "product",
            "mid_price",
            "profit_and_loss",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
        ]
    )
    return [
        c
        for c in df.columns
        if c not in excluded and not c.startswith("fwd_ret_") and pd.api.types.is_numeric_dtype(df[c])
    ]
