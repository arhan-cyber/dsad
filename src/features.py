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

    # Advanced microstructure and volatility features.
    out["mid_change"] = g["mid_price"].diff()
    out["microprice_edge_change"] = g["microprice_edge"].diff()
    out["microprice_edge_mom_5"] = g["microprice_edge"].transform(lambda s: s.diff(5))

    bid_px_change = g["bid_price_1"].diff().fillna(0)
    ask_px_change = g["ask_price_1"].diff().fillna(0)
    bid_sz_change = g["bid_volume_1"].diff().fillna(0)
    ask_sz_change = g["ask_volume_1"].diff().fillna(0)
    out["ofi_l1"] = bid_px_change * out["bid_volume_1"] - ask_px_change * out["ask_volume_1"] + bid_sz_change - ask_sz_change
    out["ofi_l1_norm"] = _safe_div(out["ofi_l1"], out["bid_volume_1"] + out["ask_volume_1"])

    out["queue_bid_delta"] = bid_sz_change
    out["queue_ask_delta"] = ask_sz_change
    out["queue_imbalance_delta"] = out["queue_bid_delta"] - out["queue_ask_delta"]

    out["depth_weighted_spread"] = _safe_div(out["spread"], bid_depth + ask_depth)
    out["bid_curve_ratio_12"] = _safe_div(out["bid_volume_1"], out["bid_volume_2"])
    out["bid_curve_ratio_23"] = _safe_div(out["bid_volume_2"], out["bid_volume_3"])
    out["ask_curve_ratio_12"] = _safe_div(out["ask_volume_1"], out["ask_volume_2"])
    out["ask_curve_ratio_23"] = _safe_div(out["ask_volume_2"], out["ask_volume_3"])
    out["book_convexity"] = (out["bid_curve_ratio_12"] - out["bid_curve_ratio_23"]) - (
        out["ask_curve_ratio_12"] - out["ask_curve_ratio_23"]
    )

    out["vol_of_vol_20"] = g["vol_10"].transform(lambda s: s.rolling(20).std())
    out["ret_autocorr_10"] = g["ret_1"].transform(lambda s: s.rolling(10).corr(s.shift(1)))
    out["signed_ret_autocorr_10"] = g["ret_1"].transform(lambda s: np.sign(s).rolling(10).corr(np.sign(s).shift(1)))
    out["range_20"] = g["mid_price"].transform(lambda s: s.rolling(20).max() - s.rolling(20).min())
    out["breakout_up_20"] = _safe_div(out["mid_price"] - g["mid_price"].transform(lambda s: s.rolling(20).max()), out["mid_price"])
    out["breakout_down_20"] = _safe_div(g["mid_price"].transform(lambda s: s.rolling(20).min()) - out["mid_price"], out["mid_price"])

    vol_p33 = g["vol_10"].transform(lambda s: s.rolling(100, min_periods=20).quantile(0.33))
    vol_p66 = g["vol_10"].transform(lambda s: s.rolling(100, min_periods=20).quantile(0.66))
    out["vol_regime"] = np.where(out["vol_10"] <= vol_p33, -1, np.where(out["vol_10"] >= vol_p66, 1, 0))

    return out


def add_trade_flow_features(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_buy_qty_20"] = 0.0
    out["trade_sell_qty_20"] = 0.0
    out["trade_ofi_20"] = 0.0
    out["trade_participation_20"] = 0.0
    if trades is None or trades.empty:
        return out

    t = trades.copy()
    t = t[t["symbol"].astype(str) == str(out["product"].iloc[0])].copy()
    if t.empty:
        return out
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0.0)
    t["buy_qty"] = np.where(t["side"] == "BUY", t["quantity"], 0.0)
    t["sell_qty"] = np.where(t["side"] == "SELL", t["quantity"], 0.0)
    t = t.sort_values("timestamp")
    t["trade_buy_qty_20"] = t["buy_qty"].rolling(20, min_periods=1).sum()
    t["trade_sell_qty_20"] = t["sell_qty"].rolling(20, min_periods=1).sum()
    t["trade_ofi_20"] = t["trade_buy_qty_20"] - t["trade_sell_qty_20"]
    t["trade_participation_20"] = t["trade_buy_qty_20"] + t["trade_sell_qty_20"]

    merged = pd.merge_asof(
        out.sort_values("timestamp"),
        t[["timestamp", "trade_buy_qty_20", "trade_sell_qty_20", "trade_ofi_20", "trade_participation_20"]].sort_values(
            "timestamp"
        ),
        on="timestamp",
        direction="backward",
    )
    for col in ["trade_buy_qty_20", "trade_sell_qty_20", "trade_ofi_20", "trade_participation_20"]:
        merged[col] = merged[col].fillna(0.0)
    return merged.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


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
