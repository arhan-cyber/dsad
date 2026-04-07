from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_simulation_data
from src.features import add_features, default_feature_columns
from src.plots import correlation_heatmap, histogram, scatter, time_series
from src.scoring import add_forward_returns, bucketed_forward_returns, score_signals

st.set_page_config(page_title="Trading Alpha Visualizer", layout="wide")
st.title("Trading Alpha Visualizer")
st.caption("Explore candidate alpha signals and evaluate predictive power across forward horizons.")

uploaded = st.file_uploader("Upload simulation log file (.log or .csv)", type=["log", "csv"])
if uploaded is None:
    st.info("Upload a .log/.csv file to begin.")
    st.stop()

parsed = load_simulation_data(uploaded)
base_df = parsed["snapshots"]
trades_df = parsed["trades"]
engine_logs_df = parsed["logs"]
meta = parsed["meta"]

if base_df.empty:
    st.error("No order-book snapshots could be parsed from this file.")
    st.stop()

if meta.get("submissionId"):
    st.caption(f"Submission ID: `{meta['submissionId']}`")

products = sorted(base_df["product"].unique().tolist())
horizons = st.sidebar.multiselect("Forward horizons (ticks)", [1, 5, 10, 20, 50, 100], default=[1, 5, 10, 20, 50])
product = st.sidebar.selectbox("Product", products)

df = base_df[base_df["product"] == product].copy()
ts_values = sorted(df["timestamp"].dropna().astype(int).unique().tolist())
default_ts = ts_values[0]
global_ts = st.sidebar.select_slider("Global timestamp", options=ts_values, value=default_ts, key="global_timestamp")
global_ts_range = st.sidebar.select_slider(
    "Global timestamp range",
    options=ts_values,
    value=(ts_values[0], ts_values[-1]),
    key="global_timestamp_range",
)
df_range = df[(df["timestamp"] >= global_ts_range[0]) & (df["timestamp"] <= global_ts_range[1])].copy()
if df_range.empty:
    df_range = df.copy()

df = add_features(df)
df = add_forward_returns(df, horizons)
df_range = add_features(df_range)
df_range = add_forward_returns(df_range, horizons)
feature_cols = default_feature_columns(df)
product_trades = trades_df[trades_df["symbol"] == product].copy() if not trades_df.empty else trades_df.copy()
product_trades_range = (
    product_trades[
        (product_trades["timestamp"] >= global_ts_range[0]) & (product_trades["timestamp"] <= global_ts_range[1])
    ].copy()
    if not product_trades.empty
    else product_trades.copy()
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Data Health", "Feature Explorer", "Diagnostics", "Leaderboard", "Trades & Moment", "Market Prices", "Volatility"]
)

with tab1:
    st.subheader("Data Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows (in range)", len(df_range))
    col2.metric("Products", len(products))
    col3.metric("Null cells (in range)", int(df_range.isna().sum().sum()))
    st.dataframe(df_range.isna().mean().sort_values(ascending=False).rename("null_ratio").to_frame())

with tab2:
    st.subheader("Feature Explorer")
    feature = st.selectbox("Feature", feature_cols, index=0)
    st.plotly_chart(time_series(df_range, x="timestamp", y=feature, title=f"{feature} over time"), use_container_width=True)
    st.plotly_chart(histogram(df_range, x=feature, title=f"{feature} distribution"), use_container_width=True)
    top_corr_cols = feature_cols[: min(20, len(feature_cols))]
    st.plotly_chart(correlation_heatmap(df_range, top_corr_cols), use_container_width=True)

with tab3:
    st.subheader("Signal Diagnostics")
    feature = st.selectbox("Feature for diagnostics", feature_cols, index=0, key="diag_feature")
    horizon = st.selectbox("Horizon", horizons, index=0, key="diag_h")
    target_col = f"fwd_ret_{horizon}"
    diag_df = df_range[[feature, target_col]].dropna()
    st.plotly_chart(
        scatter(diag_df, x=feature, y=target_col, title=f"{feature} vs {target_col}"),
        use_container_width=True,
    )
    bucket_df = bucketed_forward_returns(df_range, feature, horizon)
    st.bar_chart(bucket_df.set_index("bucket")["avg_fwd_return"])

with tab4:
    st.subheader("Signal Leaderboard")
    scores = score_signals(df_range, feature_cols, horizons)
    st.dataframe(scores, use_container_width=True)
    st.download_button(
        "Download leaderboard CSV",
        data=scores.to_csv(index=False).encode("utf-8"),
        file_name=f"alpha_scores_{product}.csv",
        mime="text/csv",
    )

with tab5:
    st.subheader("Trades and Point-in-Time Stats")
    selected_ts = global_ts

    # Use latest snapshot at or before selected timestamp for stable point-in-time state.
    snap_at_ts = df[df["timestamp"] <= selected_ts].tail(1)
    if snap_at_ts.empty:
        snap_at_ts = df.head(1)
    snap = snap_at_ts.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mid Price", f"{float(snap['mid_price']):.2f}")
    c2.metric("Spread", f"{float(snap['ask_price_1'] - snap['bid_price_1']):.2f}")
    c3.metric("PnL", f"{float(snap['profit_and_loss']):.2f}")
    l1_imb = snap["l1_imbalance"] if "l1_imbalance" in snap_at_ts.columns else 0.0
    c4.metric("L1 Imbalance", f"{float(l1_imb):.3f}")

    st.write(
        f"Book snapshot at t={int(snap['timestamp'])}: "
        f"Bid1 {snap['bid_price_1']} x {snap['bid_volume_1']} | "
        f"Ask1 {snap['ask_price_1']} x {snap['ask_volume_1']}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["mid_price"], mode="lines", name="Mid Price"))
    if not product_trades_range.empty:
        buys = product_trades_range[product_trades_range["side"] == "BUY"]
        sells = product_trades_range[product_trades_range["side"] == "SELL"]
        others = product_trades_range[product_trades_range["side"] == "OTHER"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy Trades",
                    marker=dict(symbol="triangle-up", size=9, color="green"),
                    text=buys["quantity"],
                    hovertemplate="BUY @ %{y}<br>Qty=%{text}<br>t=%{x}<extra></extra>",
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell Trades",
                    marker=dict(symbol="triangle-down", size=9, color="red"),
                    text=sells["quantity"],
                    hovertemplate="SELL @ %{y}<br>Qty=%{text}<br>t=%{x}<extra></extra>",
                )
            )
        if not others.empty:
            fig.add_trace(
                go.Scatter(
                    x=others["timestamp"],
                    y=others["price"],
                    mode="markers",
                    name="Other Trades",
                    marker=dict(size=7, color="gray"),
                    text=others["quantity"],
                    hovertemplate="OTHER @ %{y}<br>Qty=%{text}<br>t=%{x}<extra></extra>",
                )
            )
    fig.add_vline(x=selected_ts, line_dash="dash", line_color="orange")
    fig.update_layout(title=f"{product} mid-price with trades", xaxis_title="Timestamp", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    if not product_trades.empty:
        recent_trades = product_trades[(product_trades["timestamp"] >= selected_ts - 1000) & (product_trades["timestamp"] <= selected_ts + 1000)]
        st.write("Trades in +/-1000 timestamp window")
        st.dataframe(recent_trades[["timestamp", "side", "price", "quantity", "buyer", "seller"]], use_container_width=True)
    else:
        st.info(f"No tradeHistory entries found for {product}.")

    if not engine_logs_df.empty:
        log_row = engine_logs_df[engine_logs_df["timestamp"] <= selected_ts].tail(1)
        if not log_row.empty:
            st.write("Engine logs at this moment")
            st.text_area("sandboxLog", str(log_row.iloc[0].get("sandboxlog", "")), height=80)
            st.text_area("lambdaLog", str(log_row.iloc[0].get("lambdalog", "")), height=80)

with tab6:
    st.subheader("Bid / Ask / Mid Price Timeline")
    show_l2 = st.checkbox("Show level-2 bid/ask", value=True, key="show_l2")
    show_l3 = st.checkbox("Show level-3 bid/ask", value=False, key="show_l3")

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["mid_price"], mode="lines", name="Mid Price"))
    price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["bid_price_1"], mode="lines", name="Bid L1"))
    price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["ask_price_1"], mode="lines", name="Ask L1"))

    if show_l2:
        price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["bid_price_2"], mode="lines", name="Bid L2"))
        price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["ask_price_2"], mode="lines", name="Ask L2"))
    if show_l3:
        price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["bid_price_3"], mode="lines", name="Bid L3"))
        price_fig.add_trace(go.Scatter(x=df_range["timestamp"], y=df_range["ask_price_3"], mode="lines", name="Ask L3"))

    price_fig.update_layout(
        title=f"{product} order-book prices over time",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
    )
    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("Depth at Selected Timestamp")
    depth_point = df[df["timestamp"] <= selected_ts].tail(1)
    if not depth_point.empty:
        row = depth_point.iloc[0]
        depth_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Bid L1", "Bid L2", "Bid L3", "Ask L1", "Ask L2", "Ask L3"],
                    y=[
                        row["bid_volume_1"] if row["bid_volume_1"] is not None else 0,
                        row["bid_volume_2"] if row["bid_volume_2"] is not None else 0,
                        row["bid_volume_3"] if row["bid_volume_3"] is not None else 0,
                        row["ask_volume_1"] if row["ask_volume_1"] is not None else 0,
                        row["ask_volume_2"] if row["ask_volume_2"] is not None else 0,
                        row["ask_volume_3"] if row["ask_volume_3"] is not None else 0,
                    ],
                )
            ]
        )
        depth_fig.update_layout(title=f"{product} depth snapshot @ t={int(row['timestamp'])}", yaxis_title="Volume")
        st.plotly_chart(depth_fig, use_container_width=True)

with tab7:
    st.subheader("Interactive Volatility")
    vol_window = st.slider("Volatility rolling window", min_value=5, max_value=200, value=20, step=1, key="vol_window")
    tmp = df_range[["timestamp", "mid_price"]].copy()
    tmp["ret_1"] = tmp["mid_price"].pct_change()
    tmp["rolling_vol"] = tmp["ret_1"].rolling(vol_window).std()
    tmp["rolling_vol_annualized"] = tmp["rolling_vol"] * (252**0.5)

    vfig = go.Figure()
    vfig.add_trace(go.Scatter(x=tmp["timestamp"], y=tmp["rolling_vol"], mode="lines", name=f"Vol ({vol_window})"))
    vfig.add_trace(
        go.Scatter(x=tmp["timestamp"], y=tmp["rolling_vol_annualized"], mode="lines", name=f"Vol Annualized ({vol_window})")
    )
    vfig.add_vline(x=selected_ts, line_dash="dash", line_color="orange")
    vfig.update_layout(title=f"{product} volatility in selected range", xaxis_title="Timestamp", yaxis_title="Volatility")
    st.plotly_chart(vfig, use_container_width=True)

    vol_point = tmp[tmp["timestamp"] <= selected_ts].tail(1)
    if not vol_point.empty:
        p1, p2, p3 = st.columns(3)
        p1.metric("Timestamp", int(vol_point.iloc[0]["timestamp"]))
        p2.metric("Rolling Vol", f"{float(vol_point.iloc[0]['rolling_vol']) if vol_point.iloc[0]['rolling_vol'] == vol_point.iloc[0]['rolling_vol'] else 0.0:.6f}")
        p3.metric(
            "Annualized Vol",
            f"{float(vol_point.iloc[0]['rolling_vol_annualized']) if vol_point.iloc[0]['rolling_vol_annualized'] == vol_point.iloc[0]['rolling_vol_annualized'] else 0.0:.6f}",
        )
