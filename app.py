from __future__ import annotations

import streamlit as st

from src.data_loader import load_csv
from src.features import add_features, default_feature_columns
from src.plots import correlation_heatmap, histogram, scatter, time_series
from src.scoring import add_forward_returns, bucketed_forward_returns, score_signals

st.set_page_config(page_title="Trading Alpha Visualizer", layout="wide")
st.title("Trading Alpha Visualizer")
st.caption("Explore candidate alpha signals and evaluate predictive power across forward horizons.")

uploaded = st.file_uploader("Upload simulation log CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

base_df = load_csv(uploaded)
products = sorted(base_df["product"].unique().tolist())
horizons = st.sidebar.multiselect("Forward horizons (ticks)", [1, 5, 10, 20, 50, 100], default=[1, 5, 10, 20, 50])
product = st.sidebar.selectbox("Product", products)

df = base_df[base_df["product"] == product].copy()
df = add_features(df)
df = add_forward_returns(df, horizons)
feature_cols = default_feature_columns(df)

tab1, tab2, tab3, tab4 = st.tabs(["Data Health", "Feature Explorer", "Diagnostics", "Leaderboard"])

with tab1:
    st.subheader("Data Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Products", len(products))
    col3.metric("Null cells", int(df.isna().sum().sum()))
    st.dataframe(df.isna().mean().sort_values(ascending=False).rename("null_ratio").to_frame())

with tab2:
    st.subheader("Feature Explorer")
    feature = st.selectbox("Feature", feature_cols, index=0)
    st.plotly_chart(time_series(df, x="timestamp", y=feature, title=f"{feature} over time"), use_container_width=True)
    st.plotly_chart(histogram(df, x=feature, title=f"{feature} distribution"), use_container_width=True)
    top_corr_cols = feature_cols[: min(20, len(feature_cols))]
    st.plotly_chart(correlation_heatmap(df, top_corr_cols), use_container_width=True)

with tab3:
    st.subheader("Signal Diagnostics")
    feature = st.selectbox("Feature for diagnostics", feature_cols, index=0, key="diag_feature")
    horizon = st.selectbox("Horizon", horizons, index=0, key="diag_h")
    target_col = f"fwd_ret_{horizon}"
    diag_df = df[[feature, target_col]].dropna()
    st.plotly_chart(
        scatter(diag_df, x=feature, y=target_col, title=f"{feature} vs {target_col}"),
        use_container_width=True,
    )
    bucket_df = bucketed_forward_returns(df, feature, horizon)
    st.bar_chart(bucket_df.set_index("bucket")["avg_fwd_return"])

with tab4:
    st.subheader("Signal Leaderboard")
    scores = score_signals(df, feature_cols, horizons)
    st.dataframe(scores, use_container_width=True)
    st.download_button(
        "Download leaderboard CSV",
        data=scores.to_csv(index=False).encode("utf-8"),
        file_name=f"alpha_scores_{product}.csv",
        mime="text/csv",
    )
