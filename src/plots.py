from __future__ import annotations

import pandas as pd
import plotly.express as px


def time_series(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    return px.line(df, x=x, y=y, color=color, title=title)


def histogram(df: pd.DataFrame, x: str, color: str | None = None, title: str | None = None):
    return px.histogram(df, x=x, color=color, marginal="box", title=title)


def correlation_heatmap(df: pd.DataFrame, cols: list[str], title: str = "Feature Correlation"):
    corr = df[cols].corr()
    return px.imshow(corr, text_auto=".2f", aspect="auto", title=title)


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    try:
        import statsmodels.api  # noqa: F401

        return px.scatter(df, x=x, y=y, color=color, trendline="ols", title=title)
    except ModuleNotFoundError:
        # Fallback keeps diagnostics usable when optional OLS dependency is absent.
        return px.scatter(df, x=x, y=y, color=color, title=title)
