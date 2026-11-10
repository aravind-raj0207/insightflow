"""Turns a ChartSpec + query result into an interactive Plotly figure.

The renderer only draws what the spec says. It checks that the columns really exist
and falls back to a plain table if anything goes wrong.
"""
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils.formatting import pretty_label
from app.visualization.chart_selector import ChartSpec

logger = logging.getLogger(__name__)
TEMPLATE = "plotly_white"
PALETTE = ["#0F766E", "#C2410C", "#1D4ED8", "#A21CAF", "#4D7C0F", "#B45309", "#0E7490", "#BE123C"]


def _check_columns(df: pd.DataFrame, spec: ChartSpec) -> None:
    for column in (spec.x_column, spec.y_column, spec.color_column):
        if column is not None and column not in df.columns:
            raise ValueError(f"Chart column '{column}' is not in the result.")


def _sort_if_temporal(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return df.sort_values(column)
    converted = pd.to_datetime(df[column], errors="coerce", format="mixed") if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]) else None
    if converted is not None and converted.notna().all():
        df = df.assign(**{column: converted})
        return df.sort_values(column)
    if pd.api.types.is_numeric_dtype(df[column]):
        return df.sort_values(column)
    return df


def _draw(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    kind = spec.chart_type
    labels = {c: pretty_label(c) for c in df.columns}

    if kind == "indicator":
        value = df[spec.y_column].iloc[0]
        fig = go.Figure(go.Indicator(mode="number", value=float(value), title={"text": spec.y_label or ""},
                                     number={"valueformat": ",.2f" if float(value) % 1 else ",.0f"}))
        fig.update_layout(height=220)
    elif kind == "bar":
        data = df.head(spec.max_rows) if spec.max_rows else df
        if spec.horizontal:
            fig = px.bar(data, x=spec.y_column, y=spec.x_column, color=spec.color_column, orientation="h",
                         barmode="group", labels=labels, text_auto=".3~s", color_discrete_sequence=PALETTE)
            fig.update_yaxes(autorange="reversed")
        else:
            fig = px.bar(data, x=spec.x_column, y=spec.y_column, color=spec.color_column,
                         barmode="group", labels=labels, text_auto=".3~s", color_discrete_sequence=PALETTE)
    elif kind == "line":
        data = _sort_if_temporal(df, spec.x_column)
        fig = px.line(data, x=spec.x_column, y=spec.y_column, color=spec.color_column, markers=True,
                      labels=labels, color_discrete_sequence=PALETTE)
    elif kind == "pie":
        fig = px.pie(df, names=spec.x_column, values=spec.y_column, hole=0.35, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif kind == "scatter":
        fig = px.scatter(df, x=spec.x_column, y=spec.y_column, color=spec.color_column, labels=labels,
                         opacity=0.75, color_discrete_sequence=PALETTE)
    elif kind == "histogram":
        fig = px.histogram(df, x=spec.x_column, nbins=min(40, max(10, int(len(df) ** 0.5))), labels=labels,
                           color_discrete_sequence=PALETTE)
        fig.update_layout(yaxis_title="Count", bargap=0.05)
    elif kind == "box":
        fig = px.box(df, x=spec.x_column, y=spec.y_column, points="outliers", labels=labels,
                     color_discrete_sequence=PALETTE)
    elif kind == "heatmap":
        if spec.x_column is None:
            numeric_df = df.select_dtypes("number")
            matrix = numeric_df.corr()
            fig = px.imshow(matrix, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        else:
            pivot = df.pivot_table(index=spec.color_column, columns=spec.x_column, values=spec.y_column, aggfunc="mean")
            fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Teal", aspect="auto")
    else:  # "table"
        fig = go.Figure(go.Table(
            header=dict(values=[pretty_label(c) for c in df.columns], fill_color="#0F766E",
                        font=dict(color="white"), align="left"),
            cells=dict(values=[df[c].astype(str).tolist() for c in df.columns], align="left"),
        ))

    fig.update_layout(
        title=dict(text=spec.title, x=0.02), template=TEMPLATE, margin=dict(l=40, r=20, t=70, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    )
    if kind in ("bar", "line", "scatter", "histogram", "box"):
        if kind == "bar" and spec.horizontal:
            fig.update_layout(xaxis_title=spec.y_label, yaxis_title=spec.x_label)
        elif kind != "histogram":
            fig.update_layout(xaxis_title=spec.x_label, yaxis_title=spec.y_label)
        else:
            fig.update_layout(xaxis_title=spec.x_label)
    return fig


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> tuple[go.Figure, ChartSpec]:
    """Returns (figure, spec). If drawing fails, a table is returned and the spec is updated."""
    try:
        _check_columns(df, spec)
        return _draw(df, spec), spec
    except Exception:
        logger.exception("Chart rendering failed for spec %s; falling back to a table", spec.model_dump())
        fallback = ChartSpec(chart_type="table", title=spec.title or "Query Result",
                             reason="The chart could not be drawn, so the data is shown as a table.")
        return _draw(df, fallback), fallback
