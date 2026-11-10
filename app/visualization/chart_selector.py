"""Decides which chart fits a query result.

The output is a ChartSpec - a small, validated description of a chart. The LLM never writes
chart code; the renderer only draws what the spec says, using columns that really exist.

Order of decisions:
  1. If the user explicitly asked for a chart type AND it works for this data -> use it.
  2. Otherwise choose automatically from the SHAPE of the result
     (how many categorical / numeric / time columns, how many rows).
"""
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from app.utils.formatting import pretty_label

ChartType = Literal["bar", "line", "pie", "scatter", "histogram", "box", "heatmap", "indicator", "table"]

MAX_BAR_CATEGORIES = 30
MAX_PIE_SLICES = 8
TEMPORAL_NAME_HINTS = ("date", "time", "month", "year", "quarter", "week", "day", "period", "timestamp")
_TEMPORAL_TEXT = re.compile(r"^\d{4}([-/]\d{1,2})?([-/]\d{1,2})?([ T].*)?$|^\d{4}-?Q[1-4]$", re.IGNORECASE)

_REQUEST_PATTERNS: list[tuple[str, ChartType]] = [
    (r"\bpie\b|\bdonut\b|\bdoughnut\b", "pie"),
    (r"\bscatter\b|\bscatterplot\b", "scatter"),
    (r"\bhistogram\b", "histogram"),
    (r"\bbox\s*-?plot\b|\bbox and whisker\b|\bwhisker\b", "box"),
    (r"\bheat\s*-?map\b|\bcorrelation matrix\b", "heatmap"),
    (r"\bline\s*(chart|graph|plot)\b|\bas a line\b|\busing a line\b|\bwith a line\b", "line"),
    (r"\bbar\s*(chart|graph|plot)\b|\bcolumn\s*chart\b|\bas a bar\b|\busing a bar\b|\bwith bars\b", "bar"),
]


class ChartSpec(BaseModel):
    chart_type: ChartType
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    title: str = ""
    reason: str = ""
    x_label: str | None = None
    y_label: str | None = None
    horizontal: bool = False
    max_rows: int | None = None  # when only the first N rows are drawn


def detect_requested_chart(question: str) -> ChartType | None:
    """Find an explicit chart request such as 'as a pie chart' in the user's question."""
    text = question.lower()
    for pattern, chart_type in _REQUEST_PATTERNS:
        if re.search(pattern, text):
            return chart_type
    return None


# ------------------------------------------------------------ column roles

def _is_temporal(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    lowered = name.lower()
    name_hint = any(hint in lowered for hint in TEMPORAL_NAME_HINTS)
    if pd.api.types.is_numeric_dtype(series):
        # numeric year / month / quarter columns, e.g. 2021, 2022 or 1..12
        values = series.dropna()
        return bool(name_hint and values.nunique() > 1 and (values % 1 == 0).all())
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return bool(non_null.head(50).str.match(_TEMPORAL_TEXT).mean() >= 0.9)


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {"temporal": [], "numeric": [], "categorical": []}
    for column in df.columns:
        series = df[column]
        if _is_temporal(series, column):
            roles["temporal"].append(column)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            roles["numeric"].append(column)
        else:
            roles["categorical"].append(column)
    return roles


# ------------------------------------------------------------- feasibility

def _why_not(chart: ChartType, df: pd.DataFrame, roles: dict[str, list[str]]) -> str | None:
    """Returns None when the chart can be drawn from this data, otherwise the reason it cannot."""
    numeric, categorical, temporal = roles["numeric"], roles["categorical"], roles["temporal"]
    labels = categorical + temporal
    if chart in ("bar", "line"):
        if not labels or not numeric:
            return "it needs one label column and one numeric column"
    elif chart == "pie":
        if not categorical or not numeric:
            return "it needs one category column and one numeric column"
        if len(df) > MAX_PIE_SLICES:
            return f"it has {len(df)} categories and a pie chart becomes unreadable above {MAX_PIE_SLICES}"
        if len(df) < 2:
            return "it has only one category"
        if (df[numeric[0]].dropna() < 0).any():
            return "the values contain negative numbers"
    elif chart == "scatter":
        if len(numeric) < 2:
            return "it needs at least two numeric columns"
    elif chart in ("histogram", "box"):
        if not numeric:
            return "it needs a numeric column"
    elif chart == "heatmap":
        if len(numeric) < 3 and not (len(categorical) >= 2 and numeric):
            return "it needs at least three numeric columns (correlation) or two categories plus one number"
    return None


# ------------------------------------------------------------------ titles

def _default_title(spec_type: str, x: str | None, y: str | None, color: str | None = None) -> str:
    if spec_type in ("bar", "line", "box") and x and y:
        title = f"{pretty_label(y)} by {pretty_label(x)}"
    elif spec_type == "pie" and x and y:
        title = f"Share of {pretty_label(y)} by {pretty_label(x)}"
    elif spec_type == "scatter" and x and y:
        title = f"{pretty_label(y)} vs {pretty_label(x)}"
    elif spec_type == "histogram" and x:
        title = f"Distribution of {pretty_label(x)}"
    elif spec_type == "heatmap":
        title = "Correlation Between Numeric Columns"
    else:
        title = "Query Result"
    return f"{title} (by {pretty_label(color)})" if color else title


# --------------------------------------------------------------- the logic

def _build_spec(chart: ChartType, df: pd.DataFrame, roles: dict[str, list[str]], title_hint: str | None,
                intent_type: str | None) -> ChartSpec:
    numeric, categorical, temporal = roles["numeric"], roles["categorical"], roles["temporal"]
    spec = ChartSpec(chart_type=chart)

    if chart in ("bar", "line"):
        x = (temporal or categorical)[0] if chart == "line" else (categorical + temporal)[0]
        y = numeric[0]
        spec.x_column, spec.y_column = x, y
        others = [c for c in categorical + temporal if c != x]
        if others and len(df) > df[x].nunique():          # several rows per x value -> split into series
            spec.color_column = others[0]
        if chart == "bar":
            labels = df[x].astype(str)
            spec.horizontal = bool(df[x].nunique() > 8 or labels.str.len().max() > 14)
            if len(df) > MAX_BAR_CATEGORIES and not spec.color_column:
                spec.max_rows = MAX_BAR_CATEGORIES
    elif chart == "pie":
        spec.x_column, spec.y_column = categorical[0], numeric[0]
    elif chart == "scatter":
        spec.x_column, spec.y_column = numeric[0], numeric[1]
        if categorical:
            spec.color_column = categorical[0] if df[categorical[0]].nunique() <= 12 else None
    elif chart == "histogram":
        spec.x_column = numeric[0]
    elif chart == "box":
        spec.y_column = numeric[0]
        group_columns = categorical + temporal
        spec.x_column = group_columns[0] if group_columns else None
    elif chart == "heatmap":
        if len(numeric) >= 3:
            spec.x_column = None  # correlation matrix of all numeric columns
        else:
            spec.x_column, spec.color_column, spec.y_column = categorical[0], categorical[1], numeric[0]
    spec.x_label = pretty_label(spec.x_column) if spec.x_column else None
    spec.y_label = pretty_label(spec.y_column) if spec.y_column else None
    spec.title = title_hint or _default_title(chart, spec.x_column, spec.y_column,
                                              spec.color_column if chart in ("bar", "line", "scatter") else None)
    return spec


def _auto_choice(df: pd.DataFrame, roles: dict[str, list[str]], intent_type: str | None) -> tuple[ChartType, str]:
    numeric, categorical, temporal = roles["numeric"], roles["categorical"], roles["temporal"]
    rows = len(df)

    if rows == 1 and len(numeric) == 1 and not categorical and not temporal:
        return "indicator", "The result is a single number, so it is shown as a headline value."
    if not numeric:
        return "table", "The result has no numeric column to plot, so it is shown as a table."

    if temporal:
        return "line", "The result tracks a numeric measure over time, so a line chart shows the trend best."

    if intent_type == "correlation" and len(numeric) >= 3:
        return "heatmap", "Several numeric columns were compared, so a correlation heatmap shows how they relate."
    if len(numeric) >= 3 and not categorical:
        return "heatmap", "The result has several numeric columns, so a correlation heatmap shows how they relate."

    if len(numeric) >= 2 and not categorical:
        return "scatter", "Two numeric variables are compared, so a scatter plot shows their relationship."
    if len(numeric) >= 2 and categorical and rows > df[categorical[0]].nunique():
        return "scatter", "Two numeric variables are compared across many records, so a scatter plot shows their relationship."

    if categorical:
        category = categorical[0]
        unique_categories = df[category].nunique()
        if rows > unique_categories and unique_categories >= 2 and intent_type in (None, "distribution", "other"):
            return "box", "Several values exist for each group, so a box plot compares how the values are distributed."
        if intent_type == "composition" and 2 <= rows <= MAX_PIE_SLICES and (df[numeric[0]].dropna() >= 0).all():
            return "pie", "The question is about each category's share of a whole and there are few categories, so a pie chart fits."
        if rows > unique_categories and len(categorical) >= 2:
            return "bar", "The result compares a measure across two grouping columns, so a grouped bar chart is used."
        return "bar", "The result compares a numeric measure across categories, so a bar chart fits."

    if len(numeric) == 1 and rows > 1:
        return "histogram", "The result is a set of values of one numeric column, so a histogram shows how they are distributed."
    return "table", "The result shape does not match a standard chart, so it is shown as a table."


def select_chart(df: pd.DataFrame, intent_type: str | None = None, requested_chart: str | None = None,
                 title_hint: str | None = None) -> ChartSpec:
    """Main entry point: returns the chart specification for a query result."""
    roles = classify_columns(df)
    note = ""

    if requested_chart in ("bar", "line", "pie", "scatter", "histogram", "box", "heatmap"):
        problem = _why_not(requested_chart, df, roles)  # type: ignore[arg-type]
        if problem is None:
            spec = _build_spec(requested_chart, df, roles, title_hint, intent_type)  # type: ignore[arg-type]
            spec.reason = f"You asked for a {requested_chart} chart, and the data suits it."
            return spec
        note = f"You asked for a {requested_chart} chart, but it cannot be used here because {problem}. "

    chart, reason = _auto_choice(df, roles, intent_type)
    if chart in ("indicator", "table"):
        spec = ChartSpec(chart_type=chart, title=title_hint or "Query Result")
        if chart == "indicator":
            spec.y_column = roles["numeric"][0]
            spec.y_label = pretty_label(spec.y_column)
    else:
        spec = _build_spec(chart, df, roles, title_hint, intent_type)
    spec.reason = note + reason
    return spec
