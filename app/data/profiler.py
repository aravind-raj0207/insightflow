"""Calculates facts about a dataset from the real data (the AI never guesses these)."""
import json

import numpy as np
import pandas as pd

CATEGORICAL_MAX_UNIQUE = 50


def infer_kind(series: pd.Series) -> str:
    """Returns: numeric | categorical | text | datetime | boolean."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique = series.nunique(dropna=True)
    non_null = series.notna().sum()
    if unique <= CATEGORICAL_MAX_UNIQUE or (non_null and unique / non_null <= 0.2):
        return "categorical"
    return "text"


def kind_to_sql_type(kind: str, series: pd.Series) -> str:
    if kind == "numeric":
        return "BIGINT" if pd.api.types.is_integer_dtype(series) else "DOUBLE PRECISION"
    if kind == "datetime":
        return "TIMESTAMP"
    if kind == "boolean":
        return "BOOLEAN"
    return "TEXT"


def _to_python(value):
    """Make numpy / pandas values JSON-friendly."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def profile_column(series: pd.Series) -> dict:
    kind = infer_kind(series)
    missing = int(series.isna().sum())
    info = {
        "name": series.name,
        "kind": kind,
        "sql_type": kind_to_sql_type(kind, series),
        "missing_count": missing,
        "missing_pct": round(missing / len(series) * 100, 2) if len(series) else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "stats": {},
        "top_values": [],
    }
    non_null = series.dropna()
    if non_null.empty:
        return info

    if kind == "numeric":
        info["stats"] = {
            "min": _to_python(non_null.min()),
            "max": _to_python(non_null.max()),
            "mean": _to_python(non_null.astype(float).mean()),
            "median": _to_python(non_null.astype(float).median()),
            "std": _to_python(non_null.astype(float).std()),
        }
    elif kind == "datetime":
        info["stats"] = {"min": _to_python(non_null.min()), "max": _to_python(non_null.max())}
    elif kind in ("categorical", "boolean"):
        counts = non_null.astype(str).value_counts().head(15)
        info["top_values"] = [{"value": str(v), "count": int(c)} for v, c in counts.items()]
    return info


def profile_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    columns = [profile_column(df[c]) for c in df.columns]
    sample = json.loads(df.head(sample_rows).to_json(orient="records", date_format="iso"))
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "columns": columns,
        "sample_rows": sample,
    }
