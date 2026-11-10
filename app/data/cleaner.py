"""Cleans a raw DataFrame so it can be stored in PostgreSQL.

Steps: drop empty rows/columns, make safe unique column names, trim text,
convert text columns to numbers / dates when that is clearly right.
Everything that was changed is recorded in a report shown to the user.
"""
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

MAX_COLUMN_NAME_LENGTH = 50
_DATE_PATTERN = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
_NUMERIC_JUNK = re.compile(r"[,\s$€£₹%]")


@dataclass
class CleaningReport:
    renamed_columns: list[dict] = field(default_factory=list)   # [{"original": ..., "new": ...}]
    warnings: list[str] = field(default_factory=list)
    dropped_empty_rows: int = 0
    dropped_empty_columns: int = 0

    def to_dict(self) -> dict:
        return {
            "renamed_columns": self.renamed_columns,
            "warnings": self.warnings,
            "dropped_empty_rows": self.dropped_empty_rows,
            "dropped_empty_columns": self.dropped_empty_columns,
        }


def normalize_column_name(name, position: int) -> str:
    """'Avg Rating (5)' -> 'avg_rating_5'. Missing/unnamed headers become 'column_<n>'."""
    text = str(name).strip()
    if not text or text.lower().startswith("unnamed") or text.lower() == "nan":
        return f"column_{position}"
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not text:
        return f"column_{position}"
    if text[0].isdigit():
        text = f"col_{text}"
    return text[:MAX_COLUMN_NAME_LENGTH]


def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}_{seen[name]}"
            seen[candidate] = 1
            unique_names.append(candidate)
        else:
            seen[name] = 1
            unique_names.append(name)
    return unique_names


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _try_numeric(series: pd.Series, column: str, report: CleaningReport) -> pd.Series:
    """Convert text like '1,234' or '$12.5' to numbers when at least 90% of values fit."""
    non_null = series.dropna()
    if non_null.empty:
        return series
    cleaned = non_null.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    success_rate = converted.notna().mean()
    if success_rate < 0.9:
        return series
    full = pd.to_numeric(series.astype("string").str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce")
    lost = int(series.notna().sum() - full.notna().sum())
    if lost > 0:
        report.warnings.append(
            f"Column '{column}': {lost} non-numeric value(s) were treated as missing so the column could be numeric."
        )
    return full.astype("float64")


def _try_datetime(series: pd.Series) -> pd.Series:
    """Convert text columns that clearly contain dates (like 2024-03-15) to datetime."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series
    sample = non_null.head(200)
    if sample.str.match(_DATE_PATTERN).mean() < 0.9:
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.notna().sum() < 0.9 * series.notna().sum():
        return series
    return parsed


def _shrink_integers(series: pd.Series) -> pd.Series:
    """Float columns that only hold whole numbers become nullable integers (so 4.0 is stored as 4)."""
    non_null = series.dropna()
    if non_null.empty or not (non_null % 1 == 0).all():
        return series
    if non_null.abs().max() > 9e15:
        return series
    return series.astype("Int64")


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    df = df.copy()

    # 1. Drop columns and rows that are completely empty
    columns_before, rows_before = df.shape[1], df.shape[0]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    report.dropped_empty_columns = columns_before - df.shape[1]
    report.dropped_empty_rows = rows_before - df.shape[0]
    if df.empty:
        from app.utils.errors import DataLoadError
        raise DataLoadError("The file contains no usable data after removing empty rows and columns.")

    # 2. Safe, unique column names
    original_names = [str(c) for c in df.columns]
    new_names = _make_unique([normalize_column_name(c, i + 1) for i, c in enumerate(df.columns)])
    for original, new in zip(original_names, new_names):
        if original != new:
            report.renamed_columns.append({"original": original, "new": new})
    df.columns = new_names

    # 3. Trim text and convert obvious numbers / dates
    for column in df.columns:
        series = df[column]
        if not _is_text(series):
            if pd.api.types.is_float_dtype(series):
                df[column] = _shrink_integers(series)
            continue
        series = series.astype("string").str.strip().replace("", pd.NA)
        series = _try_numeric(series, column, report)
        if _is_text(series):
            series = _try_datetime(series)
        elif pd.api.types.is_float_dtype(series):
            series = _shrink_integers(series)
        df[column] = series

    df = df.reset_index(drop=True)
    return df, report
