"""Result analysis - all numbers are calculated here in Python, never by the LLM.

`analyze_result` produces a list of plain-English "facts" (with exact numbers).
The explanation step may only use these facts and the result table.
"""
import numpy as np
import pandas as pd

from app.utils.formatting import fmt_number, pretty_label
from app.visualization.chart_selector import classify_columns

ADDITIVE_HINTS = ("count", "total", "sum", "sales", "revenue", "votes", "units", "quantity", "amount", "number")


def _correlation_strength(r: float) -> str:
    size = abs(r)
    if size >= 0.7:
        strength = "strong"
    elif size >= 0.4:
        strength = "moderate"
    elif size >= 0.2:
        strength = "weak"
    else:
        return "no meaningful"
    return f"{strength} {'positive' if r > 0 else 'negative'}"


def _nice(value) -> str:
    """Show timestamps as dates (2024-01-01) instead of 2024-01-01 00:00:00."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _label(row: pd.Series, label_columns: list[str]) -> str:
    return ", ".join(str(row[c]) for c in label_columns)


def analyze_result(df: pd.DataFrame, intent_type: str | None = None) -> dict:
    facts: list[str] = []
    roles = classify_columns(df)
    numeric, temporal = roles["numeric"], roles["temporal"]
    label_columns = (temporal + roles["categorical"])[:2]
    rows = len(df)
    analysis: dict = {"row_count": rows, "numeric_columns": numeric}

    if rows == 0:
        return {**analysis, "facts": ["The query returned no rows."]}

    # A single value, e.g. "how many restaurants are there?"
    if rows == 1:
        for column in df.columns:
            facts.append(f"{pretty_label(column)}: {fmt_number(df[column].iloc[0]) if column in numeric else df[column].iloc[0]}")
        return {**analysis, "facts": facts}

    # Relationship between numeric columns (raw records, no label column)
    if len(numeric) >= 2 and not label_columns:
        corr_pairs = []
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                r = df[a].corr(df[b])
                if pd.notna(r):
                    corr_pairs.append((a, b, float(r)))
        for a, b, r in sorted(corr_pairs, key=lambda t: -abs(t[2]))[:5]:
            facts.append(
                f"Correlation between {pretty_label(a)} and {pretty_label(b)}: r = {r:.2f} "
                f"({_correlation_strength(r)} relationship, {rows:,} records)."
            )
        analysis["correlations"] = [{"a": a, "b": b, "r": r} for a, b, r in corr_pairs]
        return {**analysis, "facts": facts}

    # Distribution of a single numeric column
    if len(numeric) == 1 and not label_columns:
        s = df[numeric[0]].dropna().astype(float)
        facts.append(
            f"{pretty_label(numeric[0])} across {len(s):,} records: min {fmt_number(s.min())}, "
            f"median {fmt_number(s.median())}, mean {fmt_number(s.mean())}, max {fmt_number(s.max())}, "
            f"standard deviation {fmt_number(s.std())}."
        )
        return {**analysis, "facts": facts}

    # Tables with labels: ranking, comparison, trend, composition
    for column in numeric[:3]:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        name = pretty_label(column)
        top_row, bottom_row = df.loc[values.idxmax()], df.loc[values.idxmin()]
        facts.append(f"Highest {name}: {_label(top_row, label_columns)} ({fmt_number(values.max())}).")
        facts.append(f"Lowest {name}: {_label(bottom_row, label_columns)} ({fmt_number(values.min())}).")
        facts.append(f"Average {name} across the {rows} rows shown: {fmt_number(values.mean())}.")

        additive = any(hint in column.lower() for hint in ADDITIVE_HINTS)
        if (additive or intent_type == "composition") and (values.dropna() >= 0).all() and values.sum() > 0:
            share = values.max() / values.sum() * 100
            facts.append(f"Total {name}: {fmt_number(values.sum())}; {_label(top_row, label_columns)} is {share:.1f}% of it.")

        if rows == 2:
            first, second = df.iloc[0], df.iloc[1]
            a, b = float(first[column]), float(second[column])
            if a != b and min(abs(a), abs(b)) > 0:
                higher, lower = (first, second) if a > b else (second, first)
                diff = (max(a, b) - min(a, b)) / min(abs(a), abs(b)) * 100
                facts.append(
                    f"{_label(higher, label_columns)} is higher than {_label(lower, label_columns)} on {name} "
                    f"({fmt_number(max(a, b))} vs {fmt_number(min(a, b))}, {diff:.1f}% higher)."
                )

    if temporal:
        column = numeric[0] if numeric else None
        if column is not None:
            series = df.sort_values(temporal[0])[[temporal[0], column]].dropna()
            if len(series) >= 2:
                first, last = series.iloc[0], series.iloc[-1]
                start, end = float(first[column]), float(last[column])
                change = end - start
                direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
                text = (f"{pretty_label(column)} {direction} from {fmt_number(start)} ({_nice(first[temporal[0]])}) "
                        f"to {fmt_number(end)} ({_nice(last[temporal[0]])})")
                if start != 0:
                    text += f", a change of {change / abs(start) * 100:+.1f}%"
                facts.append(text + ".")
                slope = np.polyfit(np.arange(len(series)), series[column].astype(float).to_numpy(), 1)[0]
                analysis["trend_slope"] = float(slope)

    return {**analysis, "facts": facts}
