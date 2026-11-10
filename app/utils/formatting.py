"""Small helpers for turning numbers and column names into readable text."""
import math


def fmt_number(value) -> str:
    """1234567.891 -> '1,234,567.89'; 4.0 -> '4'; None -> 'n/a'."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.2f}"


def pretty_label(column_name: str) -> str:
    """'avg_rating' -> 'Avg Rating'."""
    return str(column_name).replace("_", " ").strip().title()
