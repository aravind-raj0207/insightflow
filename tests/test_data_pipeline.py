import io

import pandas as pd
import pytest

from app.data.cleaner import clean_dataframe, normalize_column_name
from app.data.loader import load_dataframe
from app.data.profiler import profile_dataframe
from app.utils.errors import DataLoadError


def test_column_names_are_normalised():
    assert normalize_column_name("Avg Rating (5)", 1) == "avg_rating_5"
    assert normalize_column_name("2024 Sales", 1) == "col_2024_sales"
    assert normalize_column_name("Unnamed: 3", 4) == "column_4"


def test_duplicate_and_missing_headers_are_fixed():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["Sales", "Sales", "Unnamed: 2"])
    cleaned, report = clean_dataframe(df)
    assert list(cleaned.columns) == ["sales", "sales_2", "column_3"]
    assert len(report.renamed_columns) == 3


def test_text_numbers_and_dates_are_converted():
    df = pd.DataFrame({"price": ["1,200", "3,400", "500"], "when": ["2024-01-05", "2024-02-10", "2024-03-15"],
                       "name": ["a", "b", "c"]})
    cleaned, _ = clean_dataframe(df)
    profile = {c["name"]: c["kind"] for c in profile_dataframe(cleaned)["columns"]}
    assert profile == {"price": "numeric", "when": "datetime", "name": "categorical"}


def test_profile_matches_real_data():
    df = pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "y", "y"]})
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)       # one duplicate row
    profile = profile_dataframe(df)
    assert profile["row_count"] == 5 and profile["column_count"] == 2
    assert profile["duplicate_rows"] == 1
    column_a = next(c for c in profile["columns"] if c["name"] == "a")
    assert column_a["missing_count"] == 1 and column_a["stats"]["max"] == 3


def test_csv_and_validation_errors():
    df = load_dataframe("x.csv", b"a,b\n1,2\n3,4\n")
    assert df.shape == (2, 2)
    with pytest.raises(DataLoadError):
        load_dataframe("x.txt", b"hello")
    with pytest.raises(DataLoadError):
        load_dataframe("x.csv", b"")
    with pytest.raises(DataLoadError):
        load_dataframe("x.csv", b"a,b\n")   # header only


def test_excel_is_read():
    buffer = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buffer, index=False)
    assert load_dataframe("x.xlsx", buffer.getvalue()).shape == (2, 2)
