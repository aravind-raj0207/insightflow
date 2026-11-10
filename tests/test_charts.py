import numpy as np
import pandas as pd

from app.agent.nodes.analysis import analyze_result
from app.visualization.chart_selector import detect_requested_chart, select_chart
from app.visualization.plotly_renderer import render_chart

rng = np.random.default_rng(0)


def chart_for(df, **kwargs):
    spec = select_chart(df, **kwargs)
    figure, final_spec = render_chart(df, spec)       # also proves the chart can really be drawn
    assert figure is not None
    return final_spec


def test_category_comparison_is_a_bar_chart():
    assert chart_for(pd.DataFrame({"city": ["A", "B", "C"], "avg_rating": [4.1, 3.9, 4.3]})).chart_type == "bar"


def test_time_series_is_a_line_chart():
    df = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=6, freq="MS"), "sales": [1, 3, 2, 5, 6, 8]})
    assert chart_for(df).chart_type == "line"


def test_text_months_are_recognised_as_time():
    df = pd.DataFrame({"month": ["2024-01", "2024-02", "2024-03"], "sales": [5, 7, 6]})
    assert chart_for(df).chart_type == "line"


def test_single_numeric_column_is_a_histogram():
    assert chart_for(pd.DataFrame({"rating": rng.normal(4, 0.5, 200)})).chart_type == "histogram"


def test_two_numeric_columns_make_a_scatter_plot():
    df = pd.DataFrame({"votes": rng.integers(1, 500, 100), "rating": rng.normal(4, 0.5, 100)})
    assert chart_for(df).chart_type == "scatter"


def test_group_distributions_make_a_box_plot():
    df = pd.DataFrame({"city": ["A", "B"] * 30, "rating": rng.normal(4, 0.5, 60)})
    assert chart_for(df).chart_type == "box"


def test_many_numeric_columns_make_a_heatmap():
    assert chart_for(pd.DataFrame(rng.normal(size=(50, 4)), columns=list("wxyz"))).chart_type == "heatmap"


def test_single_number_is_an_indicator():
    assert chart_for(pd.DataFrame({"restaurant_count": [9500]})).chart_type == "indicator"


def test_pie_only_for_share_questions_with_few_categories():
    df = pd.DataFrame({"region": list("NSEW"), "total_sales": [10, 20, 5, 15]})
    assert chart_for(df, intent_type="composition").chart_type == "pie"
    assert chart_for(df, intent_type="comparison").chart_type == "bar"      # not pie by default


def test_explicit_chart_request_is_respected():
    df = pd.DataFrame({"region": list("NSEW"), "total_sales": [10, 20, 5, 15]})
    spec = chart_for(df, intent_type="comparison", requested_chart="pie")
    assert spec.chart_type == "pie" and "asked for" in spec.reason
    df_time = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=4, freq="MS"), "sales": [1, 2, 3, 4]})
    assert chart_for(df_time, requested_chart="bar").chart_type == "bar"     # auto would be "line"


def test_impossible_chart_request_falls_back_and_explains():
    df = pd.DataFrame({"city": [f"c{i}" for i in range(20)], "n": range(20)})
    spec = chart_for(df, requested_chart="pie")
    assert spec.chart_type == "bar" and "cannot be used" in spec.reason


def test_chart_request_detection():
    assert detect_requested_chart("Show sales by region as a pie chart") == "pie"
    assert detect_requested_chart("Show monthly sales using a bar chart") == "bar"
    assert detect_requested_chart("Show monthly sales") is None


def test_analysis_numbers_are_computed_by_code():
    df = pd.DataFrame({"city": ["Chennai", "Bangalore"], "avg_rating": [4.0, 4.5]})
    facts = " ".join(analyze_result(df)["facts"])
    assert "Highest Avg Rating: Bangalore (4.50)" in facts
    assert "12.5% higher" in facts
