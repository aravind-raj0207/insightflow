"""Nodes that analyse the result, choose a chart and draw it."""
import json
import logging

from app.agent.nodes.analysis import analyze_result
from app.visualization.chart_selector import select_chart
from app.visualization.plotly_renderer import render_chart

logger = logging.getLogger(__name__)


def analyze_node(state: dict) -> dict:
    return {"analysis": analyze_result(state["result_df"], state.get("intent_type"))}


def select_chart_node(state: dict) -> dict:
    spec = select_chart(
        state["result_df"], intent_type=state.get("intent_type"),
        requested_chart=state.get("requested_chart"), title_hint=state.get("chart_title"),
    )
    return {"chart_spec": spec.model_dump()}


def render_chart_node(state: dict) -> dict:
    from app.visualization.chart_selector import ChartSpec
    spec = ChartSpec(**state["chart_spec"])
    figure, final_spec = render_chart(state["result_df"], spec)
    return {"figure": json.loads(figure.to_json()), "chart_spec": final_spec.model_dump()}
