"""The LangGraph workflow. Reading this file shows the whole agent at a glance.

  load_schema -> resolve_question -> understand_intent
        |-- (rejected / cannot answer / needs clarification) --> END
        '-- generate_sql -> validate_sql -> execute_sql -> analyze -> select_chart -> render_chart -> explain -> END
               ^                 |               |
               '---- retry ------'---------------'   (once, with the error message)
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.intent import intent_node, resolve_question_node
from app.agent.nodes.responses import explain_node, respond_empty_node, respond_error_node
from app.agent.nodes.schema import load_schema_node
from app.agent.nodes.sql_execution import execute_sql_node, validate_sql_node
from app.agent.nodes.sql_generation import generate_sql_node
from app.agent.nodes.visualization import analyze_node, render_chart_node, select_chart_node
from app.agent.state import AgentState
from app.llm.base import LLMClient

MAX_SQL_ATTEMPTS = 2  # first try + one retry with the error message


def route_after_intent(state: AgentState) -> str:
    return "generate_sql" if state.get("status") == "pending" else END


def route_after_generation(state: AgentState) -> str:
    return END if state.get("status") == "cannot_answer" else "validate_sql"


def route_after_validation(state: AgentState) -> str:
    if not state.get("sql_error"):
        return "execute_sql"
    return "generate_sql" if state.get("attempts", 0) < MAX_SQL_ATTEMPTS else "respond_error"


def route_after_execution(state: AgentState) -> str:
    if state.get("sql_error"):
        return "generate_sql" if state.get("attempts", 0) < MAX_SQL_ATTEMPTS else "respond_error"
    result = state.get("result_df")
    return "respond_empty" if result is None or len(result) == 0 else "analyze"


def build_graph(llm: LLMClient):
    graph = StateGraph(AgentState)

    graph.add_node("load_schema", load_schema_node)
    graph.add_node("resolve_question", partial(resolve_question_node, llm=llm))
    graph.add_node("understand_intent", partial(intent_node, llm=llm))
    graph.add_node("generate_sql", partial(generate_sql_node, llm=llm))
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("select_chart", select_chart_node)
    graph.add_node("render_chart", render_chart_node)
    graph.add_node("explain", partial(explain_node, llm=llm))
    graph.add_node("respond_empty", respond_empty_node)
    graph.add_node("respond_error", respond_error_node)

    graph.add_edge(START, "load_schema")
    graph.add_edge("load_schema", "resolve_question")
    graph.add_edge("resolve_question", "understand_intent")
    graph.add_conditional_edges("understand_intent", route_after_intent, ["generate_sql", END])
    graph.add_conditional_edges("generate_sql", route_after_generation, ["validate_sql", END])
    graph.add_conditional_edges("validate_sql", route_after_validation, ["execute_sql", "generate_sql", "respond_error"])
    graph.add_conditional_edges("execute_sql", route_after_execution,
                                ["generate_sql", "respond_error", "respond_empty", "analyze"])
    graph.add_edge("analyze", "select_chart")
    graph.add_edge("select_chart", "render_chart")
    graph.add_edge("render_chart", "explain")
    graph.add_edge("explain", END)
    graph.add_edge("respond_empty", END)
    graph.add_edge("respond_error", END)
    return graph.compile()
