"""The shared state that flows through the LangGraph workflow.

Each node reads what it needs from the state and returns only the keys it changes.
"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- input
    dataset_id: int
    question: str
    history: list[dict]              # earlier turns of this conversation (for follow-ups)

    # ---- schema
    table_name: str
    columns: list[dict]
    schema_text: str

    # ---- understanding the question
    standalone_question: str         # the question rewritten so it makes sense on its own
    intent_type: str                 # ranking / comparison / trend / distribution / ...
    requested_chart: str | None      # chart type explicitly asked for by the user
    chart_title: str | None

    # ---- SQL
    sql: str | None
    pretty_sql: str | None
    sql_error: str | None            # validation or execution error (used for the retry)
    attempts: int

    # ---- result
    result_df: Any                   # pandas DataFrame
    analysis: dict
    chart_spec: dict | None
    figure: dict | None

    # ---- output
    status: str                      # ok / cannot_answer / clarify / rejected / empty / error
    answer: str
    error: str | None
