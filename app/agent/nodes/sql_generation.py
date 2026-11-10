"""SQL generation (LLM). Receives the real schema; returns one SELECT statement."""
from app.agent.nodes.responses import cannot_answer_message
from app.agent.prompts import templates
from app.llm.base import LLMClient


def generate_sql_node(state: dict, llm: LLMClient) -> dict:
    retry_block = ""
    if state.get("sql_error"):
        retry_block = templates.RETRY_BLOCK.format(sql=state.get("sql"), error=state["sql_error"])

    data = llm.generate_json(
        templates.SQL_PROMPT.format(
            schema_text=state["schema_text"], question=state["standalone_question"],
            intent_type=state.get("intent_type", "other"), retry_block=retry_block,
        ),
        system=templates.SQL_SYSTEM, task="generate_sql",
    )
    attempts = state.get("attempts", 0) + 1
    sql = str(data.get("sql") or "").strip()
    if not sql or sql.lower() == "null":
        return {
            "sql": None, "attempts": attempts, "status": "cannot_answer",
            "answer": cannot_answer_message(data.get("cannot_answer_reason"), state["columns"]),
        }
    return {"sql": sql, "chart_title": data.get("chart_title"), "attempts": attempts, "sql_error": None}
