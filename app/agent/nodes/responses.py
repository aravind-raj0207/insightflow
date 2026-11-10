"""Final answer nodes: the LLM explains verified facts; failures produce friendly messages."""
import logging
import re

from app.agent.prompts import templates
from app.llm.base import LLMClient
from app.utils.errors import LLMError

logger = logging.getLogger(__name__)
PREVIEW_ROWS = 20


def cannot_answer_message(reason: str | None, columns: list[dict]) -> str:
    """Honest refusal that lists what the dataset really contains."""
    reason = (reason or "the dataset does not contain the information needed").strip().rstrip(".")
    available = ", ".join(c["name"] for c in columns)
    return f"I cannot answer that from the uploaded dataset: {reason}. The dataset only contains these columns: {available}."


def explain_node(state: dict, llm: LLMClient) -> dict:
    df = state["result_df"]
    facts = state["analysis"]["facts"]
    shown = df.head(PREVIEW_ROWS)
    try:
        answer = llm.generate(
            templates.EXPLAIN_PROMPT.format(
                question=state["standalone_question"], facts="\n".join(f"- {f}" for f in facts),
                shown=len(shown), total=len(df), table=shown.to_string(index=False),
            ),
            system=templates.EXPLAIN_SYSTEM, task="explain",
        ).strip()
    except LLMError:
        # The numbers are already verified, so we can still answer without the LLM
        logger.warning("Explanation step failed; using the calculated facts as the answer")
        answer = " ".join(facts[:3]) or "The query ran successfully. See the table and chart below."
    return {"answer": answer, "status": "ok"}


def respond_empty_node(state: dict) -> dict:
    return {
        "status": "empty",
        "answer": "The query ran correctly but returned no rows, so there is nothing to report. "
                  "Check the spelling of any names or values in your question.",
    }


def respond_error_node(state: dict) -> dict:
    # If the model kept using a column that does not exist, say so plainly instead of a vague error
    missing = re.search(r"The column '(.+?)' does not exist", state.get("sql_error") or "")
    if missing:
        return {
            "status": "cannot_answer", "error": state.get("sql_error"),
            "answer": cannot_answer_message(
                f"the dataset does not contain a '{missing.group(1)}' column", state.get("columns", [])),
        }
    logger.error("Giving up after %s attempts. Last error: %s | SQL: %s",
                 state.get("attempts"), state.get("sql_error"), state.get("sql"))
    return {
        "status": "error",
        "error": state.get("sql_error"),
        "answer": "I couldn't turn that question into a query I can safely run on this dataset. "
                  "Try rephrasing it, or ask about specific columns.",
    }
