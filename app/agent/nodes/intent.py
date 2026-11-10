"""Question understanding: rewrite follow-ups, then decide what kind of question this is."""
import logging

from app.agent.nodes.responses import cannot_answer_message
from app.agent.prompts import templates
from app.llm.base import LLMClient
from app.utils.errors import LLMError
from app.utils.safety import looks_like_modification_request
from app.visualization.chart_selector import detect_requested_chart

logger = logging.getLogger(__name__)
VALID_INTENTS = {"ranking", "comparison", "trend", "distribution", "relationship",
                 "composition", "correlation", "aggregate", "lookup", "other"}


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        lines.append(f'User asked: "{turn["standalone_question"]}"')
        lines.append(f'Assistant answered: "{turn["answer"][:300]}"')
    return "\n".join(lines)


def resolve_question_node(state: dict, llm: LLMClient) -> dict:
    """Turns 'What about Chennai?' into a complete question using earlier turns."""
    question = state["question"].strip()
    history = [t for t in state.get("history", []) if t.get("status") == "ok"]
    if not history:
        return {"standalone_question": question}
    try:
        data = llm.generate_json(
            templates.RESOLVE_PROMPT.format(history=_format_history(history[-3:]), question=question),
            system=templates.RESOLVE_SYSTEM, task="resolve_question",
        )
        rewritten = str(data.get("standalone_question") or "").strip()
        return {"standalone_question": rewritten or question}
    except LLMError:
        logger.warning("Follow-up rewrite failed; using the original question")
        return {"standalone_question": question}


def intent_node(state: dict, llm: LLMClient) -> dict:
    question = state["standalone_question"]

    # Deterministic guard first: requests to change data are refused immediately
    if looks_like_modification_request(state["question"]) or looks_like_modification_request(question):
        return {
            "status": "rejected",
            "answer": "I can only read and analyse your data. I can't delete, change or insert data, "
                      "so I did not run that request. Try asking a question about the data instead.",
        }

    data = llm.generate_json(
        templates.INTENT_PROMPT.format(schema_text=state["schema_text"], question=question),
        system=templates.INTENT_SYSTEM, task="intent",
    )

    if data.get("is_modification_request"):
        return {
            "status": "rejected",
            "answer": "I can only read and analyse your data. I can't delete, change or insert data, "
                      "so I did not run that request.",
        }

    # The model decides whether the data can answer the question. Column names it lists are only a hint:
    # models sometimes list derived names (like "month"), so a mismatch alone must not block the question.
    # Invented columns are still caught later by the SQL validator, which names the missing column.
    if not data.get("answerable", True):
        reason = data.get("reason")
        if not reason:
            known = {c["name"].lower() for c in state["columns"]}
            missing = [str(c) for c in (data.get("required_columns") or []) if str(c).lower() not in known]
            reason = f"the dataset has no column for: {', '.join(missing)}" if missing else None
        return {"status": "cannot_answer", "answer": cannot_answer_message(reason, state["columns"])}

    if data.get("needs_clarification") and data.get("clarifying_question"):
        return {"status": "clarify", "answer": str(data["clarifying_question"])}

    intent_type = data.get("intent_type") if data.get("intent_type") in VALID_INTENTS else "other"
    # An explicit chart request in the question always wins; the model's guess is only a fallback
    requested = detect_requested_chart(question) or detect_requested_chart(state["question"]) or data.get("requested_chart")
    return {"intent_type": intent_type, "requested_chart": requested, "status": "pending"}
