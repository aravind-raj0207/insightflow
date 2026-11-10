"""Runs the agent for one question and stores the conversation. The API calls this."""
import json
import logging
import time

import pandas as pd

from app.agent.graph import build_graph
from app.database import repository
from app.llm.base import LLMClient
from app.utils.errors import DatasetNotFoundError, InsightFlowError, LLMError

logger = logging.getLogger(__name__)
RESULT_ROWS_RETURNED = 200


class AnalystService:
    def __init__(self, llm: LLMClient):
        self.graph = build_graph(llm)

    def ask(self, dataset_id: int, question: str, conversation_id: int | None = None) -> dict:
        started = time.perf_counter()
        repository.get_dataset(dataset_id)  # raises DatasetNotFoundError if missing

        if conversation_id is not None:
            if repository.get_conversation_dataset_id(conversation_id) != dataset_id:
                raise DatasetNotFoundError("That conversation does not belong to this dataset.")
        else:
            conversation_id = repository.create_conversation(dataset_id)

        history = repository.get_recent_turns(conversation_id)
        try:
            state = self.graph.invoke({"dataset_id": dataset_id, "question": question, "history": history})
        except InsightFlowError as exc:
            self._finish(dataset_id, conversation_id, question, None, "error", exc.technical_detail or exc.user_message, started)
            raise
        except Exception as exc:
            logger.exception("Agent crashed")
            self._finish(dataset_id, conversation_id, question, None, "error", str(exc), started)
            raise InsightFlowError("Something went wrong while analysing your question. Please try again.", str(exc))

        response = self._build_response(conversation_id, state)
        repository.add_message(conversation_id, "user", question)
        repository.add_message(
            conversation_id, "assistant", response["answer"], resolved_question=response["resolved_question"],
            sql=response["sql"], chart_spec=response["chart_spec"], status=response["status"],
        )
        self._finish(dataset_id, conversation_id, question, response["sql"], response["status"],
                     state.get("error"), started)
        return response

    @staticmethod
    def _build_response(conversation_id: int, state: dict) -> dict:
        df = state.get("result_df")
        rows, columns, row_count = [], [], 0
        if isinstance(df, pd.DataFrame):
            columns, row_count = list(df.columns), len(df)
            rows = json.loads(df.head(RESULT_ROWS_RETURNED).to_json(orient="records", date_format="iso"))
        spec = state.get("chart_spec")
        return {
            "conversation_id": conversation_id,
            "status": state.get("status", "error"),
            "answer": state.get("answer", ""),
            "resolved_question": state.get("standalone_question", state.get("question")),
            "intent_type": state.get("intent_type"),
            "sql": state.get("pretty_sql"),
            "chart_type": spec["chart_type"] if spec else None,
            "chart_spec": spec,
            "chart_reason": spec["reason"] if spec else None,
            "figure": state.get("figure"),
            "columns": columns, "rows": rows, "row_count": row_count,
            "facts": (state.get("analysis") or {}).get("facts", []),
        }

    @staticmethod
    def _finish(dataset_id, conversation_id, question, sql, status, error, started) -> None:
        repository.log_query(dataset_id, question, sql, status, error, int((time.perf_counter() - started) * 1000))
