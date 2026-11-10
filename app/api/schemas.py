"""Request / response shapes for the API."""
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    dataset_id: int
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    conversation_id: int
    status: str                       # ok / cannot_answer / clarify / rejected / empty / error
    answer: str
    resolved_question: str | None = None
    intent_type: str | None = None
    sql: str | None = None
    chart_type: str | None = None
    chart_reason: str | None = None
    chart_spec: dict | None = None
    figure: dict[str, Any] | None = None
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    facts: list[str] = []
