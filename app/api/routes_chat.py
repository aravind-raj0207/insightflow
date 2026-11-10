import threading
from fastapi import APIRouter, Depends

from app.agent.runner import AnalystService
from app.api.schemas import ChatRequest, ChatResponse
from app.database import repository
from app.llm.gemini_client import get_llm_client

router = APIRouter(prefix="/api", tags=["chat"])

_service: AnalystService | None = None
_service_lock = threading.Lock()


def get_analyst_service() -> AnalystService:
    """Create the agent once. Tests replace this with a service that uses a fake LLM."""
    global _service
    with _service_lock:
        if _service is None:
            _service = AnalystService(get_llm_client())
        return _service


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, service: AnalystService = Depends(get_analyst_service)) -> dict:
    return service.ask(request.dataset_id, request.question.strip(), request.conversation_id)


@router.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int) -> list[dict]:
    return repository.get_messages(conversation_id)
