"""The interface every LLM provider must follow.

The rest of the app only knows about LLMClient. To switch from Gemini to another
provider, write a new class with a `generate` method and change `get_llm_client`.
"""
import json
import re
from abc import ABC, abstractmethod

from app.utils.errors import LLMError


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, json_output: bool = False,
                 task: str = "generic") -> str:
        """Send a prompt, return the model's text. `task` is a label used for logging/testing."""

    def generate_json(self, prompt: str, system: str | None = None, task: str = "generic") -> dict:
        """Ask for JSON and parse it. Raises LLMError if the answer is not valid JSON."""
        raw = self.generate(prompt, system=system, json_output=True, task=task)
        return parse_json_text(raw)


def parse_json_text(raw: str) -> dict:
    text = raw.strip()
    # Models sometimes wrap JSON in ```json fences even when told not to
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMError("The AI returned an answer I could not understand. Please try again.", raw[:500])
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("The AI returned an answer I could not understand. Please try again.", str(exc))
    if not isinstance(data, dict):
        raise LLMError("The AI returned an unexpected answer format.", raw[:500])
    return data
