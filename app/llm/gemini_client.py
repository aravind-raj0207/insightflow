"""Gemini implementation of LLMClient, using Google's official `google-genai` SDK.

Reliability features (Gemini is a shared service and sometimes says "high demand"):
  * temporary errors (503 overloaded, 429 rate limit, 5xx) are retried with a growing wait
  * if a model stays busy, the next best Flash model your key can use is tried
  * if a model name does not exist for your key (404), a working one is chosen automatically
"""
import logging
import re
import time

from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.base import LLMClient
from app.utils.errors import LLMError

logger = logging.getLogger(__name__)
ATTEMPTS_PER_MODEL = 3
MAX_FALLBACK_MODELS = 2
_SKIP_WORDS = ("image", "live", "tts", "audio", "embed", "thinking", "exp", "native", "robotics", "computer")


def _model_id(name: str) -> str:
    return name.split("/", 1)[-1]


def rank_flash_models(models) -> list[str]:
    """Flash models that support generateContent, best first.

    Order: stable before preview, full Flash before Flash-Lite, newer before older.
    """
    ranked: list[tuple[tuple, str]] = []
    for model in models:
        name = _model_id(getattr(model, "name", "") or "")
        actions = getattr(model, "supported_actions", None) or []
        lowered = name.lower()
        if "flash" not in lowered or "generateContent" not in actions:
            continue
        if any(word in lowered for word in _SKIP_WORDS) or lowered.endswith("-latest"):
            continue
        version = re.search(r"gemini-(\d+(?:\.\d+)?)", lowered)
        version_number = float(version.group(1)) if version else 0.0
        key = ("preview" not in lowered, "lite" not in lowered, version_number, -len(lowered))
        ranked.append((key, name))
    return [name for _, name in sorted(ranked, reverse=True)]


def pick_best_flash_model(models) -> str | None:
    ranked = rank_flash_models(models)
    return ranked[0] if ranked else None


def _code(exc: Exception) -> int | None:
    return getattr(exc, "code", None)


def _is_model_not_found(exc: Exception) -> bool:
    return _code(exc) == 404


def _is_transient(exc: Exception) -> bool:
    """Overloaded / rate limited / server trouble / network trouble - worth retrying."""
    code = _code(exc)
    return code is None or code == 429 or code >= 500


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float | None = None,
                 api_version: str | None = "v1beta"):
        if not api_key or api_key.strip() in ("your_key_here", "YOUR_KEY_HERE"):   # empty or still the template text
            raise LLMError(
                "The AI service is not configured. Add GEMINI_API_KEY to the .env file and restart the backend."
            )
        http_options = types.HttpOptions(api_version=api_version) if api_version else None
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._model = model
        self._temperature = temperature
        self._fallbacks: list[str] | None = None      # fetched from Google the first time they are needed

    @property
    def model(self) -> str:
        return self._model

    def _fallback_models(self, exclude: set[str]) -> list[str]:
        if self._fallbacks is None:
            try:
                self._fallbacks = rank_flash_models(self._client.models.list())
            except Exception as exc:
                logger.warning("Could not list Gemini models: %s", exc)
                self._fallbacks = []
        return [m for m in self._fallbacks if m not in exclude][:MAX_FALLBACK_MODELS]

    def _call(self, model: str, prompt: str, config) -> str:
        response = self._client.models.generate_content(model=model, contents=prompt, config=config)
        return response.text or ""

    def generate(self, prompt: str, system: str | None = None, json_output: bool = False,
                 task: str = "generic") -> str:
        options: dict = {"system_instruction": system}
        if json_output:
            options["response_mime_type"] = "application/json"
        if self._temperature is not None:        # otherwise the model's own default is used
            options["temperature"] = self._temperature
        config = types.GenerateContentConfig(**options)

        candidates = [self._model]
        index = 0
        last_error: Exception | None = None
        not_found: set[str] = set()
        while index < len(candidates):
            model = candidates[index]
            for attempt in range(1, ATTEMPTS_PER_MODEL + 1):
                try:
                    text = self._call(model, prompt, config)
                    if text:
                        if model != self._model:
                            logger.warning("Answered by '%s' instead of '%s'.%s", model, self._model,
                                           f" Set GEMINI_MODEL={model} in .env to make this permanent."
                                           if self._model in not_found else "")
                            if self._model in not_found:
                                self._model = model          # the configured model does not exist: stop using it
                        return text
                    last_error = LLMError("Empty response from the model.")
                except Exception as exc:  # network errors, quota, overload, safety blocks...
                    last_error = exc
                    logger.warning("Gemini call failed (task=%s, model=%s, attempt %d/%d): %s",
                                   task, model, attempt, ATTEMPTS_PER_MODEL, exc)
                    if _is_model_not_found(exc):
                        not_found.add(model)
                        break                                 # no point retrying a model that does not exist
                    if not _is_transient(exc):
                        raise LLMError(
                            "Gemini rejected the request. Check that GEMINI_API_KEY in .env is correct and "
                            "active (https://aistudio.google.com/apikey).", str(exc))
                if attempt < ATTEMPTS_PER_MODEL:
                    time.sleep(2 * attempt)                   # wait 2 s, then 4 s
            # This model is not available or stays busy: line up the next-best models (once)
            if index == len(candidates) - 1:
                candidates += self._fallback_models(exclude=set(candidates))
            index += 1

        if not_found and len(not_found) == len(candidates):
            raise LLMError(
                f"The Gemini model '{self._model}' is not available for your API key. "
                "Run: python scripts/list_models.py  and put one of the listed names in GEMINI_MODEL in .env.",
                str(last_error))
        raise LLMError(
            "Gemini is busy right now (high demand). Please wait a minute and try again.",
            str(last_error))


_cached_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Create the LLM client once. Replace this function to switch provider."""
    global _cached_client
    if _cached_client is None:
        settings = get_settings()
        _cached_client = GeminiClient(settings.gemini_api_key, settings.gemini_model,
                                      settings.gemini_temperature, settings.gemini_api_version or None)
    return _cached_client
