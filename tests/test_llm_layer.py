import pytest

from app.llm.base import parse_json_text
from app.llm.gemini_client import GeminiClient
from app.utils.errors import LLMError


def test_json_inside_code_fences_is_parsed():
    assert parse_json_text('```json\n{"sql": "SELECT 1"}\n```') == {"sql": "SELECT 1"}


def test_json_with_extra_text_is_parsed():
    assert parse_json_text('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_garbage_raises_friendly_error():
    with pytest.raises(LLMError) as error:
        parse_json_text("no json here")
    assert "Traceback" not in error.value.user_message


def test_missing_api_key_gives_clear_message():
    with pytest.raises(LLMError) as error:
        GeminiClient(api_key="", model="gemini-flash-latest")
    assert "GEMINI_API_KEY" in error.value.user_message


def test_gemini_client_builds_a_valid_request_config():
    from google.genai import types
    types.GenerateContentConfig(system_instruction="x", temperature=0.0, response_mime_type="application/json")
    GeminiClient(api_key="dummy-key", model="gemini-flash-latest")    # constructing does not call the network


def test_placeholder_key_from_env_example_is_treated_as_missing():
    with pytest.raises(LLMError) as error:
        GeminiClient(api_key="your_key_here", model="gemini-flash-latest")
    assert "GEMINI_API_KEY" in error.value.user_message


# ---- model selection / 404 recovery (no network: the Google client is replaced by a fake)
class _Model:
    def __init__(self, name, actions=("generateContent",)):
        self.name, self.supported_actions = name, list(actions)


def test_fallback_order_is_stable_full_then_lite_then_preview():
    from app.llm.gemini_client import rank_flash_models
    models = [_Model("models/gemini-2.5-flash-lite"), _Model("models/gemini-3-flash-preview"),
              _Model("models/gemini-2.5-flash"), _Model("models/gemini-2.0-flash")]
    assert rank_flash_models(models) == ["gemini-2.5-flash", "gemini-2.0-flash",
                                         "gemini-2.5-flash-lite", "gemini-3-flash-preview"]


def test_best_flash_model_prefers_stable_then_newest():
    from app.llm.gemini_client import pick_best_flash_model
    models = [_Model("models/gemini-2.0-flash"), _Model("models/gemini-2.5-flash"),
              _Model("models/gemini-2.5-flash-lite"), _Model("models/gemini-3-flash-preview"),
              _Model("models/gemini-flash-latest"), _Model("models/gemini-2.5-pro"),
              _Model("models/gemini-2.5-flash-image"), _Model("models/embedding-001", ["embedContent"])]
    assert pick_best_flash_model(models) == "gemini-2.5-flash"


def test_no_suitable_model_returns_none():
    from app.llm.gemini_client import pick_best_flash_model
    assert pick_best_flash_model([_Model("models/gemini-2.5-pro")]) is None


class _NotFound(Exception):
    code = 404


def test_client_recovers_from_unknown_model(monkeypatch):
    """The configured model gives 404 -> the client lists models, switches, and succeeds."""
    class FakeModels:
        def __init__(self): self.used = []
        def generate_content(self, model, contents, config):
            self.used.append(model)
            if model == "gemini-flash-latest":
                raise _NotFound("404 NOT_FOUND")
            return type("R", (), {"text": "hello"})()
        def list(self): return [_Model("models/gemini-2.5-flash"), _Model("models/gemini-2.5-pro")]

    client = GeminiClient(api_key="dummy-key", model="gemini-flash-latest")
    fake = FakeModels()
    client._client = type("C", (), {"models": fake})()
    assert client.generate("hi") == "hello"
    assert client.model == "gemini-2.5-flash" and fake.used == ["gemini-flash-latest", "gemini-2.5-flash"]


def test_unknown_model_with_no_alternative_gives_clear_message():
    class FakeModels:
        def generate_content(self, **kwargs): raise _NotFound("404")
        def list(self): return []

    client = GeminiClient(api_key="dummy-key", model="made-up-model")
    client._client = type("C", (), {"models": FakeModels()})()
    with pytest.raises(LLMError) as error:
        client.generate("hi")
    assert "list_models.py" in error.value.user_message


# ---- temporary overload (HTTP 503) handling
class _Overloaded(Exception):
    code = 503


class _Forbidden(Exception):
    code = 403


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.llm.gemini_client.time.sleep", lambda seconds: None)


def _client_with(fake_models):
    client = GeminiClient(api_key="dummy-key", model="gemini-flash-latest")
    client._client = type("C", (), {"models": fake_models})()
    return client


def test_short_overload_is_retried_on_the_same_model(no_sleep):
    class FakeModels:
        calls = 0
        def generate_content(self, model, contents, config):
            FakeModels.calls += 1
            if FakeModels.calls == 1:
                raise _Overloaded("503 UNAVAILABLE high demand")
            return type("R", (), {"text": "ok"})()
        def list(self): return []

    client = _client_with(FakeModels())
    assert client.generate("hi") == "ok" and FakeModels.calls == 2 and client.model == "gemini-flash-latest"


def test_busy_model_falls_back_to_another_flash_model(no_sleep):
    class FakeModels:
        used = []
        def generate_content(self, model, contents, config):
            FakeModels.used.append(model)
            if model == "gemini-flash-latest":
                raise _Overloaded("503")
            return type("R", (), {"text": "from fallback"})()
        def list(self): return [_Model("models/gemini-2.5-flash"), _Model("models/gemini-2.5-flash-lite")]

    client = _client_with(FakeModels())
    assert client.generate("hi") == "from fallback"
    assert FakeModels.used[-1] == "gemini-2.5-flash"
    assert client.model == "gemini-flash-latest"          # the busy model is still the preferred one next time


def test_everything_busy_gives_a_clear_message(no_sleep):
    class FakeModels:
        def generate_content(self, **kwargs): raise _Overloaded("503")
        def list(self): return [_Model("models/gemini-2.5-flash")]

    with pytest.raises(LLMError) as error:
        _client_with(FakeModels()).generate("hi")
    assert "busy" in error.value.user_message.lower()


def test_rejected_key_is_not_retried(no_sleep):
    class FakeModels:
        calls = 0
        def generate_content(self, **kwargs):
            FakeModels.calls += 1
            raise _Forbidden("403 PERMISSION_DENIED")
        def list(self): return []

    with pytest.raises(LLMError) as error:
        _client_with(FakeModels()).generate("hi")
    assert FakeModels.calls == 1 and "GEMINI_API_KEY" in error.value.user_message
