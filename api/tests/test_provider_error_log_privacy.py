import logging

import anyio
import pytest

from app.llms import anthropic as anthropic_module
from app.llms import google as google_module
from app.llms import ollama as ollama_module
from app.llms.models import Model

_PRIVATE_PROMPT = "private prompt content"
_PRIVATE_PROVIDER_ERROR = (
    "provider rejected key=secret at https://private-provider.example/v1"
)


@pytest.mark.parametrize(
    ("module", "function_name", "client_factory", "model"),
    [
        (
            google_module,
            "transform_query_with_google",
            "get_google_llm",
            Model.GEMINI_3_5_FLASH,
        ),
        (
            anthropic_module,
            "transform_query_with_anthropic",
            "get_anthropic_llm",
            Model.CLAUDE_HAIKU_4_5,
        ),
        (ollama_module, "transform_query_with_ollama", "get_llm", Model.QWEN3_14B),
    ],
)
def test_provider_query_failure_omits_raw_error(
    caplog,
    monkeypatch,
    module,
    function_name,
    client_factory,
    model,
):
    def fail_client(*args, **kwargs):
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)

    monkeypatch.setattr(module, client_factory, fail_client)
    caplog.set_level(logging.WARNING, logger=module.__name__)

    async def collect():
        return await getattr(module, function_name)(
            _PRIVATE_PROMPT,
            model,
            system_prompt="Transform the query.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result == _PRIVATE_PROMPT
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


@pytest.mark.parametrize(
    ("module", "function_name", "client_factory", "fallback_name", "model"),
    [
        (
            google_module,
            "stream_google_agent_response",
            "get_google_llm",
            "stream_google_response",
            Model.GEMINI_3_5_FLASH,
        ),
        (
            anthropic_module,
            "stream_anthropic_agent_response",
            "get_anthropic_llm",
            "stream_anthropic_response",
            Model.CLAUDE_HAIKU_4_5,
        ),
        (
            ollama_module,
            "stream_ollama_agent_response",
            "get_llm",
            "stream_ollama_response",
            Model.QWEN3_14B,
        ),
    ],
)
def test_provider_agent_setup_failure_omits_raw_error(
    caplog,
    monkeypatch,
    module,
    function_name,
    client_factory,
    fallback_name,
    model,
):
    fallback_response = object()

    def fail_client(*args, **kwargs):
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)

    monkeypatch.setattr(module, client_factory, fail_client)
    monkeypatch.setattr(
        module,
        fallback_name,
        lambda *args, **kwargs: fallback_response,
    )
    caplog.set_level(logging.WARNING, logger=module.__name__)

    async def collect():
        return await getattr(module, function_name)(
            _PRIVATE_PROMPT,
            model,
            system_prompt="Answer safely.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result is fallback_response
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
