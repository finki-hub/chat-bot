import logging

import anyio

from app.llms import openai as openai_module
from app.llms.models import Model

_PRIVATE_PROMPT = "private prompt content"
_PRIVATE_PROVIDER_ERROR = (
    "provider rejected key=secret at https://private-provider.example/v1"
)


def test_openai_query_failure_omits_raw_error(caplog, monkeypatch):
    def fail_client(*args, **kwargs):
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)

    monkeypatch.setattr(openai_module, "get_openai_llm", fail_client)
    caplog.set_level(logging.WARNING, logger=openai_module.__name__)

    async def collect():
        return await openai_module.transform_query_with_openai(
            _PRIVATE_PROMPT,
            Model.GPT_5_4_MINI,
            system_prompt="Transform the query.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result == _PRIVATE_PROMPT
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_openai_agent_setup_failure_omits_raw_error(caplog, monkeypatch):
    fallback_response = object()

    def fail_client(*args, **kwargs):
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)

    monkeypatch.setattr(openai_module, "get_openai_llm", fail_client)
    monkeypatch.setattr(
        openai_module,
        "stream_openai_response",
        lambda *args, **kwargs: fallback_response,
    )
    caplog.set_level(logging.WARNING, logger=openai_module.__name__)

    async def collect():
        return await openai_module.stream_openai_agent_response(
            _PRIVATE_PROMPT,
            Model.GPT_5_6_LUNA,
            system_prompt="Answer safely.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result is fallback_response
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
