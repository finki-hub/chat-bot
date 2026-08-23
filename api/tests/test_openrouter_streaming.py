import os
from collections.abc import Generator

import anyio
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from app.llms import openrouter
from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialSecret


def test_openrouter_byok_client_does_not_inherit_deployment_base_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://deployment.example/v1")

    llm = openrouter.get_openrouter_llm(
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    assert llm.openrouter_api_base is None
    assert os.environ["OPENROUTER_API_BASE"] == "https://deployment.example/v1"


def test_openrouter_regular_stream_preserves_reasoning_events(monkeypatch) -> None:
    class FakeOpenRouter:
        def stream(self, _messages) -> Generator[AIMessageChunk]:
            yield AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "thinking"},
                    ],
                },
            )
            yield AIMessageChunk(content="answer")

    monkeypatch.setattr(
        openrouter,
        "get_openrouter_llm",
        lambda *_args, **_kwargs: FakeOpenRouter(),
    )
    response = openrouter.stream_openrouter_response(
        "question",
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        reasoning=True,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    async def collect() -> str:
        chunks = [
            chunk if isinstance(chunk, str) else bytes(chunk).decode()
            async for chunk in response.body_iterator
        ]
        return "".join(chunks)

    body = anyio.run(collect)

    assert 'event: thinking\ndata: {"text": "thinking"}' in body
    assert 'event: token\ndata: {"text": "answer"}' in body


def test_openrouter_agent_fallback_preserves_upstream_model(monkeypatch) -> None:
    captured: list[str | None] = []

    async def fail_get_agent_tools() -> list[object]:
        msg = "agent tools unavailable"
        raise RuntimeError(msg)

    def fake_stream_openrouter_response(
        user_prompt: str,
        routed_model: Model,
        *,
        system_prompt: str,
        history=None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning: bool = False,
        credential: ChatCredentialSecret | None = None,
        upstream_model: str | None = None,
    ) -> StreamingResponse:
        captured.append(upstream_model)
        return StreamingResponse(iter(()), media_type="text/event-stream")

    monkeypatch.setattr(
        openrouter, "get_openrouter_llm", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(openrouter, "get_agent_tools", fail_get_agent_tools)
    monkeypatch.setattr(
        openrouter,
        "stream_openrouter_response",
        fake_stream_openrouter_response,
    )

    async def route() -> StreamingResponse:
        return await openrouter.stream_openrouter_agent_response(
            "question",
            Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
            system_prompt="system",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(
                provider="openrouter",
                api_key="sponsored-key",
            ),
            upstream_model="sponsored/upstream-model",
        )

    anyio.run(route)

    assert captured == ["sponsored/upstream-model"]
