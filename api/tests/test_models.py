from collections.abc import AsyncIterator

import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage

from app.llms import streams
from app.llms.agents import StreamObservation
from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    CHAT_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)


def test_claude_sonnet_5_is_a_supported_anthropic_chat_model():
    model = Model("claude-sonnet-5")

    assert model in CHAT_MODELS
    assert model in ANTHROPIC_QUERY_TRANSFORM_MODELS
    assert model in REASONING_CAPABLE_MODELS


@pytest.mark.anyio
async def test_claude_sonnet_5_routes_to_anthropic(monkeypatch):
    model = Model("claude-sonnet-5")

    async def fake_stream_anthropic_agent_response(
        user_prompt: str,
        routed_model: Model,
        *,
        system_prompt: str,
        history: list[BaseMessage] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning: bool = False,
        observation: StreamObservation | None = None,
    ) -> StreamingResponse:
        async def empty_body() -> AsyncIterator[bytes]:
            if False:
                yield b""

        response = StreamingResponse(empty_body(), media_type="text/event-stream")
        response.headers["x-routed-model"] = routed_model.value
        return response

    monkeypatch.setattr(
        streams,
        "stream_anthropic_agent_response",
        fake_stream_anthropic_agent_response,
    )

    response = await streams.stream_response_with_agent(
        "test",
        model,
        system_prompt="system",
        history=[],
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )

    assert response.headers["x-routed-model"] == "claude-sonnet-5"
