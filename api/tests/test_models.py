from collections.abc import AsyncIterator
from inspect import Parameter, signature

import anyio
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage

from app.llms import streams
from app.llms.agents import StreamObservation
from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    CHAT_MODELS,
    GPU_API_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)
from app.llms.pricing import is_self_hosted
from app.llms.provider_credentials import provider_for_model
from app.schemas.chat_credentials import ChatCredentialSecret


def test_stream_response_requires_explicit_interface() -> None:
    interface = signature(streams.stream_response_with_agent).parameters["interface"]

    assert interface.default is Parameter.empty


def test_claude_sonnet_5_is_a_supported_anthropic_chat_model():
    model = Model("claude-sonnet-5")

    assert model in CHAT_MODELS
    assert model in ANTHROPIC_QUERY_TRANSFORM_MODELS
    assert model in REASONING_CAPABLE_MODELS


def test_every_byok_chat_model_requires_provider_credential() -> None:
    self_hosted_models = {*GPU_API_MODELS}
    hosted_models = CHAT_MODELS - self_hosted_models

    assert hosted_models
    assert {
        model for model in hosted_models if provider_for_model(model) is None
    } == set()


def test_ollama_models_are_not_priced_as_self_hosted() -> None:
    assert not is_self_hosted(Model.MISTRAL)


def test_claude_sonnet_5_routes_to_anthropic(monkeypatch):
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
        credential: ChatCredentialSecret | None = None,
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

    async def route_response() -> StreamingResponse:
        return await streams.stream_response_with_agent(
            "test",
            model,
            system_prompt="system",
            history=[],
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            interface="discord",
        )

    response = anyio.run(route_response)

    assert response.headers["x-routed-model"] == "claude-sonnet-5"
