from collections.abc import AsyncIterator
from inspect import Parameter, signature

import anyio
import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from app.constants.defaults import (
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_QUERY_TRANSFORM_MODEL,
)
from app.llms import streams
from app.llms.agents import StreamObservation
from app.llms.models import (
    ACTIVE_EMBEDDING_MODELS,
    ALL_MODELS_EMBEDDINGS,
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    CHAT_MODEL_ORDER,
    CHAT_MODELS,
    GPU_API_MODELS,
    QUERY_TRANSFORM_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)
from app.llms.pricing import is_self_hosted
from app.llms.provider_credentials import provider_for_model
from app.schemas.chat import ChatSchema
from app.schemas.chat_credentials import ChatCredentialSecret

EXPECTED_CHAT_IDS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "qwen3:30b-a3b-thinking-2507-q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "qwen3:14b-q4_K_M",
)


def test_removed_chat_values_are_not_parseable_or_active() -> None:
    # Given superseded chat values that must not remain in runtime policy
    legacy_values = {
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.2",
        "mistral:latest",
        "llama3.3:70b",
        "deepseek-r1:70b",
        "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0",
        "hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0",
        "qwen2.5:72b",
    }

    # When active and enum values are compared with the removed values
    active_values = {model.value for model in CHAT_MODELS}
    enum_values = {model.value for model in Model}

    # Then removed IDs cannot reach provider dispatch through the runtime enum
    assert active_values == set(EXPECTED_CHAT_IDS)
    assert active_values.isdisjoint(legacy_values)
    assert enum_values.isdisjoint(legacy_values)


def test_embedding_fill_and_gpu_maps_only_activate_bge_m3() -> None:
    # Given the modernized embedding maps
    # When active fill and GPU dispatch values are read
    fill_values = tuple(model.value for model in ALL_MODELS_EMBEDDINGS)
    gpu_values = set(GPU_API_MODELS.values())

    # Then only BGE-M3 is active while legacy enum values remain readable
    assert fill_values == ("BAAI/bge-m3",)
    assert gpu_values == {"BAAI/bge-m3"}


def test_stream_response_requires_explicit_interface() -> None:
    interface = signature(streams.stream_response_with_agent).parameters["interface"]

    assert interface.default is Parameter.empty


def test_claude_sonnet_5_is_a_supported_anthropic_chat_model():
    model = Model("claude-sonnet-5")

    assert model in CHAT_MODELS
    assert model in ANTHROPIC_QUERY_TRANSFORM_MODELS
    assert model in REASONING_CAPABLE_MODELS


def test_curated_defaults_and_embedding_policy() -> None:
    values = {model.value for model in ACTIVE_EMBEDDING_MODELS}

    assert DEFAULT_INFERENCE_MODEL.value == "claude-sonnet-5"
    assert DEFAULT_QUERY_TRANSFORM_MODEL.value == "gpt-5.4-mini"
    assert DEFAULT_EMBEDDINGS_MODEL.value == "BAAI/bge-m3"
    assert values == {
        "BAAI/bge-m3",
        "gemini-embedding-001",
        "text-embedding-3-large",
    }
    assert tuple(model.value for model in ALL_MODELS_EMBEDDINGS) == ("BAAI/bge-m3",)


def test_every_curated_chat_model_supports_query_transform_and_one_provider() -> None:
    # Given the executable catalog
    # When provider and query-transform policy are inspected
    providers = [provider_for_model(model) for model in CHAT_MODEL_ORDER]

    # Then no active model can bypass either routing policy
    assert CHAT_MODELS == QUERY_TRANSFORM_MODELS
    assert all(provider is not None for provider in providers)


def test_removed_chat_model_is_rejected_by_request_schema() -> None:
    # Given a request naming a superseded but historically known model
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "inference_model": "claude-sonnet-4-6",
    }

    # When request validation runs, then provider dispatch is never reachable
    with pytest.raises(ValidationError):
        ChatSchema.model_validate(payload)


def test_every_byok_chat_model_requires_provider_credential() -> None:
    self_hosted_models = {*GPU_API_MODELS}
    hosted_models = CHAT_MODELS - self_hosted_models

    assert hosted_models
    assert {
        model for model in hosted_models if provider_for_model(model) is None
    } == set()


def test_ollama_models_are_not_priced_as_self_hosted() -> None:
    assert not is_self_hosted(Model.QWEN3_14B)


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
