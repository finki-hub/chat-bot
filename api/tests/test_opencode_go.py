import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

import anyio
import httpx
import httpx2
import pytest
from anthropic import Anthropic
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import ValidationError

from app.api import chat as chat_api
from app.api import chat_title
from app.api.chat_state import _ensure_allowed_base_url, _normalize_default_base_url
from app.api.provider_credentials import resolve_provider_credentials
from app.data.chat_credentials import upsert_chat_credential
from app.data.connection import Database
from app.llms import opencode_go, query_transform, streams
from app.llms.agents import StreamObservation, _chunk_reasoning
from app.llms.model_access import (
    ModelAccessContext,
    SponsoredSettings,
    overlay_model_access,
)
from app.llms.model_catalog import CatalogFetchError, ModelCatalogService
from app.llms.model_catalog_remote import parse_models_dev
from app.llms.models import (
    OPENCODE_GO_QUERY_TRANSFORM_MODELS,
    Model,
    is_ollama_model_tag,
)
from app.llms.provider_credentials import (
    LlmProviderCredentials,
    ProviderCredentialRequiredError,
    provider_for_model,
)
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_result import RetrievedContext
from app.schemas.chat import ChatSchema, ConversationTurn
from app.schemas.chat_credentials import ChatCredentialSecret, ChatCredentialUpsert
from app.schemas.chat_title import ChatTitleSchema
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

SESSION = "00000000-0000-4000-8000-000000000123"
USER_ID = UUID("00000000-0000-4000-8000-000000000001")
GO_KEY = "test-go-user-key"
GO_CREDENTIAL = ChatCredentialSecret(provider="opencode-go", api_key=GO_KEY)
GO_MODELS = tuple(
    model for model in Model if model in OPENCODE_GO_QUERY_TRANSFORM_MODELS
)


def make_llm(model, *, reasoning=False, credential=GO_CREDENTIAL, session=SESSION):
    return opencode_go.get_opencode_go_llm(
        model,
        0.2,
        0.8,
        8192,
        reasoning=reasoning,
        credential=credential,
        conversation_id=session,
    )


@pytest.mark.parametrize("model", GO_MODELS)
def test_go_sdk_wire_endpoint_auth_and_model(model, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "deployment-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "deployment-anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "deployment-token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://deployment.invalid/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://deployment.invalid")
    requests = []
    upstream = model.value.removeprefix("opencode-go:")

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/messages"):
            body = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": upstream,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        elif request.url.path.endswith("/responses"):
            body = {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": upstream,
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "ok", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        else:
            body = {
                "id": "chat_test",
                "object": "chat.completion",
                "created": 0,
                "model": upstream,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "reason",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        library = (
            httpx2
            if upstream in {"claude-haiku-5-5", "minimax-m3", "qwen3.8-max"}
            else httpx
        )
        return library.Response(200, json=body)

    llm = make_llm(model)
    llm.streaming = False
    library = httpx2 if isinstance(llm, ChatAnthropic) else httpx
    with library.Client(transport=library.MockTransport(handle)) as http_client:
        if isinstance(llm, ChatAnthropic):
            llm.__dict__["_client"] = Anthropic(
                **llm._client_params, http_client=http_client
            )
            endpoint = "/messages"
        else:
            root = OpenAI(
                api_key=GO_KEY,
                base_url=llm.openai_api_base,
                default_headers=llm.default_headers,
                http_client=http_client,
            )
            llm.root_client = root
            llm.client = root.chat.completions
            endpoint = "/responses" if llm.use_responses_api else "/chat/completions"
        assert llm.invoke([HumanMessage(content="test")]).text == "ok"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"https://opencode.ai/inference/go/v1{endpoint}"
    assert request.headers["user-agent"] == "finki-hub/1.0"
    assert request.headers["x-opencode-session"] == SESSION
    if endpoint == "/messages":
        assert request.headers["x-api-key"] == GO_KEY
        assert "authorization" not in request.headers
    else:
        assert request.headers["authorization"] == f"Bearer {GO_KEY}"
    payload = json.loads(request.content)
    assert payload["model"] == upstream
    assert "openrouter_provider" not in payload
    assert provider_for_model(model) == "opencode-go"
    if endpoint == "/messages":
        assert payload["thinking"] == {"type": "disabled"}
    elif model == Model.OPENCODE_GO_GPT_6_LUNA:
        assert payload["reasoning"] == {"effort": "none"}
    elif model == Model.OPENCODE_GO_GROK_4_7:
        assert payload["reasoning"] == {"effort": "low"}
    else:
        assert payload["max_completion_tokens"] == 8192
        assert payload["reasoning_effort"] == (
            "none"
            if model
            in {
                Model.OPENCODE_GO_DEEPSEEK_V4_1_FLASH,
                Model.OPENCODE_GO_DEEPSEEK_V4_PRO,
            }
            else "low"
        )


@pytest.mark.parametrize("model", GO_MODELS)
def test_go_requires_own_user_key_and_session(model, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "server-key")
    with pytest.raises(ProviderCredentialRequiredError):
        make_llm(model, credential=None)
    with pytest.raises(ValueError, match="own provider"):
        make_llm(
            model,
            credential=ChatCredentialSecret(provider="openai", api_key="other-key"),
        )
    with pytest.raises(ValueError, match="conversation ID"):
        make_llm(model, session=None)


@pytest.mark.parametrize("unknown", ["opencode-go:unknown", "opencode-go:gpt-5.6-luna"])
def test_unknown_go_models_are_not_ollama(unknown):
    assert not is_ollama_model_tag(unknown)
    with pytest.raises(ValidationError):
        ChatSchema(
            messages=[ConversationTurn(role="user", content="test")],
            inference_model=unknown,
        )
    with pytest.raises(ValueError, match="Unsupported"):
        provider_for_model(unknown)


@pytest.mark.parametrize("model", GO_MODELS)
def test_go_reasoning_uses_protocol_specific_supported_shapes(model):
    llm = make_llm(model, reasoning=True)
    if isinstance(llm, ChatAnthropic):
        payload = llm._get_request_payload([HumanMessage(content="test")])
        assert "reasoning" not in payload
        if model == Model.OPENCODE_GO_CLAUDE_HAIKU_5_5:
            assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert payload["output_config"]["effort"] == "medium"
        else:
            assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
            assert "temperature" not in payload
    else:
        payload = llm._get_request_payload([HumanMessage(content="test")])
        if llm.use_responses_api:
            assert payload["reasoning"]["effort"] in {"medium", "high"}
            assert "reasoning_effort" not in payload
            assert payload["store"] is False
        else:
            assert payload["reasoning_effort"] == (
                "max" if model == Model.OPENCODE_GO_KIMI_K3 else "high"
            )
            assert "reasoning" not in payload
    disabled = make_llm(model)
    if isinstance(disabled, ChatAnthropic):
        assert disabled.thinking == {"type": "disabled"}
    elif disabled.use_responses_api and model == Model.OPENCODE_GO_GPT_6_LUNA:
        assert disabled.reasoning == {"effort": "none"}
    elif model in {
        Model.OPENCODE_GO_DEEPSEEK_V4_1_FLASH,
        Model.OPENCODE_GO_DEEPSEEK_V4_PRO,
    }:
        assert disabled.reasoning_effort == "none"
    elif model == Model.OPENCODE_GO_GROK_4_7:
        assert disabled.reasoning == {"effort": "low"}
    else:
        assert disabled.reasoning_effort == "low"


def test_go_chat_reasoning_roundtrips_stream_and_tool_history():
    llm = make_llm(Model.OPENCODE_GO_DEEPSEEK_V4_PRO)
    chunk = llm._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "thinking",
                    }
                }
            ]
        },
        AIMessageChunk,
        None,
    )
    assert _chunk_reasoning(chunk.message) == "thinking"
    assert chunk.message.response_metadata["model_provider"] == "opencode-go"
    payload = llm._get_request_payload(
        [
            HumanMessage(content="test"),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "thinking"},
                tool_calls=[
                    {"name": "search", "args": {}, "id": "call_1", "type": "tool_call"}
                ],
            ),
        ]
    )
    assert payload["messages"][1]["reasoning_content"] == "thinking"
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call_1"


@pytest.mark.parametrize(
    "base",
    [None, "https://opencode.ai/inference/go", "https://opencode.ai/inference/go/v1/"],
)
def test_go_default_credential_base_is_normalized_without_allowlist(base):
    payload = _normalize_default_base_url(
        ChatCredentialUpsert(provider="opencode-go", api_key=GO_KEY, base_url=base)
    )
    assert payload.base_url is None
    _ensure_allowed_base_url(payload, Settings(BYOK_ALLOWED_BASE_URLS=""))


def test_go_custom_credential_base_still_requires_exact_allowlist():
    payload = ChatCredentialUpsert(
        provider="opencode-go", api_key=GO_KEY, base_url="https://proxy.example/go/v1"
    )
    with pytest.raises(HTTPException):
        _ensure_allowed_base_url(payload, Settings(BYOK_ALLOWED_BASE_URLS=""))
    assert payload.base_url is not None
    _ensure_allowed_base_url(payload, Settings(BYOK_ALLOWED_BASE_URLS=payload.base_url))
    llm = make_llm(
        Model.OPENCODE_GO_CLAUDE_HAIKU_5_5,
        credential=ChatCredentialSecret(
            provider="opencode-go", api_key=GO_KEY, base_url=payload.base_url
        ),
    )
    assert llm.anthropic_api_url == "https://proxy.example/go"


def test_go_saved_credentials_are_isolated_and_revoked_by_allowlist():
    db = FakeChatDatabase()
    settings = Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
        MCP_API_KEY="test-mcp-key",
    )

    async def run():
        await upsert_chat_credential(
            db,
            user_id=USER_ID,
            credential=ChatCredentialUpsert(
                provider="openai", api_key="openai-user-key"
            ),
            settings=settings,
        )
        result = await resolve_provider_credentials(
            db,
            user_id=USER_ID,
            providers=frozenset({"opencode-go", "openai"}),
            settings=settings,
        )
        assert result is not None
        assert result.opencode_go is None
        assert result.openai is not None
        assert result.openai.api_key == "openai-user-key"
        await upsert_chat_credential(
            db,
            user_id=USER_ID,
            credential=ChatCredentialUpsert(
                provider="opencode-go",
                api_key=GO_KEY,
                base_url="https://opencode.ai/inference/go/v1",
            ),
            settings=settings,
        )
        result = await resolve_provider_credentials(
            db, user_id=USER_ID, providers=frozenset({"opencode-go"}), settings=settings
        )
        assert result is not None
        assert result.available_providers() == {"opencode-go"}
        assert result.opencode_go is not None
        assert result.opencode_go.base_url is None
        assert result.opencode_go.api_key == GO_KEY
        go_credential = db.credentials[(USER_ID, "opencode-go")]
        go_credential["base_url"] = "https://unapproved.example/v1"
        result = await resolve_provider_credentials(
            db, user_id=USER_ID, providers=frozenset({"opencode-go"}), settings=settings
        )
        assert result is not None
        assert result.opencode_go is None
        assert result.rejected_providers == {"opencode-go"}

    anyio.run(run)


def test_go_query_and_title_auxiliary_calls_retain_session(monkeypatch):
    captured = []

    async def transform(*args, **kwargs):
        captured.append(kwargs)
        return "title"

    monkeypatch.setattr(query_transform, "transform_query_with_opencode_go", transform)
    credentials = LlmProviderCredentials(
        opencode_go=GO_CREDENTIAL, conversation_id=SESSION
    )
    monkeypatch.setattr(
        chat_title,
        "resolve_provider_credentials",
        AsyncMock(return_value=replace(credentials, conversation_id=None)),
    )

    async def run():
        await query_transform.transform_query(
            "test",
            Model.OPENCODE_GO_GPT_6_LUNA,
            temperature=0,
            top_p=1,
            max_tokens=128,
            credentials=credentials,
        )
        result = await chat_title.generate_chat_title(
            ChatTitleSchema(
                user_id=USER_ID,
                provider_model=Model.OPENCODE_GO_KIMI_K3,
                messages=[ConversationTurn(role="user", content="test")],
            ),
            FakeChatDatabase(),
            Settings(),
            conversation_id=SESSION,
        )
        assert result.title == "title"

    anyio.run(run)
    assert len(captured) == 2
    assert all(
        call["credential"] == GO_CREDENTIAL and call["conversation_id"] == SESSION
        for call in captured
    )


def test_go_stream_dispatch_tags_provider_and_retains_session(monkeypatch):
    dispatch = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(streams, "stream_opencode_go_agent_response", dispatch)
    observation = StreamObservation(distinct_id="test", response_id="test")

    async def run():
        await streams.stream_response_with_agent(
            "test",
            Model.OPENCODE_GO_GLM_5_3,
            system_prompt="system",
            history=[],
            temperature=0,
            top_p=1,
            max_tokens=128,
            interface="web",
            observation=observation,
            credential=GO_CREDENTIAL,
            conversation_id=SESSION,
        )

    anyio.run(run)
    assert observation.provider == "opencode-go"
    assert dispatch.call_args.kwargs["conversation_id"] == SESSION
    assert dispatch.call_args.kwargs["credential"] == GO_CREDENTIAL


@pytest.mark.parametrize(
    ("model", "expected_reasoning"),
    [
        (Model.OPENCODE_GO_GROK_4_7, True),
        (Model.OPENCODE_GO_GLM_5_3, True),
        (Model.OPENCODE_GO_KIMI_K3, True),
        (Model.OPENCODE_GO_DEEPSEEK_V4_1_FLASH, False),
    ],
)
def test_go_stream_reasoning_policy(monkeypatch, model, expected_reasoning):
    dispatch = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(streams, "stream_opencode_go_agent_response", dispatch)

    async def run():
        await streams.stream_response_with_agent(
            "test",
            model,
            system_prompt="system",
            history=[],
            temperature=0,
            top_p=1,
            max_tokens=128,
            reasoning=False,
            interface="web",
            credential=GO_CREDENTIAL,
            conversation_id=SESSION,
        )

    anyio.run(run)
    assert dispatch.call_args.kwargs["reasoning"] is expected_reasoning


def test_go_agent_fallback_reuses_scoped_client_and_preserves_reasoning(
    monkeypatch, caplog
):
    captured = []
    closed = []

    class FakeClient:
        def stream(self, messages):
            try:
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": "think"}
                )
                yield AIMessageChunk(content="answer")
            finally:
                closed.append(True)

    def factory(*args, **kwargs):
        captured.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(opencode_go, "get_opencode_go_llm", factory)
    monkeypatch.setattr(
        opencode_go,
        "get_agent_tools",
        AsyncMock(side_effect=RuntimeError(f"{GO_KEY} private prompt")),
    )

    async def run():
        response = await opencode_go.stream_opencode_go_agent_response(
            "private prompt",
            Model.OPENCODE_GO_GLM_5_3,
            system_prompt="system",
            temperature=0,
            top_p=1,
            max_tokens=128,
            credential=GO_CREDENTIAL,
            conversation_id=SESSION,
        )
        return "".join([str(chunk) async for chunk in response.body_iterator])

    body = anyio.run(run)
    assert "event: thinking" in body
    assert "answer" in body
    assert closed == [True]
    assert len(captured) == 1
    assert captured[0]["conversation_id"] == SESSION
    assert GO_KEY not in caplog.text
    assert "private prompt" not in caplog.text


def test_go_auxiliary_failure_is_private_and_cancellation_propagates(
    monkeypatch, caplog
):
    async def fail(*args, **kwargs):
        raise RuntimeError(f"{GO_KEY} private query")

    monkeypatch.setattr(
        opencode_go,
        "get_opencode_go_llm",
        lambda *a, **k: SimpleNamespace(ainvoke=fail),
    )

    async def run():
        result = await opencode_go.transform_query_with_opencode_go(
            "private query",
            Model.OPENCODE_GO_GPT_6_LUNA,
            system_prompt="system",
            temperature=0,
            top_p=1,
            max_tokens=32,
            credential=GO_CREDENTIAL,
            conversation_id=SESSION,
        )
        assert result == "private query"

    anyio.run(run)
    assert GO_KEY not in caplog.text
    assert "private query" not in caplog.text

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        opencode_go,
        "get_opencode_go_llm",
        lambda *a, **k: SimpleNamespace(ainvoke=cancel),
    )
    with pytest.raises(asyncio.CancelledError):
        anyio.run(run)


def test_go_catalog_and_remote_lookup_keep_provider_namespace():
    payload = json.dumps(
        {
            "opencode-go": {
                "id": "opencode-go",
                "name": "Go",
                "models": {"gpt-6-luna": {"id": "gpt-6-luna", "name": "Go Luna"}},
            }
        }
    ).encode()
    assert parse_models_dev(payload)[Model.OPENCODE_GO_GPT_6_LUNA].name == "Go Luna"

    async def unavailable():
        raise CatalogFetchError("offline")

    async def run():
        catalog = await ModelCatalogService(fetch_metadata=unavailable).get_catalog()
        go = [
            descriptor
            for descriptor in catalog.models
            if descriptor.provider == "opencode-go"
        ]
        assert len(go) == 9
        context = ModelAccessContext(
            available_providers=frozenset(),
            sponsored_settings=SponsoredSettings(enabled=False),
            personal_quota=None,
            global_quota=None,
            utc_reset=datetime.now(UTC),
        )
        assert all(
            overlay_model_access(descriptor, context).availability == "unavailable"
            for descriptor in go
        )
        with_key = replace(context, available_providers=frozenset({"opencode-go"}))
        assert all(
            overlay_model_access(descriptor, with_key).availability == "byok"
            for descriptor in go
        )

    anyio.run(run)


def test_go_migration_is_additive_and_preserves_all_providers():
    text = (
        Path(__file__).parents[1]
        / "resources/migrations/0012_allow_opencode_go_chat_credentials.sql"
    ).read_text()
    assert "chat_user_credential_provider_check" in text
    assert all(
        f"'{provider}'" in text
        for provider in (
            "openai",
            "google",
            "anthropic",
            "ollama",
            "openrouter",
            "opencode-go",
        )
    )


def _stream_fixture(path, upstream):
    events: list[dict[str, object]]
    if path.endswith("/responses"):
        events = [
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "in_progress",
                    "model": upstream,
                    "output": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "delta": "answer",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": upstream,
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    elif path.endswith("/messages"):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": upstream,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "answer"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    else:
        return (
            "data: "
            + json.dumps(
                {
                    "id": "chat_1",
                    "object": "chat.completion.chunk",
                    "model": upstream,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "answer",
                                "reasoning_content": "think",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


@pytest.mark.parametrize("model", GO_MODELS)
@pytest.mark.parametrize("phase", ["agent", "fallback", "auxiliary"])
def test_go_headers_persist_on_actual_stream_and_auxiliary_http_calls(
    model, phase, monkeypatch
):
    requests = []
    upstream = model.value.removeprefix("opencode-go:")

    def send(client, request, **kwargs):
        requests.append(request)
        library = (
            httpx2 if isinstance(client, httpx2.Client | httpx2.AsyncClient) else httpx
        )
        return library.Response(
            200,
            request=request,
            content=_stream_fixture(request.url.path, upstream),
            headers={"content-type": "text/event-stream"},
        )

    async def async_send(client, request, **kwargs):
        return send(client, request, **kwargs)

    for library in (httpx, httpx2):
        monkeypatch.setattr(library.Client, "send", send)
        monkeypatch.setattr(library.AsyncClient, "send", async_send)
    monkeypatch.setattr(
        opencode_go,
        "get_agent_tools",
        AsyncMock(
            side_effect=RuntimeError("setup") if phase == "fallback" else None,
            return_value=[],
        ),
    )

    async def run():
        if phase == "auxiliary":
            result = await opencode_go.transform_query_with_opencode_go(
                "query",
                model,
                system_prompt="system",
                temperature=0,
                top_p=1,
                max_tokens=128,
                credential=GO_CREDENTIAL,
                conversation_id=SESSION,
            )
            assert result == "answer"
        else:
            response = await opencode_go.stream_opencode_go_agent_response(
                "query",
                model,
                system_prompt="system",
                temperature=0,
                top_p=1,
                max_tokens=128,
                credential=GO_CREDENTIAL,
                conversation_id=SESSION,
            )
            text = "".join([str(chunk) async for chunk in response.body_iterator])
            assert "answer" in text
            assert "event: error" not in text

    anyio.run(run)
    assert requests
    for request in requests:
        assert request.headers["x-opencode-session"] == SESSION
        assert request.headers["user-agent"] == "finki-hub/1.0"
        assert json.loads(request.content)["model"] == upstream


@pytest.mark.parametrize("session", [SESSION, None, "invalid-id"])
def test_go_chat_request_session_reaches_main_and_retrieval(session, monkeypatch):
    provider_credentials = LlmProviderCredentials(opencode_go=GO_CREDENTIAL)
    resolver = AsyncMock(return_value=provider_credentials)
    context = AsyncMock(
        return_value=RetrievedContext(
            text="context", effective_transform_mode=QueryTransformMode.RAW
        )
    )

    async def body():
        yield 'event: token\ndata: {"text":"answer"}\n\n'
        yield "event: done\ndata: {}\n\n"

    main = AsyncMock(return_value=StreamingResponse(body()))
    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolver)
    monkeypatch.setattr(chat_api, "get_retrieved_context_with_sources", context)
    monkeypatch.setattr(chat_api, "get_links_context", AsyncMock(return_value=""))
    monkeypatch.setattr(chat_api, "handle_chat", main)
    payload = ChatSchema(
        user_id=USER_ID,
        inference_model=Model.OPENCODE_GO_GPT_6_LUNA,
        messages=[ConversationTurn(role="user", content="test")],
    )
    request_headers = (
        [] if session is None else [(b"x-conversation-id", session.encode())]
    )
    request = Request(
        {
            "type": "http",
            "headers": request_headers,
            "app": SimpleNamespace(state=SimpleNamespace(settings=Settings())),
        }
    )

    async def run():
        return "".join(
            [
                str(chunk)
                async for chunk in chat_api._chat_response_stream(
                    payload, request, cast(Database, FakeChatDatabase()), UUID(SESSION)
                )
            ]
        )

    text = anyio.run(run)
    if session == SESSION:
        assert "answer" in text
        assert context.call_args.kwargs["credentials"].conversation_id == SESSION
        assert main.call_args.kwargs["conversation_id"] == SESSION
        assert main.call_args.kwargs["inference_credential"] == GO_CREDENTIAL
        assert provider_credentials.conversation_id is None
    else:
        assert (
            "invalid_conversation_id" in text
            if session is not None
            else "missing_conversation_id" in text
        )
        resolver.assert_not_called()
        context.assert_not_called()
        main.assert_not_called()


def test_go_missing_user_key_is_a_safe_credential_denial(monkeypatch):
    monkeypatch.setattr(
        chat_api,
        "resolve_provider_credentials",
        AsyncMock(
            return_value=LlmProviderCredentials(
                openai=ChatCredentialSecret(provider="openai", api_key="other-provider")
            )
        ),
    )
    context = AsyncMock()
    monkeypatch.setattr(chat_api, "get_retrieved_context_with_sources", context)
    payload = ChatSchema(
        user_id=USER_ID,
        inference_model=Model.OPENCODE_GO_GPT_6_LUNA,
        messages=[ConversationTurn(role="user", content="test")],
    )
    request = Request(
        {
            "type": "http",
            "headers": [(b"x-conversation-id", SESSION.encode())],
            "app": SimpleNamespace(state=SimpleNamespace(settings=Settings())),
        }
    )

    async def run():
        return "".join(
            [
                str(chunk)
                async for chunk in chat_api._chat_response_stream(
                    payload, request, cast(Database, FakeChatDatabase()), UUID(SESSION)
                )
            ]
        )

    text = anyio.run(run)
    assert "credential_required" in text
    assert '"provider": "opencode-go"' in text
    assert "other-provider" not in text
    context.assert_not_called()


@pytest.mark.parametrize("outcome", ["error", "cancel"])
def test_go_agent_stream_failure_is_private_and_close_is_preserved(
    outcome, monkeypatch, caplog
):
    closed = []

    class FakeAgent:
        async def astream_events(self, *args, **kwargs):
            try:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content="partial")},
                }
                if outcome == "error":
                    raise RuntimeError(f"{GO_KEY} private prompt")
                raise AssertionError("Closing must not advance the source")
            finally:
                closed.append(True)

    monkeypatch.setattr(
        opencode_go, "get_opencode_go_llm", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(opencode_go, "get_agent_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(opencode_go, "create_agent", lambda *a, **k: FakeAgent())
    observation = StreamObservation(distinct_id="test", response_id="test")
    observation.provider = "opencode-go"

    async def run():
        response = await opencode_go.stream_opencode_go_agent_response(
            "private prompt",
            Model.OPENCODE_GO_GPT_6_LUNA,
            system_prompt="system",
            temperature=0,
            top_p=1,
            max_tokens=128,
            credential=GO_CREDENTIAL,
            conversation_id=SESSION,
            observation=observation,
        )
        iterator = cast(
            AsyncGenerator[str | bytes | memoryview], response.body_iterator
        )
        first = str(await anext(iterator))
        assert "partial" in first
        if outcome == "cancel":
            await iterator.aclose()
            return first
        return first + "".join([str(chunk) async for chunk in iterator])

    text = anyio.run(run)
    assert closed == [True]
    assert GO_KEY not in text
    assert "private prompt" not in text
    assert GO_KEY not in caplog.text
    assert "private prompt" not in caplog.text
    if outcome == "error":
        assert "interrupted" in text
        assert "event: done" in text
        assert "error_type=RuntimeError" in caplog.text
    else:
        assert "event: error" not in text
        assert "Agent streaming failed" not in caplog.text


def test_go_title_without_session_falls_back_locally(monkeypatch):
    monkeypatch.setattr(
        chat_title,
        "resolve_provider_credentials",
        AsyncMock(return_value=LlmProviderCredentials(opencode_go=GO_CREDENTIAL)),
    )
    constructor = AsyncMock()
    monkeypatch.setattr(ChatOpenAI, "model_validate", constructor)

    async def run():
        result = await chat_title.generate_chat_title(
            ChatTitleSchema(
                user_id=USER_ID,
                provider_model=Model.OPENCODE_GO_GPT_6_LUNA,
                messages=[ConversationTurn(role="user", content="local fallback")],
            ),
            FakeChatDatabase(),
            Settings(),
        )
        assert result.title == "local fallback"

    anyio.run(run)
    constructor.assert_not_called()


def test_go_direct_unknown_dispatch_is_rejected_before_ollama(monkeypatch):
    ollama = AsyncMock()
    monkeypatch.setattr(query_transform, "transform_query_with_ollama", ollama)
    monkeypatch.setattr(streams, "stream_ollama_agent_response", ollama)

    async def run():
        with pytest.raises(ValueError, match="Unsupported"):
            await query_transform.transform_query(
                "test", "opencode-go:unknown", temperature=0, top_p=1, max_tokens=128
            )
        with pytest.raises(ValueError, match="Unsupported"):
            await streams.stream_response_with_agent(
                "test",
                "opencode-go:unknown",
                system_prompt="system",
                history=[],
                temperature=0,
                top_p=1,
                max_tokens=128,
                interface="web",
            )

    anyio.run(run)
    ollama.assert_not_called()


@pytest.mark.parametrize(
    "model", [Model.OPENCODE_GO_DEEPSEEK_V4_1_FLASH, Model.OPENCODE_GO_DEEPSEEK_V4_PRO]
)
@pytest.mark.parametrize("auxiliary", ["query", "title"])
def test_go_deepseek_auxiliary_wire_explicitly_disables_default_thinking(
    model, auxiliary, monkeypatch
):
    requests = []

    async def send(client, request, **kwargs):
        requests.append(request)
        library = httpx2 if isinstance(client, httpx2.AsyncClient) else httpx
        data = {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "model": model.value.removeprefix("opencode-go:"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "OK"},
                    "finish_reason": None,
                }
            ],
        }
        return library.Response(
            200,
            request=request,
            content=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    for library in (httpx, httpx2):
        monkeypatch.setattr(library.AsyncClient, "send", send)
    credentials = LlmProviderCredentials(
        opencode_go=GO_CREDENTIAL, conversation_id=SESSION
    )
    monkeypatch.setattr(
        chat_title, "resolve_provider_credentials", AsyncMock(return_value=credentials)
    )

    async def run():
        if auxiliary == "query":
            result = await query_transform.transform_query(
                "Reply with OK.",
                model,
                temperature=0,
                top_p=1,
                max_tokens=128,
                credentials=credentials,
            )
            assert result == "OK"
        else:
            title_result = await chat_title.generate_chat_title(
                ChatTitleSchema(
                    user_id=USER_ID,
                    provider_model=model,
                    messages=[ConversationTurn(role="user", content="Reply with OK.")],
                ),
                FakeChatDatabase(),
                Settings(),
                conversation_id=SESSION,
            )
            assert title_result.title == "OK"

    anyio.run(run)
    assert len(requests) == 1
    wire = json.loads(requests[0].content)
    assert wire["reasoning_effort"] == "none"
    assert wire["max_completion_tokens"] == (128 if auxiliary == "query" else 32)
    assert requests[0].headers["x-opencode-session"] == SESSION
    assert requests[0].headers["user-agent"] == "finki-hub/1.0"
