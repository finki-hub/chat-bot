import asyncio
import json
import logging
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessageChunk

from app.api import chat as chat_api
from app.data.connection import Database
from app.data.db import get_db
from app.llms import agents
from app.llms import chat as llm_chat
from app.llms import openai as openai_module
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_result import RetrievedContext
from app.utils import posthog_client
from app.utils.auth import verify_api_key
from app.utils.timing import RequestTimings
from tests.chat_models_access_support import credentials, settings
from tests.test_agent_error_log_privacy import TestAgentGraph
from tests.test_chat_sponsored_admission import USER_ID, _admission, _payload

_PRIVATE_ERROR = "private provider body key=test-secret https://private.example/v1"


class DirectStatusError(Exception):
    status_code = 401


class NestedStatusError(Exception):
    response = SimpleNamespace(status_code=504)


@pytest.fixture
def captured(monkeypatch):
    events = []

    def capture(_distinct_id, event, properties=None):
        events.append((event, properties or {}))

    monkeypatch.setattr(posthog_client._state, "client", None)
    for module in (chat_api, agents, posthog_client):
        monkeypatch.setattr(module, "capture", capture)
    return events


def _properties(events, name):
    (properties,) = [properties for event, properties in events if event == name]
    return properties


def _assert_outcome(events, outcome, *, provider_failure):
    assert _properties(events, "$ai_generation")["outcome"] == outcome
    sponsored = _properties(events, "sponsored_stream")
    assert sponsored["outcome"] == outcome
    assert sponsored["provider_failure"] is provider_failure


def _configure_route(monkeypatch, *, case, mode):
    admitted, released, provider_calls = [], [], []
    started = asyncio.Event()

    class GraphEdge:
        async def astream_events(self, agent_input, config, *, version):
            assert version == "v2"
            provider_calls.append(agent_input)
            if case in ("after_504", "success", "cancel"):
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content="partial answer")},
                }
            if case == "cancel":
                started.set()
                await asyncio.Future()
            elif case == "before_401":
                raise DirectStatusError(_PRIVATE_ERROR)
            elif case in ("before_504", "after_504"):
                raise NestedStatusError(_PRIVATE_ERROR)

    async def tools():
        return []

    async def resolve(*args, **kwargs):
        return credentials(openai=mode == "byok")

    async def context(*args, **kwargs):
        return RetrievedContext(
            text="context",
            effective_transform_mode=QueryTransformMode.REWRITE_HYDE,
        )

    async def links(*args, **kwargs):
        return ""

    async def admit(db, **kwargs):
        admitted.append((kwargs["user_id"], kwargs["request_id"]))
        return _admission(kwargs["request_id"])

    async def release(db, *, user_id, request_id):
        released.append((user_id, request_id))

    # Keep handle_chat, dispatch, provider wrapper and the agent catcher real.
    # Only provider graph events/construction and external services are replaced.
    monkeypatch.setattr(openai_module, "get_openai_llm", lambda *a, **kw: object())
    monkeypatch.setattr(openai_module, "get_agent_tools", tools)
    monkeypatch.setattr(openai_module, "create_agent", lambda *a: GraphEdge())
    monkeypatch.setattr(llm_chat, "build_recommendation_tools", lambda db: [])
    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve)
    monkeypatch.setattr(chat_api, "get_retrieved_context_with_sources", context)
    monkeypatch.setattr(chat_api, "get_links_context", links)
    monkeypatch.setattr(chat_api, "admit_sponsored_request", admit)
    monkeypatch.setattr(chat_api, "release_sponsored_request", release)
    app = FastAPI()
    app.state.settings = settings()
    app.include_router(chat_api.router)
    app.dependency_overrides[get_db] = object
    app.dependency_overrides[verify_api_key] = lambda: None
    return app, admitted, released, provider_calls, started


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["byok", "sponsored"])
@pytest.mark.parametrize(
    "case", ["before_401", "before_504", "after_504", "success", "empty"]
)
async def test_real_agent_failure_outcomes(monkeypatch, captured, caplog, case, mode):
    caplog.set_level(logging.INFO)
    app, admitted, released, provider_calls, _ = _configure_route(
        monkeypatch,
        case=case,
        mode=mode,
    )
    response_id = uuid4()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat/",
            json=_payload().model_dump(mode="json"),
            headers={"X-Response-Id": str(response_id)},
        )
    assert response.status_code == 200
    assert response.headers["x-response-id"] == str(response_id)
    assert len(provider_calls) == 1
    expected_leases = [(USER_ID, response_id)] if mode == "sponsored" else []
    assert admitted == released == expected_leases
    frames = [frame.splitlines() for frame in response.text.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in frames]
    errors = [
        json.loads(lines[1].removeprefix("data: "))
        for lines in frames
        if lines[0] == "event: error"
    ]
    failure = case in ("before_401", "before_504", "after_504")
    expected_outcome = (
        "provider_error"
        if failure
        else "empty_answer"
        if case == "empty"
        else "completed"
    )
    _assert_outcome(captured, expected_outcome, provider_failure=failure)
    assert _properties(captured, "sponsored_stream")["mode"] == mode
    if failure:
        assert names == (["token"] if case == "after_504" else ["reset"]) + [
            "error",
            "done",
            "meta",
        ]
        (error,) = errors
        assert error["code"] == (
            "interrupted" if case == "after_504" else "agent_error"
        )
        assert set(error) == {"code", "message"}
        model_error = _properties(captured, "model_error")
        assert model_error["status_code"] == (401 if case == "before_401" else 504)
        assert model_error["provider"] == "openai"
        assert [event for event, _ in captured][-3:] == [
            "model_error",
            "$ai_generation",
            "sponsored_stream",
        ]
    else:
        assert all(event != "model_error" for event, _ in captured)
        if case == "empty":
            assert names == ["reset", "error", "done", "meta"]
            assert errors[0]["code"] == "no_answer"
        else:
            assert names == ["token", "done", "meta"]
    generation = _properties(captured, "$ai_generation")
    assert generation["answer_char_len"] == (
        14 if case in ("after_504", "success") else 0
    )
    for private in (
        _PRIVATE_ERROR,
        "test-secret",
        "private.example",
        "test-user-openai-key",
        "sponsored-secret",
    ):
        assert private not in response.text + repr(captured) + caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["byok", "sponsored"])
async def test_real_agent_cancellation_releases_lease(monkeypatch, captured, mode):
    app, admitted, released, _, started = _configure_route(
        monkeypatch,
        case="cancel",
        mode=mode,
    )
    response_id = uuid4()
    received = 0

    async def consume():
        nonlocal received
        async for chunk in chat_api._chat_response_stream(
            _payload(),
            Request({"type": "http", "app": app, "headers": []}),
            Database.__new__(Database),
            response_id,
        ):
            assert str(chunk).startswith("event: token")
            received += 1

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert received == 1
    assert (
        admitted
        == released
        == ([(USER_ID, response_id)] if mode == "sponsored" else [])
    )
    _assert_outcome(captured, "client_disconnect", provider_failure=False)
    assert all(event != "model_error" for event, _ in captured)


def _instrument(body):
    return chat_api._instrument_stream(
        body,
        payload=_payload(),
        response_id=uuid4(),
        timings=RequestTimings(),
        retrieval_hit=False,
        distinct_id="test",
        session_id=None,
        observation=agents.StreamObservation(distinct_id="test", response_id="test"),
        effective_transform_mode=QueryTransformMode.REWRITE_HYDE,
        sponsored_mode="sponsored",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "code",
    [
        "no_answer",
        "credential_required",
        "free_tier_unavailable",
        "free_quota_exhausted",
        "sponsored_request_in_progress",
        "unknown",
    ],
)
async def test_non_provider_error_frames_are_not_provider_failures(captured, code):
    async def body():
        yield agents.error_event(code, "safe")
        yield agents.DONE_EVENT

    _ = [chunk async for chunk in _instrument(body())]
    _assert_outcome(captured, "empty_answer", provider_failure=False)


@pytest.mark.anyio
@pytest.mark.parametrize("close", [False, True])
async def test_disconnect_after_caught_provider_error_takes_precedence(captured, close):
    class GraphEdge:
        async def astream_events(self, *args, **kwargs):
            yield {"event": "on_chain_start"}
            raise NestedStatusError(_PRIVATE_ERROR)

    body = agents.create_agent_token_generator(cast("TestAgentGraph", GraphEdge()), [])
    stream = _instrument(body)
    try:
        assert (await anext(stream)).startswith("event: reset")
        assert (await anext(stream)).startswith("event: error")
        if close:
            await stream.aclose()
        else:
            with pytest.raises(asyncio.CancelledError):
                await stream.athrow(asyncio.CancelledError())
    finally:
        await body.aclose()
    _assert_outcome(captured, "client_disconnect", provider_failure=True)


@pytest.mark.anyio
@pytest.mark.parametrize("data", ["not-json", "[]", '{"code":[]}'])
async def test_malformed_error_frame_does_not_break_instrumentation(captured, data):
    async def body():
        yield f"event: error\ndata: {data}\n\n"
        yield agents.DONE_EVENT

    _ = [chunk async for chunk in _instrument(body())]
    _assert_outcome(captured, "empty_answer", provider_failure=False)


@pytest.mark.anyio
@pytest.mark.parametrize("split", [False, True])
async def test_provider_error_frames_are_classified_across_chunk_boundaries(
    captured, split
):
    wire = (agents.error_event("agent_error", "safe") + agents.DONE_EVENT).encode()
    chunks = [wire[:25], memoryview(wire[25:])] if split else [wire]

    async def body():
        for chunk in chunks:
            yield chunk

    output = [chunk async for chunk in _instrument(body())]
    assert output[:-1] == chunks
    _assert_outcome(captured, "provider_error", provider_failure=True)
