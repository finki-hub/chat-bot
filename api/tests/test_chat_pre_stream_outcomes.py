import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec
from uuid import UUID, uuid4

import anyio
import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from posthog import Posthog
from starlette.requests import ClientDisconnect

from app.api import chat as chat_api
from app.data.connection import Database
from app.data.db import get_db
from app.data.sponsored_usage import (
    SponsoredQuotaExceededError,
    SponsoredRequestInProgressError,
)
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_result import RetrievalSource, RetrievedContext
from app.main import make_app
from app.utils import posthog_client
from app.utils.async_iterators import closing_stream
from tests.chat_models_access_support import (
    RESET,
    USER_WITHOUT_KEY,
    credentials,
    settings,
)
from tests.test_chat_sponsored_admission import _admission, _payload

SENTINEL = (
    "private question answer sk-secret person@example.test https://host?key=secret"
)


@pytest.fixture
def prepared(monkeypatch):
    sdk = create_autospec(Posthog, instance=True)
    monkeypatch.setattr(posthog_client._state, "client", sdk)
    monkeypatch.setattr(posthog_client._state, "revisions", {})

    async def body():
        yield 'event: token\ndata: {"text":"answer"}\n\n'
        yield "event: done\ndata: {}\n\n"

    mocks = {
        "credentials": AsyncMock(return_value=credentials(openai=True)),
        "admission": AsyncMock(
            side_effect=lambda db, **kw: _admission(kw["request_id"])
        ),
        "context": AsyncMock(
            return_value=RetrievedContext(
                text="context",
                effective_transform_mode=QueryTransformMode.REWRITE_HYDE,
            )
        ),
        "links": AsyncMock(return_value=""),
        "agent_setup": AsyncMock(return_value=StreamingResponse(body())),
        "release": AsyncMock(),
    }
    for phase, name in {
        "credentials": "resolve_provider_credentials",
        "admission": "admit_sponsored_request",
        "context": "get_retrieved_context_with_sources",
        "links": "get_links_context",
        "agent_setup": "handle_chat",
        "release": "release_sponsored_request",
    }.items():
        monkeypatch.setattr(chat_api, name, mocks[phase])
    payload = _payload()
    payload.messages[0].content = SENTINEL
    return SimpleNamespace(
        sdk=sdk,
        mocks=mocks,
        response_id=uuid4(),
        payload=payload,
        request=SimpleNamespace(
            headers={"X-Distinct-Id": SENTINEL, "X-PostHog-Session-Id": SENTINEL},
            app=SimpleNamespace(state=SimpleNamespace(settings=settings())),
        ),
    )


def _stream(prepared):
    return chat_api._chat_response_stream(
        prepared.payload,
        prepared.request,
        Database.__new__(Database),
        prepared.response_id,
    )


async def _consume(prepared):
    return [str(chunk) async for chunk in _stream(prepared)]


def _events(prepared, event):
    return [
        call.kwargs
        for call in prepared.sdk.capture.call_args_list
        if call.kwargs["event"] == event
    ]


def _assert_outcome(prepared, phase, outcome, reason):
    assert _events(prepared, "chat_pre_stream_outcome") == [
        {
            "distinct_id": str(prepared.response_id),
            "event": "chat_pre_stream_outcome",
            "properties": {
                "response_id": str(prepared.response_id),
                "phase": phase,
                "outcome": outcome,
                "reason": reason,
                "$process_person_profile": False,
                "service": "chat-bot-api",
            },
        }
    ]
    assert not _events(prepared, "$ai_generation")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "phase", ["credentials", "admission", "context", "links", "agent_setup"]
)
async def test_preparation_exception_is_joined_once_without_private_content(
    prepared, phase
):
    failure = RuntimeError(SENTINEL)
    prepared.mocks[phase].side_effect = failure
    if phase == "admission":
        prepared.mocks["credentials"].return_value = credentials(openai=False)
    if phase in {"credentials", "admission"}:
        with pytest.raises(RuntimeError) as caught:
            await _consume(prepared)
        assert caught.value is failure
    else:
        chunks = await _consume(prepared)
        assert sum('"code": "agent_error"' in chunk for chunk in chunks) == 1
        assert SENTINEL not in "".join(chunks)
    _assert_outcome(
        prepared,
        "context" if phase == "links" else phase,
        "error",
        "preparation_failed",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "phase", ["credentials", "admission", "context", "agent_setup"]
)
async def test_cancellation_is_reraised_with_phase_and_releases_only_admitted_lease(
    prepared, phase
):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    started = asyncio.Event()

    async def block(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    prepared.mocks[phase].side_effect = block
    async with asyncio.timeout(2):
        task = asyncio.create_task(_consume(prepared))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    _assert_outcome(prepared, phase, "client_disconnect", "cancelled")
    if phase in {"context", "agent_setup"}:
        prepared.mocks["release"].assert_awaited_once()
        assert (
            prepared.mocks["release"].call_args.kwargs["request_id"]
            == prepared.response_id
        )
    else:
        prepared.mocks["release"].assert_not_awaited()


@pytest.mark.anyio
async def test_close_while_sources_yielded_is_pre_stream_and_cleans_lease(prepared):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    prepared.mocks["context"].return_value = RetrievedContext(
        text="context",
        effective_transform_mode=QueryTransformMode.REWRITE_HYDE,
        sources=(RetrievalSource(id="faq", kind="faq", title="Source"),),
    )
    stream = _stream(prepared)
    assert "event: sources" in str(await anext(stream))
    await stream.aclose()
    _assert_outcome(prepared, "agent_setup", "client_disconnect", "cancelled")
    prepared.mocks["release"].assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["completed", "provider_error", "client_disconnect"])
async def test_generation_owns_terminal_outcome_after_handoff(prepared, kind):
    started = asyncio.Event()

    async def body():
        started.set()
        if kind == "client_disconnect":
            await asyncio.Event().wait()
        if kind == "provider_error":
            yield 'event: error\ndata: {"code":"agent_error","message":"safe"}\n\n'
        else:
            yield 'event: token\ndata: {"text":"answer"}\n\n'

    prepared.mocks["agent_setup"].return_value = StreamingResponse(body())
    if kind == "client_disconnect":
        async with asyncio.timeout(2):
            task = asyncio.create_task(_consume(prepared))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    else:
        await _consume(prepared)
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert [
        event["properties"]["outcome"] for event in _events(prepared, "$ai_generation")
    ] == [kind]


@pytest.mark.anyio
@pytest.mark.parametrize("code", ["credential_required", "free_tier_unavailable"])
async def test_credential_denials_never_start_retrieval_or_admission(prepared, code):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    prepared.request.app.state.settings = settings(enabled=False)
    if code == "credential_required":
        prepared.payload = prepared.payload.model_copy(update={"user_id": None})
    chunks = await _consume(prepared)
    assert code in "".join(chunks)
    _assert_outcome(prepared, "credentials", "denied", code)
    for phase in ("admission", "context", "agent_setup", "release"):
        prepared.mocks[phase].assert_not_awaited()


@pytest.mark.anyio
async def test_context_failure_cancels_sibling_and_releases_once(prepared):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    started = asyncio.Event()
    finished = asyncio.Event()

    async def retrieve(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    async def links(*args, **kwargs):
        await started.wait()
        raise RuntimeError(SENTINEL)

    prepared.mocks["context"].side_effect = retrieve
    prepared.mocks["links"].side_effect = links
    async with asyncio.timeout(2):
        await _consume(prepared)
    assert finished.is_set()
    _assert_outcome(prepared, "context", "error", "preparation_failed")
    prepared.mocks["release"].assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        SponsoredQuotaExceededError("user", RESET),
        SponsoredRequestInProgressError(USER_WITHOUT_KEY),
    ],
)
async def test_existing_sponsored_denial_is_not_duplicated(prepared, error):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    prepared.mocks["admission"].side_effect = error
    await _consume(prepared)
    assert len(_events(prepared, "sponsored_denied")) == 1
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert not _events(prepared, "$ai_generation")
    prepared.mocks["release"].assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("code", ["credential_required", "free_tier_unavailable"])
async def test_admission_denial_before_quota_does_not_consume_lease(prepared, code):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    if code == "credential_required":
        prepared.payload = prepared.payload.model_copy(update={"user_id": None})
    else:
        prepared.request.app.state.settings = settings().model_copy(
            update={"SPONSORED_DAILY_GLOBAL_LIMIT": None},
        )
    stream = _stream(prepared)
    assert code in str(await anext(stream))
    # Closing at the denial yield must not also record a disconnect.
    await stream.aclose()
    _assert_outcome(prepared, "admission", "denied", code)
    for phase in ("admission", "context", "agent_setup", "release"):
        prepared.mocks[phase].assert_not_awaited()


@pytest.mark.anyio
async def test_unstarted_stream_has_no_observed_outcome(prepared):
    stream = _stream(prepared)
    await stream.aclose()
    assert not _events(prepared, "chat_pre_stream_outcome")
    prepared.mocks["credentials"].assert_not_awaited()


def _assert_generation_disconnect(prepared):
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert [
        event["properties"]["outcome"] for event in _events(prepared, "$ai_generation")
    ] == ["client_disconnect"]
    prepared.mocks["release"].assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_close_after_first_token_closes_provider_immediately(
    prepared, cleanup_fails
):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    closed = []

    async def body():
        try:
            yield 'event: token\ndata: {"text":"answer"}\n\n'
            pytest.fail("Closing must not advance the provider")
        finally:
            await anyio.lowlevel.checkpoint()
            closed.append(True)
            if cleanup_fails:
                raise RuntimeError(SENTINEL)

    prepared.mocks["agent_setup"].return_value = StreamingResponse(body())
    stream = _stream(prepared)
    assert "event: token" in str(await anext(stream))
    await stream.aclose()
    assert closed == [True]
    _assert_generation_disconnect(prepared)


@pytest.mark.anyio
@pytest.mark.parametrize("send_error", [OSError, RuntimeError])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_asgi_send_failure_closes_all_streams_preserving_exception(
    prepared, send_error, cleanup_fails
):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    closed = []
    failure = send_error("downstream send failed")

    async def body():
        try:
            yield 'event: token\ndata: {"text":"answer"}\n\n'
            pytest.fail("Closing must not advance the provider")
        finally:
            await anyio.lowlevel.checkpoint()
            closed.append(True)
            if cleanup_fails:
                raise RuntimeError(SENTINEL)

    async def send(message):
        if message["type"] == "http.response.body":
            assert b"event: token" in message["body"]
            raise failure

    prepared.mocks["agent_setup"].return_value = StreamingResponse(body())
    response = await chat_api.chat(
        prepared.payload,
        prepared.request,
        prepared.response_id,
        Database.__new__(Database),
    )
    expected = ClientDisconnect if send_error is OSError else RuntimeError
    receive = AsyncMock()
    with pytest.raises(expected) as caught:
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    receive.assert_not_awaited()
    if send_error is RuntimeError:
        assert caught.value is failure
    else:
        assert caught.value.__context__ is failure
    assert closed == [True]
    _assert_generation_disconnect(prepared)


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["context", "generation"])
async def test_asgi_disconnect_shields_provider_sibling_and_lease_cleanup(
    prepared, phase
):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    sent = asyncio.Event()
    sibling_started = asyncio.Event()
    closed = []

    async def retrieve(*args, **kwargs):
        await sibling_started.wait()
        kwargs["on_stage"]("retrieving")
        try:
            await asyncio.Event().wait()
        finally:
            await anyio.lowlevel.checkpoint()
            closed.append("retrieval")

    async def links(*args, **kwargs):
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await anyio.lowlevel.checkpoint()
            closed.append("links")

    async def body():
        try:
            yield 'event: token\ndata: {"text":"answer"}\n\n'
            await asyncio.Event().wait()
        finally:
            await anyio.lowlevel.checkpoint()
            closed.append("provider")

    async def release(*args, **kwargs):
        await anyio.lowlevel.checkpoint()
        closed.append("lease")

    async def send(message):
        if message["type"] == "http.response.body":
            sent.set()
            await asyncio.Event().wait()

    async def receive():
        await sent.wait()
        return {"type": "http.disconnect"}

    if phase == "context":
        prepared.mocks["context"].side_effect = retrieve
        prepared.mocks["links"].side_effect = links
    prepared.mocks["agent_setup"].return_value = StreamingResponse(body())
    prepared.mocks["release"].side_effect = release
    response = await chat_api.chat(
        prepared.payload,
        prepared.request,
        prepared.response_id,
        Database.__new__(Database),
    )
    async with asyncio.timeout(2):
        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    if phase == "context":
        assert sorted(closed) == ["lease", "links", "retrieval"]
        _assert_outcome(prepared, "context", "client_disconnect", "cancelled")
        prepared.mocks["release"].assert_awaited_once()
    else:
        assert closed == ["provider", "lease"]
        _assert_generation_disconnect(prepared)


@pytest.mark.anyio
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_asgi_completion_keeps_frames_headers_and_one_completed_outcome(
    prepared, spec_version
):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        await asyncio.Event().wait()

    response = await chat_api.chat(
        prepared.payload,
        prepared.request,
        prepared.response_id,
        Database.__new__(Database),
    )
    async with asyncio.timeout(2):
        await response(
            {"type": "http", "asgi": {"spec_version": spec_version}}, receive, send
        )
    assert messages[0]["status"] == 200
    assert (
        dict(messages[0]["headers"])[b"x-response-id"]
        == str(prepared.response_id).encode()
    )
    assert (
        dict(messages[0]["headers"])[b"content-type"]
        == b"text/event-stream; charset=utf-8"
    )
    assert [message["body"].split(b"\n", 1)[0] for message in messages[1:]] == [
        b"event: token",
        b"event: done",
        b"event: meta",
        b"",
    ]
    assert messages[-1]["more_body"] is False
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert [
        event["properties"]["outcome"] for event in _events(prepared, "$ai_generation")
    ] == ["completed"]
    prepared.mocks["release"].assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("close_kind", ["absent", "success", "failure"])
async def test_provider_closes_actual_iterator_and_reports_normal_cleanup_failure(
    prepared, close_kind
):
    closed = []
    failure = RuntimeError(SENTINEL)

    class Iterator:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return 'event: token\ndata: {"text":"answer"}\n\n'

    class ClosableIterator(Iterator):
        async def aclose(self):
            closed.append(True)
            if close_kind == "failure":
                raise failure

    class Body:
        def __aiter__(self):
            return Iterator() if close_kind == "absent" else ClosableIterator()

        async def aclose(self):
            pytest.fail("Must close the consumed iterator, not its iterable")

    prepared.mocks["agent_setup"].return_value = StreamingResponse(Body())
    if close_kind == "failure":
        with pytest.raises(RuntimeError) as caught:
            await _consume(prepared)
        assert caught.value is failure
    else:
        await _consume(prepared)
    assert closed == ([] if close_kind == "absent" else [True])
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert [
        event["properties"]["outcome"] for event in _events(prepared, "$ai_generation")
    ] == ["provider_error" if close_kind == "failure" else "completed"]


@pytest.mark.anyio
@pytest.mark.parametrize("primary", ["none", "cancelled", "generator_exit", "error"])
async def test_closing_stream_cleanup_failure_preserves_primary_exception(primary):
    cleanup_failure = RuntimeError(SENTINEL)

    class Iterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise cleanup_failure

    class Body:
        def __aiter__(self):
            return Iterator()

    async def consume():
        async with closing_stream(Body()):
            if primary == "cancelled":
                raise asyncio.CancelledError
            if primary == "generator_exit":
                raise GeneratorExit
            if primary == "error":
                raise ValueError("primary")

    expected = {
        "none": RuntimeError,
        "cancelled": asyncio.CancelledError,
        "generator_exit": GeneratorExit,
        "error": ValueError,
    }[primary]
    with pytest.raises(expected) as caught:
        await consume()
    if primary == "none":
        assert caught.value is cleanup_failure


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({}, 401),
        ({"x-api-key": "wrong"}, 401),
        ({"x-api-key": "test-api-key", "X-Response-Id": "private-not-a-uuid"}, 422),
    ],
)
def test_http_auth_and_uuid_rejections_are_not_preparation_errors(
    prepared, headers, expected_status
):
    app = make_app(settings())
    app.dependency_overrides[get_db] = lambda: Database.__new__(Database)
    response = TestClient(app).post(
        "/chat/",
        headers=headers,
        json=prepared.payload.model_dump(mode="json"),
    )
    assert response.status_code == expected_status
    prepared.mocks["credentials"].assert_not_awaited()
    assert not _events(prepared, "chat_pre_stream_outcome")
    assert not _events(prepared, "$ai_generation")


def test_http_sse_denial_keeps_transport_ok_and_joins_server_uuid(prepared):
    prepared.mocks["credentials"].return_value = credentials(openai=False)
    app = make_app(settings(enabled=False))
    app.dependency_overrides[get_db] = lambda: Database.__new__(Database)
    response = TestClient(app).post(
        "/chat/",
        headers={"x-api-key": "test-api-key", "X-Distinct-Id": SENTINEL},
        json=prepared.payload.model_dump(mode="json"),
    )
    assert response.status_code == 200
    prepared.response_id = UUID(response.headers["X-Response-Id"])
    _assert_outcome(prepared, "credentials", "denied", "free_tier_unavailable")
    assert _events(prepared, "request_completed")[0]["properties"]["outcome"] == "ok"
