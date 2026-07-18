import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import anyio
import pytest
from fastapi.responses import StreamingResponse

from app.api import chat as chat_api
from app.data.connection import Database
from app.data.sponsored_usage import (
    SponsoredQuotaExceededError,
    SponsoredRequestInProgressError,
    SponsoredUsageAdmission,
)
from app.llms.models import Model
from app.llms.retrieval_result import RetrievedContext
from app.utils import posthog_client
from tests.chat_models_access_support import credentials, settings, usage_snapshot

USER_ID = UUID("00000000-0000-4000-8000-000000000002")
RESET = datetime(2026, 7, 19, tzinfo=UTC)


def _request(current_settings):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=current_settings)),
        headers={"X-Distinct-Id": "test-distinct-id"},
    )


def _payload(user_id: UUID | None = USER_ID):
    return chat_api.ChatSchema.model_validate(
        {
            "user_id": user_id,
            "interface": "web",
            "inference_model": Model.GPT_5_6_LUNA,
            "messages": [{"role": "user", "content": "Question"}],
        },
    )


def _admission(request_id: UUID) -> SponsoredUsageAdmission:
    return SponsoredUsageAdmission(
        request_id=request_id,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        snapshot=usage_snapshot(USER_ID, personal_remaining=4, global_remaining=9),
    )


async def _body() -> AsyncIterator[str]:
    yield 'event: token\ndata: {"text":"answer"}\n\n'
    yield "event: done\ndata: {}\n\n"


def _json_events(chunks: list[str]) -> list[dict[str, object]]:
    return [
        json.loads(chunk.split("data: ", 1)[1]) for chunk in chunks if "data: " in chunk
    ]


async def _run_stream(
    monkeypatch,
    *,
    current_settings,
    user_credentials,
    admit,
    retrieve=None,
    body=_body,
    release_func=None,
    handle_func=None,
):
    released: list[tuple[UUID, UUID]] = []
    captured: list[tuple[str, dict[str, object]]] = []

    async def release(db, *, user_id, request_id):
        released.append((user_id, request_id))

    async def resolve_credentials(*args, **kwargs):
        return user_credentials

    async def get_links(*args, **kwargs):
        return ""

    async def handle(*args, **kwargs):
        return StreamingResponse(body(), media_type="text/event-stream")

    async def get_context(*args, **kwargs):
        if retrieve is not None:
            return await retrieve(*args, **kwargs)
        return RetrievedContext(text="context")

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve_credentials)
    monkeypatch.setattr(chat_api, "admit_sponsored_request", admit)
    monkeypatch.setattr(
        chat_api,
        "release_sponsored_request",
        release if release_func is None else release_func,
    )
    monkeypatch.setattr(chat_api, "get_retrieved_context_with_sources", get_context)
    monkeypatch.setattr(chat_api, "get_links_context", get_links)
    monkeypatch.setattr(
        chat_api,
        "handle_chat",
        handle if handle_func is None else handle_func,
    )
    monkeypatch.setattr(
        chat_api,
        "capture_sponsored_event",
        lambda distinct_id, event, **properties: captured.append(
            (event, {"distinct_id": distinct_id, **properties}),
        ),
    )

    response_id = uuid4()
    database = Database.__new__(Database)
    chunks = [
        str(chunk)
        async for chunk in chat_api._chat_response_stream(  # noqa: SLF001
            _payload(),
            _request(current_settings),
            database,
            response_id,
        )
    ]
    return chunks, released, captured, response_id


@pytest.mark.anyio
async def test_sponsored_request_admits_and_releases_exact_lease(monkeypatch):
    calls: list[dict[str, object]] = []

    async def admit(db, **kwargs):
        calls.append(kwargs)
        return _admission(kwargs["request_id"])

    chunks, released, captured, response_id = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=False),
        admit=admit,
    )

    assert any('"text":"answer"' in chunk for chunk in chunks)
    assert calls[0]["user_id"] == USER_ID
    assert calls[0]["request_id"] == response_id
    assert released == [(USER_ID, response_id)]
    assert [event for event, _ in captured] == [
        "sponsored_admitted",
        "sponsored_stream",
    ]
    assert {properties["distinct_id"] for _, properties in captured} == {
        "test-distinct-id",
    }
    assert captured[-1][1]["input_tokens"] == 0


@pytest.mark.anyio
async def test_sponsored_quota_denial_is_safe_and_not_released(monkeypatch):
    async def admit(db, **kwargs):
        raise SponsoredQuotaExceededError("user", RESET)

    chunks, released, captured, _ = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=False),
        admit=admit,
    )

    payload = _json_events(chunks)[0]
    assert payload == {
        "code": "free_quota_exhausted",
        "message": "Бесплатната квота е искористена. Обидете се повторно по ресетирањето.",
        "resets_at": "2026-07-19T00:00:00Z",
    }
    assert released == []
    assert captured[0][0] == "sponsored_denied"
    assert captured[0][1]["denial_reason"] == "free_quota_exhausted"


@pytest.mark.anyio
async def test_sponsored_concurrent_denial_has_distinct_code(monkeypatch):
    async def admit(db, **kwargs):
        raise SponsoredRequestInProgressError(USER_ID)

    chunks, released, _, _ = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=False),
        admit=admit,
    )

    assert _json_events(chunks)[0]["code"] == "sponsored_request_in_progress"
    assert released == []


@pytest.mark.anyio
async def test_disabled_sponsored_tier_is_unavailable_before_provider(monkeypatch):
    provider_called = False

    async def admit(db, **kwargs):
        raise AssertionError("disabled sponsorship must not admit")

    async def resolve_credentials(*args, **kwargs):
        return credentials(openai=False)

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve_credentials)
    monkeypatch.setattr(chat_api, "admit_sponsored_request", admit)

    chunks = [
        str(chunk)
        async for chunk in chat_api._chat_response_stream(  # noqa: SLF001
            _payload(),
            _request(settings(enabled=False)),
            Database.__new__(Database),
            uuid4(),
        )
    ]

    assert _json_events(chunks)[0]["code"] == "free_tier_unavailable"
    assert provider_called is False


@pytest.mark.anyio
async def test_byok_luna_bypasses_sponsored_admission(monkeypatch):
    async def admit(db, **kwargs):
        raise AssertionError("BYOK Luna must not consume sponsored quota")

    chunks, released, captured, _ = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=True),
        admit=admit,
    )

    assert any('"text":"answer"' in chunk for chunk in chunks)
    assert released == []
    assert captured[-1][1]["mode"] == "byok"


@pytest.mark.anyio
async def test_pre_stream_failure_releases_admitted_lease(monkeypatch):
    async def admit(db, **kwargs):
        return _admission(kwargs["request_id"])

    async def retrieve(*args, **kwargs):
        raise RuntimeError("retrieval failed")

    _, released, _, response_id = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=False),
        admit=admit,
        retrieve=retrieve,
    )

    assert released == [(USER_ID, response_id)]


@pytest.mark.anyio
async def test_cancellation_releases_admitted_lease(monkeypatch):
    started = anyio.Event()

    async def admit(db, **kwargs):
        return _admission(kwargs["request_id"])

    async def body():
        started.set()
        await anyio.sleep_forever()
        yield "never"

    released: list[tuple[UUID, UUID]] = []

    async def release(db, *, user_id, request_id):
        released.append((user_id, request_id))

    monkeypatch.setattr(chat_api, "release_sponsored_request", release)

    async def consume():
        chunks, _, _, response_id = await _run_stream(
            monkeypatch,
            current_settings=settings(),
            user_credentials=credentials(openai=False),
            admit=admit,
            body=body,
            release_func=release,
        )
        return chunks, response_id

    with anyio.move_on_after(0.1) as scope:
        await consume()
    assert scope.cancelled_caught
    assert released


@pytest.mark.anyio
async def test_cancellation_observes_deferred_release_completion(monkeypatch):
    release_finished = anyio.Event()
    callback_called = anyio.Event()

    async def admit(db, **kwargs):
        return _admission(kwargs["request_id"])

    async def body():
        await anyio.sleep_forever()
        yield "never"

    async def release(db, *, user_id, request_id):
        await anyio.lowlevel.checkpoint()
        release_finished.set()

    def observe_release(task):
        task.result()
        callback_called.set()

    monkeypatch.setattr(
        chat_api,
        "_log_sponsored_release_failure",
        observe_release,
    )

    async def consume():
        await _run_stream(
            monkeypatch,
            current_settings=settings(),
            user_credentials=credentials(openai=False),
            admit=admit,
            body=body,
            release_func=release,
        )

    with anyio.move_on_after(0.1) as scope:
        await consume()
    assert scope.cancelled_caught
    await release_finished.wait()
    await callback_called.wait()


@pytest.mark.anyio
async def test_deferred_release_failure_is_logged(caplog):
    async def fail_release():
        raise RuntimeError("release failed")

    task = asyncio.create_task(fail_release())
    with pytest.raises(RuntimeError):
        _ = await task

    with caplog.at_level(logging.ERROR, logger=chat_api.logger.name):
        chat_api._log_sponsored_release_failure(task)  # noqa: SLF001

    assert "Failed to release sponsored request lease" in caplog.text


@pytest.mark.anyio
async def test_sixth_sponsored_request_is_denied_before_provider(monkeypatch):
    admitted = 0
    provider_calls = 0

    async def admit(db, **kwargs):
        nonlocal admitted
        admitted += 1
        if admitted == 6:
            raise SponsoredQuotaExceededError("user", RESET)
        return _admission(kwargs["request_id"])

    async def handle(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return StreamingResponse(_body(), media_type="text/event-stream")

    for _ in range(6):
        await _run_stream(
            monkeypatch,
            current_settings=settings(),
            user_credentials=credentials(openai=False),
            admit=admit,
            handle_func=handle,
        )
    assert admitted == 6
    assert provider_calls == 5


@pytest.mark.anyio
async def test_provider_error_event_releases_admitted_lease(monkeypatch):
    async def admit(db, **kwargs):
        return _admission(kwargs["request_id"])

    async def provider_failure_body():
        yield 'event: error\ndata: {"code":"agent_error","message":"safe"}\n\n'
        yield "event: done\ndata: {}\n\n"

    chunks, released, captured, response_id = await _run_stream(
        monkeypatch,
        current_settings=settings(),
        user_credentials=credentials(openai=False),
        admit=admit,
        body=provider_failure_body,
    )

    assert any('"code":"agent_error"' in chunk for chunk in chunks)
    assert released == [(USER_ID, response_id)]
    assert captured[-1][1]["provider_failure"] is True


@pytest.mark.anyio
async def test_provider_exception_releases_admitted_lease(monkeypatch):
    released: list[tuple[UUID, UUID]] = []

    async def admit(db, **kwargs):
        return _admission(kwargs["request_id"])

    async def release(db, *, user_id, request_id):
        released.append((user_id, request_id))

    async def provider_exception_body():
        yield 'event: token\ndata: {"text":"partial"}\n\n'
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await _run_stream(
            monkeypatch,
            current_settings=settings(),
            user_credentials=credentials(openai=False),
            admit=admit,
            body=provider_exception_body,
            release_func=release,
        )
    assert len(released) == 1
    assert released[0][0] == USER_ID


def test_sponsored_posthog_metrics_are_aggregate_only(monkeypatch):
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        posthog_client,
        "capture",
        lambda _distinct_id, _event, properties: captured.append(properties or {}),
    )

    posthog_client.capture_sponsored_event(
        "safe-id",
        "sponsored_stream",
        response_id="response-id",
        mode="sponsored",
        client_interface="web",
        outcome="completed",
        provider_failure=False,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
    )

    assert captured == [
        {
            "response_id": "response-id",
            "mode": "sponsored",
            "client_interface": "web",
            "outcome": "completed",
            "provider_failure": False,
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    ]
    assert "Question" not in repr(captured)
    assert "secret" not in repr(captured)
    assert "https://" not in repr(captured)
