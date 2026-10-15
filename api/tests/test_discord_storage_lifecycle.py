import asyncio
from dataclasses import replace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app import main
from app.api import chat as chat_api
from app.api import discord_storage as storage_api
from app.api.discord_storage import DiscordStorage, DiscordWriteRegistry
from app.data.connection import Database
from app.data.discord_conversations import DiscordExchange
from app.llms.retrieval_result import RetrievalSource, RetrievedContext
from tests.chat_models_access_support import settings
from tests.test_chat_pre_stream_outcomes import _consume, _stream
from tests.test_chat_pre_stream_outcomes import prepared as prepared_fixture
from tests.test_chat_sponsored_admission import credentials

prepared = prepared_fixture


@pytest.fixture
def discord_prepared(prepared, monkeypatch):
    prepared.payload.interface = "discord"
    prepared.request.headers["x-conversation-id"] = str(uuid4())
    prepared.request.app.state.discord_writes = DiscordWriteRegistry()
    monkeypatch.setattr(Database, "fetchval", AsyncMock(return_value=4))
    prepared.scheduled = []
    monkeypatch.setattr(
        chat_api,
        "schedule_discord_exchange",
        lambda _, exchange: prepared.scheduled.append(exchange),
    )
    return prepared


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["context", "links", "agent_setup"])
async def test_accepted_pre_stream_failure_is_stored_once(discord_prepared, phase):
    p = discord_prepared
    p.mocks[phase].side_effect = RuntimeError("private provider error")
    await _consume(p)
    (exchange,) = p.scheduled
    assert exchange.question == p.payload.query
    assert exchange.answer == ""
    assert exchange.response_id == p.response_id
    assert exchange.history_generation == 4
    assert exchange.metadata["diagnostics"]["outcome"] == "provider_error"
    assert "private provider error" not in str(exchange.metadata)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "phase", ["credentials", "admission", "context", "agent_setup"]
)
async def test_cancellation_persists_only_after_admission(discord_prepared, phase):
    p = discord_prepared
    p.mocks["credentials"].return_value = credentials(openai=False)
    started = asyncio.Event()

    async def block(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    p.mocks[phase].side_effect = block
    async with asyncio.timeout(2):
        task = asyncio.create_task(_consume(p))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    if phase in {"context", "agent_setup"}:
        (exchange,) = p.scheduled
        assert exchange.answer == ""
        assert exchange.metadata["diagnostics"]["outcome"] == "client_disconnect"
    else:
        assert p.scheduled == []


@pytest.mark.anyio
@pytest.mark.parametrize("capture", [None, RuntimeError("private database error")])
async def test_failed_epoch_capture_never_falls_back_to_unfenced_write(
    discord_prepared, monkeypatch, capture
):
    fetch = (
        AsyncMock(side_effect=capture)
        if isinstance(capture, Exception)
        else AsyncMock(return_value=None)
    )
    monkeypatch.setattr(Database, "fetchval", fetch)
    await _consume(discord_prepared)
    assert discord_prepared.scheduled == []


@pytest.mark.anyio
async def test_success_handoff_schedules_once_and_capture_precedes_credentials(
    discord_prepared,
):
    p = discord_prepared
    original = p.mocks["credentials"].return_value

    async def resolve(*args, **kwargs):
        cast("AsyncMock", Database.fetchval).assert_awaited_once()
        return original

    p.mocks["credentials"].side_effect = resolve
    await _consume(p)
    (exchange,) = p.scheduled
    assert exchange.answer == "answer"
    assert exchange.metadata["diagnostics"]["outcome"] == "completed"


@pytest.mark.anyio
async def test_close_at_sources_before_generator_start_stores_once(discord_prepared):
    p = discord_prepared
    p.mocks["context"].return_value = RetrievedContext(
        text="context",
        effective_transform_mode=p.payload.query_transform_mode,
        sources=(RetrievalSource(id="faq", kind="faq", title="Source"),),
    )
    stream = _stream(p)
    assert "event: sources" in str(await anext(stream))
    await stream.aclose()
    (exchange,) = p.scheduled
    assert exchange.answer == ""
    assert exchange.metadata["diagnostics"]["outcome"] == "client_disconnect"
    assert exchange.metadata["sources"][0]["id"] == "faq"


@pytest.mark.anyio
async def test_credential_denial_has_no_transcript(discord_prepared):
    p = discord_prepared
    p.mocks["credentials"].return_value = credentials(openai=False)
    p.request.app.state.settings = settings(enabled=False)
    await _consume(p)
    assert p.scheduled == []


def _exchange():
    return DiscordExchange(
        uuid4(), uuid4(), uuid4(), "model", "question", "answer", {}, 0
    )


@pytest.mark.anyio
@pytest.mark.parametrize("blocked", [False, True])
async def test_registry_drains_or_cancels_and_rejects_new_writes(monkeypatch, blocked):
    registry = DiscordWriteRegistry()
    exchange = _exchange()
    storage = DiscordStorage(
        cast("Database", object()),
        exchange.conversation_id,
        exchange.user_id,
        [],
        0,
        registry,
    )
    started = asyncio.Event()
    finished = asyncio.Event()

    async def store(*args):
        started.set()
        try:
            if blocked:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(0)
            return True
        finally:
            finished.set()

    monkeypatch.setattr(storage_api, "store_discord_exchange", store)
    registry.schedule(storage, exchange)
    async with asyncio.timeout(2):
        await started.wait()
        await registry.drain(grace_seconds=0.01)
    assert finished.is_set()
    assert not registry.pending
    registry.schedule(storage, replace(exchange, response_id=uuid4()))
    assert not registry.pending


@pytest.mark.anyio
async def test_lifespan_awaits_storage_before_database_disconnect(monkeypatch):
    events = []

    class DB:
        def __init__(self, **kwargs):
            pass

        async def init(self):
            pass

        async def disconnect(self):
            events.append("database_close")

    async def store(*args):
        await asyncio.sleep(0)
        events.append("stored")
        return True

    monkeypatch.setattr(main, "Database", DB)
    monkeypatch.setattr(main, "init_http_client", lambda: None)
    monkeypatch.setattr(main, "close_http_client", AsyncMock())
    monkeypatch.setattr(storage_api, "store_discord_exchange", store)
    app = main.make_app(main.settings)
    async with main.lifespan(app):
        exchange = _exchange()
        storage = DiscordStorage(
            app.state.db,
            exchange.conversation_id,
            exchange.user_id,
            [],
            0,
            app.state.discord_writes,
        )
        app.state.discord_writes.schedule(storage, exchange)
    assert events == ["stored", "database_close"]


@pytest.mark.anyio
@pytest.mark.parametrize("backend_failure", [False, True])
async def test_registry_suppression_is_normal_and_backend_logs_never_include_content(
    monkeypatch, caplog, backend_failure
):
    registry = DiscordWriteRegistry()
    exchange = replace(
        _exchange(), question="private question", answer="private answer"
    )
    storage = DiscordStorage(
        cast("Database", object()),
        exchange.conversation_id,
        exchange.user_id,
        [],
        0,
        registry,
    )
    backend = (
        AsyncMock(side_effect=RuntimeError("sk-private-secret"))
        if backend_failure
        else AsyncMock(return_value=False)
    )
    monkeypatch.setattr(storage_api, "store_discord_exchange", backend)
    registry.schedule(storage, exchange)
    await registry.drain()
    assert "private question" not in caplog.text
    assert "private answer" not in caplog.text
    assert "sk-private-secret" not in caplog.text
    assert ("storage failed" in caplog.text) is backend_failure
