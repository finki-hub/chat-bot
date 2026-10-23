import asyncio
from collections.abc import AsyncIterator
from typing import cast
from uuid import UUID, uuid4

import anyio
import pytest

from app.api import chat as chat_api
from app.api.discord_storage import (
    DiscordStorage,
    DiscordWriteRegistry,
    discord_storage,
)
from app.data.connection import Database
from app.data.discord_conversations import DiscordExchange
from app.llms.agents import StreamObservation
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings

_DB = cast("Database", object())
_CONVERSATION = "7d1c6f3e-3a8a-4f0e-9a37-2f5d6c1b9e01"


def _payload(**overrides: object) -> ChatSchema:
    return ChatSchema.model_validate(
        {
            "user_id": str(uuid4()),
            "messages": [
                {"role": "user", "content": "Претходно?"},
                {"role": "assistant", "content": "Претходен одговор."},
                {"role": "user", "content": "Кој е декан?"},
            ],
            "discord_context": {
                "guild_id": "1",
                "channel_id": "2",
                "message_id": "3",
            },
            **overrides,
        },
    )


async def _body() -> AsyncIterator[str]:
    yield 'event: token\ndata: {"text":"Декан е "}\n\n'
    yield 'event: token\ndata: {"text":"Петре Ламески."}\n\n'
    yield "event: done\ndata: {}\n\n"


@pytest.mark.anyio
async def test_discord_requests_with_user_and_conversation_are_stored():
    payload = _payload()

    class CaptureDB:
        async def fetchval(self, *args):
            return 3

    storage = await discord_storage(
        payload, cast("Database", CaptureDB()), _CONVERSATION, DiscordWriteRegistry()
    )
    assert storage is not None
    assert storage.conversation_id == UUID(_CONVERSATION)
    assert storage.user_id == payload.user_id
    assert storage.history_generation == 3


@pytest.mark.parametrize(
    ("overrides", "conversation_id"),
    [
        ({"interface": "web"}, _CONVERSATION),
        ({"user_id": None}, _CONVERSATION),
        ({}, None),
    ],
)
@pytest.mark.anyio
async def test_other_requests_are_not_stored(overrides, conversation_id):
    assert (
        await discord_storage(
            _payload(**overrides), _DB, conversation_id, DiscordWriteRegistry()
        )
        is None
    )


def test_discord_context_rejects_non_snowflake_ids():
    with pytest.raises(ValueError, match="pattern"):
        _payload(discord_context={"channel_id": "not-an-id"})


def test_stream_schedules_question_answer_and_diagnostics(monkeypatch):
    scheduled: list[tuple[DiscordStorage, DiscordExchange]] = []
    monkeypatch.setattr(chat_api, "capture", lambda *_args: None)
    monkeypatch.setattr(
        chat_api,
        "schedule_discord_exchange",
        lambda storage, exchange: scheduled.append((storage, exchange)),
    )
    payload = _payload()
    response_id = uuid4()
    storage = DiscordStorage(
        db=_DB,
        conversation_id=UUID(_CONVERSATION),
        user_id=cast("UUID", payload.user_id),
        history_generation=0,
        registry=DiscordWriteRegistry(),
    )

    observation = StreamObservation(distinct_id="user-1", response_id=str(response_id))
    observation.search_hit = True
    observation.sources = [{"id": "q1", "kind": "faq", "title": "Деканат"}]

    async def collect() -> list[str]:
        timings = RequestTimings()
        timings.retrieval_ids = ["faq-id"]
        timings.final_faq_count = 2
        timings.reranker_score_max = 0.91
        timings.reranker_score_min = 0.52
        timings.reranker_above_threshold = 2
        timings.lexical_faq_candidate_count = 3
        timings.transliteration_variant_added = True
        stream = chat_api._instrument_stream(  # noqa: SLF001
            _body(),
            payload=payload,
            response_id=response_id,
            timings=timings,
            distinct_id="user-1",
            session_id=None,
            observation=observation,
            storage=storage,
        )
        return [str(chunk) async for chunk in stream]

    anyio.run(collect)

    assert len(scheduled) == 1
    stored_with, exchange = scheduled[0]
    assert stored_with is storage
    assert exchange.conversation_id == UUID(_CONVERSATION)
    assert exchange.response_id == response_id
    assert exchange.question == "Кој е декан?"
    assert exchange.answer == "Декан е Петре Ламески."
    metadata = exchange.metadata
    assert metadata["responseId"] == str(response_id)
    assert metadata["sources"] == [{"id": "q1", "kind": "faq", "title": "Деканат"}]
    assert metadata["discord"] == {"guildId": "1", "channelId": "2", "messageId": "3"}
    diagnostics = cast("dict[str, object]", metadata["diagnostics"])
    assert diagnostics["outcome"] == "completed"
    assert diagnostics["historyTurns"] == 2
    assert diagnostics["retrievalHit"] is True
    assert not any("queryTransform" in key for key in diagnostics)
    assert "total_ms" in cast("dict[str, object]", diagnostics["timing"])
    retrieval = cast("dict[str, object]", diagnostics["retrieval"])
    assert retrieval["retrieval_ids"] == ["faq-id"]
    assert retrieval["final_faq_count"] == 2
    assert retrieval["reranker_score_max"] == 0.91
    assert retrieval["reranker_score_min"] == 0.52
    assert retrieval["chunks_above_threshold"] == 2
    assert retrieval["lexical_faq_candidate_count"] == 3
    assert retrieval["transliteration_variant_added"] is True


def test_stream_without_storage_schedules_nothing(monkeypatch):
    scheduled: list[object] = []
    monkeypatch.setattr(chat_api, "capture", lambda *_args: None)
    monkeypatch.setattr(
        chat_api,
        "schedule_discord_exchange",
        lambda *args: scheduled.append(args),
    )
    payload = _payload(interface="web")

    async def collect() -> None:
        stream = chat_api._instrument_stream(  # noqa: SLF001
            _body(),
            payload=payload,
            response_id=uuid4(),
            timings=RequestTimings(),
            distinct_id="user-1",
            session_id=None,
            observation=StreamObservation(distinct_id="user-1", response_id="r"),
        )
        async for _ in stream:
            pass

    anyio.run(collect)
    assert scheduled == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind", ["caught", "raised", "cancel", "close", "reset", "empty"]
)
async def test_terminal_paths_store_once_and_reset_discards_old_answer(
    monkeypatch, kind
):
    scheduled = []
    monkeypatch.setattr(chat_api, "capture", lambda *args: None)
    monkeypatch.setattr(
        chat_api,
        "schedule_discord_exchange",
        lambda _, exchange: scheduled.append(exchange),
    )
    payload = _payload()
    assert payload.user_id is not None
    storage = DiscordStorage(
        _DB, UUID(_CONVERSATION), payload.user_id, 0, DiscordWriteRegistry()
    )

    async def body():
        if kind == "empty":
            return
        yield 'event: token\ndata: {"text":"old"}\n\n'
        if kind == "raised":
            raise RuntimeError("private provider error")
        if kind == "caught":
            yield 'event: error\ndata: {"code":"agent_error"}\n\n'
        if kind == "reset":
            yield "event: reset\ndata: {}\n\n"
            yield 'event: token\ndata: {"text":"new"}\n\n'

    stream = chat_api._instrument_stream(
        body(),
        payload=payload,
        response_id=uuid4(),
        timings=RequestTimings(),
        distinct_id="test",
        session_id=None,
        observation=StreamObservation(distinct_id="test", response_id="test"),
        storage=storage,
    )
    if kind in {"cancel", "close"}:
        await anext(stream)
        if kind == "close":
            await stream.aclose()
        else:
            with pytest.raises(asyncio.CancelledError):
                await stream.athrow(asyncio.CancelledError())
    elif kind == "raised":
        with pytest.raises(RuntimeError, match="private provider error"):
            async for _ in stream:
                pass
    else:
        async for _ in stream:
            pass
    (exchange,) = scheduled
    expected_outcome = {
        "caught": "provider_error",
        "raised": "provider_error",
        "cancel": "client_disconnect",
        "close": "client_disconnect",
        "reset": "completed",
        "empty": "empty_answer",
    }[kind]
    assert exchange.metadata["diagnostics"]["outcome"] == expected_outcome
    assert exchange.answer == (
        "new" if kind == "reset" else "" if kind == "empty" else "old"
    )
    assert "private provider error" not in str(exchange.metadata)
