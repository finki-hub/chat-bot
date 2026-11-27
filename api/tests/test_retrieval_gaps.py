from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import anyio
import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage

from app.api import chat as chat_api
from app.api import retrieval_gap_writes
from app.api.retrieval_gap_writes import RetrievalGapWrites
from app.data.connection import Database
from app.data.retrieval_gaps import RetrievalGap, retrieval_gap_reason
from app.llms import agents, search_tools
from app.llms import chat as chat_module
from app.llms.agents import SearchAttempt, StreamObservation
from app.llms.models import Model
from app.llms.retrieval_result import RetrievedContext
from app.llms.search_tools import KnowledgeSearch, SearchResult
from app.llms.tools import fallback_prompts
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings
from tests.test_agent_error_log_privacy import TestAgentGraph

MISS = SearchAttempt(query="уписни рокови", hit=False, weak=False)
WEAK = SearchAttempt(query="паркинг", hit=True, weak=True, top_score=0.01)
HIT = SearchAttempt(query="школарина", hit=True, weak=False, top_score=0.8)


@pytest.mark.parametrize(
    ("searches", "reason"),
    [
        ([], None),
        ([MISS], "no_results"),
        ([MISS, MISS], "no_results"),
        ([WEAK], "weak_results"),
        ([MISS, WEAK], "weak_results"),
        ([HIT], None),
        ([MISS, HIT], None),
        ([WEAK, HIT], None),
    ],
)
def test_gap_reason(searches: list[SearchAttempt], reason: str | None) -> None:
    assert retrieval_gap_reason(searches) == reason


def _tool_end(run_id: str, artifact: dict[str, object]) -> list[dict]:
    return [
        {
            "event": "on_tool_start",
            "run_id": run_id,
            "name": "search_knowledge_base",
            "data": {"input": {"query": artifact.get("query")}},
        },
        {
            "event": "on_tool_end",
            "run_id": run_id,
            "name": "search_knowledge_base",
            "data": {
                "output": ToolMessage(
                    content="context",
                    tool_call_id=run_id,
                    artifact=artifact,
                ),
            },
        },
    ]


class _Graph:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    async def astream_events(self, *args, **kwargs):
        for event in self._events:
            yield event


@pytest.mark.anyio
async def test_agent_records_each_successful_search() -> None:
    observation = StreamObservation(distinct_id="d", response_id="r")
    events = [
        *_tool_end(
            "a",
            {"hit": False, "weak": False, "query": "упис", "sources": []},
        ),
        *_tool_end(
            "b",
            {"hit": True, "weak": True, "top_score": 0.02, "query": "паркинг"},
        ),
        *_tool_end("c", {"failed": True, "query": "грешка"}),
    ]

    async for _ in agents.create_agent_token_generator(
        cast("TestAgentGraph", _Graph(events)),
        [],
        observation,
    ):
        pass

    assert observation.searches == [
        SearchAttempt(query="упис", hit=False, weak=False),
        SearchAttempt(query="паркинг", hit=True, weak=True, top_score=0.02),
    ]


@pytest.mark.anyio
async def test_search_artifact_reports_weak_match_and_score(monkeypatch) -> None:
    retrieved = RetrievedContext(text="слаб блок", weak=True, top_score=0.01)
    monkeypatch.setattr(
        search_tools,
        "get_retrieved_context_with_sources",
        AsyncMock(return_value=retrieved),
    )

    class _Links:
        def __init__(self, _db) -> None:
            self.render = AsyncMock(return_value="")

    monkeypatch.setattr(search_tools, "RelevantLinks", _Links)
    [tool] = KnowledgeSearch(
        cast("Database", object()),
        Model.MULTILINGUAL_E5_LARGE,
        None,
    ).tools()

    message = await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "t1",
            "name": tool.name,
            "args": {"query": "паркинг"},
        },
    )

    assert message.artifact["weak"] is True
    assert message.artifact["top_score"] == 0.01
    assert message.artifact["query"] == "паркинг"


@pytest.mark.anyio
async def test_tool_less_fallback_records_its_search(monkeypatch) -> None:
    async def search(self, query: str, *, cite: bool = True) -> SearchResult:
        return SearchResult(content="", sources=[], hit=False)

    async def stream(user_prompt, model, *, system_prompt, **kwargs):
        await fallback_prompts(system_prompt, user_prompt)
        return StreamingResponse(iter(()))

    monkeypatch.setattr(KnowledgeSearch, "search", search)
    monkeypatch.setattr(chat_module, "stream_response_with_agent", stream)
    observation = StreamObservation(distinct_id="d", response_id="r")
    payload = ChatSchema.model_validate(
        {
            "interface": "web",
            "inference_model": Model.GPT_6_LUNA,
            "messages": [{"role": "user", "content": "Кога е уписот?"}],
        },
    )

    await chat_module.handle_chat(
        payload,
        "Денешен датум: 24.10.2026.",
        observation=observation,
        db=cast("Database", object()),
    )

    # The question itself is never stored; it stays joinable by response_id.
    assert observation.searches == [SearchAttempt(query=None, hit=False, weak=False)]


class _RecordingWrites(RetrievalGapWrites):
    def __init__(self) -> None:
        super().__init__()
        self.scheduled: list[RetrievalGap] = []

    def schedule(self, db: Database, gap: RetrievalGap) -> None:
        self.scheduled.append(gap)


async def _body() -> AsyncIterator[str]:
    yield 'event: token\ndata: {"text":"Немам податок."}\n\n'
    yield "event: done\ndata: {}\n\n"


async def _failed_body() -> AsyncIterator[str]:
    yield 'event: error\ndata: {"code":"agent_error"}\n\n'
    yield "event: done\ndata: {}\n\n"


def _run_stream(
    monkeypatch,
    searches: list[SearchAttempt],
    *,
    interface: str = "discord",
    body: AsyncIterator[str] | None = None,
) -> tuple[UUID, list[RetrievalGap]]:
    monkeypatch.setattr(chat_api, "capture", lambda *args, **kwargs: None)
    writes = _RecordingWrites()
    observation = StreamObservation(distinct_id="d", response_id="r")
    observation.searches.extend(searches)
    response_id = uuid4()
    payload = ChatSchema.model_validate(
        {
            "interface": interface,
            "inference_model": Model.GPT_6_LUNA,
            "messages": [{"role": "user", "content": "Прашање?"}],
        },
    )

    async def collect() -> list[str]:
        stream = chat_api._instrument_stream(  # noqa: SLF001
            _body() if body is None else body,
            payload=payload,
            response_id=response_id,
            timings=RequestTimings(),
            distinct_id="d",
            session_id=None,
            observation=observation,
            gap_writes=(writes, cast("Database", object())),
        )
        return [str(chunk) async for chunk in stream]

    anyio.run(collect)
    return response_id, writes.scheduled


def test_stream_records_a_gap_when_every_search_misses(monkeypatch) -> None:
    response_id, scheduled = _run_stream(monkeypatch, [MISS, WEAK])

    assert scheduled == [
        RetrievalGap(
            response_id=response_id,
            interface="discord",
            reason="weak_results",
            search_queries=["уписни рокови", "паркинг"],
            inference_model=Model.GPT_6_LUNA.value,
            embeddings_model=ChatSchema.model_fields["embeddings_model"].default.value,
            reranker_score_max=0.01,
            outcome="completed",
        ),
    ]


@pytest.mark.parametrize("searches", [[], [MISS, HIT]])
def test_stream_records_no_gap_when_answered_or_not_searched(
    monkeypatch,
    searches: list[SearchAttempt],
) -> None:
    _, scheduled = _run_stream(monkeypatch, searches, interface="web")

    assert scheduled == []


def test_stream_records_no_gap_for_a_failed_answer(monkeypatch) -> None:
    # It may have stopped before the searches that would have found something.
    _, scheduled = _run_stream(monkeypatch, [MISS], body=_failed_body())

    assert scheduled == []


def _gap() -> RetrievalGap:
    return RetrievalGap(
        response_id=uuid4(),
        interface="web",
        reason="no_results",
        search_queries=["упис", None],
        inference_model="m",
        embeddings_model="e",
        reranker_score_max=None,
        outcome="completed",
    )


@pytest.mark.anyio
async def test_writes_insert_the_gap_and_drain() -> None:
    db = AsyncMock()
    writes = RetrievalGapWrites()
    gap = _gap()

    writes.schedule(cast("Database", db), gap)
    await writes.drain()

    insert, sweep = [call.args for call in db.execute.await_args_list]
    sql, *args = insert
    assert "INSERT INTO retrieval_gap" in sql
    assert "ON CONFLICT (response_id) DO NOTHING" in sql
    assert args == [
        gap.response_id,
        "web",
        "no_results",
        ["упис", None],
        "m",
        "e",
        None,
        "completed",
    ]
    assert "DELETE FROM retrieval_gap" in sweep[0]
    assert "NOT EXISTS" in sweep[0]
    assert not writes.pending

    writes.schedule(cast("Database", db), gap)
    assert not writes.pending, "a drained writer accepts no new writes"


@pytest.mark.anyio
async def test_failed_write_is_logged_without_content(monkeypatch, caplog) -> None:
    db = AsyncMock()
    db.execute.side_effect = RuntimeError("упис secret")
    writes = RetrievalGapWrites()

    writes.schedule(cast("Database", db), _gap())
    await writes.drain()

    [record] = [r for r in caplog.records if r.name == retrieval_gap_writes.__name__]
    assert "RuntimeError" in record.getMessage()
    assert "упис" not in record.getMessage()
