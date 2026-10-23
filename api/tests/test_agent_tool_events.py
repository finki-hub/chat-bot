import json
from typing import cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.data.connection import Database
from app.llms import agents, search_tools
from app.llms.agents import StreamObservation
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_result import RetrievalSource, RetrievedContext
from app.llms.search_tools import build_search_tools
from tests.test_agent_error_log_privacy import TestAgentGraph


def _frames(events: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for event in events:
        name_line, data_line = event.strip().split("\n", 1)
        parsed.append(
            (name_line.removeprefix("event: "), json.loads(data_line[len("data: ") :])),
        )
    return parsed


def _text(content: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": AIMessageChunk(content=content)},
    }


def _tool_start(run_id: str, query: str) -> dict:
    return {
        "event": "on_tool_start",
        "run_id": run_id,
        "name": "search_knowledge_base",
        "data": {"input": {"query": query}},
    }


def _tool_end(run_id: str, sources: list[dict], *, hit: bool = True) -> dict:
    output = ToolMessage(
        content="context",
        tool_call_id=run_id,
        artifact={"hit": hit, "sources": sources, "summary": f"{len(sources)} sources"},
    )
    return {
        "event": "on_tool_end",
        "run_id": run_id,
        "name": "search_knowledge_base",
        "data": {"output": output},
    }


class _Graph:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    async def astream_events(self, *args, **kwargs):
        for event in self._events:
            yield event


async def _run(events: list[dict], observation: StreamObservation | None = None):
    return _frames(
        [
            event
            async for event in agents.create_agent_token_generator(
                cast("TestAgentGraph", _Graph(events)),
                [],
                observation,
            )
        ],
    )


FAQ = {"id": "1", "kind": "faq", "title": "Префрлување"}
DOC = {"id": "9", "kind": "chunk", "title": "Правилник"}


@pytest.mark.anyio
async def test_text_around_parallel_tool_calls_is_kept_in_order() -> None:
    observation = StreamObservation(distinct_id="d", response_id="r")
    frames = await _run(
        [
            _text("Проверувам. "),
            _tool_start("a", "префрлување"),
            _tool_start("b", "школарина"),
            _tool_end("b", [DOC]),
            _tool_end("a", [FAQ, DOC]),
            _text("Одговор."),
        ],
        observation,
    )

    names = [name for name, _ in frames]
    assert "reset" not in names
    assert names[:2] == ["token", "status"]
    calls = [data for name, data in frames if name == "tool_call"]
    assert [(c["id"], c["label"], c["input"]) for c in calls] == [
        ("a", "Пребарувам: префрлување", {"query": "префрлување"}),
        ("b", "Пребарувам: школарина", {"query": "школарина"}),
    ]
    results = [data for name, data in frames if name == "tool_result"]
    assert [(r["id"], r["ok"], r["summary"]) for r in results] == [
        ("b", True, "1 sources"),
        ("a", True, "2 sources"),
    ]
    sources = [data["sources"] for name, data in frames if name == "sources"]
    assert sources == [[DOC], [DOC, FAQ]]
    tokens = [data["text"] for name, data in frames if name == "token"]
    assert tokens == ["Проверувам. ", "Одговор."]
    assert observation.search_hit is True
    assert observation.sources == [DOC, FAQ]
    assert observation.tool_call_count == 2


@pytest.mark.anyio
async def test_failed_tool_reports_safe_summary_without_sources() -> None:
    frames = await _run(
        [
            _tool_start("a", "рок"),
            {
                "event": "on_tool_error",
                "run_id": "a",
                "name": "search_knowledge_base",
                "data": {"error": RuntimeError("secret detail")},
            },
            _text("Не најдов."),
        ],
    )

    result = next(data for name, data in frames if name == "tool_result")
    assert result == {
        "id": "a",
        "name": "search_knowledge_base",
        "ok": False,
        "summary": "Алатката не врати резултат",
    }
    assert not any(name == "sources" for name, _ in frames)
    assert "secret detail" not in json.dumps(frames, ensure_ascii=False)


@pytest.mark.anyio
async def test_non_search_tool_gets_generic_label() -> None:
    frames = await _run(
        [
            {
                "event": "on_tool_start",
                "run_id": "c",
                "name": "list_courses",
                "data": {"input": {"program": "КН"}},
            },
            {
                "event": "on_tool_end",
                "run_id": "c",
                "name": "list_courses",
                "data": {"output": ToolMessage(content="[]", tool_call_id="c")},
            },
            _text("Предмети."),
        ],
    )

    call = next(data for name, data in frames if name == "tool_call")
    assert call["label"] == "Користам алатка: list_courses"
    result = next(data for name, data in frames if name == "tool_result")
    assert result["ok"] is True
    assert result["summary"] == ""


def _retrieved(text: str) -> RetrievedContext:
    return RetrievedContext(
        text=text,
        effective_transform_mode=QueryTransformMode.RAW,
        sources=(RetrievalSource(id="1", kind="faq", title="Префрлување"),)
        if text
        else (),
    )


@pytest.mark.anyio
async def test_search_tool_returns_context_and_source_artifact(monkeypatch) -> None:
    retrieve = AsyncMock(return_value=_retrieved("блок"))
    monkeypatch.setattr(search_tools, "get_retrieved_context_with_sources", retrieve)
    [tool] = build_search_tools(
        cast("Database", object()),
        Model.MULTILINGUAL_E5_LARGE,
        Model.GPT_6_LUNA,
        max_searches=4,
    )

    message = await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "t1",
            "name": tool.name,
            "args": {"query": "префрлување"},
        },
    )

    assert message.content == "блок"
    assert message.artifact == {
        "hit": True,
        "sources": [{"id": "1", "kind": "faq", "title": "Префрлување"}],
        "summary": "Пронајдени извори: 1",
    }
    assert retrieve.await_args.kwargs["query"] == "префрлување"
    assert retrieve.await_args.kwargs["query_transform_mode"] == QueryTransformMode.RAW


@pytest.mark.anyio
async def test_search_tool_caps_searches_per_answer(monkeypatch) -> None:
    retrieve = AsyncMock(return_value=_retrieved(""))
    monkeypatch.setattr(search_tools, "get_retrieved_context_with_sources", retrieve)
    [tool] = build_search_tools(
        cast("Database", object()),
        Model.MULTILINGUAL_E5_LARGE,
        Model.GPT_6_LUNA,
        max_searches=2,
    )

    messages = [
        await tool.ainvoke(
            {
                "type": "tool_call",
                "id": f"t{i}",
                "name": tool.name,
                "args": {"query": "q"},
            },
        )
        for i in range(3)
    ]

    assert retrieve.await_count == 2
    assert messages[0].content == "Нема пронајдени извори за ова барање."
    assert messages[0].artifact["summary"] == "Нема пронајдени извори"
    assert "лимитот" in messages[2].content
    assert "sources" not in messages[2].artifact


def test_interleaving_keeps_each_search_best_hits_within_cap() -> None:
    first = [{"id": str(i), "kind": "chunk", "title": "A"} for i in range(10)]
    second = [{"id": f"f{i}", "kind": "faq", "title": "B"} for i in range(10)]

    merged = agents._interleave_sources([first, second])  # noqa: SLF001

    assert len(merged) == 12
    assert [s["id"] for s in merged[:4]] == ["0", "f0", "1", "f1"]
