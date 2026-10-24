import json
from typing import cast
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.data.connection import Database
from app.llms import agents, search_tools
from app.llms.agents import StreamObservation
from app.llms.models import Model
from app.llms.retrieval_result import (
    RetrievalSource,
    RetrievalSourcePayload,
    RetrievedContext,
)
from app.llms.search_tools import KnowledgeSearch
from app.llms.tools import SEARCH_TOOL_NAME, agent_middleware
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
    assert tokens == ["Проверувам. ", "\n\nОдговор."]
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
    assert call["label"] == "Ги проверувам предметите"
    result = next(data for name, data in frames if name == "tool_result")
    assert result["ok"] is True
    assert result["summary"] == ""


def _retrieved(text: str) -> RetrievedContext:
    return RetrievedContext(
        text=text,
        sources=(RetrievalSource(id="1", kind="faq", title="Префрлување"),)
        if text
        else (),
    )


def _search_tool(monkeypatch, retrieved: RetrievedContext, links: str = ""):
    retrieve = AsyncMock(return_value=retrieved)
    monkeypatch.setattr(search_tools, "get_retrieved_context_with_sources", retrieve)
    relevant_links = AsyncMock(return_value=links)

    class _Links:
        def __init__(self, _db) -> None:
            self.render = relevant_links

    monkeypatch.setattr(search_tools, "RelevantLinks", _Links)
    [tool] = KnowledgeSearch(
        cast("Database", object()),
        Model.MULTILINGUAL_E5_LARGE,
        None,
    ).tools()
    return tool, retrieve, relevant_links


async def _invoke(tool, query: str) -> ToolMessage:
    return await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "t1",
            "name": tool.name,
            "args": {"query": query},
        },
    )


@pytest.mark.anyio
async def test_search_tool_returns_context_and_source_artifact(monkeypatch) -> None:
    tool, retrieve, _ = _search_tool(monkeypatch, _retrieved("блок"))

    message = await _invoke(tool, "префрлување")

    assert message.content == "блок"
    assert message.artifact == {
        "hit": True,
        "sources": [{"id": "1", "kind": "faq", "title": "Префрлување"}],
        "summary": "Пронајдени извори: 1",
        "context_chars": 4,
    }
    assert retrieve.await_args is not None
    assert retrieve.await_args.kwargs["query"] == "префрлување"


@pytest.mark.anyio
async def test_search_tool_appends_relevant_links(monkeypatch) -> None:
    links = "Корисни линкови:\n- Распоред: https://finki-hub.com"
    tool, _, relevant_links = _search_tool(monkeypatch, _retrieved("блок"), links)

    message = await _invoke(tool, "распоред")

    assert message.content == f"блок\n\n{links}"
    assert relevant_links.await_args is not None
    assert relevant_links.await_args.args[0] == "распоред"


@pytest.mark.anyio
async def test_search_tool_without_results_says_so(monkeypatch) -> None:
    tool, _, _ = _search_tool(monkeypatch, _retrieved(""))

    message = await _invoke(tool, "q")

    assert message.content == "Нема пронајдени извори за ова барање."
    assert message.artifact == {
        "hit": False,
        "sources": [],
        "summary": "Нема пронајдени извори",
        "context_chars": 0,
    }


@pytest.mark.anyio
async def test_failed_search_becomes_a_failed_result_not_an_error(monkeypatch) -> None:
    tool, retrieve, _ = _search_tool(monkeypatch, _retrieved(""))
    retrieve.side_effect = RuntimeError("gpu-api down")

    message = await _invoke(tool, "q")

    assert message.artifact == {"failed": True}
    assert "gpu-api" not in message.content


def test_agent_middleware_limits_searches_tool_calls_and_model_calls() -> None:
    search, tools, model = agent_middleware()

    assert isinstance(search, ToolCallLimitMiddleware)
    assert (search.tool_name, search.run_limit) == (SEARCH_TOOL_NAME, 4)
    assert isinstance(tools, ToolCallLimitMiddleware)
    assert (tools.tool_name, tools.run_limit) == (None, 15)
    assert isinstance(model, ModelCallLimitMiddleware)
    assert (model.run_limit, model.exit_behavior) == (10, "end")


def test_interleaving_keeps_each_search_best_hits_within_cap() -> None:
    first: list[RetrievalSourcePayload] = [
        {"id": str(i), "kind": "chunk", "title": "A"} for i in range(10)
    ]
    second: list[RetrievalSourcePayload] = [
        {"id": f"f{i}", "kind": "faq", "title": "B"} for i in range(10)
    ]

    merged = agents._interleave_sources([first, second])  # noqa: SLF001

    assert len(merged) == 12
    assert [s["id"] for s in merged[:4]] == ["0", "f0", "1", "f1"]


def _failed_search_end(run_id: str) -> dict:
    output = ToolMessage(
        content="failed", tool_call_id=run_id, artifact={"failed": True}
    )
    return {
        "event": "on_tool_end",
        "run_id": run_id,
        "name": "search_knowledge_base",
        "data": {"output": output},
    }


@pytest.mark.anyio
async def test_failed_search_result_is_reported_as_not_ok() -> None:
    frames = await _run([_tool_start("a", "рок"), _failed_search_end("a"), _text("ok")])

    result = next(data for name, data in frames if name == "tool_result")
    assert (result["ok"], result["summary"]) == (False, "Алатката не врати резултат")


@pytest.mark.anyio
async def test_text_after_a_tool_call_starts_a_new_paragraph() -> None:
    observation = StreamObservation(distinct_id="d", response_id="r")
    frames = await _run(
        [
            _text("Проверувам."),
            _tool_start("a", "рок"),
            _tool_end("a", [FAQ]),
            _text(" Одговор."),
        ],
        observation,
    )

    tokens = [data["text"] for name, data in frames if name == "token"]
    assert tokens == ["Проверувам.", "\n\nОдговор."]
    assert observation.context_chars == 0


# The tools the MCP server exposed in production on 2026-10-23.
_LIVE_TOOLS = [
    "list_exam_session_schedules",
    "get_random_anto_quote",
    "list_timetables",
    "list_timetable_groups",
    "list_timetable_professors",
    "list_timetable_rooms",
    "list_timetable_courses",
    "list_timetable_entries",
    "list_courses",
    "get_course_data",
    "get_course_staff",
    "get_course_participants",
    "list_rooms",
    "get_room_data",
    "list_staff",
    "get_staff_member",
    "list_diplomas",
    "get_diploma",
    "list_masters",
    "get_master",
    "get_course_weekly_topics",
    "search_companies_by_name",
    "get_all_companies",
    "search_courses_by_name",
    "get_staff_details",
    "get_professor_consultations",
    "get_company_by_name",
    "search_staff_details_by_name",
    "get_all_courses",
    "search_study_programs_by_name",
    "get_study_program_by_code",
    "get_course_by_code",
    "get_all_study_programs",
    "get_academic_calendar",
    "recommend_diploma_committee",
]


@pytest.mark.parametrize("name", _LIVE_TOOLS)
def test_every_live_tool_has_a_readable_label(name: str) -> None:
    label = agents._tool_label(name, {})  # noqa: SLF001

    assert label != agents._FALLBACK_TOOL_LABEL  # noqa: SLF001
    assert name not in label


def test_specific_tool_labels_win_over_general_ones() -> None:
    assert agents._tool_label("get_course_weekly_topics", {}) == (  # noqa: SLF001
        "Ги проверувам темите по недели"
    )
    assert agents._tool_label("list_timetable_courses", {}) == (  # noqa: SLF001
        "Го проверувам распоредот на часови"
    )
    assert agents._tool_label("unknown_tool", {}) == "Проверувам податоци"  # noqa: SLF001


def test_announcement_tools_get_their_own_label() -> None:
    assert agents._tool_label(  # noqa: SLF001
        "search_announcements",
        {"query": " упис семестар "},
    ) == ("Ги проверувам огласите: упис семестар")
    assert agents._tool_label("search_announcements", {}) == "Ги проверувам огласите"  # noqa: SLF001
    assert agents._tool_label("get_announcement", {"id": 7}) == "Ги проверувам огласите"  # noqa: SLF001
