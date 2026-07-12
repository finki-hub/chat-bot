import json
from collections.abc import AsyncIterator
from uuid import uuid4

import anyio
import pytest

from app.api import chat as chat_api
from app.llms.agents import StreamObservation
from app.llms.models import ChatModel, Model
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings


async def _body() -> AsyncIterator[str]:
    yield 'event: meta\ndata: {"tokens":{"input":1000,"output":1000,"total":2000}}\n\n'
    yield 'event: token\ndata: {"text":"discarded preamble"}\n\n'
    yield "event: reset\ndata: {}\n\n"
    yield 'event: token\ndata: {"text":"answer"}\n\n'
    yield "event: done\ndata: {}\n\n"


async def _gpu_body() -> AsyncIterator[str]:
    yield "data: self-hosted answer\n\n"
    yield "event: done\ndata: {}\n\n"


async def _gpu_fragmented_body() -> AsyncIterator[str]:
    yield "data: split"
    yield " answer\n\ndata: coalesced"
    yield " answer\n\nevent: status\ndata: {}\n\n"
    yield "event: done\ndata: {}\n\n"


def _run_captured_stream(
    monkeypatch,
    body: AsyncIterator[str],
    model: ChatModel,
) -> tuple[list[str], dict[str, object]]:
    captured: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        chat_api,
        "capture",
        lambda distinct_id, event, props: captured.append((distinct_id, event, props)),
    )
    payload = ChatSchema.model_validate(
        {
            "interface": "web",
            "inference_model": model,
            "messages": [{"role": "user", "content": "Прашање?"}],
        },
    )
    observation = StreamObservation(distinct_id="user-1", response_id="response-1")

    async def collect() -> list[str]:
        stream = chat_api._instrument_stream(  # noqa: SLF001
            body,
            payload=payload,
            response_id=uuid4(),
            timings=RequestTimings(),
            retrieval_hit=True,
            distinct_id="user-1",
            session_id="session-1",
            observation=observation,
        )
        return [str(chunk) async for chunk in stream]

    chunks = anyio.run(collect)
    assert len(captured) == 1
    _, _, props = captured.pop()
    return chunks, props


def test_chat_stream_emits_cost_diagnostics_when_pricing_is_known(monkeypatch):
    chunks, props = _run_captured_stream(
        monkeypatch,
        _body(),
        Model.CLAUDE_HAIKU_4_5,
    )
    meta = json.loads(chunks[-1].split("data:", 1)[1].strip())

    assert meta["cost"]["input_usd"] == pytest.approx(0.001)
    assert meta["cost"]["output_usd"] == pytest.approx(0.005)
    assert meta["cost"]["total_usd"] == pytest.approx(0.006)
    assert props["$ai_total_cost_usd"] == pytest.approx(0.006)
    assert props["$ai_input"] == [{"role": "user", "content": "Прашање?"}]
    assert props["$session_id"] == "session-1"
    assert props["$ai_output_choices"] == [
        {"role": "assistant", "content": "answer"},
    ]


def test_chat_stream_records_bare_data_token_frames(monkeypatch):
    chunks, props = _run_captured_stream(
        monkeypatch,
        _gpu_body(),
        "qwen3:14b-q4_K_M",
    )

    assert chunks[0] == "data: self-hosted answer\n\n"
    assert props["$ai_output_choices"] == [
        {"role": "assistant", "content": "self-hosted answer"},
    ]


def test_chat_stream_records_fragmented_bare_data_token_frames(monkeypatch):
    chunks, props = _run_captured_stream(
        monkeypatch,
        _gpu_fragmented_body(),
        "qwen3:14b-q4_K_M",
    )

    assert chunks[:3] == [
        "data: split",
        " answer\n\ndata: coalesced",
        " answer\n\nevent: status\ndata: {}\n\n",
    ]
    assert props["$ai_output_choices"] == [
        {"role": "assistant", "content": "split answercoalesced answer"},
    ]


def test_chat_request_log_fields_do_not_include_message_content():
    payload = ChatSchema.model_validate(
        {
            "interface": "web",
            "inference_model": Model.CLAUDE_HAIKU_4_5,
            "messages": [
                {"role": "user", "content": "private first question"},
                {"role": "assistant", "content": "private previous answer"},
                {"role": "user", "content": "private latest question"},
            ],
        },
    )

    fields = chat_api._chat_request_log_fields(payload)  # noqa: SLF001

    assert set(fields) == {
        "embeddings_model",
        "history_turns",
        "inference_model",
        "interface",
        "max_tokens",
        "query_len",
        "query_transform_model",
        "query_transform_mode",
        "reasoning",
        "temperature",
    }
    assert "private" not in repr(fields)
    assert fields["query_len"] == len("private latest question")
    assert fields["history_turns"] == 2


def test_chat_request_posthog_fields_include_message_content():
    payload = ChatSchema.model_validate(
        {
            "interface": "web",
            "inference_model": Model.CLAUDE_HAIKU_4_5,
            "messages": [
                {"role": "user", "content": "private first question"},
                {"role": "assistant", "content": "private previous answer"},
                {"role": "user", "content": "private latest question"},
            ],
        },
    )

    fields = chat_api._chat_request_posthog_fields(payload)  # noqa: SLF001

    assert fields["messages"] == [
        {"role": "user", "content": "private first question"},
        {"role": "assistant", "content": "private previous answer"},
        {"role": "user", "content": "private latest question"},
    ]
    assert fields["query_len"] == len("private latest question")
    assert fields["history_turns"] == 2
