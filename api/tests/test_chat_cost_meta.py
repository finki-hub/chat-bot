import json
from collections.abc import AsyncIterator
from uuid import uuid4

import anyio
import pytest

from app.api import chat as chat_api
from app.llms.agents import StreamObservation
from app.llms.models import Model
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings


async def _body() -> AsyncIterator[str]:
    yield 'event: meta\ndata: {"tokens":{"input":1000,"output":1000,"total":2000}}\n\n'
    yield 'event: token\ndata: {"text":"answer"}\n\n'
    yield "event: done\ndata: {}\n\n"


def test_chat_stream_emits_cost_diagnostics_when_pricing_is_known(monkeypatch):
    captured: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        chat_api,
        "capture",
        lambda distinct_id, event, props: captured.append((distinct_id, event, props)),
    )
    payload = ChatSchema(
        interface="web",
        inference_model=Model.CLAUDE_HAIKU_4_5,
        messages=[{"role": "user", "content": "Прашање?"}],
    )
    observation = StreamObservation(distinct_id="user-1", response_id="response-1")

    async def collect() -> list[str]:
        stream = chat_api._instrument_stream(  # noqa: SLF001
            _body(),
            payload=payload,
            response_id=uuid4(),
            timings=RequestTimings(),
            retrieval_hit=True,
            distinct_id="user-1",
            observation=observation,
        )
        return [str(chunk) async for chunk in stream]

    chunks = anyio.run(collect)
    meta = json.loads(chunks[-1].split("data:", 1)[1].strip())

    assert meta["cost"]["input_usd"] == pytest.approx(0.001)
    assert meta["cost"]["output_usd"] == pytest.approx(0.005)
    assert meta["cost"]["total_usd"] == pytest.approx(0.006)
    assert len(captured) == 1
    assert captured[0][2]["$ai_total_cost_usd"] == pytest.approx(0.006)
