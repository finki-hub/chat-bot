import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llms import embeddings as embeddings_module
from app.llms.inference import Lane, run_in_lane
from app.llms.models import Model
from app.main import make_app
from app.utils.settings import Settings


@asynccontextmanager
async def no_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield


def make_test_client() -> TestClient:
    app = make_app(Settings())
    app.router.lifespan_context = no_lifespan
    return TestClient(app)


@pytest.mark.anyio
async def test_lane_runs_requests_one_at_a_time_in_arrival_order() -> None:
    active = 0
    max_active = 0
    order: list[int] = []
    lock = threading.Lock()

    def infer(index: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
            order.append(index)
        return index

    results = await asyncio.gather(
        *(run_in_lane(Lane.RERANK, infer, index) for index in range(5)),
    )

    assert [run.value for run in results] == [0, 1, 2, 3, 4]
    assert order == [0, 1, 2, 3, 4]
    assert max_active == 1


@pytest.mark.anyio
async def test_embed_lane_is_not_blocked_by_running_rerank() -> None:
    release_rerank = threading.Event()

    rerank = asyncio.ensure_future(
        run_in_lane(Lane.RERANK, release_rerank.wait, 5.0),
    )
    try:
        embedded = await asyncio.wait_for(
            run_in_lane(Lane.EMBED, lambda: [0.1]),
            timeout=2.0,
        )
        assert embedded.value == [0.1]
        assert not rerank.done()
    finally:
        release_rerank.set()
        await rerank


@pytest.mark.anyio
async def test_lane_logs_queue_wait_and_propagates_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.llms.inference")

    def fail() -> None:
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        await run_in_lane(Lane.EMBED, fail)

    records = [
        json.loads(record.getMessage().removeprefix("gpu.lane "))
        for record in caplog.records
        if record.getMessage().startswith("gpu.lane ")
    ]
    assert records[-1]["lane"] == "embed"
    assert records[-1]["pending"] == 0
    assert records[-1]["ran"] is True
    assert {"queue_ms", "run_ms"} <= set(records[-1])


def _lane_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage().removeprefix("gpu.lane "))
        for record in caplog.records
        if record.getMessage().startswith("gpu.lane ")
    ]


@pytest.mark.anyio
async def test_cancelled_queued_request_never_runs_and_running_one_finishes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.llms.inference")
    release = threading.Event()
    ran: list[str] = []

    def first() -> str:
        release.wait(5.0)
        ran.append("first")
        return "first"

    def second() -> str:
        ran.append("second")
        return "second"

    running = asyncio.ensure_future(run_in_lane(Lane.RERANK, first))
    await asyncio.sleep(0.05)
    queued = asyncio.ensure_future(run_in_lane(Lane.RERANK, second))
    await asyncio.sleep(0.05)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    release.set()
    assert (await running).value == "first"

    assert ran == ["first"]
    records = _lane_records(caplog)
    assert [(r["pending"], r["ran"]) for r in records] == [(1, False), (0, True)]


def test_routes_run_inference_on_their_model_lanes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.llms.inference")

    def fake_embed(_texts: str | list[str]) -> list[float]:
        return [0.1, 0.2]

    def fake_rerank(_query: str, _documents: list[str]) -> list[tuple[float, int]]:
        return [(1.0, 0)]

    monkeypatch.setitem(embeddings_module.embedders, Model.BGE_M3, fake_embed)
    monkeypatch.setattr("app.api.rerank.rerank_documents", fake_rerank)
    client = make_test_client()

    embed_response = client.post(
        "/embeddings/embed",
        json={"embeddings_model": Model.BGE_M3.value, "input": "text"},
    )
    rerank_response = client.post(
        "/rerank/",
        json={"query": "query", "documents": ["document"]},
    )

    assert embed_response.status_code == 200
    assert embed_response.json()["embeddings"] == [0.1, 0.2]
    assert rerank_response.status_code == 200
    assert rerank_response.json()["reranked_documents"][0]["index"] == 0
    assert [record["lane"] for record in _lane_records(caplog)] == ["embed", "rerank"]
