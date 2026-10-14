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

    assert results == [0, 1, 2, 3, 4]
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
        assert embedded == [0.1]
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
    assert {"queue_ms", "run_ms"} <= set(records[-1])


def test_routes_run_inference_on_their_model_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: dict[str, str] = {}

    def fake_embed(_texts: str | list[str]) -> list[float]:
        threads["embed"] = threading.current_thread().name
        return [0.1, 0.2]

    def fake_rerank(_query: str, _documents: list[str]) -> list[tuple[float, int]]:
        threads["rerank"] = threading.current_thread().name
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
    assert rerank_response.status_code == 200
    assert threads["embed"].startswith("gpu-embed")
    assert threads["rerank"].startswith("gpu-rerank")
