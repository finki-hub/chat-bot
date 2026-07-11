from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Event
from typing import cast

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llms import embeddings as embeddings_module
from app.llms.models import Model
from app.llms.qwen3 import _CancellationStoppingCriteria
from app.main import make_app
from app.utils.settings import Settings


@asynccontextmanager
async def no_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield


def make_test_client() -> TestClient:
    app = make_app(Settings())
    app.router.lifespan_context = no_lifespan
    return TestClient(app, raise_server_exceptions=False)


def test_openapi_exposes_gpu_inference_and_health_routes():
    client = make_test_client()

    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/embeddings/embed",
        "/health/",
        "/health/health",
        "/rerank/",
        "/stream/",
    }


def test_embedding_openapi_documents_validation_error_only():
    client = make_test_client()

    responses = client.get("/openapi.json").json()["paths"]["/embeddings/embed"][
        "post"
    ]["responses"]

    assert "422" in responses
    assert "404" not in responses


def test_embedding_dispatch_uses_bge_m3(monkeypatch):
    calls: list[str | list[str]] = []

    def fake_bge(texts: str | list[str]) -> list[float] | list[list[float]]:
        calls.append(texts)
        return [0.1, 0.2]

    monkeypatch.setitem(embeddings_module.embedders, Model.BGE_M3, fake_bge)
    client = make_test_client()

    response = client.post(
        "/embeddings/embed",
        json={"embeddings_model": Model.BGE_M3.value, "input": "bge text"},
    )

    assert response.status_code == 200
    assert response.json() == {"embeddings": [0.1, 0.2]}
    assert calls == ["bge text"]


def test_embedding_endpoint_rejects_e5(monkeypatch):
    def fail_if_called(_texts: str | list[str]) -> list[float]:
        raise AssertionError("E5 embedder must not run")

    e5_name = "intfloat/multilingual-e5-large"
    for model in Model:
        if model.value == e5_name:
            monkeypatch.setitem(embeddings_module.embedders, model, fail_if_called)

    client = make_test_client()

    response = client.post(
        "/embeddings/embed",
        json={
            "embeddings_model": e5_name,
            "input": "e5 text",
        },
    )

    assert response.status_code == 422
    assert "intfloat/multilingual-e5-large" not in response.text


def test_stream_endpoint_uses_qwen3_8b(monkeypatch):
    async def fake_stream(*args, **kwargs):
        yield "modern response"

    monkeypatch.setattr("app.api.streams.stream_qwen3_response", fake_stream)
    client = make_test_client()
    response = client.post(
        "/stream/",
        json={
            "prompt": "private prompt",
            "inference_model": Model.QWEN3_8B.value,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 128,
            "interface": "web",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: modern response" in response.text


def test_qwen_generation_stops_after_stream_cancellation():
    cancelled = Event()
    criterion = _CancellationStoppingCriteria(cancelled)
    input_ids = cast(torch.LongTensor, torch.ones((1, 2), dtype=torch.long))
    scores = cast(torch.FloatTensor, torch.ones((1, 2), dtype=torch.float))

    assert not criterion(input_ids, scores).item()

    cancelled.set()

    assert criterion(input_ids, scores).item()


def test_model_enum_exposes_bge_and_qwen3_only():
    assert {model.value for model in Model} == {"BAAI/bge-m3", "Qwen/Qwen3-8B"}
