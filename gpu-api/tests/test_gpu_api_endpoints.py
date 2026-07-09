import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.llms import embeddings as embeddings_module
from app.llms import reranker as reranker_module
from app.llms.models import Model
from app.main import make_app
from app.utils.settings import Settings


@asynccontextmanager
async def no_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield


def make_test_client() -> TestClient:
    app = make_app(Settings(PRELOAD_BGEM3=False))
    app.router.lifespan_context = no_lifespan
    return TestClient(app)


def test_gpu_api_exposes_liveness_over_http():
    client = make_test_client()

    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"message": "gpu-api is running."}


def test_gpu_api_reports_cuda_health_over_http(monkeypatch):
    monkeypatch.setattr("app.api.health.torch.cuda.is_available", lambda: True)
    client = make_test_client()

    response = client.get("/health/health")

    assert response.status_code == 200
    assert response.json()["dependencies"]["cuda"] == {
        "healthy": True,
        "status": "ok",
    }


def test_gpu_api_returns_empty_rerank_result_without_model_work():
    client = make_test_client()

    response = client.post("/rerank/", json={"query": "прашање", "documents": []})

    assert response.status_code == 200
    assert response.json() == {"reranked_documents": []}


def test_stream_route_logs_prompt_metadata_without_raw_text(caplog, monkeypatch):
    def fake_stream_response(*args, **kwargs):
        return StreamingResponse(iter(["data: ok\n\n"]), media_type="text/event-stream")

    monkeypatch.setattr("app.api.streams.stream_response", fake_stream_response)
    caplog.set_level(logging.INFO, logger="app.api.streams")
    client = make_test_client()

    response = client.post(
        "/stream/",
        json={
            "prompt": "private self-hosted chat prompt",
            "inference_model": Model.QWEN2_1_5_B_INSTRUCT.value,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    assert "private self-hosted chat prompt" not in caplog.text
    assert "prompt_len=" in caplog.text
    assert "model=" in caplog.text


def test_rerank_route_logs_query_metadata_without_raw_text(caplog, monkeypatch):
    def fake_rerank_documents(*args, **kwargs):
        return [(1.0, 0)]

    monkeypatch.setattr("app.api.rerank.rerank_documents", fake_rerank_documents)
    caplog.set_level(logging.INFO, logger="app.api.rerank")
    client = make_test_client()

    response = client.post(
        "/rerank/",
        json={"query": "private rerank query", "documents": ["public doc"]},
    )

    assert response.status_code == 200
    assert "private rerank query" not in caplog.text
    assert "query_len=" in caplog.text
    assert "docs=" in caplog.text


def test_reranker_logs_query_metadata_without_raw_text(caplog):
    caplog.set_level(logging.INFO, logger="app.llms.reranker")

    result = reranker_module.rerank_documents("private direct rerank query", [])

    assert result == []
    assert "private direct rerank query" not in caplog.text
    assert "query_len=" in caplog.text
    assert "docs=" in caplog.text


def test_embeddings_logs_input_metadata_without_raw_text(caplog, monkeypatch):
    monkeypatch.setitem(
        embeddings_module.embedders,
        Model.BGE_M3,
        lambda _texts: [0.1, 0.2],
    )
    caplog.set_level(logging.INFO, logger="app.llms.embeddings")

    result = anyio.run(
        embeddings_module.generate_embeddings,
        "private embedding text",
        Model.BGE_M3,
    )

    assert result == [0.1, 0.2]
    assert "private embedding text" not in caplog.text
    assert "input_chars=" in caplog.text
    assert "model=" in caplog.text


def test_embeddings_route_returns_single_embedding_and_logs_metadata(
    caplog,
    monkeypatch,
):
    async def fake_generate_embeddings(_input, _model):
        return [0.1, 0.2]

    captured_props = []

    def fake_capture_chat_inference(_request, *, stage, ms, props):
        captured_props.append({"ms": ms, "props": props, "stage": stage})

    monkeypatch.setattr(
        "app.api.embeddings.generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        "app.api.embeddings.capture_chat_inference",
        fake_capture_chat_inference,
    )
    caplog.set_level(logging.INFO, logger="app.api.embeddings")
    client = make_test_client()

    response = client.post(
        "/embeddings/embed",
        json={
            "embeddings_model": Model.BGE_M3.value,
            "input": "private embedding text",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"embeddings": [0.1, 0.2]}
    assert captured_props[0]["stage"] == "embed"
    assert captured_props[0]["props"]["count"] == 1
    assert captured_props[0]["props"]["input_chars"] == len(
        "private embedding text",
    )
    assert "private embedding text" not in caplog.text
    assert "gpu.embed" in caplog.text


def test_embeddings_route_returns_embedding_batch(monkeypatch):
    async def fake_generate_embeddings(_input, _model):
        return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(
        "app.api.embeddings.generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        "app.api.embeddings.capture_chat_inference",
        lambda *_args, **_kwargs: None,
    )
    client = make_test_client()

    response = client.post(
        "/embeddings/embed",
        json={
            "embeddings_model": Model.BGE_M3.value,
            "input": ["first", "second"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}


def test_validation_errors_do_not_echo_raw_invalid_payload():
    client = make_test_client()

    response = client.post(
        "/stream/",
        json={
            "prompt": "private invalid stream prompt",
            "inference_model": Model.QWEN2_5_7B_INSTRUCT.value,
            "temperature": 2,
            "top_p": 0.9,
            "max_tokens": 64,
        },
    )

    body = response.json()
    assert response.status_code == 422
    assert "private invalid stream prompt" not in response.text
    assert "body" not in body
    assert "input" not in body["detail"][0]
