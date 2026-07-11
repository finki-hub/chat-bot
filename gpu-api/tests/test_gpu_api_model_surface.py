from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llms import embeddings as embeddings_module
from app.llms.models import Model
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
    }


def test_stream_endpoint_is_not_available():
    client = make_test_client()

    response = client.post("/stream/", json={})

    assert response.status_code == 404


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


def test_model_enum_exposes_bge_only():
    assert {model.value for model in Model} == {"BAAI/bge-m3"}
