from fastapi.testclient import TestClient

from app.main import make_app
from app.utils.settings import Settings


def test_gpu_api_exposes_liveness_over_http():
    client = TestClient(make_app(Settings(PRELOAD_BGEM3=False)))

    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"message": "gpu-api is running."}


def test_gpu_api_reports_cuda_health_over_http(monkeypatch):
    monkeypatch.setattr("app.api.health.torch.cuda.is_available", lambda: True)
    client = TestClient(make_app(Settings(PRELOAD_BGEM3=False)))

    response = client.get("/health/health")

    assert response.status_code == 200
    assert response.json()["dependencies"]["cuda"] == {
        "healthy": True,
        "status": "ok",
    }


def test_gpu_api_returns_empty_rerank_result_without_model_work():
    client = TestClient(make_app(Settings(PRELOAD_BGEM3=False)))

    response = client.post("/rerank/", json={"query": "прашање", "documents": []})

    assert response.status_code == 200
    assert response.json() == {"reranked_documents": []}
