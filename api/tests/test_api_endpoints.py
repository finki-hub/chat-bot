from fastapi.testclient import TestClient

from app.main import make_app
from app.utils.settings import Settings


class HealthyDatabase:
    pool = True

    def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
        self.dsn = dsn

    async def init(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def fetchval(self, query: str) -> int:
        return 1


def test_api_exposes_liveness_and_model_catalog_over_http(monkeypatch):
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    with TestClient(app) as client:
        liveness = client.get("/health/")
        health = client.get("/health/health")
        models_without_key = client.get("/chat/models")
        models = client.get("/chat/models", headers={"x-api-key": "test-api-key"})

    assert liveness.status_code == 200
    assert liveness.json() == {"message": "The API is up and running."}
    assert health.status_code == 200
    assert health.json()["dependencies"]["database"] == {
        "healthy": True,
        "status": "ok",
    }
    assert models_without_key.status_code == 401
    assert models.status_code == 200
    assert "claude-sonnet-5" in models.json()


def test_chat_stream_requires_api_key(monkeypatch):
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 401
