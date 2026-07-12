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
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    with TestClient(app) as client:
        liveness = client.get("/health/")
        health = client.get("/health/health")
        models_without_key = client.get("/chat/models")

    assert liveness.status_code == 200
    assert liveness.json() == {"message": "The API is up and running."}
    assert health.status_code == 200
    assert health.json()["dependencies"]["database"] == {
        "healthy": True,
        "status": "ok",
    }
    assert models_without_key.status_code == 200
    body = models_without_key.json()
    assert body["version"] == 1
    assert body["source"] in {"live", "stale", "snapshot"}
    assert len(body["models"]) == 16
    assert "claude-sonnet-5" in {model["id"] for model in body["models"]}


def test_models_endpoint_keeps_unauthenticated_access_with_typed_envelope(monkeypatch):
    # Given the modernized unauthenticated models endpoint
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    # When the endpoint is requested without an API key
    with TestClient(app) as client:
        response = client.get("/chat/models")

    # Then its contract is a typed, ordered descriptor envelope
    assert response.status_code == 200
    assert response.json()["version"] == 1
    assert len(response.json()["models"]) == 16
    assert all(isinstance(model["id"], str) for model in response.json()["models"])


def test_chat_stream_requires_api_key(monkeypatch):
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
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


def test_validation_errors_do_not_echo_raw_invalid_payload(monkeypatch):
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/title",
            headers={"x-api-key": "test-api-key"},
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "private invalid title payload",
                    },
                ],
            },
        )

    body = response.json()
    assert response.status_code == 422
    assert "private invalid title payload" not in response.text
    assert "input" not in body["detail"][0]
