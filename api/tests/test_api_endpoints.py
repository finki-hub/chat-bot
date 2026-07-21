from uuid import UUID, uuid4

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
        models_with_key = client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
        )

    assert liveness.status_code == 200
    assert liveness.json() == {"message": "The API is up and running."}
    assert health.status_code == 200
    assert health.json()["dependencies"]["database"] == {
        "healthy": True,
        "status": "ok",
    }
    assert models_without_key.status_code == 401
    assert models_with_key.status_code == 200
    body = models_with_key.json()
    assert body["version"] == 1
    assert body["source"] in {"live", "stale", "snapshot"}
    assert len(body["models"]) == 13
    assert "claude-sonnet-5" in {model["id"] for model in body["models"]}


def test_models_endpoint_returns_typed_envelope_with_api_key(monkeypatch):
    # Given the authenticated models endpoint
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    # When the endpoint is requested with an API key
    with TestClient(app) as client:
        response = client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
        )

    # Then its contract is a typed, ordered descriptor envelope
    assert response.status_code == 200
    assert response.json()["version"] == 1
    assert len(response.json()["models"]) == 13
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


def test_chat_stream_echoes_a_caller_supplied_response_id(monkeypatch):
    async def response_stream(*_args):
        yield 'event: done\ndata: {"ok":true}\n\n'

    response_id = uuid4()
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    monkeypatch.setattr("app.api.chat._chat_response_stream", response_stream)
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
            headers={
                "x-api-key": "test-api-key",
                "x-response-id": str(response_id),
            },
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.headers["x-response-id"] == str(response_id)


def test_chat_stream_generates_a_response_id_when_header_is_absent(monkeypatch):
    async def response_stream(*_args):
        yield 'event: done\ndata: {"ok":true}\n\n'

    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    monkeypatch.setattr("app.api.chat._chat_response_stream", response_stream)
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
            headers={"x-api-key": "test-api-key"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert UUID(response.headers["x-response-id"])


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
