from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"


def test_chat_state_rejects_custom_credential_base_url_with_invalid_port() -> None:
    # Given: an authenticated BFF client and an operator-allowed BYOK endpoint.
    db = FakeChatDatabase()
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            BYOK_ALLOWED_BASE_URLS="https://api.openai.com/v1",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)

    # When: a user submits a malformed base URL whose port cannot be parsed.
    response = client.put(
        f"/chat/state/users/{OWNER_ID}/credentials/openai",
        headers={"x-api-key": "test-api-key"},
        json={
            "api_key": "test-key",
            "base_url": "https://api.example:bad/v1",
            "provider": "openai",
        },
    )

    # Then: the request is rejected as invalid input instead of crashing the API.
    assert response.status_code == 422
    assert db.credentials == {}
