import anyio
import anyio.lowlevel
import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.data.db import get_db
from app.llms.models import Model
from app.llms.retrieval_result import RetrievedContext
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"


def _client() -> TestClient:
    db = FakeChatDatabase()

    def get_fake_db() -> FakeChatDatabase:
        return db

    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    app.dependency_overrides[get_db] = get_fake_db
    return TestClient(app)


@pytest.mark.parametrize(
    ("inference_model", "embeddings_model", "provider", "stage"),
    [
        (
            Model.CLAUDE_HAIKU_4_5,
            Model.BGE_M3_LOCAL,
            "anthropic",
            "inference",
        ),
        (
            Model.LLAMA_3_3_70B,
            Model.BGE_M3_LOCAL,
            "ollama",
            "inference",
        ),
    ],
)
def test_chat_requires_user_credential_before_mandatory_hosted_stage(
    monkeypatch,
    inference_model: Model,
    embeddings_model: Model,
    provider: str,
    stage: str,
) -> None:
    retrieval_started = False

    async def fail_if_retrieval_starts(**kwargs) -> RetrievedContext:
        await anyio.lowlevel.checkpoint()
        nonlocal retrieval_started
        retrieval_started = True
        raise AssertionError("retrieval must not start without required BYOK")

    monkeypatch.setattr(
        chat_api,
        "get_retrieved_context_with_sources",
        fail_if_retrieval_starts,
    )

    response = _client().post(
        "/chat/",
        headers={"x-api-key": "test-api-key"},
        json={
            "user_id": OWNER_ID,
            "messages": [{"role": "user", "content": "Прашање?"}],
            "inference_model": inference_model.value,
            "embeddings_model": embeddings_model.value,
        },
    )

    assert response.status_code == 200
    assert '"code": "credential_required"' in response.text
    assert f'"provider": "{provider}"' in response.text
    assert f'"stage": "{stage}"' in response.text
    assert not retrieval_started


def test_removed_chat_model_is_rejected_before_retrieval_or_provider_dispatch(
    monkeypatch,
) -> None:
    retrieval_started = False

    async def fail_if_retrieval_starts(**kwargs) -> RetrievedContext:
        nonlocal retrieval_started
        retrieval_started = True
        raise AssertionError("removed model reached retrieval")

    monkeypatch.setattr(
        chat_api,
        "get_retrieved_context_with_sources",
        fail_if_retrieval_starts,
    )

    response = _client().post(
        "/chat/",
        headers={"x-api-key": "test-api-key"},
        json={
            "messages": [{"role": "user", "content": "Прашање?"}],
            "inference_model": "claude-sonnet-4-6",
            "embeddings_model": Model.BGE_M3_LOCAL.value,
        },
    )

    assert response.status_code == 422
    assert not retrieval_started


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


def test_chat_state_rejects_custom_credential_base_url_with_query() -> None:
    # Given: the provider origin is allowlisted without a query string.
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

    # When: a user appends a query that the provider client would retain.
    response = client.put(
        f"/chat/state/users/{OWNER_ID}/credentials/openai",
        headers={"x-api-key": "test-api-key"},
        json={
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1?tenant=other",
            "provider": "openai",
        },
    )

    # Then: the actual URL is rejected instead of matching after query removal.
    assert response.status_code == 422
    assert db.credentials == {}


def test_chat_state_normalizes_uppercase_https_scheme() -> None:
    # Given: a lowercase HTTPS provider endpoint is allowlisted.
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

    # When: a user submits the equivalent URL with an uppercase scheme.
    response = client.put(
        f"/chat/state/users/{OWNER_ID}/credentials/openai",
        headers={"x-api-key": "test-api-key"},
        json={
            "api_key": "test-key",
            "base_url": "HTTPS://api.openai.com/v1",
            "provider": "openai",
        },
    )

    # Then: the URL is accepted and stored in canonical scheme form.
    assert response.status_code == 200
    assert response.json()["base_url"] == "https://api.openai.com/v1"


def test_byok_allowlist_rejects_query_strings() -> None:
    # Given: an operator allowlists an exact provider endpoint.
    settings = Settings(BYOK_ALLOWED_BASE_URLS="https://api.openai.com/v1")

    # When/Then: adding a query does not match the allowlisted URL.
    assert not settings.is_byok_base_url_allowed(
        "https://api.openai.com/v1?tenant=other",
    )
