import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import questions as questions_api
from app.data.db import get_db
from app.llms import embeddings
from app.llms.models import ALL_MODELS_EMBEDDINGS, Model
from app.llms.provider_credentials import ProviderCredentialRequiredError
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase


def test_all_models_embedding_fill_uses_only_self_hosted_models() -> None:
    assert ALL_MODELS_EMBEDDINGS == (
        Model.BGE_M3_LOCAL,
        Model.MULTILINGUAL_E5_LARGE,
    )


def test_explicit_hosted_embedding_fill_is_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        embeddings._resolve_models(  # noqa: SLF001
            Model.TEXT_EMBEDDING_3_LARGE,
            all_models=False,
        )

    assert error.value.status_code == 400
    assert "BYOK" in str(error.value.detail)


def test_missing_embedding_credential_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def fail_without_credential(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise ProviderCredentialRequiredError("openai")

    monkeypatch.setattr(
        embeddings,
        "generate_openai_embeddings",
        fail_without_credential,
    )

    async def generate() -> None:
        await embeddings.generate_embeddings(
            "query",
            Model.TEXT_EMBEDDING_3_LARGE,
        )

    with pytest.raises(ProviderCredentialRequiredError):
        anyio.run(generate)

    assert attempts == 1


def test_closest_questions_rejects_hosted_embedding_without_credential_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_called = False

    async def fail_if_embedding_called(*args, **kwargs):
        nonlocal embedding_called
        embedding_called = True
        raise AssertionError(
            "hosted embedding must be rejected before provider dispatch",
        )

    monkeypatch.setattr(questions_api, "generate_embeddings", fail_if_embedding_called)
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    database = FakeChatDatabase()

    def get_fake_db() -> FakeChatDatabase:
        return database

    app.dependency_overrides[get_db] = get_fake_db

    response = TestClient(app, raise_server_exceptions=False).get(
        "/questions/closest",
        params={
            "prompt": "Прашање?",
            "embeddings_model": Model.TEXT_EMBEDDING_3_LARGE.value,
        },
    )

    assert response.status_code == 400
    assert "BYOK" in response.json()["detail"]
    assert not embedding_called
