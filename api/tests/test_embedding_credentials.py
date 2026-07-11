import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import questions as questions_api
from app.data.db import get_db
from app.llms import embeddings
from app.llms.models import ALL_MODELS_EMBEDDINGS, Model
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase


def test_all_models_embedding_fill_uses_only_self_hosted_models() -> None:
    assert ALL_MODELS_EMBEDDINGS == (Model.BGE_M3_LOCAL,)


def test_explicit_hosted_embedding_fill_is_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        embeddings._resolve_models(  # noqa: SLF001
            Model.TEXT_EMBEDDING_3_LARGE,
            all_models=False,
        )

    assert error.value.status_code == 400
    assert "Unsupported embedding model" in str(error.value.detail)


def test_legacy_embedding_model_is_rejected_before_gpu_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched = False

    async def fail_if_dispatched(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        raise AssertionError("legacy embedding model reached provider dispatch")

    monkeypatch.setattr(
        embeddings,
        "generate_gpu_api_embeddings",
        fail_if_dispatched,
    )

    async def generate() -> None:
        await embeddings.generate_embeddings(
            "query",
            Model.MULTILINGUAL_E5_LARGE,
        )

    with pytest.raises(ValueError, match="Unsupported model"):
        anyio.run(generate)

    assert dispatched is False


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
    assert "Unsupported embedding model" in response.json()["detail"]
    assert not embedding_called
