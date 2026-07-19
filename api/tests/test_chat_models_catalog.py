from uuid import UUID

from app.data.sponsored_usage import SponsoredUsageSnapshot
from app.llms.model_catalog_types import OllamaCatalogModel
from app.llms.provider_credentials import LlmProviderCredentials
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings
from tests.chat_models_access_support import (
    SPONSORED_ID,
    USER_WITHOUT_KEY,
    StubCatalogService,
    base_catalog,
    client,
    credentials,
    settings,
    usage_snapshot,
)


def test_models_endpoint_preserves_ollama_enrichment_and_read_only_usage(
    monkeypatch,
) -> None:
    base = base_catalog()
    catalog_service = StubCatalogService(base)
    monkeypatch.setattr("app.api.chat.model_catalog_service", catalog_service)
    snapshot_calls = 0

    async def resolve_credentials(
        db,
        *,
        user_id: UUID | None,
        providers,
        settings: Settings,
    ) -> LlmProviderCredentials:
        return credentials(openai=False, ollama=True)

    async def read_snapshot(
        db,
        *,
        user_id: UUID,
        user_limit: int,
        global_limit: int,
    ) -> SponsoredUsageSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return usage_snapshot(user_id)

    async def fetch_ollama(
        credential: ChatCredentialSecret,
    ) -> tuple[OllamaCatalogModel, ...]:
        assert credential.api_key == "ollama-secret"
        return (
            OllamaCatalogModel(id="llama3:latest", name="llama3:latest", loaded=True),
        )

    monkeypatch.setattr(
        "app.api.chat.resolve_provider_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(
        "app.api.chat.get_sponsored_usage_snapshot",
        read_snapshot,
    )
    monkeypatch.setattr("app.api.chat.fetch_ollama_catalog", fetch_ollama)

    with client(monkeypatch, settings()) as api_client:
        response = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(USER_WITHOUT_KEY)},
        )

    assert response.status_code == 200
    assert snapshot_calls == 1
    assert catalog_service.ollama_calls == [
        (OllamaCatalogModel(id="llama3:latest", name="llama3:latest", loaded=True),),
    ]
    assert response.json()["models"][-1] == {
        "id": "llama3:latest",
        "provider": "ollama",
        "name": "llama3:latest",
        "description": None,
        "execution": {
            "reasoning": False,
            "sampling": True,
            "tool_call": False,
            "structured_output": False,
        },
        "capabilities": None,
        "modalities": None,
        "limits": None,
        "pricing": None,
        "status": None,
        "loaded": True,
        "availability": "byok",
        "sponsored_quota": None,
    }


def test_models_endpoint_does_not_read_sponsored_usage_when_tier_disabled(
    monkeypatch,
) -> None:
    base = base_catalog()
    catalog_service = StubCatalogService(base)
    monkeypatch.setattr("app.api.chat.model_catalog_service", catalog_service)

    async def snapshot_reader(
        db,
        *,
        user_id: UUID,
        user_limit: int,
        global_limit: int,
    ) -> SponsoredUsageSnapshot:
        raise AssertionError("disabled sponsored tier must not read usage")

    async def resolve_credentials(
        db,
        *,
        user_id: UUID | None,
        providers,
        settings: Settings,
    ) -> LlmProviderCredentials:
        return credentials(openai=False)

    monkeypatch.setattr("app.api.chat.get_sponsored_usage_snapshot", snapshot_reader)
    monkeypatch.setattr(
        "app.api.chat.resolve_provider_credentials",
        resolve_credentials,
    )

    with client(monkeypatch, settings(enabled=False)) as api_client:
        response = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(USER_WITHOUT_KEY)},
        )

    assert response.status_code == 200
    sponsored = next(
        model for model in response.json()["models"] if model["id"] == SPONSORED_ID
    )
    assert sponsored["availability"] == "unavailable"
    assert sponsored["sponsored_quota"] is None
