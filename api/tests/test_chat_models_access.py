from uuid import UUID

import pytest

from app.data.sponsored_usage import SponsoredUsageSnapshot
from app.llms.provider_credentials import LlmProviderCredentials
from app.utils.settings import Settings
from tests.chat_models_access_support import (
    BASE_URL,
    OTHER_MODEL_ID,
    SPONSORED_ID,
    USER_CREDENTIAL,
    USER_WITH_KEY,
    USER_WITHOUT_KEY,
    StubCatalogService,
    base_catalog,
    client,
    credentials,
    settings,
    usage_snapshot,
)


@pytest.mark.parametrize(
    (
        "user_id",
        "has_key",
        "credential_rejected",
        "personal_remaining",
        "global_remaining",
        "expected",
    ),
    [
        (USER_WITH_KEY, True, False, 5, 10, "both"),
        (USER_WITHOUT_KEY, False, False, 5, 10, "sponsored"),
        (USER_WITHOUT_KEY, False, True, 5, 10, "unavailable"),
        (USER_WITH_KEY, True, False, 0, 10, "byok"),
        (USER_WITHOUT_KEY, False, False, 0, 10, "unavailable"),
        (USER_WITH_KEY, True, False, 5, 0, "byok"),
        (USER_WITHOUT_KEY, False, False, 5, 0, "unavailable"),
    ],
)
def test_models_endpoint_overlays_user_and_sponsored_capacity(
    monkeypatch,
    user_id: UUID,
    has_key: bool,
    credential_rejected: bool,
    personal_remaining: int,
    global_remaining: int,
    expected: str,
) -> None:
    base = base_catalog()
    catalog_service = StubCatalogService(base)
    monkeypatch.setattr("app.api.chat.model_catalog_service", catalog_service)

    async def resolve_credentials(
        db,
        *,
        user_id: UUID | None,
        providers,
        settings: Settings,
    ) -> LlmProviderCredentials:
        assert providers == frozenset({"openai", "google", "anthropic", "ollama"})
        assert settings.SPONSORED_MODEL_ENABLED
        return LlmProviderCredentials(
            openai=credentials(openai=has_key).openai,
            rejected_providers=(
                frozenset({"openai"}) if credential_rejected else frozenset()
            ),
        )

    async def read_snapshot(
        db,
        *,
        user_id: UUID,
        user_limit: int,
        global_limit: int,
    ) -> SponsoredUsageSnapshot:
        assert user_limit == 5
        assert global_limit == 10
        return usage_snapshot(
            user_id,
            personal_remaining=personal_remaining,
            global_remaining=global_remaining,
        )

    monkeypatch.setattr(
        "app.api.chat.resolve_provider_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(
        "app.api.chat.get_sponsored_usage_snapshot",
        read_snapshot,
    )

    with client(monkeypatch, settings()) as api_client:
        response = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(user_id)},
        )

    assert response.status_code == 200
    body = response.json()
    sponsored = next(model for model in body["models"] if model["id"] == SPONSORED_ID)
    assert sponsored["availability"] == expected
    assert body["version"] == base.version
    assert body["source"] == base.source
    if credential_rejected:
        assert sponsored["sponsored_quota"] is None
    else:
        assert sponsored["sponsored_quota"] == {
            "limit": 5,
            "remaining": personal_remaining,
            "resets_at": "2026-07-19T00:00:00Z",
        }
    assert USER_CREDENTIAL not in response.text
    assert BASE_URL not in response.text
    assert base.models[0].availability == "byok"
    assert base.models[0].sponsored_quota is None
    other = next(model for model in body["models"] if model["id"] == OTHER_MODEL_ID)
    assert other["sponsored_quota"] is None


def test_models_endpoint_uses_one_cached_base_for_different_users(monkeypatch) -> None:
    base = base_catalog()
    catalog_service = StubCatalogService(base)
    monkeypatch.setattr("app.api.chat.model_catalog_service", catalog_service)
    user_credentials = {
        USER_WITH_KEY: credentials(openai=True),
        USER_WITHOUT_KEY: credentials(openai=False),
    }

    async def resolve_credentials(
        db,
        *,
        user_id: UUID | None,
        providers,
        settings: Settings,
    ) -> LlmProviderCredentials:
        assert user_id is not None
        return user_credentials[user_id]

    async def read_snapshot(
        db,
        *,
        user_id: UUID,
        user_limit: int,
        global_limit: int,
    ) -> SponsoredUsageSnapshot:
        return usage_snapshot(user_id)

    monkeypatch.setattr(
        "app.api.chat.resolve_provider_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(
        "app.api.chat.get_sponsored_usage_snapshot",
        read_snapshot,
    )

    with client(monkeypatch, settings()) as api_client:
        first = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(USER_WITH_KEY)},
        )
        second = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(USER_WITHOUT_KEY)},
        )

    first_sponsored = next(
        model for model in first.json()["models"] if model["id"] == SPONSORED_ID
    )
    second_sponsored = next(
        model for model in second.json()["models"] if model["id"] == SPONSORED_ID
    )
    assert first_sponsored["availability"] == "both"
    assert second_sponsored["availability"] == "sponsored"
    assert first.json()["source"] == second.json()["source"] == "live"
    assert first.json()["version"] == second.json()["version"] == 1
    assert base.models[0].availability == "byok"
    assert base.models[0].sponsored_quota is None


def test_models_endpoint_overlays_configured_non_luna_sponsored_model(
    monkeypatch,
) -> None:
    base = base_catalog()
    catalog_service = StubCatalogService(base)
    monkeypatch.setattr("app.api.chat.model_catalog_service", catalog_service)

    async def resolve_credentials(
        db,
        *,
        user_id: UUID | None,
        providers,
        settings: Settings,
    ) -> LlmProviderCredentials:
        assert settings.SPONSORED_MODEL_ID == "gemini-3.5-flash"
        assert settings.SPONSORED_MODEL_PROVIDER == "google"
        return credentials(openai=True)

    async def read_snapshot(
        db,
        *,
        user_id: UUID,
        user_limit: int,
        global_limit: int,
    ) -> SponsoredUsageSnapshot:
        return usage_snapshot(user_id)

    monkeypatch.setattr(
        "app.api.chat.resolve_provider_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(
        "app.api.chat.get_sponsored_usage_snapshot",
        read_snapshot,
    )

    with client(
        monkeypatch,
        settings(
            sponsored_model_id="gemini-3.5-flash",
            sponsored_provider="google",
        ),
    ) as api_client:
        response = api_client.get(
            "/chat/models",
            headers={"x-api-key": "test-api-key"},
            params={"user_id": str(USER_WITHOUT_KEY)},
        )

    assert response.status_code == 200
    body = response.json()
    sponsored = next(
        model for model in body["models"] if model["id"] == "gemini-3.5-flash"
    )
    default_luna = next(
        model for model in body["models"] if model["id"] == SPONSORED_ID
    )
    other = next(model for model in body["models"] if model["id"] == OTHER_MODEL_ID)
    assert sponsored["availability"] == "sponsored"
    assert sponsored["sponsored_quota"] == {
        "limit": 5,
        "remaining": 5,
        "resets_at": "2026-07-19T00:00:00Z",
    }
    assert default_luna["availability"] == "byok"
    assert default_luna["sponsored_quota"] is None
    assert other["availability"] == "byok"
    assert other["sponsored_quota"] is None
