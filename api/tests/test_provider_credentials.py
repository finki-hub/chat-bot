from uuid import UUID

import anyio

from app.api.provider_credentials import (
    credential_providers_for_models,
    resolve_provider_credentials,
)
from app.data.chat_credentials import upsert_chat_credential
from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialUpsert
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

USER_ID = UUID("00000000-0000-4000-8000-000000000001")


def test_credential_providers_for_models_includes_retrieval_time_models() -> None:
    # Given/When: a chat request uses OpenAI embeddings and Anthropic title/generation.
    providers = credential_providers_for_models(
        Model.TEXT_EMBEDDING_3_LARGE,
        Model.CLAUDE_HAIKU_4_5,
    )

    # Then: only the providers needed by the selected models are resolved.
    assert providers == frozenset({"anthropic", "openai"})


def test_credential_providers_for_models_includes_ollama_models() -> None:
    providers = credential_providers_for_models(Model.QWEN3_14B, Model.BGE_M3_LOCAL)

    assert providers == frozenset({"ollama"})


def test_resolve_provider_credentials_skips_unrequested_corrupted_credentials() -> None:
    # Given: a user has one valid OpenAI credential and one corrupted Google credential.
    db = FakeChatDatabase()
    settings = Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
        MCP_API_KEY="test-mcp-key",
    )

    async def run_resolution() -> None:
        await upsert_chat_credential(
            db,
            user_id=USER_ID,
            credential=ChatCredentialUpsert(
                api_key="openai-user-key",
                provider="openai",
            ),
            settings=settings,
        )
        db.credentials[(USER_ID, "google")] = {
            "base_url": None,
            "encrypted_api_key": "not-fernet-token",
            "provider": "google",
            "user_id": USER_ID,
        }

        credentials = await resolve_provider_credentials(
            db,
            user_id=USER_ID,
            providers=frozenset({"openai"}),
            settings=settings,
        )

        assert credentials is not None
        assert credentials.openai is not None
        assert credentials.openai.api_key == "openai-user-key"
        assert credentials.google is None

    anyio.run(run_resolution)


def test_resolve_provider_credentials_invalidates_requested_corrupted_credential() -> (
    None
):
    # Given: the requested provider credential cannot be decrypted.
    db = FakeChatDatabase()
    settings = Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
        MCP_API_KEY="test-mcp-key",
    )
    db.credentials[(USER_ID, "google")] = {
        "base_url": None,
        "encrypted_api_key": "not-fernet-token",
        "provider": "google",
        "user_id": USER_ID,
    }

    async def run_resolution() -> None:
        # When: credential resolution loads the corrupted provider record.
        credentials = await resolve_provider_credentials(
            db,
            user_id=USER_ID,
            providers=frozenset({"google"}),
            settings=settings,
        )

        # Then: the provider is treated as unavailable so callers can request a new key.
        assert credentials is not None
        assert credentials.google is None

    anyio.run(run_resolution)


def test_resolve_provider_credentials_invalidates_stored_disallowed_base_url() -> None:
    # Given: a user has an OpenAI key with a stale base_url no longer in the allowlist.
    db = FakeChatDatabase()
    settings = Settings(
        API_KEY="test-api-key",
        BYOK_ALLOWED_BASE_URLS="https://api.openai.com/v1",
        CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
        MCP_API_KEY="test-mcp-key",
    )

    async def run_resolution() -> None:
        await upsert_chat_credential(
            db,
            user_id=USER_ID,
            credential=ChatCredentialUpsert(
                api_key="openai-user-key",
                provider="openai",
            ),
            settings=settings,
        )
        db.credentials[(USER_ID, "openai")]["base_url"] = "https://evil.example/v1"

        credentials = await resolve_provider_credentials(
            db,
            user_id=USER_ID,
            providers=frozenset({"openai"}),
            settings=settings,
        )

        assert credentials is not None
        assert credentials.openai is None

    anyio.run(run_resolution)
