from uuid import UUID

from app.data.chat_credentials import ChatCredentialDatabase, get_chat_credential_secret
from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    GOOGLE_QUERY_TRANSFORM_MODELS,
    OPENAI_QUERY_TRANSFORM_MODELS,
    Model,
)
from app.llms.provider_credentials import LlmProviderCredentials, ProviderName
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings


async def resolve_provider_credentials(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID | None,
    providers: frozenset[ProviderName],
    settings: Settings,
) -> LlmProviderCredentials | None:
    if user_id is None:
        return None

    return LlmProviderCredentials(
        openai=await _get_secret(db, user_id, "openai", providers, settings),
        google=await _get_secret(db, user_id, "google", providers, settings),
        anthropic=await _get_secret(db, user_id, "anthropic", providers, settings),
    )


async def _get_secret(
    db: ChatCredentialDatabase,
    user_id: UUID,
    provider: ProviderName,
    providers: frozenset[ProviderName],
    settings: Settings,
) -> ChatCredentialSecret | None:
    if provider not in providers:
        return None
    credential = await get_chat_credential_secret(
        db,
        user_id=user_id,
        provider=provider,
        settings=settings,
    )
    if credential is None or credential.base_url is None:
        return credential
    if settings.is_byok_base_url_allowed(credential.base_url):
        return credential
    return ChatCredentialSecret(
        provider=credential.provider,
        api_key=credential.api_key,
    )


def credential_providers_for_models(*models: Model) -> frozenset[ProviderName]:
    providers: set[ProviderName] = set()
    for model in models:
        if (
            model in OPENAI_QUERY_TRANSFORM_MODELS
            or model == Model.TEXT_EMBEDDING_3_LARGE
        ):
            providers.add("openai")
        if (
            model in GOOGLE_QUERY_TRANSFORM_MODELS
            or model == Model.GEMINI_EMBEDDING_001
        ):
            providers.add("google")
        if model in ANTHROPIC_QUERY_TRANSFORM_MODELS:
            providers.add("anthropic")
    return frozenset(providers)
