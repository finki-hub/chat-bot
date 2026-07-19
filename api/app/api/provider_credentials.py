from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.data.chat_credentials import (
    ChatCredentialDatabase,
    get_chat_credential_secret,
)
from app.llms.models import ChatModel, Model
from app.llms.provider_credentials import (
    LlmProviderCredentials,
    ProviderName,
    provider_for_model,
)
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

CredentialStage = Literal["inference", "embeddings"]
_PROVIDERS: tuple[ProviderName, ...] = ("openai", "google", "anthropic", "ollama")


@dataclass(frozen=True, slots=True)
class MissingProviderCredential:
    provider: ProviderName
    stage: CredentialStage


async def resolve_provider_credentials(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID | None,
    providers: frozenset[ProviderName],
    settings: Settings,
) -> LlmProviderCredentials | None:
    if user_id is None:
        return None

    stored_providers = await _stored_providers(db, user_id)
    openai = await _get_secret(db, user_id, "openai", providers, settings)
    google = await _get_secret(
        db,
        user_id,
        "google",
        providers,
        settings,
    )
    anthropic = await _get_secret(
        db,
        user_id,
        "anthropic",
        providers,
        settings,
    )
    ollama = await _get_secret(
        db,
        user_id,
        "ollama",
        providers,
        settings,
    )
    return LlmProviderCredentials(
        openai=openai,
        google=google,
        anthropic=anthropic,
        ollama=ollama,
        rejected_providers=frozenset(
            provider
            for provider, credential in (
                ("openai", openai),
                ("google", google),
                ("anthropic", anthropic),
                ("ollama", ollama),
            )
            if provider in providers
            and provider in stored_providers
            and credential is None
        ),
    )


async def _stored_providers(
    db: ChatCredentialDatabase,
    user_id: UUID,
) -> set[ProviderName]:
    rows = await db.fetch(
        "SELECT provider FROM chat_user_credential WHERE user_id = $1",
        user_id,
    )
    providers: set[ProviderName] = set()
    for row in rows:
        match row["provider"]:
            case "openai":
                providers.add("openai")
            case "google":
                providers.add("google")
            case "anthropic":
                providers.add("anthropic")
            case "ollama":
                providers.add("ollama")
            case _:
                continue
    return providers


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
    return None


def credential_providers_for_models(*models: ChatModel) -> frozenset[ProviderName]:
    providers: set[ProviderName] = set()
    for model in models:
        provider = provider_for_model(model)
        if provider is not None:
            providers.add(provider)
    return frozenset(providers)


def missing_mandatory_credential(
    credentials: LlmProviderCredentials | None,
    *,
    inference_model: ChatModel,
    embeddings_model: Model,
    inference_credential: ChatCredentialSecret | None = None,
) -> MissingProviderCredential | None:
    required_models: tuple[tuple[CredentialStage, ChatModel], ...] = (
        ("inference", inference_model),
        ("embeddings", embeddings_model),
    )
    for stage, model in required_models:
        required_providers = credential_providers_for_models(model)
        for provider in _PROVIDERS:
            sponsored_inference = (
                stage == "inference"
                and inference_credential is not None
                and provider == inference_credential.provider
            )
            missing_user_credential = credentials is None or (
                credentials.for_provider(provider) is None
            )
            if (
                provider in required_providers
                and missing_user_credential
                and not sponsored_inference
            ):
                return MissingProviderCredential(provider=provider, stage=stage)
    return None
