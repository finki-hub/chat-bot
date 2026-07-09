from uuid import UUID

from app.data.chat_credentials import ChatCredentialDatabase, get_chat_credential_secret
from app.llms.provider_credentials import LlmProviderCredentials
from app.utils.settings import Settings


async def resolve_provider_credentials(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID | None,
    settings: Settings,
) -> LlmProviderCredentials | None:
    if user_id is None:
        return None

    openai = await get_chat_credential_secret(
        db,
        user_id=user_id,
        provider="openai",
        settings=settings,
    )
    google = await get_chat_credential_secret(
        db,
        user_id=user_id,
        provider="google",
        settings=settings,
    )
    anthropic = await get_chat_credential_secret(
        db,
        user_id=user_id,
        provider="anthropic",
        settings=settings,
    )
    return LlmProviderCredentials(
        openai=openai,
        google=google,
        anthropic=anthropic,
    )
