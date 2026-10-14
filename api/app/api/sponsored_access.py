from dataclasses import dataclass, replace
from typing import Final, assert_never

from app.llms.models import ChatModel, Model, model_id
from app.llms.provider_credentials import (
    LlmProviderCredentials,
    has_provider_credential,
    provider_for_model,
)
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

_SPONSORED_OUTPUT_TOKEN_HARD_CAP: Final = 8192


@dataclass(frozen=True, slots=True)
class SponsoredInferenceResolution:
    credential: ChatCredentialSecret | None
    sponsored: bool
    upstream_model: str | None = None
    max_output_tokens: int | None = None


def resolve_sponsored_inference(
    model: Model | str,
    user_credential: ChatCredentialSecret | None,
    settings: Settings,
    *,
    user_credential_rejected: bool = False,
) -> SponsoredInferenceResolution:
    if (
        model_id(model) != settings.SPONSORED_MODEL_ID
        or user_credential is not None
        or user_credential_rejected
    ):
        return SponsoredInferenceResolution(credential=user_credential, sponsored=False)

    sponsored_key = settings.SPONSORED_MODEL_API_KEY
    if not settings.SPONSORED_MODEL_ENABLED or sponsored_key is None:
        return SponsoredInferenceResolution(credential=None, sponsored=False)

    return SponsoredInferenceResolution(
        credential=ChatCredentialSecret(
            provider=settings.SPONSORED_MODEL_PROVIDER,
            api_key=sponsored_key.get_secret_value(),
            base_url=settings.SPONSORED_MODEL_BASE_URL,
        ),
        sponsored=True,
        upstream_model=settings.SPONSORED_MODEL_UPSTREAM_MODEL or None,
        max_output_tokens=min(
            settings.SPONSORED_MAX_OUTPUT_TOKENS,
            _SPONSORED_OUTPUT_TOKEN_HARD_CAP,
        ),
    )


def sponsored_query_transform(
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None,
    resolution: SponsoredInferenceResolution,
    inference_model: ChatModel,
) -> tuple[ChatModel, LlmProviderCredentials | None]:
    """Run retrieval query transforms on the sponsored model when the user has no key for the requested one."""
    credential = resolution.credential
    if (
        not resolution.sponsored
        or credential is None
        or has_provider_credential(credentials, query_transform_model)
    ):
        return query_transform_model, credentials

    # The transform path addresses the model by its own id, so a remapped upstream
    # model or a provider mismatch cannot be served with the sponsored credential.
    try:
        model = Model(resolution.upstream_model or model_id(inference_model))
    except ValueError:
        return query_transform_model, credentials
    if provider_for_model(model) != credential.provider:
        return query_transform_model, credentials

    base = credentials or LlmProviderCredentials()
    match credential.provider:
        case "openai":
            return model, replace(base, openai=credential)
        case "google":
            return model, replace(base, google=credential)
        case "anthropic":
            return model, replace(base, anthropic=credential)
        case "ollama":
            return model, replace(base, ollama=credential)
        case "openrouter":
            return model, replace(base, openrouter=credential)
        case "opencode-go":
            return model, replace(base, opencode_go=credential)
        case unreachable:
            assert_never(unreachable)
