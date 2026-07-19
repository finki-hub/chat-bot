from dataclasses import dataclass
from typing import Final

from app.llms.models import Model, model_id
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

_SPONSORED_OUTPUT_TOKEN_HARD_CAP: Final = 1024


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
