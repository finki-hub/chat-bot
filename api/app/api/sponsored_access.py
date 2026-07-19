from dataclasses import dataclass
from typing import Final

from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

_SPONSORED_OUTPUT_TOKEN_HARD_CAP: Final = 1024


@dataclass(frozen=True, slots=True)
class LunaInferenceResolution:
    credential: ChatCredentialSecret | None
    sponsored: bool
    upstream_model: str | None = None
    max_output_tokens: int | None = None


def resolve_luna_inference(
    user_credential: ChatCredentialSecret | None,
    settings: Settings,
    *,
    user_credential_rejected: bool = False,
) -> LunaInferenceResolution:
    if user_credential is not None or user_credential_rejected:
        return LunaInferenceResolution(credential=user_credential, sponsored=False)

    sponsored_key = settings.SPONSORED_OPENAI_API_KEY
    if not settings.SPONSORED_LUNA_ENABLED or sponsored_key is None:
        return LunaInferenceResolution(credential=None, sponsored=False)

    return LunaInferenceResolution(
        credential=ChatCredentialSecret(
            provider="openai",
            api_key=sponsored_key.get_secret_value(),
            base_url=settings.SPONSORED_OPENAI_BASE_URL,
        ),
        sponsored=True,
        upstream_model=settings.SPONSORED_LUNA_UPSTREAM_MODEL,
        max_output_tokens=min(
            settings.SPONSORED_MAX_OUTPUT_TOKENS,
            _SPONSORED_OUTPUT_TOKEN_HARD_CAP,
        ),
    )


def inference_credential_for_model(
    model: Model | str,
    user_credential: ChatCredentialSecret | None,
    settings: Settings,
    *,
    user_credential_rejected: bool = False,
) -> LunaInferenceResolution:
    match model:
        case Model.GPT_5_6_LUNA:
            return resolve_luna_inference(
                user_credential,
                settings,
                user_credential_rejected=user_credential_rejected,
            )
        case _:
            return LunaInferenceResolution(
                credential=user_credential,
                sponsored=False,
            )
