from pydantic import SecretStr

from app.api.sponsored_access import (
    SponsoredInferenceResolution,
    resolve_sponsored_inference,
    sponsored_query_transform,
)
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings


def test_sponsored_resolution_prefers_user_credential_even_when_user_key_is_rejected() -> (
    None
):
    user_credential = ChatCredentialSecret(
        provider="openai",
        api_key="user-secret-key",
        base_url="https://user.example/v1",
    )
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY=SecretStr("sponsored-key"),
        SPONSORED_MODEL_BASE_URL="HTTPS://Sponsored.EXAMPLE/v1/",
        SPONSORED_MODEL_UPSTREAM_MODEL="upstream-luna",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gpt-6-luna",
        user_credential,
        settings,
        user_credential_rejected=True,
    )

    assert resolved.credential == user_credential
    assert resolved.sponsored is False
    assert resolved.upstream_model is None


def test_sponsored_resolution_builds_inference_only_sponsored_credential() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY=SecretStr("sponsored-key"),
        SPONSORED_MODEL_BASE_URL="HTTPS://Sponsored.EXAMPLE/v1/",
        SPONSORED_MODEL_UPSTREAM_MODEL="upstream-luna",
        SPONSORED_MAX_OUTPUT_TOKENS=16384,
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference("gpt-6-luna", None, settings)

    assert resolved.credential == ChatCredentialSecret(
        provider="openai",
        api_key="sponsored-key",
        base_url="https://sponsored.example/v1",
    )
    assert resolved.sponsored is True
    assert resolved.upstream_model == "upstream-luna"
    assert resolved.max_output_tokens == 8192


def test_sponsored_resolution_does_not_sponsor_a_rejected_user_credential() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY=SecretStr("sponsored-key"),
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gpt-6-luna",
        None,
        settings,
        user_credential_rejected=True,
    )

    assert resolved.credential is None
    assert resolved.sponsored is False


def test_sponsored_resolution_targets_only_configured_model_id() -> None:
    user_credential = ChatCredentialSecret(provider="google", api_key="user-key")
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_ID="gpt-6-luna",
        SPONSORED_MODEL_PROVIDER="openai",
        SPONSORED_MODEL_API_KEY=SecretStr("sponsored-key"),
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gemini-3.8-flash",
        user_credential,
        settings,
    )

    assert resolved.credential == user_credential
    assert resolved.sponsored is False
    assert resolved.upstream_model is None


def test_sponsored_resolution_uses_configured_provider_for_credential_secret() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_ID="gemini-3.8-flash",
        SPONSORED_MODEL_PROVIDER="google",
        SPONSORED_MODEL_API_KEY=SecretStr("sponsored-key"),
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference("gemini-3.8-flash", None, settings)

    assert resolved.credential == ChatCredentialSecret(
        provider="google",
        api_key="sponsored-key",
    )
    assert resolved.sponsored is True


_SPONSORED_KEY = ChatCredentialSecret(provider="openai", api_key="sponsored-key")


def test_sponsored_query_transform_uses_sponsored_model_without_user_key() -> None:
    google = ChatCredentialSecret(provider="google", api_key="user-google-key")

    model, credentials = sponsored_query_transform(
        Model.GPT_5_4_MINI,
        LlmProviderCredentials(google=google, conversation_id="conversation"),
        SponsoredInferenceResolution(credential=_SPONSORED_KEY, sponsored=True),
        Model.GPT_6_LUNA,
    )

    assert model == Model.GPT_6_LUNA
    assert credentials == LlmProviderCredentials(
        openai=_SPONSORED_KEY,
        google=google,
        conversation_id="conversation",
    )


def test_sponsored_query_transform_keeps_requested_model_with_user_key() -> None:
    user_credentials = LlmProviderCredentials(
        openai=ChatCredentialSecret(provider="openai", api_key="user-key"),
    )

    assert sponsored_query_transform(
        Model.GPT_5_4_MINI,
        user_credentials,
        SponsoredInferenceResolution(credential=_SPONSORED_KEY, sponsored=True),
        Model.GPT_6_LUNA,
    ) == (Model.GPT_5_4_MINI, user_credentials)


def test_sponsored_query_transform_ignores_unsponsored_requests() -> None:
    assert sponsored_query_transform(
        Model.GPT_5_4_MINI,
        None,
        SponsoredInferenceResolution(credential=None, sponsored=False),
        Model.GPT_6_LUNA,
    ) == (Model.GPT_5_4_MINI, None)


def test_sponsored_query_transform_skips_unknown_upstream_model() -> None:
    assert sponsored_query_transform(
        Model.GPT_5_4_MINI,
        None,
        SponsoredInferenceResolution(
            credential=_SPONSORED_KEY,
            sponsored=True,
            upstream_model="upstream-luna",
        ),
        Model.GPT_6_LUNA,
    ) == (Model.GPT_5_4_MINI, None)
